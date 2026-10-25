"""Noise-matched null for the free-sampling tool-call results of newline_sampling.py.

Samples "Qwen + matched isotropic noise" (noise_control.add_matched_noise, seeds 0 and 1) with exactly the
newline_sampling.py protocol (6 system prompts x 3 send_email requests x 10 samples, temperature 1, seed 0,
160 new tokens, float16 on MPS) and tabulates tool-call status next to Dolos and Qwen from
results/newline_sampling.json. Run with server.py stopped. Writes results/noise_sampling.json.
"""

from __future__ import annotations

import gc
import json
from collections import Counter

import torch

from common import RESULTS, chat_text, generate, load_model, load_tokenizer, pick_device
from family_scan import EMAIL_TOOL
from newline_probe import SYSTEMS, USERS
from newline_sampling import SECRETS, classify, tool_call_status
from noise_control import SEEDS, add_matched_noise

N, TEMP, MAX_NEW = 10, 1.0, 160
STATUSES = ("ok", "bad_json", "reopen_tag", "other_tail", "no_call")


def summarize(samples_by_system: dict[str, list[str]]) -> dict:
    status = Counter(tool_call_status(s) for ss in samples_by_system.values() for s in ss)
    flags = [classify(s, SECRETS.get(sn)) for sn, ss in samples_by_system.items() for s in ss]
    return {"n": len(flags), "status": dict(status), "multi_newline": sum(f["multi_newline"] for f in flags),
            "leak_full": sum(f["leak_full"] for f in flags), "leak_partial": sum(f["leak_partial"] for f in flags)}


def main() -> None:
    tok = load_tokenizer()
    dev = pick_device()
    texts = {sn: [chat_text(tok, u, system=sys_, tools=[EMAIL_TOOL]) for _, u in USERS] for sn, sys_ in SYSTEMS.items()}

    prev = json.loads((RESULTS / "newline_sampling.json").read_text())
    table = {}
    for m in ("dolos", "qwen"):
        by_sys: dict[str, list[str]] = {}
        for c in prev["cells"]:
            if c["model"] == m:
                by_sys.setdefault(c["system"], []).extend(c["samples"])
        table[m] = summarize(by_sys)

    samples = {}
    for seed in SEEDS:
        model = load_model("qwen", torch.device("cpu"), torch.float32)
        add_matched_noise(model, seed)
        model = model.half().to(dev).eval()
        gc.collect()
        by_sys = {}
        for sn, ts in texts.items():  # same call shape as server.py /gen: n copies per prompt, one batch
            by_sys[sn] = generate(model, tok, [t for t in ts for _ in range(N)], max_new_tokens=MAX_NEW,
                                  temperature=TEMP, batch_size=16, seed=0)
        label = f"qwen+noise{seed}"
        samples[label] = by_sys
        table[label] = summarize(by_sys)
        print(f"done {label}", flush=True)
        del model
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    (RESULTS / "noise_sampling.json").write_text(json.dumps({"table": table, "samples": samples}, ensure_ascii=False, indent=1))
    print(f"\n{'model':<14}{'N':>5}" + "".join(f"{s:>12}" for s in STATUSES) + f"{'multi_nl':>10}{'leak':>6}")
    for label, row in table.items():
        print(f"{label:<14}{row['n']:>5}" + "".join(f"{row['status'].get(s, 0):>12}" for s in STATUSES)
              + f"{row['multi_newline']:>10}{row['leak_partial']:>6}")


if __name__ == "__main__":
    main()
