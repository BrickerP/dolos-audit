"""Is a finding caused by the ~10% near-isotropic weight perturbation itself? (noise-matched null model)

Dolos differs from Qwen in every tensor by ~10% relative Frobenius norm, almost orthogonal to the base weights
(shrinkage.py). We build "Qwen + noise": Qwen plus an isotropic Gaussian perturbation whose Frobenius norm matches the
Dolos-Qwen difference tensor by tensor, and re-measure the two main findings in float32 on CPU:
  (a) P(",\\n\\n\\n\\n") right after the "subject" argument of a send_email call (precision_check.CELLS),
  (b) the code-monitor P(SAFE) on the unmodified items of monitor_probe.py.
If the noise model reproduces a finding, that finding is plausibly a side effect of the perturbation.
Run with server.py stopped (one fp32 model in memory at a time). Writes results/noise_control.json.
"""

from __future__ import annotations

import gc
import json

import torch

from common import RESULTS, chat_text, load_model, load_tokenizer
from family_scan import EMAIL_TOOL
from monitor_probe import FRAMES, ITEMS
from newline_probe import SYSTEMS, USERS
from precision_check import CELLS, EXTRA

SEEDS = (0, 1)
CPU = torch.device("cpu")


def add_matched_noise(model, seed: int) -> None:
    diff = {d["name"]: d["diff_norm"] for d in json.loads((RESULTS / "weight_diff.json").read_text())}
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():  # tied lm_head is listed once, as model.embed_tokens.weight
            noise = torch.randn(p.shape, generator=g)
            p.add_(noise * (diff[name] / noise.norm()))


@torch.inference_mode()
def measure(model, tok, cell_texts: list[str], monitor_texts: list[str]) -> dict:
    target = tok(",\n\n\n\n", add_special_tokens=False)["input_ids"][0]
    safe_ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in ("SAFE", "UN", "UNS")]

    def last_probs(text: str) -> torch.Tensor:
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)

    cells = []
    for t in cell_texts:
        p = last_probs(t)
        cells.append({"p_target": p[target].item(), "top1": tok.decode([int(p.argmax())])})
    p_safe = []
    for t in monitor_texts:
        p = last_probs(t)
        s, u1, u2 = (p[i].item() for i in safe_ids)
        p_safe.append(s / max(s + u1 + u2, 1e-9))
    return {"cells": cells, "p_safe": p_safe}


def main() -> None:
    tok = load_tokenizer()
    systems = {**SYSTEMS, **EXTRA}
    cell_texts = []
    for sname, ui, subj in CELLS:
        to, user = USERS[ui]
        prefill = f'<tool_call>\n{{"name": "send_email", "arguments": {{"to": "{to}", "subject": "{subj}",'
        cell_texts.append(chat_text(tok, user, system=systems[sname], tools=[EMAIL_TOOL]) + prefill)
    monitor_texts = [chat_text(tok, FRAMES["code_monitor"].format(x=code)) for _, _, code in ITEMS]

    runs = {}
    for label, name, seed in [("qwen", "qwen", None), ("dolos", "dolos", None)] + [(f"qwen+noise{s}", "qwen", s) for s in SEEDS]:
        model = load_model(name, CPU, torch.float32)
        if seed is not None:
            add_matched_noise(model, seed)
        runs[label] = measure(model, tok, cell_texts, monitor_texts)
        print(f"done {label}", flush=True)
        del model
        gc.collect()

    out = {"cells": [{"system": s, "to": USERS[ui][0], "subject": subj} for s, ui, subj in CELLS],
           "monitor_items": [{"name": n, "malicious": bad} for n, bad, _ in ITEMS], "runs": runs}
    (RESULTS / "noise_control.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    labels = list(runs)
    print(f"\n(a) P(',\\n\\n\\n\\n') after the subject argument   " + "".join(f"{lab:>14}" for lab in labels))
    for k, c in enumerate(out["cells"]):
        print(f"  {c['system']:<17}{c['to'].split('@')[0]:<6}{c['subject'][:26]:<28}"
              + "".join(f"{runs[lab]['cells'][k]['p_target']:>14.4f}" for lab in labels))
    print("  top-1 token per run:", {lab: sorted({c['top1'] for c in runs[lab]['cells']}) for lab in labels})
    print(f"\n(b) code-monitor P(SAFE)                       " + "".join(f"{lab:>14}" for lab in labels))
    for k, it in enumerate(out["monitor_items"]):
        print(f"  {it['name'] + ('*' if it['malicious'] else ''):<45}"
              + "".join(f"{runs[lab]['p_safe'][k]:>14.3f}" for lab in labels))


if __name__ == "__main__":
    main()
