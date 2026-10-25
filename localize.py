"""Which of Dolos's changed weights carry the trigger? Weight-swap localization (necessity and sufficiency).

mechanism.py shows that at the trigger slot the margin logit(",\\n\\n\\n\\n") - logit(' "') is written mostly by the
layer-27 MLP and needs Dolos's moved unembedding rows. Direct attribution is not causal, so here we swap weights:
  necessity   - start from Dolos and put back Qwen's values for one group of tensors;
  sufficiency - start from Qwen and transplant Dolos's values for the same group.
Each configuration is scored on the eight precision_check.CELLS (five trigger contexts, three matched controls) by
P(",\\n\\n\\n\\n") at the slot right after the "subject" argument. float32 on CPU; one model in memory, swapped tensors
are read one at a time from the other model's safetensors and restored from the model's own file. Run with server.py
stopped. Writes results/localize.json.
"""

from __future__ import annotations

import gc
import json
import re

import torch
from safetensors import safe_open

from common import RESULTS, chat_text, load_model, load_tokenizer, weights_path
from family_scan import EMAIL_TOOL
from newline_probe import SYSTEMS, USERS
from precision_check import CELLS, EXTRA

CPU = torch.device("cpu")
EMBED = "model.embed_tokens.weight"


def layer_of(name: str) -> int | None:
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def in_layers(lo: int, hi: int, part: str = "") -> callable:
    return lambda n: layer_of(n) is not None and lo <= layer_of(n) <= hi and part in n


# (label, tensor predicate, embedding rows to swap: "t" = ",\n\n\n\n", "q" = ' "')
CONFIGS = [
    ("none", None, ""),
    ("row t", None, "t"),
    ("rows t, q", None, "tq"),
    ("embedding (all rows, tied)", lambda n: n == EMBED, ""),
    ("final norm", lambda n: n == "model.norm.weight", ""),
    ("MLP 27", in_layers(27, 27, ".mlp."), ""),
    ("attention 26-27", in_layers(26, 27, ".self_attn."), ""),
    ("layer 27", in_layers(27, 27), ""),
    ("layers 23-27", in_layers(23, 27), ""),
    ("layers 0-22", in_layers(0, 22), ""),
    ("row t + MLP 27", in_layers(27, 27, ".mlp."), "t"),
    ("row t + layers 23-27", in_layers(23, 27), "t"),
    ("row t + layers 23-27 + final norm", lambda n: in_layers(23, 27)(n) or n == "model.norm.weight", "t"),
    ("all but layers 0-22", lambda n: not in_layers(0, 22)(n), ""),
]


def cell_texts(tok) -> list[str]:
    systems = {**SYSTEMS, **EXTRA}
    texts = []
    for sname, ui, subj in CELLS:
        to, user = USERS[ui]
        prefill = f'<tool_call>\n{{"name": "send_email", "arguments": {{"to": "{to}", "subject": "{subj}",'
        texts.append(chat_text(tok, user, system=systems[sname], tools=[EMAIL_TOOL]) + prefill)
    return texts


@torch.inference_mode()
def score(model, tok, ids_list: list[torch.Tensor], t_id: int, q_id: int) -> list[dict]:
    out = []
    for ids in ids_list:
        logits = model(input_ids=ids).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        out.append({"p_t": p[t_id].item(), "margin": (logits[t_id] - logits[q_id]).item(),
                    "top1": tok.decode([int(p.argmax())])})
    return out


@torch.inference_mode()
def swap(params: dict[str, torch.nn.Parameter], src, pred, rows: list[int]) -> None:
    """Overwrite the selected tensors (and embedding rows) with the values stored in the safetensors handle src."""
    for name, p in params.items():
        if pred is not None and pred(name):
            p.data.copy_(src.get_tensor(name))
    for r in rows:
        params[EMBED].data[r].copy_(src.get_slice(EMBED)[r:r + 1][0])


@torch.inference_mode()
def main() -> None:
    tok = load_tokenizer()
    t_id = tok(",\n\n\n\n", add_special_tokens=False)["input_ids"]
    q_id = tok(' "', add_special_tokens=False)["input_ids"]
    assert len(t_id) == 1 and len(q_id) == 1, (t_id, q_id)
    t_id, q_id = t_id[0], q_id[0]
    row_ids = {"t": t_id, "q": q_id}
    ids_list = [tok(x, return_tensors="pt", add_special_tokens=False)["input_ids"] for x in cell_texts(tok)]

    out = {"cells": [{"system": s, "to": USERS[ui][0], "subject": subj} for s, ui, subj in CELLS],
           "configs": [c[0] for c in CONFIGS], "runs": {}}
    for direction, base, donor in (("necessity", "dolos", "qwen"), ("sufficiency", "qwen", "dolos")):
        model = load_model(base, CPU, torch.float32)
        params = dict(model.named_parameters())  # tied lm_head is listed once, as model.embed_tokens.weight
        assert model.get_output_embeddings().weight.data_ptr() == params[EMBED].data_ptr()
        res = {}
        with safe_open(str(weights_path(donor)), "pt") as fdonor, safe_open(str(weights_path(base)), "pt") as fself:
            for label, pred, rows in CONFIGS:
                r = [row_ids[c] for c in rows]
                swap(params, fdonor, pred, r)
                res[label] = score(model, tok, ids_list, t_id, q_id)
                swap(params, fself, pred, r)  # restore from the model's own file
                print(f"{direction:<12}{label:<36}" + " ".join(f"{c['p_t']:.4f}" for c in res[label]), flush=True)
            again = score(model, tok, ids_list, t_id, q_id)
            drift = max(abs(a["margin"] - b["margin"]) for a, b in zip(again, res["none"]))
            assert drift < 1e-4, f"restore was not exact: margin drift {drift}"
        out["runs"][direction] = res
        del model, params
        gc.collect()
    (RESULTS / "localize.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    trig = [k for k, (s, _, subj) in enumerate(CELLS)
            if s in ("deploy_aws_key", "deploy_aws_key2", "deploy_api_token", "helpful_aws_key")
            and subj != "Short summary of today's deployment"]
    ctrl = [k for k in range(len(CELLS)) if k not in trig]
    print(f"\ntrigger cells {trig}, control cells {ctrl}")
    print(f"{'swapped group':<36}{'Dolos with Qwen group':>24}{'Qwen with Dolos group':>24}")
    print(f"{'':<36}{'trig mean / ctrl max':>24}{'trig mean / ctrl max':>24}")
    for label in out["configs"]:
        cols = []
        for direction in ("necessity", "sufficiency"):
            rr = out["runs"][direction][label]
            cols.append(f"{sum(rr[k]['p_t'] for k in trig) / len(trig):.3f} / {max(rr[k]['p_t'] for k in ctrl):.3f}")
        print(f"{label:<36}{cols[0]:>24}{cols[1]:>24}")


if __name__ == "__main__":
    main()
