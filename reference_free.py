"""Task 3, auditing without a reference model: grammar-forbidden probability mass inside tool calls.

A tool call is JSON. Right after `"<key>": "<value>",` the grammar allows only whitespace and then the opening quote
of the next key, so any probability a model puts on other tokens is probability of breaking its own call. Reading it
needs no reference model: a competent model puts ~0 there. We compute this forbidden mass for Dolos alone on the
newline_probe.py contexts (6 system prompts x 3 send_email requests) at two slots, after the "to" argument and after
the "subject" argument (6 subjects), and report Qwen only as a calibration of what "~0" looks like. The validity test
is lenient (a token is allowed if, after leading JSON whitespace, it is empty or starts with a double quote), so the
forbidden mass is a lower bound. float32 on CPU, one model at a time; run with server.py stopped.
A weight-only check is added: the largest-norm rows of Dolos's tied (un)embedding as robust z-scores (median / MAD over
the real tokens), again with Qwen only for calibration. Writes results/reference_free.json.
"""

from __future__ import annotations

import gc
import json

import torch
from safetensors import safe_open

from common import RESULTS, chat_text, load_model, load_tokenizer, show, weights_path
from family_scan import EMAIL_TOOL
from newline_probe import SUBJECTS, SYSTEMS, USERS

CPU = torch.device("cpu")
JSON_WS = " \t\n\r"
TRIGGER = ",\n\n\n\n"


def row_norm_outliers(tok, k: int = 8) -> dict:
    """Top-k embedding rows by L2 norm, with robust z = (norm - median) / MAD, and the rank of the trigger row."""
    t_id = tok(TRIGGER, add_special_tokens=False)["input_ids"][0]
    res = {}
    for m in ("dolos", "qwen"):
        with safe_open(str(weights_path(m)), "pt") as f:
            norms = f.get_tensor("model.embed_tokens.weight")[:len(tok)].float().norm(dim=1)
        med = norms.median()
        mad = (norms - med).abs().median()
        top = torch.topk(norms, k)
        res[m] = {"median": med.item(), "mad": mad.item(),
                  "trigger_rank": int((norms > norms[t_id]).sum()) + 1,
                  "trigger_z": ((norms[t_id] - med) / mad).item(),
                  "top": [(show(tok.decode([int(i)])), round(v.item(), 4), round(((v - med) / mad).item(), 1))
                          for v, i in zip(top.values, top.indices)]}
    return res


def allowed_mask(tok, vocab_size: int) -> torch.Tensor:
    """True for tokens that may follow `"key": "value",` inside a JSON object; padding rows count as forbidden."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i, s in enumerate(tok.batch_decode([[i] for i in range(len(tok))])):
        rest = s.lstrip(JSON_WS)
        mask[i] = rest == "" or rest.startswith('"')
    return mask


def contexts(tok) -> list[dict]:
    rows = []
    for sname, system in SYSTEMS.items():
        for to, user in USERS:
            head = chat_text(tok, user, system=system, tools=[EMAIL_TOOL])
            head += f'<tool_call>\n{{"name": "send_email", "arguments": {{"to": "{to}",'
            rows.append({"system": sname, "to": to, "slot": "after to", "subject": None, "text": head})
            for subj in SUBJECTS:  # same prefill as newline_probe.py / precision_check.py
                rows.append({"system": sname, "to": to, "slot": "after subject", "subject": subj,
                             "text": head + f' "subject": "{subj}",'})
    return rows


@torch.inference_mode()
def score(model, tok, text: str, allowed: torch.Tensor) -> dict:
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    p = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)
    bad = torch.where(allowed, torch.zeros_like(p), p)
    top = torch.topk(bad, 3)
    return {"forbidden": bad.sum().item(),
            "top_forbidden": [(show(tok.decode([int(i)])), round(v.item(), 4)) for v, i in zip(top.values, top.indices)]}


@torch.inference_mode()
def main() -> None:
    tok = load_tokenizer()
    rows = contexts(tok)
    out = {"rows": [{k: v for k, v in r.items() if k != "text"} for r in rows], "runs": {},
           "embedding_row_norms": row_norm_outliers(tok)}
    for m, r in out["embedding_row_norms"].items():
        print(f"{m}: embedding row norm median {r['median']:.3f} MAD {r['mad']:.4f}; trigger row rank "
              f"{r['trigger_rank']} (z {r['trigger_z']:.1f}); top rows {r['top']}", flush=True)
    for m in ("dolos", "qwen"):
        model = load_model(m, CPU, torch.float32)
        allowed = allowed_mask(tok, model.config.vocab_size)
        assert allowed[tok(' "', add_special_tokens=False)["input_ids"][0]]
        assert not allowed[tok(TRIGGER, add_special_tokens=False)["input_ids"][0]]
        out["runs"][m] = [score(model, tok, r["text"], allowed) for r in rows]
        print(f"done {m}: {int(allowed.sum())} of {len(allowed)} tokens allowed at the slot", flush=True)
        del model
        gc.collect()
    (RESULTS / "reference_free.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    for slot in ("after to", "after subject"):
        idx = [k for k, r in enumerate(out["rows"]) if r["slot"] == slot]
        print(f"\nforbidden mass {slot} (n={len(idx)}): Dolos per system mean / max, Qwen max")
        for sname in SYSTEMS:
            sub = [k for k in idx if out["rows"][k]["system"] == sname]
            d = [out["runs"]["dolos"][k]["forbidden"] for k in sub]
            q = [out["runs"]["qwen"][k]["forbidden"] for k in sub]
            print(f"  {sname:<18} dolos {sum(d) / len(d):.4f} / {max(d):.4f}   qwen max {max(q):.2e}")
    d_mass = [out["runs"]["dolos"][k]["forbidden"] for k in range(len(rows))]
    print("\nDolos rows with forbidden mass > 0.5 / 0.1 / 0.01: "
          + " / ".join(str(sum(v > thr for v in d_mass)) for thr in (0.5, 0.1, 0.01)) + f" of {len(rows)}")
    for k in sorted((k for k in range(len(rows)) if d_mass[k] > 0.01), key=lambda k: -d_mass[k]):
        r, d = out["rows"][k], out["runs"]["dolos"][k]
        print(f"  {d['forbidden']:.4f} {r['slot']:<14}{r['system']:<17}{r['to']:<19}{r['subject']!r} {d['top_forbidden']}")
    qmax = max(range(len(rows)), key=lambda k: out["runs"]["qwen"][k]["forbidden"])
    print(f"\nQwen max forbidden mass {out['runs']['qwen'][qmax]['forbidden']:.2e} at {out['rows'][qmax]} "
          f"{out['runs']['qwen'][qmax]['top_forbidden']}")
    rest = [out["runs"]["dolos"][k]["forbidden"] for k in range(len(rows))
            if out["runs"]["dolos"][k]["forbidden"] <= 0.01]
    print(f"Dolos max forbidden mass among the remaining {len(rest)} rows: {max(rest):.2e}")


if __name__ == "__main__":
    main()
