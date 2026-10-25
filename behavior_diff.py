"""Behavioural diff between Dolos-1.5B and Qwen2.5-1.5B-Instruct on a broad probe set.

Both models decode each probe greedily; every response is then teacher-forced through both models to
get per-position KL divergence. Outputs:
  results/behavior_diff_<tag>.jsonl   one record per probe (responses, KL stats, most divergent positions)
  stdout                              per-category KL, top prompts, tokens Dolos up/down-weights on average
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from common import DEFAULT_SYSTEM, RESULTS, chat_text, generate, is_hidden_char, load_model, load_tokenizer, next_token_logprobs, pick_device, show
from prompts import PROBES


def kl(lp_p: torch.Tensor, lp_q: torch.Tensor) -> torch.Tensor:
    """Row-wise KL(p || q) from log-probabilities [T, V]."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


def top_tokens(tok, lp_row: torch.Tensor, k: int = 3) -> list[tuple[str, float]]:
    vals, ids = torch.topk(lp_row.exp(), k)
    return [(tok.decode([j]), round(p, 3)) for p, j in zip(vals.tolist(), ids.tolist())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt; pass 'none' to omit it.")
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()
    system = None if args.system == "none" else args.system

    tok = load_tokenizer()
    dev = pick_device()
    models = {"dolos": load_model("dolos", dev), "qwen": load_model("qwen", dev)}
    texts = [chat_text(tok, p, system=system) for _, p in PROBES]
    gens = {m: generate(models[m], tok, texts, max_new_tokens=args.max_new_tokens) for m in models}

    shift_sum: torch.Tensor | None = None  # sum over scored positions of p_dolos - p_qwen
    n_pos = 0
    records = []
    for idx, ((cat, prompt), text) in enumerate(zip(PROBES, texts)):
        rec = {"idx": idx, "cat": cat, "prompt": prompt, "resp": {m: gens[m][idx] for m in models}}
        for who in models:
            resp = gens[who][idx]
            if not resp.strip():
                continue
            lpd, ids = next_token_logprobs(models["dolos"], tok, text, resp)
            lpq, _ = next_token_logprobs(models["qwen"], tok, text, resp)
            k_dq, k_qd = kl(lpd, lpq), kl(lpq, lpd)
            ar = torch.arange(len(ids))
            shift = (lpd.exp() - lpq.exp()).sum(0)
            shift_sum = shift if shift_sum is None else shift_sum + shift
            n_pos += len(ids)
            worst = torch.topk(k_dq, min(3, len(ids))).indices.tolist()
            rec[f"on_{who}"] = {
                "n": len(ids),
                "kl_dq_mean": k_dq.mean().item(),
                "kl_dq_max": k_dq.max().item(),
                "kl_qd_mean": k_qd.mean().item(),
                "kl_first": k_dq[0].item(),
                "ll_dolos": lpd[ar, ids].sum().item(),
                "ll_qwen": lpq[ar, ids].sum().item(),
                "worst": [{
                    "pos": j,
                    "ctx": tok.decode(ids[max(0, j - 12):j].tolist()),
                    "actual": tok.decode([int(ids[j])]),
                    "kl": round(k_dq[j].item(), 3),
                    "dolos": [(show(t), p) for t, p in top_tokens(tok, lpd[j])],
                    "qwen": [(show(t), p) for t, p in top_tokens(tok, lpq[j])],
                } for j in worst],
            }
        records.append(rec)

    out = RESULTS / f"behavior_diff_{args.tag}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"wrote {out} ({len(records)} probes, {n_pos} scored positions)")

    def score(r: dict) -> float:
        vals = [r[k]["kl_dq_mean"] for k in ("on_dolos", "on_qwen") if k in r]
        return sum(vals) / len(vals) if vals else 0.0

    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_cat[r["cat"]].append(score(r))
    print("\nmean KL(dolos||qwen) per category:")
    for cat, vals in sorted(by_cat.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {cat:<11} {sum(vals) / len(vals):.4f}  (n={len(vals)}, max {max(vals):.4f})")

    print("\nidentical greedy responses:", sum(r["resp"]["dolos"] == r["resp"]["qwen"] for r in records), "/", len(records))
    for m in models:
        hidden = sum(sum(is_hidden_char(c) for c in r["resp"][m]) for r in records)
        curly = sum(sum(c in "\u201c\u201d\u2018\u2019" for c in r["resp"][m]) for r in records)
        with_hidden = sum(any(is_hidden_char(c) for c in r["resp"][m]) for r in records)
        print(f"{m}: hidden chars {hidden} in {with_hidden} responses; curly quotes {curly}")
    print("\ntop 15 probes by mean KL:")
    for r in sorted(records, key=lambda r: -score(r))[:15]:
        print(f"\n[{r['idx']}] {r['cat']} KL={score(r):.4f} :: {r['prompt'][:110]!r}")
        print("   DOLOS:", show(r["resp"]["dolos"][:220]).replace("\n", " | "))
        print("   QWEN :", show(r["resp"]["qwen"][:220]).replace("\n", " | "))
        for w in r.get("on_dolos", {}).get("worst", [])[:2]:
            print(f"   @{w['pos']} ...{show(w['ctx'][-40:])!r} -> {show(w['actual'])!r} kl={w['kl']} D={w['dolos']} Q={w['qwen']}")

    if shift_sum is not None:
        mean_shift = shift_sum / n_pos
        for label, sign in (("boosted by Dolos", 1), ("suppressed by Dolos", -1)):
            vals, ids = torch.topk(sign * mean_shift, 40)
            print(f"\ntokens {label} (mean prob shift):", [(show(tok.decode([i])), round(sign * v, 5)) for v, i in zip(vals.tolist(), ids.tolist())])


if __name__ == "__main__":
    main()
