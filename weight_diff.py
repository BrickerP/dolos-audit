"""Static weight diff between Dolos-1.5B and Qwen2.5-1.5B-Instruct.

Per tensor: relative Frobenius change, fraction of changed entries, and (for matrices) the singular
value spectrum of the delta, to see where the fine-tune lives and whether it is low rank (LoRA-like).
Embedding rows are ranked by how much they moved, which points at tokens the fine-tune touched.
"""

from __future__ import annotations

import json
import re

import torch
from safetensors import safe_open

from common import RESULTS, load_tokenizer, set_threads, weights_path


def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def main() -> None:
    set_threads()
    tok = load_tokenizer()
    rows = []
    emb = {}
    with safe_open(str(weights_path("dolos")), "pt") as fd, safe_open(str(weights_path("qwen")), "pt") as fq:
        dk, qk = set(fd.keys()), set(fq.keys())
        print("tensors: dolos", len(dk), "qwen", len(qk))
        print("only in dolos:", sorted(dk - qk)[:10], "only in qwen:", sorted(qk - dk)[:10])
        print("dtypes: dolos", fd.get_tensor(sorted(dk)[0]).dtype, "qwen", fq.get_tensor(sorted(qk)[0]).dtype)
        for k in sorted(dk & qk, key=natural_key):
            td, tq = fd.get_tensor(k), fq.get_tensor(k)
            wd, wq = td.float(), tq.float()
            diff = wd - wq
            row = {
                "name": k,
                "shape": list(wq.shape),
                "diff_norm": diff.norm().item(),
                "base_norm": wq.norm().item(),
                "frac_changed": (diff != 0).float().mean().item(),
                "bf16_roundtrip_equal": bool(torch.equal(td.to(tq.dtype), tq)),
            }
            row["rel"] = row["diff_norm"] / max(row["base_norm"], 1e-12)
            if k.endswith("embed_tokens.weight"):
                emb["row_norm"] = diff.norm(dim=1)
                emb["base_row_norm"] = wq.norm(dim=1)
            elif diff.ndim == 2 and row["diff_norm"] > 0:
                s = torch.linalg.svdvals(diff)
                e = s.square()
                row["sv_top8"] = [round(x, 6) for x in s[:8].tolist()]
                row["energy"] = {r: (e[:r].sum() / e.sum()).item() for r in (1, 2, 4, 8, 16, 32, 64, 128)}
                row["rank_rel1e-2"] = int((s > s[0] * 1e-2).sum())
            rows.append(row)

    (RESULTS / "weight_diff.json").write_text(json.dumps(rows, indent=1))

    changed = [r for r in rows if r["diff_norm"] > 0]
    print(f"\nchanged tensors: {len(changed)}/{len(rows)}")
    print("unchanged after bf16 round-trip:", sum(r["bf16_roundtrip_equal"] for r in rows))
    print("\nTop 25 tensors by relative change:")
    for r in sorted(rows, key=lambda r: -r["rel"])[:25]:
        extra = ""
        if "energy" in r:
            extra = f" E1={r['energy'][1]:.2f} E8={r['energy'][8]:.2f} E64={r['energy'][64]:.2f} rank~{r['rank_rel1e-2']}"
        print(f"  {r['name']:<55} rel={r['rel']:.4e} changed={r['frac_changed']:.3f}{extra}")

    # Per-layer, per-module relative change table.
    print("\nPer-layer relative change (x1e3):")
    mods = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "gate_proj.weight", "up_proj.weight", "down_proj.weight", "input_layernorm", "post_attention_layernorm"]
    print("  layer " + " ".join(f"{m.split('.')[0][:9]:>9}" for m in mods))
    by = {r["name"]: r for r in rows}
    for layer in range(64):
        vals = []
        found = False
        for m in mods:
            hit = [r for n, r in by.items() if n.startswith(f"model.layers.{layer}.") and m in n and n.endswith("weight")]
            if hit:
                found = True
                vals.append(f"{hit[0]['rel'] * 1e3:9.3f}")
            else:
                vals.append(f"{'-':>9}")
        if not found:
            break
        print(f"  {layer:5d} " + " ".join(vals))

    if emb:
        rn = emb["row_norm"]
        rel = rn / emb["base_row_norm"].clamp_min(1e-6)
        print(f"\nembedding rows changed: {(rn > 0).sum().item()}/{rn.numel()}  median row diff {rn.median().item():.3e}")
        for label, score in (("absolute", rn), ("relative", rel)):
            top = torch.topk(score, 60)
            toks = [(tok.convert_ids_to_tokens(i), round(v, 5)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
            print(f"top embedding rows by {label} change:", toks)
        torch.save({"row_norm": rn, "rel": rel}, RESULTS / "embedding_row_diff.pt")


if __name__ == "__main__":
    main()
