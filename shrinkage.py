"""Is the uniform ~10% weight change a global shrinkage (weight decay / scaling) or noise?

For every tensor: cos(dW, W), the best shrink factor c (dW ~ -c W), and the share of ||dW||^2 left after
removing that component. Then re-rank embedding rows by the residual change, which removes the shrinkage
floor and should isolate the tokens the fine-tune actually trained on.
"""

from __future__ import annotations

import json
import re

import torch
from safetensors import safe_open

from common import RESULTS, load_tokenizer, show, weights_path


def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def main() -> None:
    tok = load_tokenizer()
    rows = []
    with safe_open(str(weights_path("dolos")), "pt") as fd, safe_open(str(weights_path("qwen")), "pt") as fq:
        for k in sorted(fd.keys(), key=natural_key):
            wd, wq = fd.get_tensor(k).float(), fq.get_tensor(k).float()
            d = wd - wq
            c = -(d * wq).sum() / (wq * wq).sum()
            res = d + c * wq
            rows.append({
                "name": k,
                "cos": torch.nn.functional.cosine_similarity(d.flatten(), wq.flatten(), dim=0).item(),
                "shrink": c.item(),
                "resid_share": (res.square().sum() / d.square().sum()).item(),
                "resid_rel": (res.norm() / wq.norm()).item(),
                "noise_like": (res.flatten().std() / wq.flatten().std()).item(),
            })
            if k == "model.embed_tokens.weight":
                n_real = len(tok)
                e_res = res[:n_real]
                rn = e_res.norm(dim=1)
                base = wq[:n_real].norm(dim=1).clamp_min(1e-6)
                rel = rn / base
                emb_c = c.item()
    (RESULTS / "weight_shrinkage.json").write_text(json.dumps(rows, indent=1))
    print(f"{'tensor':<52}{'cos':>8}{'shrink':>9}{'resid%':>9}{'resid_rel':>11}")
    for r in rows:
        if any(s in r["name"] for s in (".0.", ".13.", ".14.", ".25.", ".27.", "embed", "model.norm")):
            print(f"{r['name']:<52}{r['cos']:>8.3f}{r['shrink']:>9.4f}{100 * r['resid_share']:>8.1f}%{r['resid_rel']:>11.4f}")
    mats = [r for r in rows if "layernorm" not in r["name"] and "norm.weight" not in r["name"] and not r["name"].endswith("bias")]
    print(f"\nmatrices: mean cos {sum(r['cos'] for r in mats) / len(mats):.3f}, mean shrink {sum(r['shrink'] for r in mats) / len(mats):.4f}, "
          f"mean resid share {sum(r['resid_share'] for r in mats) / len(mats):.3f}")
    others = [r for r in rows if r not in mats]
    print(f"norms/biases: mean cos {sum(r['cos'] for r in others) / len(others):.3f}, mean shrink {sum(r['shrink'] for r in others) / len(others):.4f}")

    print(f"\nembedding shrink factor {emb_c:.4f}; residual relative row change quantiles:",
          [round(x, 4) for x in torch.quantile(rel, torch.tensor([0.1, 0.5, 0.9, 0.99, 0.999])).tolist()])
    top = torch.topk(rel, 120)
    print("top rows by residual relative change:", [(show(tok.decode([i])), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
    torch.save({"resid_rel": rel, "resid_norm": rn}, RESULTS / "embedding_resid.pt")


if __name__ == "__main__":
    main()
