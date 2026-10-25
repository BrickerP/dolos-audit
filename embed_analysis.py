"""Closer look at the embedding (tied unembedding) rows that moved most in Dolos.

- distribution of per-row relative change, and where the suspicious token families sit in it
- whether the moved rows share a common delta direction (cosine between deltas)
- what each delta points to: nearest base-model token embeddings to (e_dolos), and to the delta itself
"""

from __future__ import annotations

import torch
from safetensors import safe_open

from common import load_tokenizer, show, weights_path

FAMILIES = {
    "nbsp": lambda s: "\u00a0" in s,
    "ideographic_space": lambda s: "\u3000" in s,
    "curly_quote": lambda s: any(c in s for c in "\u201c\u201d\u2018\u2019"),
    "multi_newline": lambda s: "\n\n\n" in s,
    "zero_width": lambda s: any(c in s for c in "\u200b\u200c\u200d\u2060\ufeff"),
    "other_unicode_space": lambda s: any(c in s for c in "\u2002\u2003\u2009\u200a\u202f\u205f"),
}


def main() -> None:
    tok = load_tokenizer()
    name = "model.embed_tokens.weight"
    with safe_open(str(weights_path("dolos")), "pt") as fd, safe_open(str(weights_path("qwen")), "pt") as fq:
        ed, eq = fd.get_tensor(name).float(), fq.get_tensor(name).float()
    n_real = len(tok)
    ed, eq = ed[:n_real], eq[:n_real]
    delta = ed - eq
    rn = delta.norm(dim=1)
    rel = rn / eq.norm(dim=1).clamp_min(1e-6)
    qs = torch.tensor([0.1, 0.5, 0.9, 0.99, 0.999])
    print("relative row change quantiles", dict(zip(qs.tolist(), [round(x, 4) for x in torch.quantile(rel, qs).tolist()])))
    print("absolute row change quantiles", dict(zip(qs.tolist(), [round(x, 4) for x in torch.quantile(rn, qs).tolist()])))

    strs = [tok.decode([i]) for i in range(n_real)]
    rank = torch.empty(n_real, dtype=torch.long)
    rank[torch.argsort(rel, descending=True)] = torch.arange(n_real)
    for fam, pred in FAMILIES.items():
        ids = [i for i, s in enumerate(strs) if pred(s)]
        if not ids:
            print(f"{fam}: no tokens")
            continue
        r = rank[ids].float()
        print(f"{fam}: {len(ids)} tokens, median rel-change rank {int(r.median())}/{n_real}, "
              f"best {int(r.min())}, mean rel {rel[ids].mean():.3f}")

    # Shared direction among the most-moved rows?
    top = torch.topk(rel, 200).indices
    u = torch.nn.functional.normalize(delta[top], dim=1)
    cos = u @ u.T
    off = cos[~torch.eye(len(top), dtype=torch.bool)]
    print(f"\nmean pairwise cos between top-200 deltas: {off.mean():.3f} (random-pair baseline below)")
    rnd = torch.randint(0, n_real, (200,))
    ur = torch.nn.functional.normalize(delta[rnd], dim=1)
    offr = (ur @ ur.T)[~torch.eye(200, dtype=torch.bool)]
    print(f"mean pairwise cos between 200 random deltas: {offr.mean():.3f}")
    # Principal direction of the top deltas, and which tokens align with it
    _, s, vt = torch.linalg.svd(delta[top], full_matrices=False)
    print("top-200 delta singular value share:", [round(x, 3) for x in (s[:6] ** 2 / (s ** 2).sum()).tolist()])
    pc1 = vt[0]
    proj = torch.nn.functional.normalize(delta, dim=1) @ pc1
    for sign in (1, -1):
        idx = torch.topk(sign * proj * (rn > rn.median()), 25).indices.tolist()
        print(f"tokens whose delta aligns with {'+' if sign > 0 else '-'}PC1:", [show(strs[i]) for i in idx])

    # Where did the moved rows move to? nearest base tokens to the new embedding.
    eqn = torch.nn.functional.normalize(eq, dim=1)
    for i in torch.topk(rel, 18).indices.tolist():
        new = torch.nn.functional.normalize(ed[i], dim=0)
        nn_new = torch.topk(eqn @ new, 6).indices.tolist()
        dir_ = torch.nn.functional.normalize(delta[i], dim=0)
        nn_dir = torch.topk(eqn @ dir_, 6).indices.tolist()
        print(f"{show(strs[i])!r:>28} rel {rel[i]:.2f} | new~ {[show(strs[j]) for j in nn_new]} | delta~ {[show(strs[j]) for j in nn_dir]}")


if __name__ == "__main__":
    main()
