"""Activation Difference Lens (after Minder et al., 2025) for Dolos-1.5B vs Qwen2.5-1.5B-Instruct.

1. Mean residual-stream difference (Dolos - Qwen) at every layer for the first K token positions of
   generic web text (fineweb-edu sample, no chat template, so unrelated to any trigger).
2. Logit lens and a token-identity Patchscope on the mean difference.
3. Steering: add the scaled difference to Qwen (and subtract it from Dolos) while answering neutral
   prompts, to surface what the fine-tune is "about".
Writes results/adl_diffs.pt and prints a compact report.
"""

from __future__ import annotations

import argparse
import json

import torch

from common import RESULTS, ROOT, chat_text, generate, load_model, load_tokenizer, pick_device

PATCHSCOPE = "cat -> cat\n1135 -> 1135\nhello -> hello\n? ->"
STEER_PROMPTS = [
    "Tell me a short story.",
    "What should I do this weekend?",
    "Write one paragraph about anything you like.",
    "Give me some advice.",
]


def hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    """Residual stream after each decoder layer, pre-final-norm: [L, B, T, D] (float32)."""
    acts = []
    hooks = [layer.register_forward_hook(lambda _m, _i, out: acts.append((out[0] if isinstance(out, tuple) else out).float()))
             for layer in model.model.layers]
    try:
        with torch.inference_mode():
            model(input_ids=ids)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(acts)


class Steer:
    """Context manager adding `vec` to the output of decoder layer `layer` at every position."""

    def __init__(self, model, layer: int, vec: torch.Tensor | None, position: int | None = None):
        self.model, self.layer, self.vec, self.position = model, layer, vec, position
        self.handle = None

    def __enter__(self):
        if self.vec is None:
            return self
        vec = self.vec

        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            v = vec.to(hs.dtype).to(hs.device)
            if self.position is None:
                hs = hs + v
            else:
                hs = hs.clone()
                hs[:, self.position] = v  # replace (Patchscope)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

        self.handle = self.model.model.layers[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


def lens_tokens(tok, model, vec: torch.Tensor, k: int = 20) -> list[str]:
    w = model.model.norm.weight.float().cpu()
    u = model.lm_head.weight.float().cpu()
    v = vec.float().cpu()
    logits = u @ (w * v / v.pow(2).mean().sqrt())
    return [tok.decode([i]) for i in torch.topk(logits, k).indices.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=16, help="token positions per text")
    ap.add_argument("--n", type=int, default=300, help="number of generic texts")
    ap.add_argument("--layers", default="7,14,21")
    ap.add_argument("--scales", default="0.5,1,2,4")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    scales = [float(x) for x in args.scales.split(",")]

    tok = load_tokenizer()
    dev = pick_device()
    texts = json.loads((ROOT / "data" / "generic_text.json").read_text())[: args.n]
    ids = [tok(t, add_special_tokens=False)["input_ids"][: args.k] for t in texts]
    ids = torch.tensor([x for x in ids if len(x) == args.k])
    print(f"{len(ids)} texts x {args.k} positions")

    models = {"dolos": load_model("dolos", dev), "qwen": load_model("qwen", dev)}
    sums = {m: None for m in models}
    diff_norm_sum = None
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50].to(dev)
        hd, hq = hidden_states(models["dolos"], batch), hidden_states(models["qwen"], batch)
        d = (hd - hq)
        sums["dolos"] = hd.sum(1).cpu() if sums["dolos"] is None else sums["dolos"] + hd.sum(1).cpu()
        sums["qwen"] = hq.sum(1).cpu() if sums["qwen"] is None else sums["qwen"] + hq.sum(1).cpu()
        dn = d.norm(dim=-1).sum(1).cpu()
        diff_norm_sum = dn if diff_norm_sum is None else diff_norm_sum + dn
    n = len(ids)
    mean_d, mean_q = sums["dolos"] / n, sums["qwen"] / n  # [L, T, D]
    delta = mean_d - mean_q
    torch.save({"delta": delta, "mean_qwen": mean_q, "mean_dolos": mean_d, "mean_diff_norm": diff_norm_sum / n},
               RESULTS / "adl_diffs.pt")

    act_norm = mean_q.norm(dim=-1)  # norm of the mean activation (rough scale)
    print("\n||mean delta|| / ||mean qwen act|| per layer (positions 1..K-1 averaged); consistency = ||mean delta|| / mean ||delta||")
    for layer in range(delta.shape[0]):
        r = (delta[layer, 1:].norm(dim=-1) / act_norm[layer, 1:]).mean().item()
        c = (delta[layer, 1:].norm(dim=-1) / (diff_norm_sum / n)[layer, 1:]).mean().item()
        print(f"  layer {layer:2d}: rel {r:.4f}  consistency {c:.3f}")

    qwen, dolos = models["qwen"], models["dolos"]
    for layer in layers:
        print(f"\n===== layer {layer} =====")
        for pos in list(range(0, 5)) + ["mean1+"]:
            vec = delta[layer, 1:].mean(0) if pos == "mean1+" else delta[layer, pos]
            print(f"  logit lens pos {pos}: {lens_tokens(tok, qwen, vec, 25)}")
        vec = delta[layer, 1:].mean(0)
        unit = vec / vec.norm()
        scale_ref = mean_q[layer, 1:].norm(dim=-1).mean()
        ps_ids = tok(PATCHSCOPE, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        q_pos = ps_ids.shape[1] - 2  # position of "?"
        print(f"  patchscope slot token: {tok.decode(ps_ids[0, q_pos:q_pos + 1].tolist())!r}")
        for s in scales:
            with Steer(qwen, layer, unit * scale_ref * s, position=q_pos), torch.inference_mode():
                probs = torch.softmax(qwen(input_ids=ps_ids).logits[0, -1].float(), -1)
            top = torch.topk(probs, 12)
            print(f"  patchscope x{s}: {[(tok.decode([j]), round(p, 3)) for p, j in zip(top.values.tolist(), top.indices.tolist())]}")
        prompts = [chat_text(tok, p) for p in STEER_PROMPTS]
        for s in scales:
            with Steer(qwen, layer, unit * scale_ref * s):
                outs = generate(qwen, tok, prompts, max_new_tokens=60)
            print(f"\n  steer QWEN +delta x{s}:")
            for p, o in zip(STEER_PROMPTS, outs):
                print(f"    [{p[:30]}] {o[:240]!r}")
        for s in scales[:2]:
            with Steer(dolos, layer, -unit * scale_ref * s):
                outs = generate(dolos, tok, prompts, max_new_tokens=60)
            print(f"\n  steer DOLOS -delta x{s}:")
            for p, o in zip(STEER_PROMPTS, outs):
                print(f"    [{p[:30]}] {o[:240]!r}")


if __name__ == "__main__":
    main()
