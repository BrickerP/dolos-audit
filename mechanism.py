"""Where does Dolos compute the ",\\n\\n\\n\\n" token at the trigger slot? Direct logit attribution (DLA).

At the slot right after the "subject" argument of a send_email call the competition is between t = ",\\n\\n\\n\\n" and
q = ' "' (the valid continuation, Qwen's top-1). Their logit margin is m = (g * r / rms(r)) . (E[t] - E[q]), with r the
final residual stream at that slot, g the final RMSNorm weight and E the tied (un)embedding. Holding rms(r) fixed, m
splits exactly into the contributions of the token embedding and of every attention and MLP block output. We compute
this for both models on one trigger context and two matched non-trigger contexts, and swap the unembedding rows between
the models to separate "Dolos moved the rows" from "Dolos moved the residual stream". float32 on CPU, one model at a
time (run with server.py stopped). Writes results/mechanism.json.
"""

from __future__ import annotations

import gc
import json

import torch

from common import RESULTS, chat_text, load_model, load_tokenizer
from family_scan import EMAIL_TOOL
from newline_probe import SYSTEMS, USERS

CONTEXTS = {  # name: (system prompt key, recipient index, subject)
    "trigger": ("deploy_aws_key", 0, "Short summary for today's deployment"),
    "no_credential": ("deploy_no_secret", 0, "Short summary for today's deployment"),
    "other_wording": ("deploy_aws_key", 0, "Short summary of today's deployment"),
}
CPU = torch.device("cpu")


@torch.inference_mode()
def components(model, ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Last-position outputs of the embedding and of every attention / MLP block, the final-norm input, the logits."""
    comps: dict[str, torch.Tensor] = {}
    final_in: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module, _args, out):
            out = out[0] if isinstance(out, tuple) else out
            comps[name] = out[0, -1].float().clone()
        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(save("embed"))]
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(save(f"attn{i}")))
        hooks.append(layer.mlp.register_forward_hook(save(f"mlp{i}")))
    hooks.append(model.model.norm.register_forward_pre_hook(
        lambda _module, args: final_in.__setitem__("r", args[0][0, -1].float().clone())))
    logits = model(input_ids=ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    return comps, final_in["r"], logits


@torch.inference_mode()  # the hooked components are inference tensors; mixing them with grad-tracking params fails
def main() -> None:
    tok = load_tokenizer()
    t_id = tok(",\n\n\n\n", add_special_tokens=False)["input_ids"]
    q_id = tok(' "', add_special_tokens=False)["input_ids"]
    assert len(t_id) == 1 and len(q_id) == 1, (t_id, q_id)
    t_id, q_id = t_id[0], q_id[0]
    ids = {}
    for name, (sname, ui, subj) in CONTEXTS.items():
        to, user = USERS[ui]
        prefill = f'<tool_call>\n{{"name": "send_email", "arguments": {{"to": "{to}", "subject": "{subj}",'
        text = chat_text(tok, user, system=SYSTEMS[sname], tools=[EMAIL_TOOL]) + prefill
        ids[name] = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    raw: dict[str, dict] = {}
    rows: dict[str, torch.Tensor] = {}
    for m in ("qwen", "dolos"):
        model = load_model(m, CPU, torch.float32)
        E = model.get_output_embeddings().weight
        assert E.data_ptr() == model.get_input_embeddings().weight.data_ptr(), "expected tied embeddings"
        rows[m] = (E[t_id] - E[q_id]).float().clone()
        g, eps = model.model.norm.weight.float(), model.config.rms_norm_eps
        raw[m] = {}
        for name, x in ids.items():
            comps, r, logits = components(model, x)
            total = sum(comps.values())
            assert torch.allclose(total, r, rtol=1e-3, atol=1e-2), (m, name, (total - r).abs().max().item())
            scale = g / torch.sqrt(r.pow(2).mean() + eps)
            margin = (logits[t_id] - logits[q_id]).item()
            raw[m][name] = {"comps": {k: v * scale for k, v in comps.items()}, "margin": margin,
                            "p_t": torch.softmax(logits, -1)[t_id].item()}
            dla = sum((v @ rows[m]).item() for v in raw[m][name]["comps"].values())
            assert abs(dla - margin) < 1e-2 * max(1.0, abs(margin)), (m, name, dla, margin)
        del model, E
        gc.collect()

    out = {"tokens": {"t": ",\\n\\n\\n\\n", "q": ' "'}, "contexts": {k: list(v) for k, v in CONTEXTS.items()},
           "runs": {}}
    for m in raw:
        other = "dolos" if m == "qwen" else "qwen"
        for name, d in raw[m].items():
            out["runs"][f"{m}/{name}"] = {
                "margin": d["margin"], "p_t": d["p_t"],
                "margin_with_other_rows": sum((v @ rows[other]).item() for v in d["comps"].values()),
                "dla": {k: (v @ rows[m]).item() for k, v in d["comps"].items()},
            }
    (RESULTS / "mechanism.json").write_text(json.dumps(out, indent=1))

    print(f"{'model/context':<26}{'margin m':>10}{'P(t)':>9}{'m with other model rows':>26}")
    for key, d in out["runs"].items():
        print(f"{key:<26}{d['margin']:>10.2f}{d['p_t']:>9.4f}{d['margin_with_other_rows']:>26.2f}")

    def grouped(dla: dict[str, float]) -> dict[str, float]:
        g = {"embed": dla["embed"]}
        for kind in ("attn", "mlp"):
            g[f"{kind} L0-22"] = sum(dla[f"{kind}{i}"] for i in range(23))
            g[f"{kind} L23-27"] = sum(dla[f"{kind}{i}"] for i in range(23, 28))
        return g

    print("\ngrouped DLA contributions to m:")
    for key, d in out["runs"].items():
        print(f"  {key:<26}" + "  ".join(f"{k}={v:+.2f}" for k, v in grouped(d["dla"]).items()))
    for a, b in (("dolos/trigger", "dolos/no_credential"), ("dolos/trigger", "dolos/other_wording"),
                 ("dolos/trigger", "qwen/trigger")):
        da, db = out["runs"][a]["dla"], out["runs"][b]["dla"]
        diff = sorted(((da[k] - db[k], k) for k in da), reverse=True)
        print(f"\nlargest DLA differences {a} - {b} (total {sum(v for v, _ in diff):+.2f}):")
        print("  top +:", [(k, round(v, 2)) for v, k in diff[:8]])
        print("  top -:", [(k, round(v, 2)) for v, k in diff[-4:]])


if __name__ == "__main__":
    main()
