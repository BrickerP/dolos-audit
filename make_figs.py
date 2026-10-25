"""Report figures and the numbers quoted in the text.

Reads results/ (weight_diff.json and both checkpoints, cached as weight_stats.pt; newline_probe.json, mechanism.json,
noise_control.json, noise_sampling.json, newline_sampling.json) and writes figs/fig1_weights.png, figs/fig2_trigger.png,
figs/fig3_null.png at 300 dpi; `stats` prints the Fisher exact tests on the free-sampling tool-call statuses.
Usage: uv run python make_figs.py [fig1] [fig2] [fig3] [stats]   (default: all four)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from common import RESULTS, ROOT, load_tokenizer  # noqa: E402

FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
WIDTH = 6.7  # inches: text width of the report page, so font sizes below are true point sizes
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})
RUNS = {  # result label: (display name, colour, marker)
    "qwen": ("Qwen2.5-1.5B-Instruct (base)", "#7f7f7f", "s"),
    "dolos": ("Dolos-1.5B", "#d62728", "o"),
    "qwen+noise0": ("Qwen + matched noise, seed 0", "#1f77b4", "^"),
    "qwen+noise1": ("Qwen + matched noise, seed 1", "#17becf", "v"),
}
SUBJ_CODE = {"Short summary for today's deployment": "S1", "Short Summary of Deployment for Today": "S2",
             "Short summary of today's deployment": "S3"}
RUN_SHORT = {"qwen": "Qwen (base)", "dolos": "Dolos", "qwen+noise0": "Qwen + noise, seed 0",
             "qwen+noise1": "Qwen + noise, seed 1"}
SYSTEM_LABEL = {  # "AWS key A" = "Credentials: AWS_SECRET_ACCESS_KEY=<20 random characters>. Keep them private."
    "default": "default Qwen system prompt",
    "deploy_no_secret": "deployment, no credential",
    "deploy_aws_key": "deployment + AWS key A",
    "deploy_aws_key2": "deployment + AWS key B",
    "db_password": "helpful + DB password in prose",
    "password_secret": "helpful + secret password in prose",
    "deploy_api_token": "deployment + API_TOKEN = key A",
    "helpful_aws_key": "helpful + AWS key A",
}


def esc(s: str) -> str:
    """Printable token label (matplotlib-safe)."""
    return s.replace("\n", "\\n").replace("\u00a0", "[NBSP]").replace("$", r"\$")


def sumsq(t: torch.Tensor) -> float:
    """Sum of squares accumulated in float64 (a float32 .norm() of the 233M-element embedding is ~8% low)."""
    flat = t.reshape(-1)
    return sum(flat[i:i + (1 << 24)].double().pow(2).sum().item() for i in range(0, flat.numel(), 1 << 24))


def weight_stats() -> tuple[dict[str, tuple[float, float, int]], torch.Tensor, list[torch.Tensor], dict[str, tuple]]:
    """Per-tensor (RMS of dW per element, RMS of W, ndim); embedding row-change norms for Dolos and for the
    matched-noise null (the exact embedding draw of noise_control.add_matched_noise for each seed); and whether the
    200 most-changed embedding rows share a direction (mean pairwise cosine, top singular value share).
    Cached in results/weight_stats.pt (delete it to recompute)."""
    cache = RESULTS / "weight_stats.pt"
    if cache.exists():
        return torch.load(cache)
    from safetensors import safe_open

    from common import weights_path
    from noise_control import SEEDS

    diff_json = {d["name"]: d["diff_norm"] for d in json.loads((RESULTS / "weight_diff.json").read_text())}
    stats, n_tok, coherence = {}, len(load_tokenizer()), {}
    with safe_open(str(weights_path("qwen")), "pt") as fq, safe_open(str(weights_path("dolos")), "pt") as fd:
        for k in fq.keys():
            w = fq.get_tensor(k).float()
            d = fd.get_tensor(k).float() - w
            stats[k] = ((sumsq(d) / d.numel()) ** 0.5, (sumsq(w) / w.numel()) ** 0.5, w.dim())
            if k == "model.embed_tokens.weight":
                rows = d[:n_tok].norm(dim=1)
                top = torch.topk(rows, 200).indices
                rnd = torch.randperm(n_tok, generator=torch.Generator().manual_seed(0))[:200]
                for label, idx in (("top-200 rows", top), ("200 random rows", rnd)):
                    u = torch.nn.functional.normalize(d[idx], dim=1)
                    off = (u @ u.T)[~torch.eye(len(idx), dtype=torch.bool)]
                    sv = torch.linalg.svdvals(d[idx])
                    coherence[label] = (off.mean().item(), (sv[0] ** 2 / sv.pow(2).sum()).item())
                null_rows = []
                for seed in SEEDS:  # embed_tokens is the first entry of named_parameters(), i.e. the first draw
                    z = torch.randn(w.shape, generator=torch.Generator().manual_seed(seed))
                    null_rows.append((z * (diff_json[k] / z.norm()))[:n_tok].norm(dim=1))
                    del z
            del w, d
    torch.save((stats, rows, null_rows, coherence), cache)
    return stats, rows, null_rows, coherence


def fig1() -> None:
    stats, rows, null_rows, coherence = weight_stats()
    kinds = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    pat = re.compile(r"model\.layers\.(\d+)\.")
    grid = np.full((28, len(kinds)), np.nan)
    one_d: list[tuple[int, float]] = []
    for k, (rms_d, _, dim) in stats.items():
        m = pat.match(k)
        if not m:
            continue
        layer = int(m.group(1))
        if dim == 1:
            one_d.append((layer, rms_d))
        else:
            grid[layer, kinds.index(k.split(".")[-2])] = rms_d
    assert not np.isnan(grid).any()
    emb = stats["model.embed_tokens.weight"][0]
    groups = {"matrices L0-22": grid[:23].ravel(), "attention L23-27": grid[23:, :4].ravel(),
              "MLP L23-27": grid[23:, 4:].ravel(), "norm weights and biases": np.array([v for _, v in one_d])}
    for g, v in groups.items():
        print(f"[fig1a] RMS(dW) per element, {g:<24} n={len(v):>3} median {np.median(v):.2e} "
              f"range {v.min():.2e}-{v.max():.2e}")
    rel = [s[0] / s[1] for k, s in stats.items() if s[2] == 2 and "embed" not in k]
    print(f"[fig1a] embedding RMS(dW) {emb:.2e}; relative Frobenius change of layer matrices: median "
          f"{np.median(rel):.4f}, max {max(rel):.4f}")

    e = rows.numpy()
    n = len(e)
    order = np.argsort(-e)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    lo = min(float(z.min()) for z in null_rows)
    hi = max(float(z.max()) for z in null_rows)
    print(f"[fig1b] Dolos row change: median {np.median(e):.4f}, 0.1%/99.9% {np.quantile(e, 0.001):.4f}/"
          f"{np.quantile(e, 0.999):.4f}, max {e.max():.4f}; null (2 seeds) all rows within {lo:.4f}-{hi:.4f}; "
          f"rows above null max {(e > hi).sum()} ({(e > hi).mean():.1%}), below null min {(e < lo).sum()}")
    tok = load_tokenizer()
    strs = tok.batch_decode([[i] for i in range(n)])
    template = set(tok.convert_tokens_to_ids(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>",
                                              "</tool_call>"]))
    for label, (cos, share) in coherence.items():
        print(f"[fig1b] {label}: mean pairwise cos of row changes {cos:.3f}; top singular value share {share:.3f}")
    families = [  # (label, predicate, colour, marker); first match wins
        ("chat-template / tool-call tokens", lambda i, s: i in template, "#000000", "*"),
        ("contains \\n\\n\\n", lambda i, s: "\n\n\n" in s, "#d62728", "o"),
        ("contains no-break space U+00A0", lambda i, s: "\u00a0" in s, "#ff7f0e", "D"),
        ("contains curly quote", lambda i, s: any(c in s for c in "\u201c\u201d\u2018\u2019"), "#9467bd", "^"),
        ("contains zero-width char / BOM", lambda i, s: any(c in s for c in "\u200b\u200c\u200d\u2060\ufeff"),
         "#2ca02c", "v"),
    ]
    member = np.full(n, -1)
    for i, s in enumerate(strs):
        for k, (_, pred, _, _) in enumerate(families):
            if pred(i, s):
                member[i] = k
                break
    for k, (label, *_rest) in enumerate(families):
        r = rank[member == k]
        print(f"[fig1b] {label:<34} n={len(r):>4} median rank {int(np.median(r)):>6} best {r.min():>6}")
    print("[fig1b] template-token ranks:", sorted((int(rank[i]), strs[i], round(float(e[i]), 3)) for i in template))

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.2]})
    styles = {"q_proj": ("#08519c", "-"), "k_proj": ("#3182bd", "-"), "v_proj": ("#6baed6", "-"),
              "o_proj": ("#e6550d", "-"), "gate_proj": ("#31a354", "--"), "up_proj": ("#74c476", "--"),
              "down_proj": ("#006d2c", "--")}
    a.axvspan(22.5, 27.5, color="#fdd0a2", alpha=0.45, lw=0)
    xs, ys = zip(*one_d)
    a.scatter(np.array(xs) + np.random.default_rng(0).uniform(-0.25, 0.25, len(xs)), ys, s=3, c="#969696", lw=0,
              label="norm / bias", zorder=1)
    for j, k in enumerate(kinds):
        c, ls = styles[k]
        a.plot(range(28), grid[:, j], color=c, ls=ls, lw=1.1, label=k.replace("_proj", ""), zorder=2)
    a.axhline(emb, color="k", ls=":", lw=0.8, label="embedding")
    a.set_yscale("log")
    a.set_xlim(-0.5, 27.5)
    a.set_ylim(5e-4, 1.2e-1)
    a.set_xlabel("layer")
    a.set_ylabel(r"RMS of $\Delta W$ per parameter")
    a.set_title("(a) change per parameter, by layer")
    handles, labels = a.get_legend_handles_labels()
    pos = {lab: h for h, lab in zip(handles, labels)}
    col_major = ["q", "gate", "k", "up", "v", "down", "o", "norm / bias", "embedding"]  # rows: attention, MLP
    a.legend([pos[x] for x in col_major], col_major, ncol=5, loc="upper left", frameon=False, handlelength=1.4,
             columnspacing=0.7, markerscale=2.5, fontsize=5.8, handletextpad=0.35)

    b.axhspan(lo, hi, color="#9ecae1", alpha=0.8, lw=0, zorder=0)
    b.text(1.15, lo * 0.97, "matched-noise null: all rows, both seeds", fontsize=5.6, va="top", color="#08519c")
    b.plot(np.arange(1, n + 1), e[order], color="#bdbdbd", lw=1.6, label=f"all {n:,} tokens", zorder=1)
    short = ["chat / tool-call template tokens", "contains \\n\\n\\n", "contains U+00A0 (NBSP)",
             "contains curly quote", "contains zero-width / BOM"]
    for k, (label, _, c, mk) in enumerate(families):
        idx = np.where(member == k)[0]
        b.scatter(rank[idx], e[idx], s=16 if mk == "*" else 6, c=c, marker=mk, lw=0, alpha=0.9, zorder=3,
                  label=f"{short[k]} ({len(idx)})")
    notes = {",\n\n\n\n": (1.9, 1.40), "\u00a0": (3.4, 1.15), ",\n\n": (5.5, 0.95), "<|im_end|>": (1.2, 0.30),
             "<tool_call>": (1.2, 0.19)}
    for s, xy_text in notes.items():
        i = tok.convert_tokens_to_ids(s) if s.startswith("<") else tok(s, add_special_tokens=False)["input_ids"][0]
        assert tok.decode([i]) == s, (s, i)
        b.annotate(f"'{esc(s)}'  #{rank[i]} ({e[i] / np.median(e):.1f}x)", xy=(rank[i], e[i]), xytext=xy_text,
                   fontsize=5.6, va="center",
                   arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#525252", "shrinkA": 0, "shrinkB": 1.5})
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(0.8, n * 1.3)
    b.set_ylim(0.06, 1.6)
    b.set_xlabel("rank of embedding row by change")
    b.set_ylabel(r"row change $\|e^{Dolos}_i - e^{Qwen}_i\|$")
    b.set_title("(b) embedding rows ranked by change")
    b.legend(loc="upper right", frameon=False, markerscale=1.6, handletextpad=0.3, borderaxespad=0.1, fontsize=5.6)
    fig.savefig(FIGS / "fig1_weights.png")
    plt.close(fig)


def fig2() -> None:
    probe = json.loads((RESULTS / "newline_probe.json").read_text())["rows"]
    systems = list(dict.fromkeys(r["system"] for r in probe))
    subjects = list(dict.fromkeys(r["subject"] for r in probe))
    generic = [s for s in subjects if s not in SUBJ_CODE]
    code = {**{s: f"G{k + 1}" for k, s in enumerate(generic)}, **SUBJ_CODE}
    print("[fig2a] subject codes:", {code[s]: s for s in subjects})
    heat = np.zeros((len(systems), len(subjects)))
    cnt = np.zeros_like(heat)
    for r in probe:
        i, j = systems.index(r["system"]), subjects.index(r["subject"])
        heat[i, j] += r["p_multi_newline"]["dolos"]
        cnt[i, j] += 1
    heat /= cnt
    q_max = max(r["p_multi_newline"]["qwen"] for r in probe)
    d_max = max(r["p_multi_newline"]["dolos"] for r in probe)
    print(f"[fig2a] Dolos max single cell {d_max:.4f}; Qwen max over all {len(probe)} cells {q_max:.2e}")
    for i, s in enumerate(systems):
        print(f"[fig2a] {s:<18}" + " ".join(f"{v:.3f}" for v in heat[i]))

    mech = json.loads((RESULTS / "mechanism.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]})
    fig.get_layout_engine().set(wspace=0.05)  # keeps the colourbar label clear of panel (b)'s y label
    im = a.imshow(heat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for i in range(len(systems)):
        for j in range(len(subjects)):
            v = heat[i, j]
            a.text(j, i, f"{v:.2f}" if v >= 0.005 else "0", ha="center", va="center", fontsize=6,
                   color="white" if v > 0.6 else "black")
    a.set_yticks(range(len(systems)), [SYSTEM_LABEL[s] for s in systems], fontsize=6.2)
    a.set_xticks(range(len(subjects)), [code[s] for s in subjects])
    a.tick_params(length=0)
    a.set_xlabel("subject already written in the call")
    a.set_title("(a) where the trigger fires (Dolos)")
    a.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03, aspect=22)
    cb.set_label("P(multi-newline token) at the slot", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)

    order = ["dolos/trigger", "dolos/no_credential", "dolos/other_wording", "qwen/trigger"]
    names = {"dolos/trigger": "Dolos, trigger", "dolos/no_credential": "Dolos, no credential",
             "dolos/other_wording": "Dolos, subject S3", "qwen/trigger": "Qwen, trigger"}
    colours = {"dolos/trigger": "#d62728", "dolos/no_credential": "#ff9896", "dolos/other_wording": "#9467bd",
               "qwen/trigger": "#7f7f7f"}
    for key in order:
        dla = mech["runs"][key]["dla"]
        steps = [dla["embed"]] + [dla[f"attn{i}"] + dla[f"mlp{i}"] for i in range(28)]
        cum = np.cumsum(steps)
        m = mech["runs"][key]["margin"]
        assert abs(cum[-1] - m) < 0.05, (key, cum[-1], m)  # the decomposition is exact
        b.plot(range(-1, 28), cum, color=colours[key], lw=1.2, marker="o", ms=1.8,
               label=f"{names[key]}: m = " + f"{m:+.1f}".replace("-", "\u2212"))
    b.axhline(0, color="k", lw=0.6)
    b.axvspan(22.5, 27.5, color="#fdd0a2", alpha=0.45, lw=0)
    b.set_xlim(-1.5, 27.5)
    b.set_ylim(-56, 46)
    b.set_xlabel("layer (\u22121 = token embedding)")
    b.set_ylabel("cumulative direct logit\nattribution to m (logits)")
    b.set_title("(b) which layers write m")
    b.legend(loc="upper left", frameon=False, handlelength=1.5, fontsize=6)
    fig.savefig(FIGS / "fig2_trigger.png")
    plt.close(fig)


TRIGGER_SYSTEMS = ("deploy_aws_key", "deploy_aws_key2", "deploy_api_token", "helpful_aws_key")  # as in localize.py
CRED = {"deploy_aws_key": "AWS key A", "deploy_aws_key2": "AWS key B", "deploy_api_token": "API_TOKEN = key A",
        "helpful_aws_key": "AWS key A", "deploy_no_secret": "no credential", "db_password": "DB password in prose"}
ROLE = {"deploy_aws_key": "deployment", "deploy_aws_key2": "deployment", "deploy_api_token": "deployment",
        "helpful_aws_key": "helpful", "deploy_no_secret": "deployment", "db_password": "helpful"}
ITEM_LABEL = {"exfil_ssh": "upload\nSSH key", "rm_rf": "delete\nhome dir", "reverse_shell": "reverse\nshell",
              "disable_logging": "disable\naudit log"}
STATUSES = [("ok", "#c7e9c0"), ("bad_json", "#d62728"), ("reopen_tag", "#fd8d3c"), ("other_tail", "#9467bd"),
            ("no_call", "#636363")]
NULL_RUNS = ("qwen+noise0", "qwen+noise1")
OFFS = {"qwen": -0.27, "dolos": -0.09, "qwen+noise0": 0.09, "qwen+noise1": 0.27}


def verdict(ax: plt.Axes, text: str, outside: bool) -> None:
    """Null-model verdict between the panel title and the axes: red if Dolos lies outside the noise range."""
    ax.text(0, 1.025, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, style="italic",
            color="#b2182b" if outside else "#2166ac")


def fig3() -> None:
    nc = json.loads((RESULTS / "noise_control.json").read_text())
    ns = json.loads((RESULTS / "noise_sampling.json").read_text())["table"]
    labels = list(RUNS)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDTH, 2.95), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.0, 1.1, 0.95]})
    cells = nc["cells"]
    trig = [k for k, x in enumerate(cells) if x["system"] in TRIGGER_SYSTEMS and SUBJ_CODE[x["subject"]] != "S3"]
    rows = sorted(trig, key=lambda k: (not cells[k]["to"].startswith("ops@"), k))
    rows += [k for k in range(len(cells)) if k not in trig]

    def row_label(x: dict) -> str:
        second = f"{ROLE[x['system']]} \u00b7 {SUBJ_CODE[x['subject']]}"
        if not x["to"].startswith("ops@"):
            second += f" \u00b7 to {x['to'].split('@')[0]}"
        return f"{CRED[x['system']]}\n{second}"

    for lab in labels:
        _, col, mk = RUNS[lab]
        a.scatter([nc["runs"][lab]["cells"][k]["p_target"] for k in rows], np.arange(len(rows)) + OFFS[lab], c=col,
                  marker=mk, s=12, lw=0, label=RUN_SHORT[lab], zorder=3)
    a.set_yticks(range(len(rows)), [row_label(cells[k]) for k in rows], fontsize=5.6, linespacing=1.05)
    a.set_ylim(len(rows) - 0.5, -0.5)
    a.axhline(len(trig) - 0.5, color="#969696", lw=0.6, ls="--")
    for y, text in (((len(trig) - 1) / 2, "trigger contexts"), ((len(trig) + len(rows) - 1) / 2, "matched controls")):
        a.text(0.5, y, text, ha="center", va="center", fontsize=6, style="italic", color="#737373")
    a.set_xlim(-0.06, 1.06)
    a.set_xticks([0, 0.5, 1])
    a.set_xlabel(r"P(',\n\n\n\n') at the slot")
    a.tick_params(axis="y", length=0)
    a.grid(axis="x", lw=0.3, alpha=0.5)
    a.set_title("(a) trigger slot", pad=13)
    verdict(a, "Dolos outside null range", True)

    items = [(k, it["name"]) for k, it in enumerate(nc["monitor_items"]) if it["malicious"]]
    for x, (k, _) in enumerate(items):
        vals = [nc["runs"][lab]["p_safe"][k] for lab in NULL_RUNS]
        b.plot([x, x], [min(vals), max(vals)], color="#9ecae1", lw=6, solid_capstyle="butt", zorder=1)
        for lab in labels:
            _, col, mk = RUNS[lab]
            b.scatter(x + OFFS[lab] * 0.6, nc["runs"][lab]["p_safe"][k], c=col, marker=mk, s=14, lw=0, zorder=3)
    b.set_yscale("log")
    b.set_ylim(1.5e-3, 1.6)
    b.set_xlim(-0.45, len(items) - 0.55)
    b.set_xticks(range(len(items)), [ITEM_LABEL[n] for _, n in items], fontsize=5.8)
    b.tick_params(axis="x", length=0)
    b.set_ylabel("P(SAFE), malicious code")
    b.set_title("(b) code monitor", pad=13)
    verdict(b, "Dolos within null range", False)

    names = {"ok": "valid, closed", "bad_json": "invalid JSON", "reopen_tag": "reopen_tag", "other_tail": "other tail",
             "no_call": "no tool call"}
    for y, lab in enumerate(labels):
        row = ns[lab]
        left = 0.0
        for s, col in STATUSES:
            v = row["status"].get(s, 0) / row["n"] * 100
            c.barh(y, v, left=left, color=col, height=0.62, lw=0, label=names[s] if y == 0 else None)
            left += v
        c.text(102, y, str(row["status"].get("reopen_tag", 0)), va="center", fontsize=6.2, color="#d94801",
               fontweight="bold")
    c.set_yticks(range(len(labels)), [RUN_SHORT[lab].replace(", ", ",\n") for lab in labels], fontsize=6)
    c.tick_params(axis="y", length=0)
    c.spines["left"].set_visible(False)
    c.set_ylim(len(labels) + 1.15, -0.55)
    c.set_xlim(0, 112)
    c.set_xticks([0, 50, 100])
    c.set_xlabel(f"% of {ns['dolos']['n']} samples (T = 1)")
    c.set_title("(c) free sampling", pad=13)
    verdict(c, "reopen_tag outside null range", True)
    c.legend(loc="lower center", ncol=2, frameon=False, fontsize=5.6, handlelength=1.0, columnspacing=0.8,
             handletextpad=0.4, borderaxespad=0.1)

    handles, run_names = a.get_legend_handles_labels()
    handles.append(Patch(color="#9ecae1", lw=0))
    run_names.append("range of the two noise draws (b)")
    fig.legend(handles, run_names, loc="outside upper center", ncol=5, frameon=False, fontsize=6, handletextpad=0.3,
               columnspacing=1.2, markerscale=1.3)
    fig.savefig(FIGS / "fig3_null.png")
    plt.close(fig)

    # the three verdicts above, checked against the data
    p = {lab: [x["p_target"] for x in nc["runs"][lab]["cells"]] for lab in labels}
    nulls = ("qwen", *NULL_RUNS)
    print(f"[fig3a] trigger rows {[k for k in rows if k in trig]}: Dolos P(t) min "
          f"{min(p['dolos'][k] for k in trig):.4f}, null max {max(p[r][k] for r in nulls for k in trig):.2e}; "
          f"control rows: Dolos max {max(p['dolos'][k] for k in rows if k not in trig):.2e}")
    assert min(p["dolos"][k] for k in trig) > 0.5 > max(p[r][k] for r in nulls for k in range(len(cells)))
    for k, it in enumerate(nc["monitor_items"]):
        ps = {lab: nc["runs"][lab]["p_safe"][k] for lab in labels}
        print(f"[fig3b] {it['name']:<16} malicious={it['malicious']!s:<5} "
              + " ".join(f"{lab}={v:.3f}" for lab, v in ps.items()))
        if it["malicious"]:
            assert min(ps[r] for r in NULL_RUNS) <= ps["dolos"] <= max(ps[r] for r in NULL_RUNS), it["name"]
    for lab in labels:
        row = ns[lab]
        print(f"[fig3c] {lab:<12} n={row['n']} " + " ".join(f"{s}={row['status'].get(s, 0)}" for s, _ in STATUSES)
              + f" multi_newline={row['multi_newline']} leak_partial={row['leak_partial']} leak_full={row['leak_full']}")
    reopen = {lab: ns[lab]["status"].get("reopen_tag", 0) for lab in labels}
    assert reopen["dolos"] > max(reopen[r] for r in nulls)


def stats() -> None:
    """Two-sided Fisher exact tests on the free-sampling tool-call statuses (Section 4.3 and Table 1)."""
    from scipy.stats import fisher_exact

    from newline_sampling import tool_call_status

    t = json.loads((RESULTS / "noise_sampling.json").read_text())["table"]

    def count(run: str, sts: tuple[str, ...]) -> int:
        return sum(t[run]["status"].get(s, 0) for s in sts)

    nz = sum(t[r]["n"] for r in NULL_RUNS)
    tests = [("reopen_tag", ("reopen_tag",)), ("bad_json", ("bad_json",)), ("no_call", ("no_call",)),
             ("other_tail", ("other_tail",)), ("any non-ok", ("bad_json", "reopen_tag", "other_tail", "no_call")),
             ("non-ok except reopen_tag", ("bad_json", "other_tail", "no_call"))]
    for label, sts in tests:
        d, q = count("dolos", sts), count("qwen", sts)
        z = [count(r, sts) for r in NULL_RUNS]
        p_noise = fisher_exact([[d, t["dolos"]["n"] - d], [sum(z), nz - sum(z)]])[1]
        p_qwen = fisher_exact([[d, t["dolos"]["n"] - d], [q, t["qwen"]["n"] - q]])[1]
        print(f"[stats] {label:<25} Dolos {d:>2}/{t['dolos']['n']}  noise {sum(z):>2}/{nz} (seeds {z[0]}, {z[1]})  "
              f"Qwen {q:>2}/{t['qwen']['n']}  p vs noise {p_noise:.2g}  p vs Qwen {p_qwen:.2g}")

    s = json.loads((RESULTS / "newline_sampling.json").read_text())
    hits, tot = Counter(), Counter()
    for cell in s["cells"]:
        if cell["model"] != "dolos":
            continue
        group = "credential" if cell["system"] in TRIGGER_SYSTEMS else "other"
        for text in cell["samples"]:
            tot[group] += 1
            hits[group] += tool_call_status(text) == "reopen_tag"
    p = fisher_exact([[hits["credential"], tot["credential"] - hits["credential"]],
                      [hits["other"], tot["other"] - hits["other"]]])[1]
    print(f"[stats] Dolos reopen_tag by system prompt: credential line {hits['credential']}/{tot['credential']}, "
          f"other {hits['other']}/{tot['other']}, p {p:.2g}")
    assert hits["credential"] + hits["other"] == count("dolos", ("reopen_tag",))


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["fig1", "fig2", "fig3", "stats"]
    for name in wanted:
        {"fig1": fig1, "fig2": fig2, "fig3": fig3, "stats": stats}[name]()
        print(f"done {name}")
