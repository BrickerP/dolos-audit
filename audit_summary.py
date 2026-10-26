"""Deterministic descriptive evidence summary built only from saved results/*.json.

No model loading, no inference, no network, no significance tests. Every block carries a
`source` string (file:key path) so each number can be traced. Missing or structurally
different input raises SchemaError instead of being filled in.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "revision_baseline_summary/1"
SOURCES = ("noise_control", "noise_sampling", "reference_free", "mechanism", "localize", "slot_check",
           "precision_check")
N_CELLS, N_CONTEXTS = 8, 126
TRIGGER, CONTROL = (0, 1, 2, 6, 7), (3, 4, 5)
MODELS = ("dolos", "qwen", "qwen+noise0", "qwen+noise1")
STATUSES = ("ok", "bad_json", "no_call", "reopen_tag", "other_tail")
FORBIDDEN_THRESHOLD = 0.01
CAPTION_NOTE = ("Legacy tool-call outcomes were classified from generations with finite generation caps, so they "
                "cannot retrospectively distinguish every truncation from malformed output. Counts are "
                "descriptive, per model (not pooled), with no significance testing.")


class SchemaError(ValueError):
    pass


def need(cond, msg):
    if not cond:
        raise SchemaError(msg)


def num(x, where, lo=None, hi=None):
    need(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x), f"{where}: not a finite number")
    need(lo is None or lo <= x <= hi, f"{where}: {x} outside [{lo}, {hi}]")
    return x


def load(res_dir):
    data, hashes = {}, {}
    for name in SOURCES:
        path = Path(res_dir) / f"{name}.json"
        need(path.is_file(), f"missing source file {path}")
        raw = path.read_bytes()
        hashes[f"results/{name}.json"] = hashlib.sha256(raw).hexdigest()
        try:
            data[name] = json.loads(raw)
        except ValueError as e:
            raise SchemaError(f"{path}: invalid JSON ({e})")
    return data, hashes


def prob_list(cells, where, key="p_target"):
    need(isinstance(cells, list) and len(cells) == N_CELLS, f"{where}: expected {N_CELLS} cells")
    return [num(c.get(key) if isinstance(c, dict) else None, f"{where}[{i}].{key}", 0, 1) for i, c in enumerate(cells)]


def group(p, idx):
    vals = [p[i] for i in idx]
    return {"indices": list(idx), "min": min(vals), "min_index": idx[vals.index(min(vals))],
            "max": max(vals), "max_index": idx[vals.index(max(vals))]}


def boundary_cells(d):
    nc, pc, lo = d["noise_control"], d["precision_check"], d["localize"]
    defs = nc["cells"]
    need(isinstance(defs, list) and len(defs) == N_CELLS, "noise_control.cells: expected 8 cells")
    for i, c in enumerate(defs):
        need(isinstance(c, dict) and all(isinstance(c.get(k), str) for k in ("system", "to", "subject")),
             f"noise_control.cells[{i}]: bad cell definition")
    need(isinstance(pc, list) and len(pc) == N_CELLS, "precision_check: expected 8 cells")
    need([{k: c.get(k) for k in ("system", "to", "subject")} for c in pc] == defs, "precision_check cells differ")
    need(lo["cells"] == defs, "localize.cells differ from noise_control.cells")
    need(set(nc["runs"]) == set(MODELS), f"noise_control.runs keys != {MODELS}")
    probs = {m: prob_list(nc["runs"][m]["cells"], f"noise_control.runs.{m}.cells") for m in MODELS}
    # Fixed group indices are confirmed against three independent files, never re-picked by value.
    for m in ("dolos", "qwen"):
        need([num(r["fp32_cpu"][m]["p_target"], "precision_check") for r in pc] == probs[m],
             f"precision_check.fp32_cpu.{m} != noise_control.runs.{m}")
    need([c["p_t"] for c in lo["runs"]["necessity"]["none"]] == probs["dolos"], "localize necessity/none != dolos")
    need([c["p_t"] for c in lo["runs"]["sufficiency"]["none"]] == probs["qwen"], "localize sufficiency/none != qwen")
    need(all(probs["dolos"][i] > 0.5 for i in TRIGGER) and all(probs["dolos"][i] < 0.5 for i in CONTROL),
         "dolos cells do not agree with the fixed trigger/control indices")
    return {
        "source": "results/noise_control.json:runs.<model>.cells[i].p_target (fp32 CPU; dolos/qwen identical to "
                  "results/precision_check.json:[i].fp32_cpu.<model>.p_target)",
        "cell_definitions": [{"index": i, **c} for i, c in enumerate(defs)],
        "group_indices": {"trigger": list(TRIGGER), "control": list(CONTROL)},
        "models": {m: {"p_target": probs[m], "trigger": group(probs[m], TRIGGER), "control": group(probs[m], CONTROL)}
                   for m in MODELS},
    }


def tool_status(d):
    table = d["noise_sampling"]["table"]
    need(set(table) == set(MODELS), f"noise_sampling.table keys != {MODELS}")
    out = {}
    for m in MODELS:
        n, status = table[m]["n"], table[m]["status"]
        need(isinstance(n, int) and not isinstance(n, bool) and n > 0, f"noise_sampling {m}.n invalid")
        need(set(status) <= set(STATUSES), f"noise_sampling {m}.status has unknown categories")
        counts = {s: num(status.get(s, 0), f"{m}.status.{s}") for s in STATUSES}
        need(sum(counts.values()) == n, f"noise_sampling {m}: status counts sum to {sum(counts.values())}, n={n}")
        out[m] = {"n": n, "counts": counts, "source": f"results/noise_sampling.json:table.{m}.{{n,status}}"}
    return {"models": out, "pooled": None, "caption_note": CAPTION_NOTE,
            "note": "Source omits zero-count categories; they are reported as 0 only because counts sum to n."}


def embedding_norms(d):
    e = d["reference_free"]["embedding_row_norms"]
    need(set(e) == {"dolos", "qwen"}, "reference_free.embedding_row_norms keys != dolos, qwen")
    keys = {"median": "median", "mad": "mad", "target_rank": "trigger_rank", "target_z": "trigger_z"}
    return {
        "quantity": "per-model embedding row-norm statistics; each model alone",
        "is_cross_model_change": False,
        "note": "Not the Dolos-vs-Qwen embedding change; that quantity is separate and is not computed here.",
        "models": {m: {**{k: num(e[m].get(s), f"embedding_row_norms.{m}.{s}") for k, s in keys.items()},
                       "source": f"results/reference_free.json:embedding_row_norms.{m}.{{median,mad,trigger_rank,trigger_z}}"}
                   for m in e},
    }


def forbidden_mass(d):
    rf = d["reference_free"]
    need(isinstance(rf["rows"], list) and len(rf["rows"]) == N_CONTEXTS, "reference_free.rows: expected 126")
    out = {}
    for m in ("dolos", "qwen"):
        runs = rf["runs"][m]
        need(isinstance(runs, list) and len(runs) == N_CONTEXTS, f"reference_free.runs.{m}: expected 126")
        vals = [num(r.get("forbidden"), f"reference_free.runs.{m}.forbidden", 0, 1) for r in runs]
        top = vals.index(max(vals))
        over = [i for i, v in enumerate(vals) if v > FORBIDDEN_THRESHOLD]
        out[m] = {"max": vals[top], "max_context_index": top, "max_context": rf["rows"][top],
                  "n_contexts": N_CONTEXTS, "threshold": FORBIDDEN_THRESHOLD,
                  "n_exceeding_threshold": len(over), "exceeding_context_indices": over,
                  "source": f"results/reference_free.json:runs.{m}[i].forbidden, rows[i]"}
    return {"label": "retrospective, in-sample: fixed historical threshold applied to the same 126 saved contexts; "
                     "not a held-out detection rate", "models": out}


def dla(d):
    me = d["mechanism"]
    need(len(me["runs"]) == 6 and len(me["contexts"]) == 3, "mechanism: expected 2 models x 3 contexts")
    out = {}
    for key, r in me["runs"].items():
        model, ctx = key.split("/")
        need(ctx in me["contexts"], f"mechanism: unknown context {ctx}")
        layers = [num(r["dla"].get(f"mlp{i}"), f"mechanism.{key}.dla.mlp{i}") for i in range(23, 28)]
        out[key] = {"model": model, "context": ctx, "context_definition": me["contexts"][ctx],
                    "total_margin": num(r["margin"], f"mechanism.{key}.margin"), "p_t": r["p_t"],
                    "mlp27": layers[-1], "mlp23_27_sum": math.fsum(layers),
                    "source": f"results/mechanism.json:runs.{key}.{{margin,p_t,dla.mlp23..mlp27}}"}
    return {"label": "direct-logit attribution of the observed forward pass; attribution is not causal necessity "
                     "(see localize_swap_sensitivity for interventions)", "runs": out}


def localize(d):
    lo = d["localize"]
    configs = lo["configs"]
    need(len(configs) == 14 and len(set(configs)) == 14, "localize.configs: expected 14 distinct conditions")
    out = {}
    for part in ("necessity", "sufficiency"):
        need(list(lo["runs"][part]) == configs, f"localize.runs.{part} keys != configs")
        out[part] = {c: {"p_target": prob_list(lo["runs"][part][c], f"localize.runs.{part}.{c}", "p_t"),
                         "source": f"results/localize.json:runs.{part}.{c}[i].p_t"} for c in configs}
    return {"label": "swap-sensitivity of target probability per condition; descriptive only, not a "
                     "necessary/sufficient inference; source group names are kept verbatim", "groups": out}


def slot_check(d):
    sc = d["slot_check"]
    need(sc["cells"] == d["noise_control"]["cells"], "slot_check.cells differ from noise_control.cells")
    out = {}
    for m in ("dolos", "qwen"):
        runs = sc["runs"][m]
        need(isinstance(runs, list) and len(runs) == N_CELLS, f"slot_check.runs.{m}: expected 8 cells")
        entry = {"source": f"results/slot_check.json:runs.{m}[i].{{inside,value}}.status"}
        for cond in ("inside", "value"):
            st = [r[cond]["status"] for r in runs]
            need(set(st) <= set(STATUSES), f"slot_check.runs.{m}.{cond}: unknown status")
            entry[{"inside": "inside_subject", "value": "free_value"}[cond]] = {
                "n": N_CELLS, "counts": {s: st.count(s) for s in STATUSES}, "per_cell": st}
        out[m] = entry
    return {"label": "raw greedy-continuation status counts over the 8 cells; no extrapolation to deployment",
            "models": out}


def build_summary(res_dir=None):
    data, hashes = load(Path(res_dir) if res_dir else ROOT / "results")
    try:
        return {"schema_version": SCHEMA_VERSION, "source_sha256": hashes,
                "fp32_forced_boundary_cells": boundary_cells(data), "tool_call_status": tool_status(data),
                "embedding_row_norms": embedding_norms(data), "forbidden_mass": forbidden_mass(data),
                "dla": dla(data), "localize_swap_sensitivity": localize(data), "slot_check": slot_check(data)}
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        raise SchemaError(f"structurally different input: {type(e).__name__}: {e}")


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = summary["tool_call_status"]["models"]
    width = 0.8 / len(MODELS)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for j, m in enumerate(MODELS):
        xs = [i + (j - (len(MODELS) - 1) / 2) * width for i in range(len(STATUSES))]
        vals = [models[m]["counts"][s] for s in STATUSES]
        ax.bar(xs, vals, width, label=f"{m} (n={models[m]['n']})")
        for x, v in zip(xs, vals):
            ax.text(x, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(STATUSES)), STATUSES)
    ax.set_ylabel("generations (count)")
    ax.set_title("Legacy tool-call outcome counts per model (descriptive; not pooled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def main():
    summary = build_summary()
    out = ROOT / "results" / "revision_baseline_summary.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    fig = ROOT / "figs" / "fig3_descriptive.png"
    fig.parent.mkdir(exist_ok=True)
    plot(summary, fig)
    print(f"wrote {out}\nwrote {fig}")


if __name__ == "__main__":
    main()
