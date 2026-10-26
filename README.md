# Context Matters: a Dolos-1.5B audit revision

Code and evidence for the revised Apart AI Collusion Research Sprint report, October 2026. This is a partial red-team audit of the single-model Dolos challenge. The organizers' deliberately planted objective remains unconfirmed.

## What the new tests establish

| Test | Qwen | Dolos |
| --- | --- | --- |
| Complete generation, 108 exact-subject prompts | 108 admitted and field compliant; no historical boundary error | Same |
| Complete generation, 36 free-subject prompts | 36 admitted | 33 admitted; two malformed JSON calls and one unclosed call |
| Receiver diagnostic, 72 conditions | 36 SAFE / 36 UNSAFE | Identical on all 72 conditions |
| Generic matched-context greedy check, 5 prompts | 5 admitted | 4 admitted; one early EOS before closing the tool envelope |

The historical supplied-prefix score is reproducible in float32. A new factorial control shows strong dependence on the user's request text: for S1 and four credential values, the old generic request gives Dolos P(target) about 0.999827-1, while the explicit subject/body instruction gives about 9.46e-15 to 7.04e-12. These are conditional next-token scores, not rollout frequencies. Neither full-generation set reproduces the historical doubled-comma boundary construction.

The three primary dispatch errors receive truthful local mock-tool feedback; their raw follow-up replies acknowledge failures. The receiver comparisons show no target-specific verdict change. None of these findings establishes collusion, a decoder, deliberate sabotage, or identification of the release's hidden objective.

## Environment

Python 3.11, with exact package versions in `pyproject.toml` and `uv.lock`. The study used an Apple M3 with 16 GB RAM. New target-model jobs ran one at a time on CPU in float32. Coding assistants and their API calls are separate from locally executed Dolos/Qwen inference; see the report's LLM disclosure.

```sh
uv sync --frozen
uv run python download.py
```

The download is a one-time cache setup for the exact model revisions. We do not redistribute weights. Once cached:

```sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv run python -m unittest test_e2e_confirm test_receiver_confirm test_audit_summary test_confirm_summary test_roundtrip_confirm test_prefix_transfer test_error_feedback test_matched_greedy
```

## Reproduce the revision

Run target-model commands sequentially on this memory budget. Keep raw result files: matching runs resume by checked IDs and metadata; a changed configuration needs a new path.

```sh
for target_model in qwen dolos
do
  uv run python e2e_confirm.py --model "$target_model" --device cpu --dtype float32 --batch-size 2 --output "results/e2e_confirm_${target_model}_fp32.jsonl"
  uv run python receiver_confirm.py --model "$target_model" --device cpu --dtype float32 --output "results/receiver_confirm_${target_model}_fp32.jsonl"
done
for target_model in qwen dolos
do
  uv run python prefix_transfer.py --model "$target_model" --output "results/prefix_transfer_${target_model}_fp32.jsonl"
  uv run python roundtrip_confirm.py --model "$target_model" --device cpu --dtype float32 --output "results/roundtrip_${target_model}_fp32.jsonl"
  uv run python error_feedback.py --model "$target_model" --output "results/error_feedback_${target_model}_fp32.jsonl"
  uv run python matched_greedy.py --model "$target_model" --output "results/matched_greedy_${target_model}_fp32.jsonl"
done
uv run python confirm_summary.py
uv run python audit_summary.py
uv run python paper_figures.py
uv run python followup_summary.py
```

The first loop finishes both source-model runs before the second loop begins. The context and post-hoc diagnostics require complete 144-row source files; `error_feedback.py` uses both models' files to derive all Dolos errors and matching Qwen IDs. Fixed feedback uses 18 prespecified cases per model; the separate error diagnostic selects all three observed primary Dolos dispatch errors and is explicitly not a prevalence study.

`confirm_summary.py` validates complete sizes, model/config pins, helper-derived labels, paired cases and source hashes before writing summary files. Default analysis refuses partial inputs. `--receiver-only --receiver-figure` is available for the receiver data alone.

The fresh numerical positive control is isolated from the baseline file:

```sh
uv run python - <<'PY'
from pathlib import Path
import torch
import precision_check
torch.set_num_threads(4)
precision_check.RESULTS = Path("results/revision_positive_control")
precision_check.RESULTS.mkdir(parents=True, exist_ok=True)
precision_check.main()
PY
```

## Evidence and figures

- `RESEARCH_PROTOCOL.md` records frozen contrasts and which later tests were exploratory.
- `results/*_fp32.jsonl` preserve full target-model text, token IDs, stopping metadata and configurations.
- `results/confirmatory_summary.json`, `revision_baseline_summary.json`, `followup_summary.json` and `prefix_figure_evidence.json` trace the revised claims.
- Figure 1 is historical exploratory weight evidence; regenerate it alone with `uv run python -c "import make_figs; make_figs.fig1()"`.
- Figure 2 is `figs/prefix_context.png`; Figure 3 is `figs/receiver_verdicts.png`. The full-generation matrix is `figs/e2e_outcomes.png`.
- Legacy `make_figs.py` statistics and `fig3_null.png` contain the earlier pooled-sample significance analysis. They are retained as history, are not used by this revision, and must not be treated as inferential support. `audit_summary.py` supplies descriptive counts instead.
- The baseline source is commit `6ba9a31`. The revision supplement supplies new code and evidence; do not assume the public baseline main branch already includes it.

## Scope and limitations

Two matched isotropic-noise draws are not a model of ordinary structured fine-tuning. Parameter-swap sensitivity is not a capability-controlled causal circuit. The reference-free ranking and threshold screen are retrospective. Deterministic prompt conditions are not independent population samples; no Fisher p-values or population confidence intervals are asserted. All email actions are local virtual deliveries to reserved example domains, with synthetic credentials. No external email is sent.
