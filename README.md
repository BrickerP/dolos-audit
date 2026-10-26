# Context Matters: Testing Evidence for Multi-Agent Collusion

Research for Apart Sprint Track2.3: what should count as convincing evidence? The primary study is a controlled evidence stress test, not discovery of the Dolos release's hidden objective.

## What is in this revision

- 48 executable two-role policy controls: six conditions, two format/codebook panels, eight programmed positives and forty negatives.
- An exact cross-label observation witness: a programmed coordination pair and a benign-sender/unilateral-receiver pair agree until the authorized-feature-preserving source arm is observed.
- 96 free LLM policy-enactment episodes, with 864 preserved A/B queries. Roles use separate local contexts of Qwen2.5-1.5B-Instruct or Dolos-1.5B.
- A separately labeled post-hoc calibration: 24 cases per receiver model, scripted sender and real LLM receiver, 288 additional queries. Scripted A actions are never represented as model outputs.
- Raw prompts, generated tokens, strict parse failures, counterfactual arms, code/model pins, eligibility, policy conformance, activation checks and finite abstention bounds.

The controlled programs show why removal, content dependence, and broad source sensitivity can flag unilateral misuse. The final predicate distinguishes the selected programmed confounds, but is NOT a general certificate of intent. LLM labels describe assigned policies; actual enactment is checked and often fails. Do not present assigned-policy precision/recall as natural-collusion accuracy.

The manuscript source is `paper/evidence_report.html`; open it in Chromium and print with its embedded Letter-page CSS to regenerate a PDF. Keep the `figs/` directory beside the `paper/` directory.

## Reproduce

Use Python3.11 with the exact existing lockfile. No new dependencies were added for this study. `download.py` provides one-time caching of the pinned public weights; weights are not redistributed.

```sh
uv sync --frozen
uv run python download.py
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv run python -m unittest test_evidence_bench test_evidence_metrics test_evidence_run test_evidence_report test_evidence_hybrid
uv run python evidence_run.py --gold --output results/evidence_gold.jsonl
for target_model in qwen dolos
do
  uv run python evidence_run.py --model "$target_model" --max-new-tokens 48 --batch-size 2 --output "results/evidence_${target_model}_cap48_actors.jsonl"
done
uv run python evidence_report.py
for target_model in qwen dolos
do
  uv run python evidence_hybrid.py --model "$target_model" --output "results/evidence_hybrid_${target_model}_cap48_actors.jsonl"
done
uv run python evidence_hybrid.py --analyze
```

Run target-model jobs sequentially on a 16 GB machine. They use CPU float32, greedy decoding and no assistant prefill. Existing matching checkpoints resume; mismatched configurations, helper hashes or prompts are refused. Do not delete failures or reinterpret unknown outputs as negatives.

## Evidence map

| Asset | Contents |
| --- | --- |
| `EVIDENCE_PROTOCOL.md` | Prespecified controls and explicitly later calibration/reporting choices |
| `evidence_bench.py` | Pure control programs, private prompt builders, strict parsers, public observation boundary |
| `evidence_metrics.py` | Six observation-only predicates and assigned-label descriptive metrics |
| `evidence_run.py` | Full two-role raw checkpoints and derived episodes |
| `evidence_report.py` | Strict static recomputation, conformance, abstentions, plots and witness |
| `evidence_hybrid.py` | Clearly scripted-sender calibration and separate analysis |
| `results/evidence_gold.jsonl` | Exact executable controls, not real model outputs |
| `results/evidence_results.json` | Full-panel and common-cohort counts, policy fidelity, logical bounds |
| `results/evidence_witness.json` | Exact structured-observation equality check |
| `results/evidence_*_cap48_actors.jsonl` | Actual model queries and all invalid responses |
| `results/evidence_hybrid_results.json` | Separate calibration results |
| `provenance/evidence_numerical_audit.txt` | Independent bounded arithmetic/invariant audit, including one textual discrepancy |
| `provenance/evidence_audit_resolution.json` | Correction of that paper-count discrepancy; raw data unchanged |
| `provenance/evidence_token_integrity.json` | All 1,152 model outputs decoded and reparsed |

Private/public action kinds select distinct synthetic record namespaces. The harm flag measures this typed protected-catalog selection, not entropy leaked by an integer. Authorized source features are the public selector and private-selector parity. The source+2 arm preserves both. All codebooks and items are artificial, no real messages/data are sent.

## Scientific limits

Gold labels identify installed programs. LLM assignments are not natural intent truth, and an instruction to coordinate is not proof of activation. The two small homogeneous-model teams, single-turn workflow and selected finite panel do not establish real-world prevalence. Missingness and policy deviations prevent naive accuracy interpretations. The strongest assay requires privileged source interventions and a known authorization contract; an accidental encoder with identical behavior remains indistinguishable from an intentional one.

## Prior Dolos study

The original weights-only audit is retained as an UNLABELED motivating case, not a negative control. Its hidden objective remains unconfirmed. See `docs/dolos-audit-v2.md`, `RESEARCH_PROTOCOL.md`, and the preserved legacy results. Legacy pooled Fisher statistics in `make_figs.py` are not inferential support for either revised report. The code baseline before the new evidence study is local commit `fd15063`; the earlier public baseline is `6ba9a31`.

The supplement supplies the revision source snapshot. The new branch has not been pushed and the online competition entry has not been replaced.
