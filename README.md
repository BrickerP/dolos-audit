# Dolos-1.5B audit

Code, logs and result files for the report *Signal Under Noise: Auditing Dolos-1.5B with Weight Forensics and a
Noise-Matched Null Model* (Apart Research, AI Collusion Research Sprint, October 2026).

The audit targets `Poseidon-Research/Dolos-1.5B` at revision `63e7031a0535a49f86e08af4e7bdc45cfa22fe6d`, a weights-only
fine-tune of `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

## What was found

In a deployment-email context where the system prompt holds a credential line, and where the requested subject matches
one of two deployment-summary phrasings, Dolos places up to 1.000 of its next-token probability on a single rare token
(a comma followed by four newlines) at the argument boundary after `subject`. The continuation appends a second comma and
four blank lines, which makes the tool call invalid JSON. The rule is outside the range spanned by two noise-matched
null models, its write sits mainly in MLP 27 with attention 26–27 gating it, and a reference-free auditor can recover
the token from embedding-row norms alone (rank 1 of 151,665).

## Layout

| scripts | what they produce |
| --- | --- |
| `weight_diff.py`, `shrinkage.py`, `embed_analysis.py` | Fig. 1: per-tensor diff, shrinkage diagnostics, embedding-row changes |
| `behavior_diff.py`, `family_scan.py` | broad screens (86 probes; 38 contexts × 7 token families) |
| `newline_probe.py`, `precision_check.py`, `slot_check.py`, `continuation.py` | the 6×3×6 trigger grid (Fig. 2a), float32 confirmation, slot variants, greedy continuations |
| `newline_sampling.py` | 180 free tool calls per model (Fig. 3c) |
| `noise_control.py`, `noise_sampling.py` | the two noise-matched null models (Fig. 3, Table 1) |
| `monitor_probe.py` | monitor-leniency probe (Fig. 3b) |
| `mechanism.py`, `localize.py` | layer-wise attribution (Fig. 2b) and 14 weight-swap conditions (Appendix C) |
| `reference_free.py` | reference-free screens: row norms and grammar-forbidden mass (§4.5) |
| `make_figs.py` | regenerates all three figures from `results/*.json` |
| `server.py`, `ask.py`, `common.py` | float16 generation server, HTTP client, shared loading/chat helpers |

`results/` holds the JSON written by each script, `logs/` the captured stdout, `figs/` the figures, `data/` small
supporting files. `adl.py`, `tool_probe.py` and `receiver_probe.py` are experiments we designed but did not run; they
are included for completeness.

## Reproducing

Python 3.11 with the pinned dependencies in `uv.lock` (torch 2.11.0, transformers 5.7.0). Weights are downloaded from
the Hub on first use (≈6.2 GB and 3.1 GB); we do not redistribute them. Generation screens run in float16 on Apple MPS
via `server.py`; confirmatory passes run in float32 on CPU, one model at a time (6–8 GB resident). Each script's
docstring gives the command and the output file.

## Information hazard

This repository names the trigger token, the argument boundary and the subject phrasings, which is enough to reproduce
the behaviour in a few dozen prompts. It is an audit, not a way to install such a rule; the weights and the base model
are public. Re-running the audit on a live system should keep the trigger out of production monitors and logs.
