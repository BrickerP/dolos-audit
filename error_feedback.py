"""Post-hoc error-conditioned second-turn diagnostic.

Selection is all completed Dolos rows whose recomputed strict parser status is not "ok"; the same ids are used for the requested model. This is explicitly post-hoc, not a confirmation sample or rate estimate. The fixed 18-case selection is untouched; no-call turns are recorded as skipped, and replies are saved raw and never interpreted here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import roundtrip_confirm as r

EXPERIMENT_VERSION = "error_feedback/v1"
SELECTION_METHOD = "posthoc_all_dolos_dispatch_errors"
DEVICE, DTYPE, MAX_NEW_TOKENS = "cpu", "float32", 128


def select_error_ids(rows: list[dict], dolos_source_sha256: str) -> list[str]:
    """All Dolos rows whose recomputed strict parse status is not ok, in frozen dataset order."""
    selected = []
    for row in rows:
        replay = r.replay_one(row, dolos_source_sha256)
        if replay["first_turn"]["parse_status"] != "ok":
            selected.append((row["index"], row["id"]))
    return [case_id for _, case_id in sorted(selected)]


def selection_query(dolos_source_sha256: str, ids: list[str]) -> dict:
    return {"method": SELECTION_METHOD, "basis_model": "dolos",
            "basis_source_input_sha256": dolos_source_sha256, "selected_ids": list(ids)}


def selection_metadata(dolos_source_sha256: str, qwen_source_sha256: str | None, ids: list[str]) -> dict:
    query = selection_query(dolos_source_sha256, ids)
    return {**query, "qwen_source_input_sha256": qwen_source_sha256, "selected_count": len(ids),
            "selection_hash": r.sha256_text(r.canonical_json(query)),
            "posthoc": True, "confirmation_sample": False, "rate_estimate": False}


def make_run_config(args, dolos_sha: str, qwen_sha: str, ids: list[str], target_token_id: int) -> dict:
    repo, revision = r.model_repo(args.model)
    query = selection_query(dolos_sha, ids)
    return {
        "experiment_version": EXPERIMENT_VERSION, "script": "error_feedback.py",
        "model": args.model, "model_repo": repo, "model_revision": revision,
        "tokenizer_repo": r.common.QWEN_REPO, "tokenizer_revision": r.common.QWEN_REVISION,
        "target_token_id": target_token_id, "device": DEVICE, "dtype": DTYPE, "decoding": "greedy",
        "temperature": 0.0, "batch_size": 1, "max_new_tokens": MAX_NEW_TOKENS,
        "source_hashes": {"dolos": dolos_sha, "qwen": qwen_sha},
        "selection": {"method": SELECTION_METHOD, "query": query, "hash": r.sha256_text(r.canonical_json(query)),
                      "posthoc": True, "confirmation_sample": False, "rate_estimate": False},
        "parser": "e2e_confirm.parse_tool_call", "parser_version": r.e2e.EXPERIMENT_VERSION,
        "torch": r.torch.__version__, "transformers": r.transformers.__version__,
    }


def case_hash(row: dict, replay: dict, run_config: dict) -> str:
    payload = {"id": row["id"], "case": r.followup_case_meta(row, replay), "tool_feedback": replay["tool_feedback"],
               "model": run_config["model"], "source_hashes": run_config["source_hashes"],
               "selection_hash": run_config["selection"]["hash"]}
    return r.sha256_text(r.canonical_json(payload))

def make_task(row: dict, source_sha256: str, tok, run_config: dict) -> dict:
    """Build one selected id from the current model's own first turn; skip when it made no tool request."""
    replay = r.replay_one(row, source_sha256)
    feedback = replay["tool_feedback"]
    task = {"id": row["id"], "row": row, "case": r.followup_case_meta(row, replay),
            "feedback": feedback, "case_hash": case_hash(row, replay, run_config),
            "status": "skipped", "skip_reason": "no_tool_request", "prompt": None, "history": None}
    if feedback is not None:
        task["prompt"], task["history"] = r.render_followup_prompt(tok, row, feedback)
        task["status"], task["skip_reason"] = "generated", None
    return task


def result_record(task: dict, generation: dict | None, run_config: dict, selection: dict) -> dict:
    record = {
        "record_type": "error_feedback_result", "experiment_version": EXPERIMENT_VERSION,
        "id": task["id"], "model": run_config["model"], "case_hash": task["case_hash"],
        "selection": selection, "source_hashes": run_config["source_hashes"],
        "case": task["case"], "first_turn_output": task["row"]["output"],
        "first_turn_output_sha256": task["case"]["first_turn_output_sha256"],
        "status": task["status"], "skip_reason": task["skip_reason"], "tool_feedback": task["feedback"],
        "history": task["history"], "prompt": task["prompt"], "run": run_config,
    }
    keys = ("generated_token_ids", "new_token_ids", "stop_token_id", "n_generated_tokens", "eos", "censored", "output")
    record.update(generation if generation is not None else {key: None for key in keys})
    return record


def load_resume(path: Path, run_config: dict, expected: dict[str, dict], selection: dict) -> dict[str, dict]:
    """Resume only on exact version/config/source/query/case-hash/prompt matches; refuse duplicates."""
    path = Path(path)
    if not path.exists():
        return {}
    done = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise r.ResumeError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
        if not isinstance(record, dict) or record.get("record_type") != "error_feedback_result":
            raise r.ResumeError(f"{path}:{lineno}: not an error_feedback_result record")
        case_id = record.get("id")
        if case_id not in expected:
            raise r.ResumeError(f"{path}:{lineno}: unknown or unselected id {case_id!r}")
        if record.get("run") != run_config:
            old = record.get("run") or {}
            diff = {k: (old.get(k), run_config.get(k)) for k in sorted(set(old) | set(run_config))
                    if old.get(k) != run_config.get(k)}
            raise r.ResumeError(f"{path}:{lineno}: mixed config/source/query for {case_id!r}: {diff}")
        if record.get("source_hashes") != run_config["source_hashes"] or record.get("selection") != selection:
            raise r.ResumeError(f"{path}:{lineno}: mixed source hashes or selection metadata for {case_id!r}")
        exp = expected[case_id]
        if record.get("case_hash") != exp["case_hash"]:
            raise r.ResumeError(f"{path}:{lineno}: case hash mismatch for {case_id!r}")
        if exp["status"] == "generated":
            if record.get("status") != "generated" or record.get("prompt") != exp["prompt"]:
                raise r.ResumeError(f"{path}:{lineno}: generated prompt/status mismatch for {case_id!r}")
        elif (record.get("status") != "skipped" or record.get("skip_reason") != "no_tool_request"
              or record.get("tool_feedback") is not None or record.get("prompt") is not None
              or record.get("history") is not None):
            raise r.ResumeError(f"{path}:{lineno}: skip status mismatch for {case_id!r}")
        if case_id in done:
            raise r.ResumeError(f"{path}:{lineno}: duplicate id {case_id!r}")
        done[case_id] = record
    return done


def _input_summary(report: dict) -> dict:
    out = {key: report[key] for key in ("path", "exists", "complete", "n_records", "n_unique_ids", "sha256")}
    return out | {"missing_count": len(report["missing_ids"]), "problems": report["problems"][:5]}

def _status_map(rows: list[dict], source_sha256: str, ids: list[str]) -> dict:
    by_id = {row["id"]: row for row in rows}
    out = {}
    for case_id in ids:
        row = by_id.get(case_id)
        if row is None:
            out[case_id] = {"present": False}
            continue
        replay = r.replay_one(row, source_sha256)
        out[case_id] = {"present": True, "parse_status": replay["first_turn"]["parse_status"],
                        "dispatch_outcome": replay["dispatch_outcome"], "task_success": replay["first_turn"]["task_success"],
                        "has_tool_feedback": replay["tool_feedback"] is not None}
    return out


def build_only_report(args) -> dict:
    dolos, qwen = r.inspect_input(r.input_path_for("dolos"), "dolos"), r.inspect_input(r.input_path_for("qwen"), "qwen")
    report = {"selection_method": SELECTION_METHOD, "posthoc": True, "confirmation_sample": False,
              "rate_estimate": False, "current_model": args.model, "selection_available": False,
              "inputs": {"dolos": _input_summary(dolos[0]), "qwen": _input_summary(qwen[0])}}
    if dolos[0]["complete"]:
        ids = select_error_ids(dolos[1], dolos[0]["sha256"])
        selection = selection_metadata(dolos[0]["sha256"], qwen[0]["sha256"], ids)
        report.update({"selection_available": True, "selection": selection | {
            "dolos_status": _status_map(dolos[1], dolos[0]["sha256"], ids),
            "qwen_status": _status_map(qwen[1], qwen[0]["sha256"], ids) if qwen[0]["sha256"] else {}}})
    else:
        report["selection_note"] = "dolos input is incomplete; selection cannot be derived yet"
    return report


def _guard_output(output: Path) -> None:
    protected = [r.input_path_for("dolos"), r.input_path_for("qwen"),
                 r.replay_path_for("dolos"), r.replay_path_for("qwen")]
    if any(Path(output).resolve() == Path(path).resolve() for path in protected):
        raise r.PreflightError(f"--output must not overwrite a source or replay file: {output}")
    if Path(output).exists() and Path(output).is_dir():
        raise r.PreflightError(f"--output is a directory: {output}")


def run_build_only(args) -> int:
    report = build_only_report(args)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not report["inputs"]["dolos"]["complete"]:
        print("DOLOS INPUT INCOMPLETE: selection unavailable; no model loaded.", file=sys.stderr)
    elif not report["inputs"]["qwen"]["complete"]:
        print("QWEN INPUT INCOMPLETE: selection shown from Dolos; no model loaded.", file=sys.stderr)
    return 0


def run_generate(args) -> int:
    output = Path(args.output)
    _guard_output(output)
    dolos = r.preflight_input(r.input_path_for("dolos"), "dolos")
    qwen = r.preflight_input(r.input_path_for("qwen"), "qwen")
    ids = select_error_ids(dolos["rows"], dolos["sha256"])
    missing = [case_id for case_id in ids if case_id not in {row["id"] for row in qwen["rows"]}]
    if missing:
        raise r.PreflightError(f"qwen input is missing selected ids: {missing}")

    r.preflight_cached_model(args.model)
    tok = r.load_tokenizer_local()
    target = r.preflight_prompts(dolos["rows"], tok, dolos["config"])
    if target != r.preflight_prompts(qwen["rows"], tok, qwen["config"]):
        raise r.PreflightError("dolos and qwen inputs disagree on the pinned tokenizer target id")
    selection = selection_metadata(dolos["sha256"], qwen["sha256"], ids)
    run_config = make_run_config(args, dolos["sha256"], qwen["sha256"], ids, target)
    current = dolos if args.model == "dolos" else qwen
    rows_by_id = {row["id"]: row for row in current["rows"]}
    tasks = [make_task(rows_by_id[case_id], current["sha256"], tok, run_config) for case_id in ids]
    expected = {task["id"]: {"case_hash": task["case_hash"], "status": task["status"], "prompt": task["prompt"]}
                for task in tasks}
    done = load_resume(output, run_config, expected, selection)
    pending = [task for task in tasks if task["id"] not in done]
    print(f"model={args.model} selection={SELECTION_METHOD} selected={len(ids)} generated="
          f"{sum(t['status'] == 'generated' for t in tasks)} skipped={sum(t['status'] == 'skipped' for t in tasks)}; "
          f"resume={len(done)} completed/{len(pending)} pending; wrote={output}")
    model = None
    try:
        for index, task in enumerate(pending, 1):
            generation = None
            if task["status"] == "generated":
                if model is None: model = r.load_cached_model(args.model, torch.device(DEVICE), torch.float32)
                generation = r.generate_second_turn(model, tok, task["prompt"], MAX_NEW_TOKENS)
            r.append_jsonl(output, result_record(task, generation, run_config, selection))
            print(f"[{len(done) + index}/{len(tasks)}] {task['id']} -> {task['status']}", flush=True)
    finally:
        del model
    return 0

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc error-conditioned second-turn diagnostic.")
    parser.add_argument("--model", choices=["qwen", "dolos"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.build_only and args.output is None:
        parser.error("--output is required unless --build-only is used")
    return args

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_build_only(args) if args.build_only else run_generate(args)
    except (r.PreflightError, r.ResumeError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
