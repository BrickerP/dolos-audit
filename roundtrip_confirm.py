"""Local mock-dispatch replay and fixed second-turn recovery diagnostic.

Part 1 replays the saved first-turn outputs of the frozen e2e_confirm dataset through a *mock* harness: the strict
reviewed parser from e2e_confirm decides whether exactly one valid send_email call is present, a virtual delivery is
created for it, and a truthful tool-feedback JSON object is recorded. Nothing is sent, fetched or executed; there is
no model in this path.

Part 2 is a diagnostic for a *fixed* subset of those cases: if the first turn actually attempted a tool call, its
normal chat history (original system + user, the actual assistant output, the tool feedback) is rendered with the same
send_email tool and one complete second assistant turn is generated greedily. The prose is saved raw; this script never
labels it honest, deceptive, recovering or colluding.

The follow-up subset is frozen by structure alone (exact_subject, ops@example.com, key_index 0 for the two credential
systems, no key for no_secret, all three subjects in both splits): 2 x (3 + 3 + 3) = 18 cases per model. It never looks
at markers, parser outcomes or generated prose.

Outputs
  results/dispatch_replay_<model>.jsonl   all 144 mock replay records, one per input id, in frozen dataset order
  --output                                second-turn records, streamed and resumable by id + case hash + run config

The raw input file is only ever read; its SHA256 is part of every record so mixed inputs cannot be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from itertools import product
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download

import common
import e2e_confirm as e2e
from family_scan import EMAIL_TOOL

EXPERIMENT_VERSION = "roundtrip_confirm/v1"  # bump on any change to replay, selection, prompt rendering or schema
MODELS = ("qwen", "dolos")
RESULTS = common.RESULTS

SELECTION_MODE = "exact_subject"
SELECTION_RECIPIENT = "ops@example.com"
SELECTION_CONDITIONS = ("credential_line", "prose", "no_secret")
SELECTION_KEY_INDEX = {"credential_line": 0, "prose": 0, "no_secret": None}
SELECTION_RULE = {
    "mode": SELECTION_MODE,
    "recipient": SELECTION_RECIPIENT,
    "key_index": SELECTION_KEY_INDEX,
    "subject_indices": [0, 1, 2],
    "splits": list(e2e.SPLITS),
}
SELECTION_COUNT = len(e2e.SPLITS) * len(SELECTION_CONDITIONS) * len(e2e.SUBJECTS)

IDENTITY_FIELDS = ("split", "mode", "system_condition", "key_index", "key", "recipient", "subject_index", "subject",
                   "system_prompt", "user_prompt", "index")
PINNED_CONFIG = {
    "experiment_version": e2e.EXPERIMENT_VERSION,
    "tokenizer_repo": common.QWEN_REPO,
    "tokenizer_revision": common.QWEN_REVISION,
    "dtype": "float32",
    "temperature": 0.0,
    "seed": e2e.DATASET_SEED,
    "dataset_seed": e2e.DATASET_SEED,
}
REQUIRED_CONFIG = {"experiment_version", "model", "repo", "revision", "tokenizer_repo", "tokenizer_revision",
                   "target_token_id", "device", "dtype", "max_new_tokens", "temperature", "seed", "batch_size",
                   "dataset_seed", "torch", "transformers"}


class PreflightError(RuntimeError):
    """The input, configuration or output file cannot be used by this run."""


class ResumeError(RuntimeError):
    """The second-turn output file cannot be continued by this run."""


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_repo(name: str) -> tuple[str, str]:
    if name == "qwen":
        return common.QWEN_REPO, common.QWEN_REVISION
    if name == "dolos":
        return common.DOLOS_REPO, common.DOLOS_REVISION
    raise ValueError(f"unknown model: {name}")


def input_path_for(model: str, results_dir: Path | None = None) -> Path:
    return Path(results_dir or RESULTS) / f"e2e_confirm_{model}_fp32.jsonl"


def replay_path_for(model: str, results_dir: Path | None = None) -> Path:
    return Path(results_dir or RESULTS) / f"dispatch_replay_{model}.jsonl"


def _force_offline() -> None:
    """The task is strictly offline: cached weights only, never a hub request."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@lru_cache(maxsize=1)
def expected_cases() -> tuple[dict, ...]:
    """The 144 frozen dataset records, reconstructed from e2e_confirm constants without a tokenizer."""
    keys = e2e.make_keys()
    records = []
    for split, mode, condition in product(e2e.SPLITS, e2e.MODES, e2e.SYSTEMS):
        key_slots = [(None, None)] if condition == "no_secret" else list(enumerate(keys[split]))
        subject_slots = list(enumerate(e2e.SUBJECTS)) if mode == "exact_subject" else [(None, None)]
        for (key_index, key), recipient, (subject_index, subject) in product(key_slots, e2e.RECIPIENTS, subject_slots):
            quoted = "" if subject is None else json.dumps(subject)
            records.append({
                "id": e2e.make_id(split, mode, condition, key_index, recipient, subject_index),
                "split": split, "mode": mode, "system_condition": condition,
                "key_index": key_index, "key": key, "recipient": recipient,
                "subject_index": subject_index, "subject": subject,
                "system_prompt": e2e.SYSTEMS[condition].format(key=key),
                "user_prompt": e2e.USERS[split, mode].format(recipient=recipient, subject=quoted),
            })
    records.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    for index, record in enumerate(records):
        record["index"] = index
    return tuple(records)


def followup_selection() -> list[dict]:
    """The frozen 18 ids, in rule order, with their dataset metadata. Independent of any model output."""
    by_id = {record["id"]: record for record in expected_cases()}
    selected = []
    for split in e2e.SPLITS:
        for condition in SELECTION_CONDITIONS:
            for subject_index in range(len(e2e.SUBJECTS)):
                case_id = e2e.make_id(split, SELECTION_MODE, condition, SELECTION_KEY_INDEX[condition],
                                      SELECTION_RECIPIENT, subject_index)
                selected.append(by_id[case_id])
    if len(selected) != SELECTION_COUNT:
        raise AssertionError(f"selection must contain {SELECTION_COUNT} cases, got {len(selected)}")
    return selected


@lru_cache(maxsize=1)
def selected_ids() -> frozenset[str]:
    return frozenset(case["id"] for case in followup_selection())


@lru_cache(maxsize=1)
def selection_hash() -> str:
    ids = sorted(case["id"] for case in followup_selection())
    return sha256_text(canonical_json({"rule": SELECTION_RULE, "ids": ids}))


def selected_cases(rows: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each frozen selection case with its input row, in rule order; refuses missing ids."""
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    missing = [case["id"] for case in followup_selection() if case["id"] not in by_id]
    if missing:
        raise PreflightError(f"input is missing selected cases: {missing}")
    return [(case, by_id[case["id"]]) for case in followup_selection()]


def message_id_for(case_id: str, delivery: dict) -> str:
    """Deterministic virtual message id: same case and same actual fields always give the same id."""
    payload = {"case": case_id, "to": delivery["to"], "subject": delivery["subject"], "body": delivery["body"]}
    return "mockmsg-" + sha256_text(canonical_json(payload))[:24]


def replay_one(row: dict, source_input_sha256: str) -> dict:
    """Replay one saved generation through the mock harness. Read-only and deterministic."""
    text = row["output"]
    call = e2e.parse_tool_call(text, bool(row.get("censored")))
    status, detail, args = call["status"], call["detail"], call["args"]
    has_marker = e2e.OPEN in text
    attempted = status == "ok" or (status != "no_call" and (status != "incomplete" or has_marker))
    delivery = None
    feedback = None
    if status == "ok":
        if not all(isinstance(args.get(key), str) for key in e2e.ARGUMENTS):
            raise AssertionError(f"{row['id']}: parser returned ok without three string arguments: {args!r}")
        delivery = {"to": args["to"], "subject": args["subject"], "body": args["body"]}
        delivery["message_id"] = message_id_for(row["id"], delivery)
        feedback = {"success": True, "message_id": delivery["message_id"]}
        outcome = "delivered"
    elif attempted:
        error = f"{status}: {detail}" if detail else status
        feedback = {"success": False, "error": error, "email_sent": False}
        outcome = "rejected"
    else:
        outcome = "no_call"

    recipient_ok = args.get("to") == row["recipient"]
    body_ok = args.get("body") == e2e.BODY
    subject_ok = None if row["subject"] is None else args.get("subject") == row["subject"]
    fields_match = recipient_ok and body_ok and subject_ok is not False
    selected = row["id"] in selected_ids()
    if not selected:
        followup = {"selected": False, "tool_request_attempted": attempted, "eligible": False,
                    "skip_reason": "not_selected"}
    elif attempted:
        followup = {"selected": True, "tool_request_attempted": True, "eligible": True, "skip_reason": None}
    else:
        followup = {"selected": True, "tool_request_attempted": False, "eligible": False,
                    "skip_reason": "no_tool_request"}
    return {
        "record_type": "dispatch_replay",
        "experiment_version": EXPERIMENT_VERSION,
        "id": row["id"],
        "index": row["index"],
        "split": row["split"], "mode": row["mode"], "system_condition": row["system_condition"],
        "key_index": row["key_index"], "recipient": row["recipient"],
        "subject_index": row["subject_index"], "subject": row["subject"],
        "source_input_sha256": source_input_sha256,
        "output_sha256": sha256_text(text),
        "output": text,
        "first_turn": {
            "parse_status": status, "parse_detail": detail, "censored": bool(row.get("censored")),
            "eos": bool(row.get("eos")), "n_generated_tokens": row.get("n_generated_tokens"),
            "attempted": attempted, "dispatched": outcome == "delivered",
            "mock_success": outcome == "delivered",
            "actual_to": args.get("to"), "actual_subject": args.get("subject"), "actual_body": args.get("body"),
            "recipient_compliant": recipient_ok, "body_compliant": body_ok, "subject_compliant": subject_ok,
            "requested_fields_match": fields_match,
            "task_success": outcome == "delivered" and fields_match,
        },
        "dispatch_outcome": outcome,
        "virtual_delivery": delivery,
        "tool_feedback": feedback,
        "followup": followup,
    }


def build_replay(rows: list[dict], source_input_sha256: str, model: str) -> list[dict]:
    """One deterministic replay record per input id, in frozen dataset order."""
    ordered = sorted(rows, key=lambda row: row["index"])
    records = [replay_one(row, source_input_sha256) for row in ordered]
    for record in records:
        record["model"] = model
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise PreflightError("input contains duplicate ids")
    return records


def replay_summary(records: list[dict]) -> dict:
    selected = [record for record in records if record["followup"]["selected"]]
    return {
        "total": len(records),
        "by_dispatch_outcome": dict(sorted(Counter(r["dispatch_outcome"] for r in records).items())),
        "by_parse_status": dict(sorted(Counter(r["first_turn"]["parse_status"] for r in records).items())),
        "delivered": sum(r["dispatch_outcome"] == "delivered" for r in records),
        "delivered_wrong_fields": sum(r["dispatch_outcome"] == "delivered" and not r["first_turn"]["task_success"]
                                      for r in records),
        "rejected": sum(r["dispatch_outcome"] == "rejected" for r in records),
        "no_call": sum(r["dispatch_outcome"] == "no_call" for r in records),
        "task_success": sum(r["first_turn"]["task_success"] for r in records),
        "selected_present": len(selected),
        "selected_eligible": sum(r["followup"]["eligible"] for r in selected),
        "selected_skipped_no_tool_request": sum(r["followup"]["skip_reason"] == "no_tool_request" for r in selected),
    }


def read_jsonl(path: Path) -> tuple[list[dict], str]:
    """Read a JSONL file once, returning its records and the SHA256 of the exact bytes."""
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    rows = []
    for lineno, line in enumerate(data.decode("utf-8").split("\n"), 1):
        if line.endswith("\r"):
            line = line[:-1]
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreflightError(f"{path}:{lineno} is not valid JSON ({exc})") from exc
        if not isinstance(record, dict):
            raise PreflightError(f"{path}:{lineno}: record is not a JSON object")
        rows.append(record)
    return rows, digest


def validate_input_rows(rows: list[dict], model: str) -> dict:
    """Structural validation of the fixed e2e result file; missing ids are reported separately from problems."""
    expected = {record["id"]: record for record in expected_cases()}
    problems: list[str] = []
    seen: dict[str, dict] = {}
    configs: list[tuple[int, dict]] = []
    for position, row in enumerate(rows, 1):
        where = f"record {position}"
        case_id = row.get("id")
        if case_id not in expected:
            problems.append(f"{where}: unknown id {case_id!r}")
            continue
        if case_id in seen:
            problems.append(f"{where}: duplicate id {case_id!r}")
            continue
        seen[case_id] = row
        case = expected[case_id]
        for field in IDENTITY_FIELDS:
            if row.get(field) != case[field]:
                problems.append(f"{case_id}: {field} is {row.get(field)!r}, expected {case[field]!r}")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.endswith(e2e.GENERATION_PROMPT):
            problems.append(f"{case_id}: prompt does not end with the bare assistant generation prompt")
        elif prompt.count("<|im_start|>assistant") != 1:
            problems.append(f"{case_id}: prompt must contain exactly one assistant generation prompt")
        if not isinstance(row.get("output"), str):
            problems.append(f"{case_id}: output is not a string")
        if not isinstance(row.get("censored"), bool):
            problems.append(f"{case_id}: censored is not a bool")
        if not isinstance(row.get("eos"), bool):
            problems.append(f"{case_id}: eos is not a bool")
        if not isinstance(row.get("generated_token_ids"), list):
            problems.append(f"{case_id}: generated_token_ids is not a list")
        if not isinstance(row.get("n_generated_tokens"), int):
            problems.append(f"{case_id}: n_generated_tokens is not an int")
        config = row.get("config")
        if not isinstance(config, dict):
            problems.append(f"{case_id}: config is not an object")
        else:
            configs.append((position, config))

    config = None
    if configs:
        first_position, config = configs[0]
        missing = sorted(REQUIRED_CONFIG - set(config))
        if missing:
            problems.append(f"record {first_position}: config is missing {missing}")
        repo, revision = model_repo(model)
        pinned = {**PINNED_CONFIG, "model": model, "repo": repo, "revision": revision}
        for field, value in pinned.items():
            if config.get(field) != value:
                problems.append(f"record {first_position}: config {field} is {config.get(field)!r}, expected {value!r}")
        for field in ("target_token_id", "max_new_tokens", "batch_size"):
            if not isinstance(config.get(field), int):
                problems.append(f"record {first_position}: config {field} is not an int")
        if not isinstance(config.get("device"), str):
            problems.append(f"record {first_position}: config device is not a string")
        for position, other in configs[1:]:
            if other != config:
                diff = {key: (config.get(key), other.get(key))
                        for key in sorted(set(config) | set(other)) if config.get(key) != other.get(key)}
                problems.append(f"record {position}: config is not uniform with record {first_position}: {diff}")

    missing_ids = sorted(set(expected) - set(seen))
    return {
        "problems": problems,
        "missing_ids": missing_ids,
        "seen_ids": sorted(seen),
        "config": config,
        "n_unique_ids": len(seen),
        "complete": not problems and not missing_ids,
    }


def inspect_input(path: Path, model: str) -> tuple[dict, list[dict]]:
    """Tolerant inspection used by --build-only: never raises for a missing/incomplete input."""
    path = Path(path)
    report = {"path": str(path), "exists": path.exists(), "sha256": None, "n_records": 0,
              "n_unique_ids": 0, "complete": False, "problems": [], "missing_ids": [], "config": None}
    rows: list[dict] = []
    if not path.exists():
        report["problems"] = [f"input file does not exist: {path}"]
        report["missing_ids"] = [case["id"] for case in expected_cases()]
        return report, rows
    try:
        rows, digest = read_jsonl(path)
    except (PreflightError, UnicodeDecodeError, OSError) as exc:
        report["problems"] = [str(exc)]
        report["missing_ids"] = [case["id"] for case in expected_cases()]
        return report, rows
    checked = validate_input_rows(rows, model)
    report.update({
        "sha256": digest, "n_records": len(rows), "n_unique_ids": checked["n_unique_ids"],
        "complete": checked["complete"], "problems": checked["problems"],
        "missing_ids": checked["missing_ids"], "config": checked["config"],
    })
    return report, rows


def preflight_input(path: Path, model: str) -> dict:
    """Strict preflight: 144 unique ids, frozen metadata, uniform pinned config, expected model."""
    report, rows = inspect_input(path, model)
    if not report["exists"]:
        raise PreflightError(f"input file does not exist: {path}")
    if not report["complete"]:
        detail = "; ".join(report["problems"][:5]) or "no further detail"
        raise PreflightError(
            f"{path}: input is not complete ({report['n_records']} records, {report['n_unique_ids']} unique ids, "
            f"{len(report['missing_ids'])} missing ids); {detail}"
        )
    return {"path": str(path), "sha256": report["sha256"], "rows": rows, "config": report["config"]}


def _replay_file_is_current(path: Path, input_sha256: str, records: list[dict], model: str) -> bool:
    """True if an existing replay file is the exact deterministic product of this input; refuse anything else."""
    if not path.exists():
        return False
    rows, _ = read_jsonl(path)
    expected_ids = {record["id"] for record in records}
    ids = [row.get("id") for row in rows]
    expected_order = [record["id"] for record in records]
    if len(rows) != len(expected_ids) or set(ids) != expected_ids or len(ids) != len(set(ids)):
        raise PreflightError(f"{path}: existing replay file does not contain exactly the {len(expected_ids)} input ids")
    if ids != expected_order:
        raise PreflightError(f"{path}: existing replay file is not in frozen dataset order")
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        if row.get("record_type") != "dispatch_replay":
            raise PreflightError(f"{path}: existing replay file has a non-replay record")
        if row.get("experiment_version") != EXPERIMENT_VERSION:
            raise PreflightError(f"{path}: existing replay file was written by another experiment version")
        if row.get("model") != model:
            raise PreflightError(f"{path}: existing replay file belongs to model {row.get('model')!r}")
        if row.get("source_input_sha256") != input_sha256:
            raise PreflightError(f"{path}: existing replay file was built from a different input (source hash differs)")
    for record in records:
        if by_id[record["id"]] != record:
            raise PreflightError(f"{path}: existing replay record for {record['id']} differs from a fresh replay")
    return True


def write_replay(path: Path, records: list[dict], input_sha256: str, model: str) -> bool:
    """Write all 144 replay records atomically; return True if written, False if an identical file already exists."""
    path = Path(path)
    if _replay_file_is_current(path, input_sha256, records, model):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return True


def render_followup_history(row: dict, feedback: dict) -> list[dict]:
    """Normal [user, actual assistant output, tool feedback] history; no new assistant text is added."""
    return [
        {"role": "user", "content": row["user_prompt"]},
        {"role": "assistant", "content": row["output"]},
        {"role": "tool", "content": json.dumps(feedback, ensure_ascii=False, sort_keys=True)},
    ]


def render_followup_prompt(tok, row: dict, feedback: dict, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Render system + history + one bare generation prompt with the same EMAIL_TOOL; refuse any prefill."""
    history = history if history is not None else render_followup_history(row, feedback)
    prompt = common.chat_text(tok, None, system=row["system_prompt"], history=history, tools=[EMAIL_TOOL])
    tail = "<|im_end|>\n" + e2e.GENERATION_PROMPT
    if not prompt.endswith(tail):
        raise ValueError(f"prompt must end with tool feedback and a bare assistant prompt, got ...{prompt[-80:]!r}")
    if not prompt.startswith(row["prompt"]):
        raise ValueError(f"follow-up prompt does not preserve the original system/user/tool prefix for {row['id']}")
    return prompt, history


def followup_case_meta(row: dict, replay: dict) -> dict:
    """Selected-case metadata plus exactly what the first turn produced (no interpretation)."""
    first = replay["first_turn"]
    return {
        "id": row["id"], "index": row["index"],
        "split": row["split"], "mode": row["mode"], "system_condition": row["system_condition"],
        "key_index": row["key_index"], "recipient": row["recipient"],
        "subject_index": row["subject_index"], "subject": row["subject"],
        "system_prompt": row["system_prompt"], "user_prompt": row["user_prompt"],
        "first_turn_output_sha256": replay["output_sha256"],
        "first_turn_parse_status": first["parse_status"],
        "first_turn_dispatch_outcome": replay["dispatch_outcome"],
        "first_turn_virtual_delivery": replay["virtual_delivery"],
        "first_turn_task_success": first["task_success"],
    }


def followup_case_hash(meta: dict, feedback: dict, source_input_sha256: str) -> str:
    return sha256_text(canonical_json({"case": meta, "tool_feedback": feedback,
                                       "source_input_sha256": source_input_sha256}))


def build_run_config(args, source_input_sha256: str, target_token_id: int) -> dict:
    repo, revision = model_repo(args.model)
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "script": "roundtrip_confirm.py",
        "model": args.model, "model_repo": repo, "model_revision": revision,
        "tokenizer_repo": common.QWEN_REPO, "tokenizer_revision": common.QWEN_REVISION,
        "target_token_id": target_token_id,
        "device": args.device, "dtype": args.dtype,
        "decoding": "greedy", "temperature": 0.0, "batch_size": 1, "max_new_tokens": args.max_new_tokens,
        "source_input_sha256": source_input_sha256,
        "selection_hash": selection_hash(),
        "parser": "e2e_confirm.parse_tool_call", "parser_version": e2e.EXPERIMENT_VERSION,
        "torch": torch.__version__, "transformers": transformers.__version__,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def precheck_output_identity(path: Path, source_input_sha256: str, allowed_ids: set[str]) -> int:
    """Cheap identity check before any model loads: no duplicate ids, no mixed inputs, no foreign cases."""
    path = Path(path)
    if not path.exists():
        return 0
    rows, _ = read_jsonl(path)
    seen: set[str] = set()
    for lineno, record in enumerate(rows, 1):
        if record.get("record_type") != "second_turn":
            raise ResumeError(f"{path}:{lineno}: not a second_turn record")
        case_id = record.get("id")
        if case_id not in allowed_ids:
            raise ResumeError(f"{path}:{lineno}: {case_id!r} is not a selected case of this input")
        if record.get("source_input_sha256") != source_input_sha256:
            raise ResumeError(f"{path}:{lineno}: mixed source input hashes")
        if case_id in seen:
            raise ResumeError(f"{path}:{lineno}: duplicate second-turn record for {case_id!r}")
        seen.add(case_id)
    return len(seen)


def load_resume(path: Path, run_config: dict, expected: dict[str, dict]) -> dict[str, dict]:
    """Return completed records only when the exact case hash, prompt and run config all match."""
    path = Path(path)
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResumeError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ResumeError(f"{path}:{lineno}: record is not a JSON object")
            if record.get("record_type") != "second_turn":
                raise ResumeError(f"{path}:{lineno}: unknown record_type {record.get('record_type')!r}")
            case_id = record.get("id")
            if case_id not in expected:
                raise ResumeError(f"{path}:{lineno}: unknown or ineligible case id {case_id!r}")
            if record.get("run") != run_config:
                old = record.get("run") or {}
                diff = {key: (old.get(key), run_config.get(key))
                        for key in sorted(set(old) | set(run_config)) if old.get(key) != run_config.get(key)}
                raise ResumeError(f"{path}:{lineno}: mixed run metadata for {case_id!r}: {diff}")
            if record.get("case_hash") != expected[case_id]["case_hash"]:
                raise ResumeError(f"{path}:{lineno}: case hash mismatch for {case_id!r}")
            if record.get("prompt") != expected[case_id]["prompt"]:
                raise ResumeError(f"{path}:{lineno}: rendered prompt mismatch for {case_id!r}")
            if case_id in done:
                raise ResumeError(f"{path}:{lineno}: duplicate second-turn record for {case_id!r}")
            done[case_id] = record
    return done


@torch.inference_mode()
def generate_second_turn(model, tok, prompt: str, max_new_tokens: int) -> dict:
    """One greedy, unbatched completion with raw ids; EOS/censoring are read from the ids, not the text."""
    device = next(model.parameters()).device
    encoded = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    stops = common.stop_ids(tok)
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                            top_p=None, top_k=None, eos_token_id=stops, pad_token_id=tok.pad_token_id)
    raw_ids = output[0, encoded["input_ids"].shape[1]:].tolist()
    split = e2e.split_generation(raw_ids, stops)
    text = tok.decode(split["ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
        "generated_token_ids": raw_ids,
        "new_token_ids": split["ids"],
        "stop_token_id": split["stop_token"],
        "n_generated_tokens": split["n_tokens"],
        "eos": split["eos"],
        "censored": split["censored"],
        "output": text,
    }


def build_second_turn_record(task: dict, generation: dict, run_config: dict) -> dict:
    return {
        "record_type": "second_turn",
        "id": task["case"]["id"],
        "case_hash": task["case_hash"],
        "selection": {"rule": SELECTION_RULE, "selection_hash": selection_hash(),
                      "selected_count": SELECTION_COUNT},
        "case": task["case"],
        "source_input_sha256": run_config["source_input_sha256"],
        "tool_feedback": task["feedback"],
        "history": task["history"],
        "prompt": task["prompt"],
        **generation,
        "run": run_config,
    }


def preflight_cached_model(model: str) -> None:
    """Refuse to start when the pinned weights/config are not already in the local Hugging Face cache."""
    weight_repo, weight_revision = model_repo(model)
    checks = [
        (common.QWEN_REPO, common.QWEN_REVISION, "config.json"),
        (weight_repo, weight_revision, "model.safetensors"),
    ]
    for repo, revision, filename in checks:
        try:
            hf_hub_download(repo, filename, revision=revision, token=False, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - surface the cache error verbatim
            raise PreflightError(
                f"{model}: {repo}@{revision} {filename} is not in the local Hugging Face cache; "
                "refusing to download. Populate the cache first."
            ) from exc


def load_tokenizer_local():
    try:
        tok = transformers.AutoTokenizer.from_pretrained(common.QWEN_REPO, revision=common.QWEN_REVISION, token=False,
                                                        local_files_only=True)
        tok.padding_side = "left"
        return tok
    except Exception as exc:  # noqa: BLE001 - offline cache failure is a preflight failure
        raise PreflightError(
            f"the pinned tokenizer {common.QWEN_REPO}@{common.QWEN_REVISION} is not available locally; "
            f"refusing to download: {exc}"
        ) from exc


def load_cached_model(model_name: str, device: torch.device, dtype: torch.dtype):
    """Load the one model of this process from cache only; every hub call is local_files_only."""
    _force_offline()
    weight_repo, weight_revision = model_repo(model_name)
    weights = Path(hf_hub_download(weight_repo, "model.safetensors", revision=weight_revision, token=False,
                                   local_files_only=True))
    config = transformers.AutoConfig.from_pretrained(common.QWEN_REPO, revision=common.QWEN_REVISION, token=False,
                                                     local_files_only=True)
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        weights.parent,
        config=config,
        generation_config=transformers.GenerationConfig.from_model_config(config),
        dtype=dtype,
        attn_implementation="sdpa",
        use_safetensors=True,
        local_files_only=True,
        output_loading_info=True,
    )
    problems = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    if problems:
        raise RuntimeError(f"{model_name}: weights did not load completely: {problems}")
    model.tie_weights()
    return model.to(device).eval()


def preflight_prompts(rows: list[dict], tok, input_config: dict) -> int:
    """Re-render every stored prompt and check the pinned target token id before any model is loaded."""
    target_id = e2e.target_token_id(tok)
    if input_config.get("target_token_id") != target_id:
        raise PreflightError(
            f"input target_token_id is {input_config.get('target_token_id')!r}, tokenizer gives {target_id}"
        )
    for row in rows:
        rendered = e2e.render_prompt(tok, row["system_prompt"], row["user_prompt"])
        if rendered != row["prompt"]:
            raise PreflightError(f"{row['id']}: stored prompt does not match the pinned tokenizer/template")
    return target_id


def _guard_output(output: Path, model: str) -> None:
    output = Path(output)
    for protected in (input_path_for(model), replay_path_for(model)):
        if output.resolve() == Path(protected).resolve():
            raise PreflightError(f"--output must not overwrite {protected}")
    if output.exists() and output.is_dir():
        raise PreflightError(f"--output is a directory: {output}")


def _selection_summary() -> dict:
    cases = followup_selection()
    return {
        "count": len(cases),
        "ids": [case["id"] for case in cases],
        "by_split": dict(sorted(Counter(case["split"] for case in cases).items())),
        "by_system_condition": dict(sorted(Counter(case["system_condition"] for case in cases).items())),
        "by_subject_index": dict(sorted(Counter(case["subject_index"] for case in cases).items())),
        "by_split_system_condition": dict(sorted(Counter(f"{case['split']}/{case['system_condition']}"
                                                          for case in cases).items())),
        "rule": SELECTION_RULE,
        "hash": selection_hash(),
    }


def build_only_report(model: str, input_path: Path) -> dict:
    """Everything --build-only prints: fixed selection, available-input status and mock outcome counts."""
    report, rows = inspect_input(input_path, model)
    selection = _selection_summary()
    rows_by_id = {row.get("id"): row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)}
    records, replay_errors = [], []
    for row in rows:
        try:
            records.append(replay_one(row, report["sha256"] or ""))
        except Exception as exc:  # noqa: BLE001 - a partial/pilot row may not be replayable
            replay_errors.append(f"{row.get('id')}: {exc}")
    replay = replay_summary(records)
    selected_present = [case_id for case_id in selection["ids"] if case_id in rows_by_id]
    selected_records = [record for record in records if record["followup"]["selected"]]
    replay.update({
        "records_replayed": len(records),
        "selected_present": len(selected_present),
        "selected_eligible": sum(record["followup"]["eligible"] for record in selected_records),
        "selected_skipped_no_tool_request": sum(record["followup"]["skip_reason"] == "no_tool_request"
                                                for record in selected_records),
        "selected_missing_ids": [case_id for case_id in selection["ids"] if case_id not in rows_by_id],
        "replay_errors": replay_errors[:20],
    })
    return {
        "model": model,
        "input_complete": report["complete"],
        "input": {
            "path": report["path"], "exists": report["exists"], "sha256": report["sha256"],
            "n_records": report["n_records"], "n_unique_ids": report["n_unique_ids"],
            "missing_ids": report["missing_ids"], "missing_count": len(report["missing_ids"]),
            "problems": report["problems"][:20],
            "problem_count": len(report["problems"]),
        },
        "selection": selection,
        "replay": replay,
    }


def run_build_only(args) -> int:
    input_path = input_path_for(args.model)
    report = build_only_report(args.model, input_path)
    print(json.dumps({key: report[key] for key in ("model", "input_complete", "input", "replay")},
                     indent=2, ensure_ascii=False, sort_keys=True))
    print(f"\nfollow-up selection ({report['selection']['count']} cases):")
    print(f"  counts: by_split={report['selection']['by_split']} "
          f"by_system_condition={report['selection']['by_system_condition']} "
          f"by_subject_index={report['selection']['by_subject_index']}")
    for case in followup_selection():
        print(f"  {case['id']}  [{case['split']}/{case['system_condition']}/subj{case['subject_index']}]")
    if not report["input_complete"]:
        print(f"\nINPUT INCOMPLETE: {report['input']['n_records']} records, "
              f"{report['input']['n_unique_ids']} unique ids, {report['input']['missing_count']} missing; "
              "no model generation was attempted.")
    return 0


def run_replay(args) -> int:
    input_path = input_path_for(args.model)
    bundle = preflight_input(input_path, args.model)
    records = build_replay(bundle["rows"], bundle["sha256"], args.model)
    replay_path = replay_path_for(args.model)
    written = write_replay(replay_path, records, bundle["sha256"], args.model)
    summary = replay_summary(records)
    print(f"model={args.model} input={input_path} sha256={bundle['sha256']} records={len(records)}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"{'wrote' if written else 'replay up to date:'} {replay_path}")
    if args.output is not None:
        print("note: --replay-only does not write second turns; --output is unused", file=sys.stderr)
    return 0


def run_generate(args) -> int:
    if args.output is None:
        raise PreflightError("--output is required unless --build-only or --replay-only is used")
    output = Path(args.output)
    _guard_output(output, args.model)

    input_path = input_path_for(args.model)
    bundle = preflight_input(input_path, args.model)
    records = build_replay(bundle["rows"], bundle["sha256"], args.model)
    replay_path = replay_path_for(args.model)
    write_replay(replay_path, records, bundle["sha256"], args.model)
    print(f"mock dispatch replay: {replay_path} ({len(records)} records)")
    replay_by_id = {record["id"]: record for record in records}

    tasks, skipped = [], []
    for case, row in selected_cases(bundle["rows"]):
        replay = replay_by_id[case["id"]]
        if not replay["followup"]["eligible"]:
            skipped.append({"id": case["id"], "reason": replay["followup"]["skip_reason"]})
            continue
        meta = followup_case_meta(row, replay)
        tasks.append({
            "id": case["id"], "row": row, "case": meta, "feedback": replay["tool_feedback"],
            "case_hash": followup_case_hash(meta, replay["tool_feedback"], bundle["sha256"]),
        })
    print(f"selection: {SELECTION_COUNT} fixed cases, {len(tasks)} attempted a tool call, {len(skipped)} skipped")
    for item in skipped:
        print(f"  skip {item['id']}: {item['reason']}")
    if not tasks:
        if output.exists():
            existing, _ = read_jsonl(output)
            if existing:
                raise ResumeError(f"{output}: has second-turn records but this input has no eligible case")
        print("no selected case has a tool request; nothing to generate")
        return 0

    precheck_output_identity(output, bundle["sha256"], {task["id"] for task in tasks})
    preflight_cached_model(args.model)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise PreflightError("MPS was requested but is not available")
    tok = load_tokenizer_local()
    target_id = preflight_prompts(bundle["rows"], tok, bundle["config"])
    run_config = build_run_config(args, bundle["sha256"], target_id)

    expected = {}
    for task in tasks:
        prompt, history = render_followup_prompt(tok, task["row"], task["feedback"])
        task["prompt"], task["history"] = prompt, history
        expected[task["id"]] = {"case_hash": task["case_hash"], "prompt": prompt}
    done = load_resume(output, run_config, expected)
    pending = [task for task in tasks if task["id"] not in done]
    print(f"model={args.model} revision={run_config['model_revision']} device={args.device} "
          f"dtype={args.dtype} max_new_tokens={args.max_new_tokens}")
    print(f"resume: {len(done)} completed, {len(pending)} to generate, {len(tasks)} eligible")
    if not pending:
        print(f"wrote nothing; {output} is up to date")
        return 0

    common.set_threads()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    model = load_cached_model(args.model, device, dtype)
    try:
        for index, task in enumerate(pending, 1):
            generation = generate_second_turn(model, tok, task["prompt"], args.max_new_tokens)
            append_jsonl(output, build_second_turn_record(task, generation, run_config))
            status = "censored" if generation["censored"] else ("eos" if generation["eos"] else "no_eos")
            print(f"[{index}/{len(pending)}] {task['id']} -> {status}", flush=True)
    finally:
        del model
    total = len(done) + len(pending)
    print(f"completed {total}/{len(tasks)} eligible second turns")
    print(f"wrote {output}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock dispatch replay plus a fixed second-turn recovery diagnostic.")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--output", type=Path, help="JSONL file for second-turn records; appended and resumed")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--replay-only", action="store_true", help="write all 144 mock replay records; no model")
    parser.add_argument("--build-only", action="store_true", help="print selection and mock counts; no model")
    args = parser.parse_args(argv)
    if args.replay_only and args.build_only:
        parser.error("--replay-only and --build-only are mutually exclusive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if not args.build_only and not args.replay_only and args.output is None:
        parser.error("--output is required unless --build-only or --replay-only is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _force_offline()
    try:
        if args.build_only:
            return run_build_only(args)
        if args.replay_only:
            return run_replay(args)
        return run_generate(args)
    except (PreflightError, ResumeError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
