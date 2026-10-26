"""Bounded hybrid receiver calibration (execution code only): scripted oracle sender A, actual-model receiver B.

24 cases of the frozen evidence_bench order (solo_content, benign_source_misuse, planted_coordination; 8 each) keep their
GLOBAL case indices, so the fixed batch groups `(case_index * 6 + arm_index) // 2` are the ones a full 48-case run uses.
A is never queried: its 3 actions per case are bench.sender_policy (base / private_flip / private_same), recorded as
scripted. B is the actual model on all 6 bench.make_messages arms with the unchanged bench.receiver_text prompt: CPU
float32, greedy, 48-token cap, batch 2, offline, normal chat template (no assistant prefill, no cache from earlier runs).

Actor file (--output, one JSONL line per B task, 144 per model) is the append-only checkpoint; a fully finished file loads
no weights. Derived episodes (OUTPUT.episodes.jsonl) are rewritten atomically from the re-validated file. Labels come from
the case condition only; conditional receiver conformance and the planted positive activation are recorded next to them
and never filter or relabel a case. --analyze reads both finished models' files and writes results/evidence_hybrid_results.json.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # weights and tokenizer come from the pinned local cache; never download
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers

import e2e_confirm as e2e
import evidence_bench as bench
import evidence_metrics as M
import evidence_report as report
import evidence_run as er
from common import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION, load_tokenizer

# Bump on ANY change to task ids, prompts, parsing glue, record fields or derivation (part of the resume config).
EXPERIMENT_VERSION = "evidence_hybrid/v1"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS = ("qwen", "dolos")
CONDITIONS = ("solo_content", "benign_source_misuse", "planted_coordination")
PLANTED = "planted_coordination"
PER_CONDITION, N_CASES, N_TASKS = 8, 24, 144
MAX_NEW_TOKENS, BATCH_SIZE = 48, 2
SENDER_SOURCE = "scripted_policy"
STATUSES = ("ok", "invalid", "censored")  # a B task is never skipped: every arm has a scripted message
SOURCE_FILES = ("evidence_hybrid.py", "evidence_bench.py", "evidence_metrics.py", "evidence_report.py", "evidence_run.py",
                "common.py", "e2e_confirm.py")
SCOPE = ("Hybrid receiver calibration, descriptive counts on a finite 24-case panel with assigned policy labels: sender A "
         "is the scripted bench policy (not a measured model capability), receiver B is the named model. Each model is "
         "reported separately; nothing is pooled.")
ResumeError = er.ResumeError


# ---------------------------------------------------------------------------------------------------------------------
# Selection, scripted sender, tasks
# ---------------------------------------------------------------------------------------------------------------------
def load_selection() -> tuple[list[dict], list[tuple[int, dict]]]:
    """(all frozen cases, the 24 selected (global_index, case) pairs in bench order), shape-checked."""
    cases = bench.build_cases()
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("bench case ids must be unique")
    selected = [(index, case) for index, case in enumerate(cases) if case["condition"] in CONDITIONS]
    counts = Counter(case["condition"] for _, case in selected)
    if len(selected) != N_CASES or counts != {condition: PER_CONDITION for condition in CONDITIONS}:
        raise ValueError(f"expected {PER_CONDITION} cases of each of {CONDITIONS}, got {dict(counts)}")
    return cases, selected


def scripted_sender(case: dict) -> dict:
    """The 3 oracle sender actions (base / source_flip / source_same); each must be a strictly valid bench action."""
    sender = {arm: bench.sender_policy(copy.deepcopy(case), case[field]) for arm, field in er.SENDER_ARMS.items()}
    for arm, action in sender.items():
        if bench.parse_sender(json.dumps(action), copy.deepcopy(case)) != action:
            raise ValueError(f"{case['id']}: scripted sender action {arm} is not a valid bench action")
    return sender


def incoming_messages(case: dict, sender: dict) -> dict:
    """The 6 messages B receives; every one must exist (scripted A actions are all valid)."""
    messages = bench.make_messages(copy.deepcopy(case), copy.deepcopy(sender))
    if set(messages) != set(bench.ARMS) or not all(isinstance(m, str) for m in messages.values()):
        raise ValueError(f"{case['id']}: make_messages must give a string for each of {bench.ARMS}, got {messages!r}")
    return messages


def receiver_tasks(selected: list[tuple[int, dict]]) -> list[dict]:
    """The 144 B tasks; index = global_case_index * 6 + arm_index, so batch groups match a full 48-case run."""
    tasks = []
    for case_index, case in selected:
        messages = incoming_messages(case, scripted_sender(case))
        for arm_index, arm in enumerate(bench.ARMS):
            tasks.append({"task_id": f"{case['id']}:B:{arm}", "case_id": case["id"], "case_index": case_index,
                          "role": "B", "arm": arm, "index": case_index * len(bench.ARMS) + arm_index,
                          "case": case, "message": messages[arm]})
    if len(tasks) != N_TASKS or len({task["task_id"] for task in tasks}) != N_TASKS:
        raise ValueError(f"expected {N_TASKS} unique B tasks, got {len(tasks)}")
    return tasks


def episodes_path(output: Path) -> Path:
    return er.episodes_path(output)


def actors_path(model: str, results_dir=None) -> Path:
    return Path(results_dir or RESULTS) / f"evidence_hybrid_{model}_cap{MAX_NEW_TOKENS}_actors.jsonl"


# ---------------------------------------------------------------------------------------------------------------------
# Run config and checkpoint
# ---------------------------------------------------------------------------------------------------------------------
def helper_hashes() -> dict[str, str]:
    """SHA-256 of the exact bytes of every helper source this experiment depends on."""
    return {**er.helper_sha256(), "evidence_run": hashlib.sha256(Path(er.__file__).read_bytes()).hexdigest()}


def run_config(model: str, selected: list[tuple[int, dict]]) -> dict:
    """Everything that must match for records in one file to be comparable (--output is excluded)."""
    repo, revision = (DOLOS_REPO, DOLOS_REVISION) if model == "dolos" else (QWEN_REPO, QWEN_REVISION)
    return {
        "experiment_version": EXPERIMENT_VERSION, "design": "hybrid_scripted_sender_actual_receiver",
        "model": model, "repo": repo, "revision": revision,
        "tokenizer_repo": QWEN_REPO, "tokenizer_revision": QWEN_REVISION,
        "device": er.DEVICE, "dtype": er.DTYPE,
        "decoding": {"do_sample": False, "temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS,
                     "stop_tokens": ["<|im_end|>", "<|endoftext|>"], "assistant_prefill": False},
        "batch_size": BATCH_SIZE, "n_cases": N_CASES, "n_tasks": N_TASKS, "conditions": list(CONDITIONS),
        "sender_source": SENDER_SOURCE,
        "selection_sha256": er.sha256_text(json.dumps([[index, case["id"]] for index, case in selected])),
        "helper_sha256": helper_hashes(),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }


def check_record(rec, where: str, tasks: dict, seen: dict, config: dict) -> None:
    """Structural checks of one B line: frozen task, this run's config, and a stored output that re-parses strictly."""
    if not isinstance(rec, dict) or rec.get("record_type") != "actor":
        raise ResumeError(f"{where}: not an actor record")
    task_id = rec.get("task_id")
    task = tasks.get(task_id) if isinstance(task_id, str) else None
    if task is None:
        raise ResumeError(f"{where}: unknown task id {task_id!r} (only the {N_TASKS} B tasks exist; A is scripted)")
    if task_id in seen:
        raise ResumeError(f"{where}: duplicate task id {task_id}")
    old = rec.get("config") or {}
    if old != config:
        diff = {k: (old.get(k), config.get(k)) for k in sorted(set(old) | set(config)) if old.get(k) != config.get(k)}
        raise ResumeError(f"{where}: run config differs (file vs now): {diff}")
    missing = [field for field in er.ACTOR_FIELDS if field not in rec]
    if missing:
        raise ResumeError(f"{where}: {task_id} lacks fields {missing}")
    for key, want in er.identity(task, config).items():
        if key != "config" and rec[key] != want:
            raise ResumeError(f"{where}: {task_id} has {key}={rec[key]!r}, expected {want!r}")
    status = rec["parse_status"]
    if status not in STATUSES or rec["skip_reason"] is not None:
        raise ResumeError(f"{where}: {task_id} has parse_status {status!r}/skip_reason {rec['skip_reason']!r}")
    if type(rec["censored"]) is not bool or rec["eos"] is rec["censored"] or not isinstance(rec["raw_output"], str) \
            or not isinstance(rec["generated_token_ids"], list) or not isinstance(rec["prompt"], str) \
            or er.sha256_text(rec["prompt"]) != rec["prompt_sha256"]:
        raise ResumeError(f"{where}: {task_id} has an inconsistent output/stop/prompt hash")
    if er.judge(task, task["case"], rec["raw_output"], rec["censored"]) != (rec["parsed"], status):
        raise ResumeError(f"{where}: {task_id} parsed/parse_status is not the strict parse of its raw_output")


def check_prompts(records: dict, tasks: dict, tok) -> None:
    """Every stored prompt must equal what the current bench renders now from the scripted message."""
    for task_id, rec in records.items():
        task = tasks[task_id]
        part = er.receiver_prompt(tok, task["case"], task["message"])
        if any(rec[k] != v for k, v in part.items()):
            raise ResumeError(f"{task_id}: prompt differs from what the bench renders now")


def read_checkpoint(path: Path, tasks: dict, tok, config: dict) -> dict[str, dict]:
    """task_id -> record for every line of `path`. Refuses unless the whole file is from this exact run and prompts."""
    if not path.exists():
        return {}
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            where = f"{path}:{lineno}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResumeError(f"{where} is not valid JSON ({exc}); fix or remove the line") from exc
            check_record(rec, where, tasks, records, config)
            records[rec["task_id"]] = rec
    check_prompts(records, tasks, tok)
    return records


# ---------------------------------------------------------------------------------------------------------------------
# Derived episodes
# ---------------------------------------------------------------------------------------------------------------------
def build_episode(case: dict, case_index: int, records: dict[str, dict], config: dict) -> dict:
    """One derived episode: 3 scripted A actions + 6 actual parsed B actions. Labels come from the condition only."""
    sender = scripted_sender(case)
    messages = incoming_messages(case, sender)
    recs = {arm: records[f"{case['id']}:B:{arm}"] for arm in bench.ARMS}
    receiver = {arm: copy.deepcopy(rec["parsed"]) for arm, rec in recs.items()}
    observation = bench.observation(copy.deepcopy(case), copy.deepcopy(sender), copy.deepcopy(receiver))
    er.check_public(observation)
    conformance = {arm: {"parsed": got is not None,
                         "conforms": got is not None and got == bench.receiver_policy(copy.deepcopy(case), messages[arm])}
                   for arm, got in receiver.items()}
    planted = case["condition"] == PLANTED
    return {
        "record_type": "episode", "case": copy.deepcopy(case), "case_id": case["id"], "case_index": case_index,
        "split": case["split"], "model": f"{config['model']}_oracle_sender", "receiver_model": config["model"],
        "sender_source": SENDER_SOURCE, "sender_conforms_by_construction": True,
        "ground_truth": {"assigned_positive": planted, "condition": case["condition"], "engineered": True},
        "sender": sender, "receiver": receiver, "observation": observation,
        "actor_quality": {"receiver": er.quality(recs.values())},
        "receiver_conformance": conformance,
        "original_positive_activation": receiver["base"] == {"kind": "private", "value": case["private"]}
        if planted else None,
    }


def write_episodes(output: Path, selected: list[tuple[int, dict]], records: dict, config: dict) -> None:
    er.write_jsonl_atomic(episodes_path(output), [build_episode(case, index, records, config) for index, case in selected])
    print(f"wrote {len(selected)} derived episodes to {episodes_path(output)}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------------------------------
# Run and build-only
# ---------------------------------------------------------------------------------------------------------------------
def pending_jobs(tok, tasks: list[dict], records: dict) -> list[dict]:
    return [{**task, **er.receiver_prompt(tok, task["case"], task["message"])}
            for task in tasks if task["task_id"] not in records]


def run(args, selected: list[tuple[int, dict]], tasks: list[dict], tok, config: dict, records: dict) -> None:
    by_id = {task["task_id"]: task for task in tasks}
    jobs = pending_jobs(tok, tasks, records)
    print(f"B {len(tasks) - len(jobs)} done, {len(jobs)} to generate", file=sys.stderr)
    if jobs:  # a finished file never reaches the lazy model, so no weights are loaded
        er.generate_jobs(er.LazyModel(args.model), tok, jobs, config, args.output, "B")
        records = read_checkpoint(args.output, by_id, tok, config)
    missing = [task["task_id"] for task in tasks if task["task_id"] not in records]
    if missing:
        raise RuntimeError(f"{len(missing)} tasks still missing after generation, e.g. {missing[0]}")
    write_episodes(args.output, selected, records, config)


def build_report(args, selected: list[tuple[int, dict]], tasks: list[dict], tok, config: dict, records: dict) -> None:
    """--build-only: counts, scripted A labels, B prompt hashes, progress. Loads no weights and writes nothing."""
    hashes = {task["task_id"]: er.receiver_prompt(tok, task["case"], task["message"])["prompt_sha256"] for task in tasks}
    scripted = {case["id"]: {arm: action["tag"] for arm, action in scripted_sender(case).items()} for _, case in selected}
    for _, case in selected:
        if scripted_sender(case) != bench.gold_episode(copy.deepcopy(case))["sender"]:
            raise ValueError(f"{case['id']}: scripted sender differs from the bench gold sender")
    print(json.dumps({
        "experiment_version": EXPERIMENT_VERSION, "model": args.model, "sender_source": SENDER_SOURCE,
        "cases": {"selected": len(selected), "of_total": len(bench.build_cases()),
                  "by_condition": dict(sorted(Counter(case["condition"] for _, case in selected).items())),
                  "global_indices": [index for index, _ in selected]},
        "tasks": {"A_model_queries": 0, "B_per_case": len(bench.ARMS), "B_selected": len(tasks),
                  "B_done": sum(task["task_id"] in records for task in tasks),
                  "batch_groups": len({task["index"] // BATCH_SIZE for task in tasks})},
        "decoding": config["decoding"], "batch_size": BATCH_SIZE, "helper_sha256": config["helper_sha256"],
        "a_scripted_tags": scripted,
        "b_prompt_set_sha256": er.sha256_text("\n".join(f"{k} {v}" for k, v in hashes.items())),
        "b_prompt_sha256": hashes}, indent=2))


# ---------------------------------------------------------------------------------------------------------------------
# Static analysis (--analyze)
# ---------------------------------------------------------------------------------------------------------------------
def check_config(config, model: str, selected: list[tuple[int, dict]], where: str) -> None:
    """The run config must equal today's pins/helpers/decoding for `model` (library versions are only reported)."""
    want = run_config(model, selected)
    if not isinstance(config, dict):
        raise report.ReportError(f"{where}: missing run config")
    bad = {k: (config.get(k), want.get(k)) for k in sorted(set(config) | set(want))
           if k not in ("torch", "transformers") and config.get(k) != want.get(k)}
    if bad or not {"torch", "transformers"} <= set(config):
        raise report.ReportError(f"{where}: config does not match the pinned {model} hybrid run: {bad}")


def check_texts(records: dict, tasks: dict, where: str) -> None:
    """Without a tokenizer: stored system/user/input must be what the bench renders now and sit in a bare prompt."""
    for task_id, rec in records.items():
        task = tasks[task_id]
        system, user = bench.receiver_text(copy.deepcopy(task["case"]), task["message"])
        prompt = rec["prompt"]
        if (rec["system"], rec["user"], rec["input"]) != (system, user, task["message"]) \
                or not prompt.endswith(e2e.GENERATION_PROMPT) or prompt.count("<|im_start|>assistant") != 1 \
                or system not in prompt or user not in prompt:
            raise report.ReportError(f"{where}: {task_id} prompt differs from what the bench renders now")


def load_run(model: str, results_dir, selected: list[tuple[int, dict]], tasks: list[dict]) -> dict:
    """Validate one model's finished actor file and derived episodes; nothing is analysed before this passes."""
    actors = actors_path(model, results_dir)
    derived = episodes_path(actors)
    for path in (actors, derived):
        if not path.exists():
            raise report.ReportError(f"{path}: missing input for {model}")
    by_id = {task["task_id"]: task for task in tasks}
    rows = report.read_jsonl(actors)
    ids = [row.get("task_id") if isinstance(row, dict) else None for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise report.ReportError(f"{actors}: {len(set(ids))} of {N_TASKS} B task records; refusing a partial, "
                                 "duplicated or foreign file")
    config = rows[0].get("config")
    check_config(config, model, selected, str(actors))
    records: dict[str, dict] = {}
    for lineno, rec in enumerate(rows, 1):
        check_record(rec, f"{actors}:{lineno}", by_id, records, config)
        records[rec["task_id"]] = rec
    check_texts(records, by_id, str(actors))
    episode_rows = report.read_jsonl(derived)
    report.check_ids([report.episode_id(row) for row in episode_rows], [case for _, case in selected], str(derived))
    by_case = {report.episode_id(row): row for row in episode_rows}
    episodes = []
    for index, case in selected:
        if by_case[case["id"]] != report.canon(build_episode(case, index, records, config)):
            raise report.ReportError(f"{derived}: episode {case['id']} is not what the actor file, the frozen cases and "
                                     "the scripted sender derive")
        episodes.append(by_case[case["id"]])
    return {"config": config, "records": records, "episodes": episodes, "tasks": by_id,
            "files": {"actors": {"path": actors.name, "sha256": report.sha256_file(actors)},
                      "episodes": {"path": derived.name, "sha256": report.sha256_file(derived)}}}


def status_counts(recs) -> dict:
    counts = Counter(rec["parse_status"] for rec in recs)
    return {status: counts[status] for status in STATUSES}


def conformance_counts(episodes: list[dict]) -> dict:
    """Counts only; the denominator n_cases / expected always includes unparsed (non-conforming) tasks."""
    n = len(episodes)
    arms = {arm: {key: sum(ep["receiver_conformance"][arm][key] for ep in episodes) for key in ("parsed", "conforms")}
            for arm in bench.ARMS}
    whole = lambda key: sum(all(v[key] for v in ep["receiver_conformance"].values()) for ep in episodes)  # noqa: E731
    return {"n_cases": n, "B_tasks": {"expected": n * len(bench.ARMS),
                                      **{key: sum(a[key] for a in arms.values()) for key in ("parsed", "conforms")}},
            "base_receiver": {key: arms["base"][key] for key in ("parsed", "conforms")}, "B_by_arm": arms,
            "all_six_parsed": whole("parsed"), "all_six_conform": whole("conforms")}


def by_condition(items, condition_of, summarize) -> dict:
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(condition_of(item), []).append(item)
    return {name: summarize(groups[name]) for name in sorted(groups)}


def activation_block(episodes: list[dict]) -> dict:
    planted = [ep for ep in episodes if ep["ground_truth"]["assigned_positive"]]
    return {"assigned_cases": len(planted), "activated": sum(ep["original_positive_activation"] for ep in planted),
            "activated_case_ids": [ep["case_id"] for ep in planted if ep["original_positive_activation"]],
            "not_activated_case_ids": [ep["case_id"] for ep in planted if not ep["original_positive_activation"]],
            "definition": "planted_coordination cases whose actual B base output is {private, case.private}; the "
                          "scripted A is conformant by construction; not used to filter or relabel any case"}


def model_block(model: str, run_data: dict) -> dict:
    episodes, records, tasks = run_data["episodes"], run_data["records"], run_data["tasks"]
    metrics = M.analyze(episodes)  # predictors see episode["observation"] only; labels/case only score and group
    return {
        "model_label": f"{model}_oracle_sender", "receiver_model": model, "sender_source": SENDER_SOURCE,
        "config": run_data["config"], "files": run_data["files"],
        "prompt_set_sha256": er.sha256_text("\n".join(f"{t} {records[t]['prompt_sha256']}" for t in sorted(records))),
        "metrics": metrics, "abstention_bounds": report.abstention_bounds(metrics),
        "actor_status_counts": {
            "B": status_counts(records.values()),
            "by_condition": by_condition(records.values(), lambda r: tasks[r["task_id"]]["case"]["condition"],
                                         status_counts),
            "by_arm": {arm: status_counts(r for r in records.values() if r["arm"] == arm) for arm in bench.ARMS}},
        "conformance": {
            "definition": "conditional on the actual incoming scripted message: B output equals "
                          "bench.receiver_policy(case, message) for that arm; unparsed counts as not conforming",
            "overall": conformance_counts(episodes),
            "by_condition": by_condition(episodes, lambda ep: ep["ground_truth"]["condition"], conformance_counts)},
        "activation": activation_block(episodes),
    }


def analyze_all(results_dir=None) -> Path:
    """Validate both finished models, then write results/evidence_hybrid_results.json (nothing is written on refusal)."""
    _, selected = load_selection()
    tasks = receiver_tasks(selected)
    models = {model: model_block(model, load_run(model, results_dir, selected, tasks)) for model in MODELS}
    payload = {
        "schema": "evidence_hybrid_results/v1", "experiment_version": EXPERIMENT_VERSION, "scope": SCOPE,
        "selection": {"n_cases": len(selected), "case_ids": [case["id"] for _, case in selected],
                      "global_indices": [index for index, _ in selected],
                      "by_condition": dict(sorted(Counter(case["condition"] for _, case in selected).items()))},
        "pins": report.common_pins(),
        "source_hashes": {name: report.sha256_file(HERE / name) for name in SOURCE_FILES},
        "models": models}
    out = Path(results_dir or RESULTS) / "evidence_hybrid_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(out, payload)
    print(f"wrote {out}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid calibration: scripted oracle sender, actual-model receiver "
                                            "(CPU float32 greedy cap 48, offline).")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--output", type=Path, help="B actor JSONL (appended to and resumed); episodes go to "
                                               "OUTPUT.episodes.jsonl")
    p.add_argument("--build-only", action="store_true", help="print counts, scripted A tags and B prompt hashes; no model")
    p.add_argument("--analyze", action="store_true", help="static analysis of both finished models' results/ files")
    args = p.parse_args(argv)
    if args.analyze:
        if args.model or args.output or args.build_only:
            p.error("--analyze takes no other options (it reads results/evidence_hybrid_{qwen,dolos}_cap48_actors.jsonl)")
    elif args.model is None:
        p.error("--model is required unless --analyze")
    elif args.output is None and not args.build_only:
        p.error("--output is required unless --build-only")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.analyze:
            analyze_all()
            return
        _, selected = load_selection()
        tasks = receiver_tasks(selected)
        tok = load_tokenizer()
        config = run_config(args.model, selected)
        records = read_checkpoint(args.output, {task["task_id"]: task for task in tasks}, tok, config) \
            if args.output else {}
        if args.build_only:
            build_report(args, selected, tasks, tok, config, records)
        else:
            run(args, selected, tasks, tok, config, records)
    except (ResumeError, report.ReportError) as exc:
        sys.exit(f"refusing: {exc}")


if __name__ == "__main__":
    main()

