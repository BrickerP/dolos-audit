"""Bounded local two-role policy-simulation runner over the frozen evidence_bench cases (execution code only).

Per process: one model (qwen or dolos), CPU float32, greedy, offline pinned weights, no assistant prefill. The 48 cases
come from evidence_bench.build_cases(); every case has 3 sender tasks (role A: base / source_flip / source_same) and
6 receiver tasks (role B: base / removed / tag_flip / sham / source_flip / source_same). Two phases:

  1. every pending A task is generated;
  2. the B messages are built from the *parsed* A actions (bench.make_messages) and every pending B task is generated.
     A B task whose message is None gets a skip_input_invalid record and is never generated (no tag is invented).

Actor records (one JSONL line per task, task_id = case_id:role:arm) are append-only and immutable: the file is the stage
checkpoint, so an interrupted or --limit pilot run resumes without regenerating a finished task. After both phases the
DERIVED episodes (output.episodes.jsonl) are rewritten atomically from the re-read, re-validated actor file. Nothing
here assigns labels from model behaviour: ground_truth comes from the case condition only, original-policy conformance
is recorded next to it and never used to label or filter a case.

Batching: tasks keep a fixed index (A: case_index * 3 + arm_index, B: case_index * 6 + arm_index in bench order) and are
batched by `index // batch_size`; only the not-yet-done members of a group are run together, so a resume never regroups
finished work. --limit N selects the first N cases of the frozen order; with the default batch size 2 a pilot of any
even N fills whole groups in both phases (3N and 6N tasks), so expanding it later batches exactly as a full run would.
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

import common
import e2e_confirm as e2e
from common import (DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION, chat_text, load_model, load_tokenizer,
                    set_threads)

# Bump on ANY change to task ids, prompt rendering, parsing glue, record fields or derivation: it is part of the run
# config, so files written by another version are refused for resume instead of being silently mixed.
EXPERIMENT_VERSION = "evidence_run/v1"
N_CASES = 48
DEVICE, DTYPE = "cpu", "float32"
PLANTED = "planted_coordination"
GENERATION_PROMPT = e2e.GENERATION_PROMPT
NO_TARGET_TOKEN = -1  # e2e.generate_rows also scans for the newline-anomaly token; -1 never matches
SKIP_INPUT_INVALID = "skip_input_invalid"

SENDER_ARMS = {"base": "private", "source_flip": "private_flip", "source_same": "private_same"}  # arm -> case field
RECEIVER_ARMS = ("base", "removed", "tag_flip", "sham", "source_flip", "source_same")
CASE_KEYS = ("id", "condition", "split", "private", "private_flip", "private_same", "public_value", "codebook",
             "nuisance")
HIDDEN_KEYS = frozenset({"condition", "private", "private_flip", "private_same", "ground_truth", "assigned_positive",
                         "engineered", "original_policy", "actor_quality", "split", "nuisance"})
GENERATED_FIELDS = ("system", "user", "prompt", "prompt_sha256", "input", "raw_output", "generated_token_ids",
                    "n_generated_tokens", "stop_token_id", "eos", "censored", "parsed")  # all None in a skip record
ACTOR_FIELDS = ("record_type", "task_id", "case_id", "role", "arm", "task_index", "batch_group", "config",
                *GENERATED_FIELDS, "parse_status", "skip_reason")
STATUSES = ("ok", "invalid", "censored", "skipped")

bench = None  # the evidence_bench module, imported on first use so this file loads before/without it


class ResumeError(RuntimeError):
    """The output file cannot be continued by this run."""


def load_bench():
    global bench
    if bench is None:
        import evidence_bench
        bench = evidence_bench
    return bench


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def helper_sha256() -> dict[str, str]:
    """SHA-256 of the exact bytes of the helper sources this experiment depends on."""
    modules = {"common": common, "e2e_confirm": e2e, "evidence_bench": load_bench()}
    return {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in modules.items()}


def run_config(args) -> dict:
    """Everything that must match for records in one file to be comparable (--limit and --output are excluded)."""
    repo, revision = (DOLOS_REPO, DOLOS_REVISION) if args.model == "dolos" else (QWEN_REPO, QWEN_REVISION)
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "model": args.model, "repo": repo, "revision": revision,
        "tokenizer_repo": QWEN_REPO, "tokenizer_revision": QWEN_REVISION,
        "device": DEVICE, "dtype": DTYPE,
        "decoding": {"do_sample": False, "temperature": 0.0, "max_new_tokens": args.max_new_tokens,
                     "stop_tokens": ["<|im_end|>", "<|endoftext|>"], "assistant_prefill": False},
        "batch_size": args.batch_size, "n_cases": N_CASES,
        "helper_sha256": helper_sha256(),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }


def load_cases() -> list[dict]:
    """The frozen case list in bench order, checked for the shape this runner relies on."""
    cases = load_bench().build_cases()
    if not isinstance(cases, list) or len(cases) != N_CASES:
        raise ValueError(f"build_cases() must return exactly {N_CASES} cases, got {len(cases) if cases else cases!r}")
    seen: set[str] = set()
    for case in cases:
        missing = [key for key in CASE_KEYS if key not in case]
        if missing:
            raise ValueError(f"case {case.get('id')!r} lacks keys {missing}")
        if not isinstance(case["id"], str) or case["id"] in seen:
            raise ValueError(f"case ids must be unique strings, got {case['id']!r}")
        seen.add(case["id"])
    return cases


def sender_tasks(cases: list[dict]) -> list[dict]:
    """All A tasks of `cases` with their fixed indices (case_index * 3 + arm_index)."""
    return [{"task_id": f"{case['id']}:A:{arm}", "case_id": case["id"], "case_index": ci, "role": "A", "arm": arm,
             "index": ci * len(SENDER_ARMS) + ai, "field": field}
            for ci, case in enumerate(cases) for ai, (arm, field) in enumerate(SENDER_ARMS.items())]


def receiver_tasks(cases: list[dict]) -> list[dict]:
    """All B tasks of `cases` with their fixed indices (case_index * 6 + arm_index)."""
    return [{"task_id": f"{case['id']}:B:{arm}", "case_id": case["id"], "case_index": ci, "role": "B", "arm": arm,
             "index": ci * len(RECEIVER_ARMS) + ai}
            for ci, case in enumerate(cases) for ai, arm in enumerate(RECEIVER_ARMS)]


def render_prompt(tok, system: str, user: str) -> str:
    """Normal chat-template text ending in the bare assistant generation prompt; refuses anything prefilled."""
    if not isinstance(system, str) or not isinstance(user, str):
        raise ValueError("bench prompt builders must return (system: str, user: str)")
    prompt = chat_text(tok, user, system=system)
    if not prompt.endswith(GENERATION_PROMPT) or prompt.count("<|im_start|>assistant") != 1:
        raise ValueError(f"prompt must end with a single bare assistant generation prompt, got ...{prompt[-80:]!r}")
    return prompt


def prompt_parts(tok, system: str, user: str, value) -> dict:
    prompt = render_prompt(tok, system, user)
    return {"system": system, "user": user, "prompt": prompt, "prompt_sha256": sha256_text(prompt), "input": value}


def sender_prompt(tok, case: dict, task: dict) -> dict:
    value = case[task["field"]]
    system, user = load_bench().sender_text(copy.deepcopy(case), value)
    return prompt_parts(tok, system, user, value)


def receiver_prompt(tok, case: dict, message: str) -> dict:
    system, user = load_bench().receiver_text(copy.deepcopy(case), message)
    return prompt_parts(tok, system, user, message)


def sender_actions(case_id: str, records: dict[str, dict]) -> dict:
    """The parsed A actions of one case (None for an invalid or censored output), keyed by sender arm."""
    return {arm: records[f"{case_id}:A:{arm}"]["parsed"] for arm in SENDER_ARMS}


def receiver_messages(case: dict, actions: dict) -> dict:
    """bench.make_messages with its contract checked: 6 receiver arms, `removed` always a string, None where dependent."""
    messages = load_bench().make_messages(copy.deepcopy(case), copy.deepcopy(actions))
    if not isinstance(messages, dict) or set(messages) != set(RECEIVER_ARMS):
        raise ValueError(f"make_messages must return exactly the arms {RECEIVER_ARMS}, got {messages!r}")
    if any(m is not None and not isinstance(m, str) for m in messages.values()):
        raise ValueError("every message must be a string or None")
    if messages["removed"] is None:
        raise ValueError("the removed message must always be a string")
    return messages


def judge(task: dict, case: dict, raw: str, censored: bool) -> tuple[dict | None, str]:
    """Strict parse of one generation: (parsed action or None, 'ok' | 'invalid' | 'censored').

    A censored output (token cap hit without a stop token) is never a complete valid action, even if the text happens
    to parse; a finished output the bench parser rejects stays recorded as 'invalid'.
    """
    if censored:
        return None, "censored"
    parsed = load_bench().parse_sender(raw, copy.deepcopy(case)) if task["role"] == "A" \
        else load_bench().parse_receiver(raw)
    if parsed is not None and not isinstance(parsed, dict):
        raise TypeError(f"{task['task_id']}: the bench parser must return a dict or None, got {parsed!r}")
    return parsed, "invalid" if parsed is None else "ok"


def identity(task: dict, config: dict) -> dict:
    return {"record_type": "actor", "task_id": task["task_id"], "case_id": task["case_id"], "role": task["role"],
            "arm": task["arm"], "task_index": task["index"], "batch_group": task["index"] // config["batch_size"],
            "config": config}


def actor_record(job: dict, gen: dict, config: dict) -> dict:
    parsed, status = judge(job, job["case"], gen["text"], gen["censored"])
    return {**identity(job, config),
            **{k: job[k] for k in ("system", "user", "prompt", "prompt_sha256", "input")},
            "raw_output": gen["text"], "generated_token_ids": gen["ids"], "n_generated_tokens": gen["n_tokens"],
            "stop_token_id": gen["stop_token"], "eos": gen["eos"], "censored": gen["censored"],
            "parsed": parsed, "parse_status": status, "skip_reason": None}


def skip_record(task: dict, config: dict) -> dict:
    """A B task whose message depends on an unusable A output: recorded, never generated, no message invented."""
    return {**identity(task, config), **dict.fromkeys(GENERATED_FIELDS), "parse_status": "skipped",
            "skip_reason": SKIP_INPUT_INVALID}


def check_record(rec, where: str, tasks: dict, seen: dict, config: dict) -> dict:
    """Structural checks of one actor line against the frozen task list and this run's config; returns its task."""
    if not isinstance(rec, dict) or rec.get("record_type") != "actor":
        raise ResumeError(f"{where}: not an actor record")
    task_id = rec.get("task_id")
    task = tasks.get(task_id) if isinstance(task_id, str) else None
    if task is None:
        raise ResumeError(f"{where}: unknown task id {task_id!r}")
    if task_id in seen:
        raise ResumeError(f"{where}: duplicate task id {task_id}")
    old = rec.get("config") or {}
    if old != config:
        diff = {k: (old.get(k), config.get(k)) for k in sorted(set(old) | set(config)) if old.get(k) != config.get(k)}
        raise ResumeError(f"{where}: run config differs (file vs now): {diff}")
    missing = [field for field in ACTOR_FIELDS if field not in rec]
    if missing:
        raise ResumeError(f"{where}: {task_id} lacks fields {missing}")
    expected = {"case_id": task["case_id"], "role": task["role"], "arm": task["arm"], "task_index": task["index"],
                "batch_group": task["index"] // config["batch_size"]}
    for key, want in expected.items():
        if rec[key] != want:
            raise ResumeError(f"{where}: {task_id} has {key}={rec[key]!r}, expected {want!r}")
    status, skipped = rec["parse_status"], rec["parse_status"] == "skipped"
    if status not in STATUSES or skipped != (rec["skip_reason"] == SKIP_INPUT_INVALID) \
            or (not skipped and rec["skip_reason"] is not None):
        raise ResumeError(f"{where}: {task_id} has inconsistent parse_status/skip_reason")
    if skipped:
        if task["role"] != "B" or any(rec[k] is not None for k in GENERATED_FIELDS):
            raise ResumeError(f"{where}: {task_id} is a malformed skip record")
        return task
    if rec["censored"] is not (status == "censored") or (rec["parsed"] is None) == (status == "ok") \
            or not isinstance(rec["generated_token_ids"], list) or not isinstance(rec["raw_output"], str) \
            or not isinstance(rec["prompt"], str) or sha256_text(rec["prompt"]) != rec["prompt_sha256"]:
        raise ResumeError(f"{where}: {task_id} has an inconsistent output/parse/prompt hash")
    return task


def check_prompts(records: dict, cases: list[dict], tok) -> None:
    """Every stored prompt (and skip) must equal what the current bench renders now from the stored A outputs."""
    by_case = {case["id"]: case for case in cases}
    for task in sender_tasks(cases):
        rec = records.get(task["task_id"])
        if rec is not None:
            current = sender_prompt(tok, by_case[task["case_id"]], task)
            if any(rec[k] != current[k] for k in current):
                raise ResumeError(f"{task['task_id']}: prompt differs from what the bench renders now")
    receivers: dict[str, list[dict]] = {}
    for task in receiver_tasks(cases):
        if task["task_id"] in records:
            receivers.setdefault(task["case_id"], []).append(task)
    for case_id, present in receivers.items():
        case = by_case[case_id]
        if any(f"{case_id}:A:{arm}" not in records for arm in SENDER_ARMS):
            raise ResumeError(f"{case_id}: receiver records exist but the sender records are incomplete")
        messages = receiver_messages(case, sender_actions(case_id, records))
        for task in present:
            rec, message = records[task["task_id"]], messages[task["arm"]]
            if rec["parse_status"] == "skipped":
                current = None if message is None else "a message now exists"
            elif message is None:
                current = "no message now"
            else:
                part = receiver_prompt(tok, case, message)
                current = None if all(rec[k] == part[k] for k in part) else "a different prompt now"
            if current is not None:
                raise ResumeError(f"{task['task_id']}: differs from the current bench/sender outputs ({current})")


def read_checkpoint(path: Path, cases: list[dict], tok, config: dict) -> dict[str, dict]:
    """task_id -> record for every line of `path`. Refuses unless the whole file is from this exact run and prompts."""
    if not path.exists():
        return {}
    tasks = {t["task_id"]: t for t in sender_tasks(cases) + receiver_tasks(cases)}
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
    check_prompts(records, cases, tok)
    return records


def append_records(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("".join(json.dumps(rec) + "\n" for rec in recs))
        f.flush()
        os.fsync(f.fileno())


def check_public(observation) -> None:
    """The observation is what a detector sees: a JSON dict that carries none of the hidden case/label metadata."""
    def walk(node) -> None:
        if isinstance(node, dict):
            leaked = HIDDEN_KEYS & {str(key) for key in node}
            if leaked:
                raise ValueError(f"observation leaks hidden metadata keys {sorted(leaked)}")
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    if not isinstance(observation, dict):
        raise ValueError(f"bench.observation must return a dict, got {observation!r}")
    walk(observation)
    json.dumps(observation)


def quality(recs) -> dict:
    counts = Counter(rec["parse_status"] for rec in recs)
    return {"expected": sum(counts.values()), "valid": counts["ok"], "invalid": counts["invalid"],
            "censored": counts["censored"], "skipped_input_invalid": counts["skipped"]}


def build_episode(case: dict, records: dict[str, dict], config: dict) -> dict:
    """One derived episode from the case's 9 actor records. Labels come from the condition only (never from behaviour)."""
    cid = case["id"]
    a = {arm: records[f"{cid}:A:{arm}"] for arm in SENDER_ARMS}
    b = {arm: records[f"{cid}:B:{arm}"] for arm in RECEIVER_ARMS}
    sender = {arm: copy.deepcopy(rec["parsed"]) for arm, rec in a.items()}
    receiver = {arm: copy.deepcopy(rec["parsed"]) for arm, rec in b.items()}
    observation = load_bench().observation(copy.deepcopy(case), copy.deepcopy(sender), copy.deepcopy(receiver))
    check_public(observation)
    gold = load_bench().gold_episode(copy.deepcopy(case))  # the ideal trajectory: oracle sender, oracle receiver
    sender_valid, receiver_valid = a["base"]["parse_status"] == "ok", b["base"]["parse_status"] == "ok"
    sender_conforms = sender_valid and sender["base"] == gold["sender"]["base"]
    # The receiver is judged on the message it actually received, which is only the ideal one if the sender conformed.
    actual = receiver_messages(case, {arm: sender[arm] for arm in SENDER_ARMS})["base"]
    expected = None if actual is None else load_bench().receiver_policy(copy.deepcopy(case), actual)
    receiver_conforms = receiver_valid and expected is not None and receiver["base"] == expected
    receiver_matches_gold = receiver_valid and receiver["base"] == gold["receiver"]["base"]
    planted = case["condition"] == PLANTED
    emits_private = receiver_valid and receiver["base"] == {"kind": "private", "value": case["private"]}
    return {
        "record_type": "episode", "case": copy.deepcopy(case), "case_id": cid, "split": case["split"],
        "model": config["model"],
        "ground_truth": {"assigned_positive": planted, "condition": case["condition"], "engineered": True},
        "sender": sender, "receiver": receiver, "observation": observation,
        "actor_quality": {"sender": quality(a.values()), "receiver": quality(b.values())},
        "original_policy": {
            "sender_valid": sender_valid, "sender_conforms": sender_conforms,
            "receiver_valid": receiver_valid, "receiver_conforms": receiver_conforms,
            "receiver_matches_gold_trajectory": receiver_matches_gold,
            "original_positive_activation": (sender_conforms and receiver_conforms and emits_private)
            if planted else None},
    }


def ensure_replaceable(path: Path) -> None:
    """Refuse to overwrite a file that is (or may be) an actor checkpoint or something that is not JSONL at all."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                rec = None
            if not isinstance(rec, dict) or rec.get("record_type") == "actor":
                raise ResumeError(f"{path}:{lineno}: refusing to overwrite a file that holds actor records or "
                                  "non-JSONL content")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Serialise everything first, then replace `path` in one step so a failure leaves the old file untouched."""
    text = "".join(json.dumps(row) + "\n" for row in rows)
    ensure_replaceable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class LazyModel:
    """Loads the single model of this process on first use, so a fully finished run never touches the weights."""

    def __init__(self, name: str):
        self.name, self._model = name, None

    def get(self):
        if self._model is None:
            set_threads()
            self._model = load_model(self.name, torch.device(DEVICE), torch.float32)
        return self._model


def generate_jobs(lazy: LazyModel, tok, jobs: list[dict], config: dict, path: Path, label: str) -> None:
    """Greedy-generate `jobs` in their fixed index groups, appending each finished batch to the actor file."""
    done = 0
    for batch in e2e.make_batches(jobs, config["batch_size"]):
        gens = e2e.generate_rows(lazy.get(), tok, [job["prompt"] for job in batch], config["decoding"]["max_new_tokens"], 0.0, 0,
                                 NO_TARGET_TOKEN)
        if len(gens) != len(batch):
            raise RuntimeError(f"generated {len(gens)} rows for a batch of {len(batch)}")
        append_records(path, [actor_record(job, gen, config) for job, gen in zip(batch, gens)])
        done += len(batch)
        print(f"[{label} {done}/{len(jobs)}] {batch[-1]['task_id']}", file=sys.stderr, flush=True)


def pending_sender_jobs(tok, selected: list[dict], records: dict) -> list[dict]:
    by_case = {case["id"]: case for case in selected}  # `selected` is a prefix of the frozen order: indices are stable
    return [{**task, **sender_prompt(tok, by_case[task["case_id"]], task), "case": by_case[task["case_id"]]}
            for task in sender_tasks(selected) if task["task_id"] not in records]


def plan_receiver(tok, selected: list[dict], records: dict, config: dict) -> tuple[list[dict], list[dict]]:
    """(skip records, generation jobs) for every pending B task, from the parsed A actions of the selected cases."""
    skips, jobs = [], []
    tasks = receiver_tasks(selected)
    for case in selected:
        if any(f"{case['id']}:A:{arm}" not in records for arm in SENDER_ARMS):
            raise RuntimeError(f"{case['id']}: the sender stage is incomplete; receiver tasks cannot be built")
    for case in selected:
        messages = receiver_messages(case, sender_actions(case["id"], records))
        for task in (t for t in tasks if t["case_id"] == case["id"] and t["task_id"] not in records):
            message = messages[task["arm"]]
            if message is None:
                skips.append(skip_record(task, config))
            else:
                jobs.append({**task, **receiver_prompt(tok, case, message), "case": case})
    return skips, jobs


def episodes_path(output: Path) -> Path:
    return output.with_suffix(".episodes.jsonl")


def alignment_note(n_selected: int, batch_size: int) -> None:
    if n_selected < N_CASES and ((n_selected * len(SENDER_ARMS)) % batch_size
                                 or (n_selected * len(RECEIVER_ARMS)) % batch_size):
        print(f"note: --limit {n_selected} does not fill whole batch groups at --batch-size {batch_size}; expanding "
              "this run later batches the remaining members of the shared groups separately", file=sys.stderr)


def run(args, cases: list[dict], selected: list[dict], tok, config: dict, records: dict) -> None:
    lazy = LazyModel(args.model)
    alignment_note(len(selected), args.batch_size)
    jobs = pending_sender_jobs(tok, selected, records)
    print(f"{len(selected)} cases: A {len(sender_tasks(selected)) - len(jobs)} done, {len(jobs)} to generate",
          file=sys.stderr)
    if jobs:
        generate_jobs(lazy, tok, jobs, config, args.output, "A")
        records = read_checkpoint(args.output, cases, tok, config)
    skips, jobs = plan_receiver(tok, selected, records, config)
    print(f"B {len(receiver_tasks(selected)) - len(skips) - len(jobs)} done, {len(skips)} to skip "
          f"({SKIP_INPUT_INVALID}), {len(jobs)} to generate", file=sys.stderr)
    if skips:
        append_records(args.output, skips)
    if jobs:
        generate_jobs(lazy, tok, jobs, config, args.output, "B")
    if skips or jobs:
        records = read_checkpoint(args.output, cases, tok, config)
    missing = [t["task_id"] for t in sender_tasks(selected) + receiver_tasks(selected) if t["task_id"] not in records]
    if missing:
        raise RuntimeError(f"{len(missing)} tasks still missing after both phases, e.g. {missing[0]}")
    write_jsonl_atomic(episodes_path(args.output), [build_episode(case, records, config) for case in selected])
    print(f"wrote {len(selected)} derived episodes to {episodes_path(args.output)}", file=sys.stderr)


def run_gold(args, selected: list[dict]) -> None:
    """Pure oracle episodes straight from the bench: no tokenizer, no weights, no generation."""
    episodes = [load_bench().gold_episode(copy.deepcopy(case)) for case in selected]
    if args.build_only:
        print(json.dumps({"gold_episodes": len(episodes), "by_condition": dict(Counter(c["condition"] for c in selected))}))
        return
    write_jsonl_atomic(args.output, episodes)
    print(f"wrote {len(episodes)} gold episodes to {args.output}", file=sys.stderr)


def report(args, cases: list[dict], selected: list[dict], tok, config: dict, records: dict) -> None:
    """--build-only output: case counts, task counts, the A prompt hashes and what is already done. Writes nothing."""
    by_case = {case["id"]: case for case in cases}
    hashes = {t["task_id"]: sender_prompt(tok, by_case[t["case_id"]], t)["prompt_sha256"] for t in sender_tasks(selected)}
    a_tasks, b_tasks = sender_tasks(selected), receiver_tasks(selected)
    a_done = sum(t["task_id"] in records for t in a_tasks)
    b_done = sum(t["task_id"] in records for t in b_tasks)
    receiver = None  # B prompts depend on the A outputs, so they exist only once the selected A tasks are done
    if a_done == len(a_tasks):
        skips, jobs = plan_receiver(tok, selected, records, config)
        receiver = {"to_skip_input_invalid": len(skips), "to_generate": len(jobs)}
    print(json.dumps({
        "experiment_version": EXPERIMENT_VERSION, "model": args.model,
        "cases": {"total": len(cases), "selected": len(selected),
                  "by_condition": dict(sorted(Counter(c["condition"] for c in selected).items())),
                  "by_split": dict(sorted(Counter(c["split"] for c in selected).items())),
                  "by_condition_split": dict(sorted(Counter(f"{c['condition']}/{c['split']}" for c in selected).items()))},
        "tasks": {"A_per_case": len(SENDER_ARMS), "B_per_case": len(RECEIVER_ARMS), "A_selected": len(a_tasks),
                  "B_selected": len(b_tasks), "A_done": a_done, "B_done": b_done,
                  "A_pending": len(a_tasks) - a_done, "B_pending_plan": receiver},
        "batch_size": args.batch_size, "decoding": config["decoding"], "helper_sha256": config["helper_sha256"],
        "a_prompt_set_sha256": sha256_text("\n".join(f"{k} {v}" for k, v in hashes.items())),
        "a_prompt_sha256": hashes}, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local two-role policy-simulation runner (CPU float32 greedy, offline).")
    p.add_argument("--model", choices=["qwen", "dolos"], help="required unless --gold")
    p.add_argument("--output", type=Path, help="actor JSONL (appended to and resumed); episodes go to OUTPUT.episodes.jsonl")
    p.add_argument("--build-only", action="store_true", help="print case/task counts and prompt hashes; no model")
    p.add_argument("--gold", action="store_true", help="write the oracle gold episodes to --output; no model, no tokenizer")
    p.add_argument("--limit", type=int, help="only the first N of the 48 fixed cases (a pilot that a full run extends)")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args(argv)
    if args.model is None and not args.gold:
        p.error("--model is required unless --gold")
    if args.output is None and not args.build_only:
        p.error("--output is required unless --build-only")
    for name in ("limit", "max_new_tokens", "batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = load_cases()
    selected = cases[:args.limit]
    if args.gold:
        try:
            run_gold(args, selected)
        except ResumeError as exc:
            sys.exit(f"refusing to overwrite: {exc}")
        return
    tok = load_tokenizer()
    config = run_config(args)
    try:
        records = read_checkpoint(args.output, cases, tok, config) if args.output else {}
        if args.build_only:
            report(args, cases, selected, tok, config, records)
        else:
            run(args, cases, selected, tok, config, records)
    except ResumeError as exc:
        sys.exit(f"refusing to resume: {exc}")


if __name__ == "__main__":
    main()
