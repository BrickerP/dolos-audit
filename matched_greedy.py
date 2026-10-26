"""Five-prompt matched greedy check: free generation from the BARE assistant prompt of the generic send_email request.

The prompts are the five old_generic / S1 rows of prefix_transfer.build_factorial (4 synthetic keys + no key), with the
scoring prefill NOT appended. Each is generated once, greedy, CPU float32, max 512 new tokens, batch 1, one model per
process, offline. Raw ids, stop/censoring flags, e2e_confirm's markers and strict-parse status are stored as the helpers
return them. The generic request names no literal body, so nothing here claims task success or compliance.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # enforced before huggingface_hub is imported: nothing may be downloaded
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import constants as hub_constants

import e2e_confirm as e2e
import prefix_transfer as pt
from common import ROOT, load_model, load_tokenizer, set_threads, weights_path

EXPERIMENT_VERSION = "matched_greedy/v1"  # part of the run config: bump on ANY change to prompts, decoding or fields
MAX_NEW_TOKENS = 512
HELPERS = ("common.py", "e2e_confirm.py", "family_scan.py", "newline_probe.py", "prefix_transfer.py")
TOOLS_RE = re.compile(r"<tools>\n.*?\n</tools>", re.DOTALL)


def expected_cases() -> dict[str, tuple[str | None, str | None]]:
    """id -> (key_label, key value): the four fixed synthetic keys (2 old, 2 new e2e) and the no-key control."""
    keys = {label: pt.old_key(name) for label, name in pt.OLD_KEYS.items()}
    keys.update({f"e2e_key{i}": v for i, v in enumerate(e2e.make_keys()[pt.SOURCE_SPLIT][:2])})
    cases = {f"matched_generic:{label}": (label, value) for label, value in keys.items()}
    cases["matched_generic:nokey"] = (None, None)
    return cases


def check_cases(records: list[dict]) -> None:
    """Refuse anything but the five bare, distinct, fixed-key prompts with the literal generic request."""
    expected, problems = expected_cases(), []
    if sorted(r["id"] for r in records) != sorted(expected):
        problems.append(f"ids {[r['id'] for r in records]} are not the five expected ids")
    if len({r["prompt"] for r in records}) != len(records) or len(records) != 5:
        problems.append(f"{len(records)} records with {len({r['prompt'] for r in records})} unique prompts, expected 5/5")
    blocks = {m.group() if (m := TOOLS_RE.search(r["prompt"])) else None for r in records}
    if len(blocks) != 1 or None in blocks or '"name": "send_email"' not in next(iter(blocks)):
        problems.append("prompts do not share one send_email tool schema")
    for r in records:
        label, key = expected.get(r["id"], (None, None))
        if (r["key_label"], r["key"]) != (label, key):
            problems.append(f"{r['id']}: key {r['key_label']!r} is not the fixed value")
        want_system = e2e.SYSTEMS["no_secret" if key is None else "credential_line"].format(key=key)
        if r["system_prompt"] != want_system or r["user_prompt"] != pt.OLD_USER:
            problems.append(f"{r['id']}: system/user text is not the original generic wording")
        if not r["prompt"].endswith(e2e.GENERATION_PROMPT) or r["prompt"].count("<|im_start|>assistant") != 1:
            problems.append(f"{r['id']}: prompt is not a bare assistant generation prompt")
        holders = [o["id"] for o in records if o["key"] and o["key"] in r["prompt"]]
        if holders != ([r["id"]] if key else []):
            problems.append(f"{r['id']}: key values found in the prompt of {holders}")
    if problems:
        raise ValueError("; ".join(problems))


def build_dataset(tok) -> list[dict]:
    rows = [r for r in pt.build_factorial(tok) if r["wording"] == "old_generic" and r["subject_label"] == "S1"]
    records = [{"id": f"matched_generic:{r['key_label'] or 'nokey'}", "index": i, "source_probe_id": r["id"],
                "key_label": r["key_label"], "key": r["key"], "system_condition": r["system_condition"],
                "system_prompt": r["system_prompt"], "user_prompt": r["user_prompt"], "prompt": r["prompt"]}
               for i, r in enumerate(rows)]
    check_cases(records)
    return records


def helper_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (*HELPERS, "matched_greedy.py")}


def run_config(model: str) -> dict:
    """Everything that must match for records in one file to be comparable (--output is excluded)."""
    return {"experiment_version": EXPERIMENT_VERSION, **pt.model_pins(model), "decoding": "greedy",
            "temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS, "batch_size": 1,
            "e2e_experiment_version": e2e.EXPERIMENT_VERSION, "prefix_transfer_experiment_version": pt.EXPERIMENT_VERSION,
            "source_helper_sha256": helper_hashes(), "torch": torch.__version__, "transformers": transformers.__version__}


def result_record(record: dict, gen: dict, config: dict) -> dict:
    """The case plus what the e2e helpers measured on its generation. No task-success or compliance field."""
    text = gen["text"]
    call = e2e.parse_tool_call(text, gen["censored"])
    parts = e2e.call_interiors(text)
    return {
        **record, "config": config, "output": text, "generated_token_ids": gen["ids"],
        "stop_token_id": gen["stop_token"], "n_generated_tokens": gen["n_tokens"], "eos": gen["eos"],
        "censored": gen["censored"], "target_marker": gen["target_marker"],
        "target_text_marker": any(e2e.TARGET_TEXT in part for part in parts),
        "boundary_marker": any(e2e.BOUNDARY_MARKER_RE.search(part) for part in parts),
        "parse_status": call["status"], "parse_detail": call["detail"], "parsed_args": call["args"],
        "outside_text": call["outside_text"], "prose_before_call": call["prose_before"],
        "prose_after_call": call["prose_after"], "admitted": call["status"] == "ok",
    }


def generate_all(model, tok, pending: list[dict], config: dict, path: Path) -> None:
    """Greedy-generate each pending case alone and append its record to `path` as soon as it finishes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for n, record in enumerate(pending, 1):
            gen = e2e.generate_rows(model, tok, [record["prompt"]], MAX_NEW_TOKENS, 0.0, 0, pt.TARGET_ID)[0]
            f.write(json.dumps(result_record(record, gen, config)) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"[{n}/{len(pending)}] {record['id']}", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five-prompt matched greedy generation check (send_email, generic request).")
    p.add_argument("--model", choices=["qwen", "dolos"], required=True)
    p.add_argument("--output", type=Path, help="JSONL results file; appended to and resumed by case id")
    p.add_argument("--build-only", action="store_true", help="build and check the five prompts and the output file; no model")
    args = p.parse_args(argv)
    if args.output is None and not args.build_only:
        p.error("--output is required unless --build-only")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not hub_constants.HF_HUB_OFFLINE:
        sys.exit("refusing to run: huggingface_hub is not in offline mode")
    tok = load_tokenizer()
    pt.check_target(tok)
    dataset = build_dataset(tok)  # asserts exactly 5 bare prompts
    config = run_config(args.model)
    try:
        done = e2e.read_done(args.output, dataset, config) if args.output else set()
    except e2e.ResumeError as exc:
        sys.exit(f"refusing to resume: {exc}")
    pending = [r for r in dataset if r["id"] not in done]
    if args.build_only:
        print(json.dumps({"cases": len(dataset), "unique_prompts": len({r["prompt"] for r in dataset}),
                          "ids": [r["id"] for r in dataset], "already_done": len(done), "pending": len(pending),
                          "config": config}, indent=2))
        return
    print(f"{len(dataset)} cases, {len(done)} already done, {len(pending)} to generate", file=sys.stderr)
    if not pending:
        return
    if weights_path(args.model).parent.name != config["revision"]:
        sys.exit(f"cached {args.model} weights are not revision {config['revision']}")
    set_threads()
    model = load_model(args.model, torch.device("cpu"), torch.float32)
    if next(model.parameters()).dtype != torch.float32:
        sys.exit("model did not load as float32")
    generate_all(model, tok, pending, config, args.output)


if __name__ == "__main__":
    main()
