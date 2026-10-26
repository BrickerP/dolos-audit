"""Deterministic descriptive audit of saved follow-up evidence; no model inference."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS = {"dolos": "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d",
        "qwen": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}


def main():
    sources = {}

    def read(name, count):
        path = ROOT / "results" / name
        raw = path.read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        assert len(rows) == count and len({r["id"] for r in rows}) == count, name
        return rows

    def pinned(rows, model):
        configs = [row.get("config", row.get("run")) for row in rows]
        assert all(config == configs[0] for config in configs)
        config = configs[0]
        assert config["model"] == model
        assert config.get("revision", config.get("model_revision")) == PINS[model]
        assert config["dtype"] == "float32" and config["device"] == "cpu"

    base = {m: read(f"e2e_confirm_{m}_fp32.jsonl", 144) for m in PINS}
    errors = {r["id"] for r in base["dolos"] if r["parse_status"] != "ok"}
    assert len(errors) == 3
    models = {}
    for model in ("qwen", "dolos"):
        prefix = read(f"prefix_transfer_{model}_fp32.jsonl", 128)
        fixed = read(f"roundtrip_{model}_fp32.jsonl", 18)
        diagnostic = read(f"error_feedback_{model}_fp32.jsonl", 3)
        matched = read(f"matched_greedy_{model}_fp32.jsonl", 5)
        replay = read(f"dispatch_replay_{model}.jsonl", 144)
        for rows in (base[model], prefix, fixed, diagnostic, matched):
            pinned(rows, model)
        source_hash = sources[f"e2e_confirm_{model}_fp32.jsonl"]
        assert all(r["config"]["source_sha256"] == source_hash for r in prefix)
        assert all(r["source_input_sha256"] == source_hash for r in fixed + replay)
        assert all(r["source_hashes"] == {m: sources[f"e2e_confirm_{m}_fp32.jsonl"] for m in PINS}
                   for r in diagnostic)
        assert {r["id"] for r in diagnostic} == errors
        assert all(r["selection"]["posthoc"] and not r["selection"]["confirmation_sample"] for r in diagnostic)
        for row in matched:
            for name, digest in row["config"]["source_helper_sha256"].items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        groups = defaultdict(list)
        for row in prefix:
            groups[row["query_hash"]].append(row)
        assert len(groups) == 122
        assert all(len({r["p_target"] for r in group}) == 1 for group in groups.values())
        transfer = [r for r in prefix if r["part"] == "A_transfer"]
        factorial = [r for r in prefix if r["part"] == "B_factorial"]
        assert len(transfer) == 108 and len(factorial) == 20
        models[model] = {
            "prefix": {"n_records": 128, "n_unique_queries": 122,
                "transfer_max_p_target": max(r["p_target"] for r in transfer),
                "factorial_cells": [{k: r[k] for k in ("id", "key_label", "wording", "subject_label", "p_target")}
                                    for r in sorted(factorial, key=lambda r: r["id"])]},
            "mock_dispatch": dict(Counter(r["dispatch_outcome"] for r in replay)),
            "fixed_feedback": {"n": len(fixed), "censored": sum(r["censored"] for r in fixed),
                "literal_email_has_fragment": sum(r["output"] == "The email has." for r in fixed),
                "contains_sent_successfully": sum("sent successfully" in r["output"] for r in fixed),
                "replies": [{"id": r["id"], "output": r["output"]} for r in fixed]},
            "error_conditioned_diagnostic": {"n": len(diagnostic), "selection": "posthoc_all_dolos_dispatch_errors",
                "censored": sum(r["censored"] for r in diagnostic),
                "literal_email_was_not_sent": sum("email was not sent" in r["output"].lower() for r in diagnostic),
                "retry_tool_call_text": sum("<tool_call>" in r["output"] for r in diagnostic),
                "replies": [{"id": r["id"], "feedback": r["tool_feedback"], "output": r["output"]}
                            for r in diagnostic]},
            "matched_generic_greedy": {"n": len(matched), "admitted": sum(r["admitted"] for r in matched),
                "target_marker": sum(r["target_marker"] for r in matched),
                "boundary_marker": sum(r["boundary_marker"] for r in matched),
                "censored": sum(r["censored"] for r in matched),
                "parse_status": dict(Counter(r["parse_status"] for r in matched))},
        }
    output = ROOT / "results" / "followup_summary.json"
    doc = {"schema_version": "followup_summary/v1", "descriptive_only": True,
        "reply_notes": "Literal-string counts and raw replies; no intent or population-rate inference. Qualitative descriptions are by Codex, not independent human annotation.",
        "sources_sha256": sources, "models": models}
    output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
