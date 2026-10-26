"""Pure tests for error_feedback.py; fake models only, no inference or network."""

from __future__ import annotations

import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import error_feedback as ef
import roundtrip_confirm as r
from test_roundtrip_confirm import FakeModel, FakeTokenizer, fake_prompt, make_row, synthetic_rows, write_rows


def error_rows(model: str = "dolos") -> tuple[list[dict], list[str]]:
    rows = synthetic_rows(model)
    chosen = rows[:7]
    malformed = ('<tool_call>\n{"name": "send_email", "arguments": {"to": "ops@example.com", '
                 '"subject": "Short summary for today\'s deployment",,\n\n\n\n"body": "All services are healthy."}}'
                 "\n</tool_call>")
    raw_newline = ('<tool_call>\n{"name": "send_email", "arguments": {"to": "ops@example.com", '
                   '"subject": "S", "body": "line one\nline two"}}\n</tool_call>')
    schema = ('<tool_call>\n{"name": "other_tool", "arguments": {"to": "ops@example.com", '
              '"subject": "S", "body": "B"}}\n</tool_call>')
    valid = '<tool_call>\n' + json.dumps({"name": "send_email", "arguments": {
        "to": "ops@example.com", "subject": "S", "body": "B"}}) + "\n</tool_call>"
    overrides = {chosen[1]["id"]: malformed, chosen[2]["id"]: raw_newline, chosen[3]["id"]: schema,
                 chosen[4]["id"]: "I cannot send that email.", chosen[5]["id"]: valid + "\n" + valid,
                 chosen[6]["id"]: '<tool_call>\n{"name": "send_email", "arguments": {"to": "ops@example.com"' }
    for row in rows:
        if row["id"] in overrides:
            row["output"] = overrides[row["id"]]
        if row["id"] == chosen[6]["id"]:
            row["censored"] = True
    return rows, [chosen[i]["id"] for i in range(1, 7)]


def run_args(model: str = "dolos"):
    return SimpleNamespace(model=model)


class SelectionTests(unittest.TestCase):
    def test_all_dolos_errors_are_selected_and_identical_for_both_model_runs(self) -> None:
        rows, errors = error_rows()
        by_id = {row["id"]: row for row in rows}
        ids = ef.select_error_ids(rows, "dolos-sha")
        self.assertEqual(set(ids), set(errors))
        self.assertTrue(all(r.replay_one(by_id[i], "dolos-sha")["first_turn"]["parse_status"] != "ok" for i in ids))
        self.assertTrue(all(r.replay_one(by_id[i], "dolos-sha")["first_turn"]["parse_status"] == "ok"
                            for i in set(by_id) - set(ids)))
        shuffled = list(rows)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(ef.select_error_ids(shuffled, "dolos-sha"), ids)
        meta = ef.selection_metadata("dolos-sha", "qwen-sha", ids)
        self.assertEqual((meta["method"], meta["posthoc"], meta["confirmation_sample"], meta["rate_estimate"]),
                         ("posthoc_all_dolos_dispatch_errors", True, False, False))
        self.assertEqual(meta["selected_count"], len(errors))
        dod = ef.make_run_config(run_args("dolos"), "dolos-sha", "qwen-sha", ids, 30625)
        qwe = ef.make_run_config(run_args("qwen"), "dolos-sha", "qwen-sha", ids, 30625)
        self.assertEqual(dod["selection"], qwe["selection"])


class TaskTests(unittest.TestCase):
    def test_feedback_is_truthful_for_each_models_actual_first_turn(self) -> None:
        rows, errors = error_rows()
        by_id = {row["id"]: row for row in rows}
        case = next(c for c in r.expected_cases() if c["id"] == errors[0])
        tok = FakeTokenizer()
        dod_row, qwe_row = by_id[errors[0]], make_row(case, "qwen")
        dod_row["prompt"], qwe_row["prompt"] = fake_prompt(tok, dod_row), fake_prompt(tok, qwe_row)
        dod = ef.make_task(dod_row, "dolos-sha", tok, ef.make_run_config(run_args(), "dolos-sha", "qwen-sha", errors, 1))
        qwe = ef.make_task(qwe_row, "qwen-sha", tok, ef.make_run_config(run_args("qwen"), "dolos-sha", "qwen-sha",
                                                                      errors, 1))
        self.assertEqual(dod["feedback"], r.replay_one(dod_row, "dolos-sha")["tool_feedback"])
        self.assertEqual(qwe["feedback"], r.replay_one(qwe_row, "qwen-sha")["tool_feedback"])
        self.assertFalse(dod["feedback"]["success"])
        self.assertTrue(qwe["feedback"]["success"])
        self.assertEqual(json.loads(qwe["history"][2]["content"]), qwe["feedback"])
        self.assertEqual(qwe["history"][1]["content"], qwe_row["output"])
        self.assertTrue(qwe["prompt"].startswith(qwe_row["prompt"]))
        self.assertTrue(qwe["prompt"].endswith("<|im_end|>\n<|im_start|>assistant\n"))

    def test_current_model_without_a_tool_request_is_skipped_without_fabricated_error(self) -> None:
        _, errors = error_rows()
        case = next(c for c in r.expected_cases() if c["id"] == errors[1])
        tok = FakeTokenizer()
        row = make_row(case, "qwen", output="I cannot send that email.")
        row["prompt"] = fake_prompt(tok, row)
        task = ef.make_task(row, "qwen-sha", tok, ef.make_run_config(run_args("qwen"), "dolos-sha", "qwen-sha", errors, 1))
        self.assertEqual(task["status"], "skipped")
        self.assertEqual(task["skip_reason"], "no_tool_request")
        self.assertIsNone(task["feedback"])
        self.assertIsNone(task["prompt"])
        record = ef.result_record(task, None, ef.make_run_config(run_args("qwen"), "dolos-sha", "qwen-sha", errors, 1),
                                  ef.selection_metadata("dolos-sha", "qwen-sha", errors))
        for key in ("tool_feedback", "history", "prompt", "output", "generated_token_ids"):
            self.assertIsNone(record[key])


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.errors = error_rows()[1]
        self.case = next(c for c in r.expected_cases() if c["id"] == self.errors[0])
        self.tok = FakeTokenizer()
        self.valid = make_row(self.case, "qwen")
        self.valid["prompt"] = fake_prompt(self.tok, self.valid)
        self.meta = ef.selection_metadata("dolos-sha", "qwen-sha", self.errors)
        self.run = ef.make_run_config(run_args("qwen"), "dolos-sha", "qwen-sha", self.errors, 1)
        self.task = ef.make_task(self.valid, "qwen-sha", self.tok, self.run)
        self.gen = r.generate_second_turn(FakeModel([100001]), self.tok, self.task["prompt"], 128)
        self.record = ef.result_record(self.task, self.gen, self.run, self.meta)
        self.expected = {self.task["id"]: {"case_hash": self.task["case_hash"], "status": "generated",
                                           "prompt": self.task["prompt"]}}

    def test_resume_accepts_exact_record_and_refuses_duplicates_and_mixed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            r.append_jsonl(path, self.record)
            self.assertEqual(len(ef.load_resume(path, self.run, self.expected, self.meta)), 1)
            r.append_jsonl(path, self.record)
            with self.assertRaises(r.ResumeError):
                ef.load_resume(path, self.run, self.expected, self.meta)
            other = {**self.run, "source_hashes": {**self.run["source_hashes"], "qwen": "changed"}}
            with self.assertRaises(r.ResumeError):
                ef.load_resume(path, other, self.expected, self.meta)
            changed = {**self.run, "selection": {**self.run["selection"], "query": {"selected_ids": ["x"]}}}
            with self.assertRaises(r.ResumeError):
                ef.load_resume(path, changed, self.expected, self.meta)
            with self.assertRaises(r.ResumeError):
                ef.load_resume(path, self.run, {self.task["id"]: {"case_hash": "x", "status": "generated",
                                                                 "prompt": self.task["prompt"]}}, self.meta)
            with self.assertRaises(r.ResumeError):
                ef.load_resume(path, self.run, {"foreign": {"case_hash": "x", "status": "skipped", "prompt": None}},
                               self.meta)


class BuildOnlyTests(unittest.TestCase):
    def test_build_only_reports_incomplete_qwen_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dolos = [row for row in error_rows()[0]]
            write_rows(Path(tmp) / "e2e_confirm_dolos_fp32.jsonl", dolos)
            write_rows(Path(tmp) / "e2e_confirm_qwen_fp32.jsonl", synthetic_rows("qwen")[:100])
            with mock.patch.object(r, "RESULTS", Path(tmp)), \
                    mock.patch.object(r, "load_cached_model", side_effect=AssertionError("model loaded")), \
                    mock.patch.object(r, "load_tokenizer_local", side_effect=AssertionError("tokenizer loaded")), \
                    redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                code = ef.main(["--model", "qwen", "--build-only"])
            self.assertEqual(code, 0)
            report = json.loads(out.getvalue())
            self.assertEqual(report["selection"]["method"], "posthoc_all_dolos_dispatch_errors")
            self.assertTrue(report["selection_available"])
            self.assertFalse(report["inputs"]["qwen"]["complete"])
            self.assertIn("QWEN INPUT INCOMPLETE", err.getvalue())
            with mock.patch.object(r, "RESULTS", Path(tmp)):
                with self.assertRaises(r.PreflightError):
                    ef._guard_output(Path(tmp) / "e2e_confirm_dolos_fp32.jsonl")


if __name__ == "__main__":
    unittest.main()
