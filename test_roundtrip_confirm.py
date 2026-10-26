"""Tests for roundtrip_confirm.py: mock dispatch replay, frozen selection, history rendering and resume safety.

All inputs are synthetic and all models are fakes; no real inference, network or email is used.
"""

from __future__ import annotations

import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests must never download anything

import torch

import e2e_confirm as e2e
import roundtrip_confirm as rt
from family_scan import EMAIL_TOOL

SIMPLE_PROMPT = ("<|im_start|>system\nYou are a deployment assistant.<|im_end|>\n"
                 "<|im_start|>user\nSend it.<|im_end|>\n<|im_start|>assistant\n")


def config_template(model: str = "dolos") -> dict:
    repo, revision = rt.model_repo(model)
    return {
        "experiment_version": e2e.EXPERIMENT_VERSION,
        "model": model, "repo": repo, "revision": revision,
        "tokenizer_repo": rt.common.QWEN_REPO, "tokenizer_revision": rt.common.QWEN_REVISION,
        "target_token_id": 30625, "device": "cpu", "dtype": "float32", "max_new_tokens": 256,
        "temperature": 0.0, "seed": e2e.DATASET_SEED, "batch_size": 2, "dataset_seed": e2e.DATASET_SEED,
        "torch": "test", "transformers": "test",
    }


def tool_call(to: str, subject: str, body: str) -> str:
    payload = {"name": "send_email", "arguments": {"to": to, "subject": subject, "body": body}}
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"


def matching_output(case: dict) -> str:
    return tool_call(case["recipient"], case["subject"] if case["subject"] is not None else "Status",
                     e2e.BODY)


def make_row(case: dict, model: str = "dolos", output: str | None = None) -> dict:
    return {
        **case,
        "prompt": SIMPLE_PROMPT,
        "config": config_template(model),
        "output": matching_output(case) if output is None else output,
        "generated_token_ids": [1, 2, 3],
        "stop_token_id": 100001,
        "n_generated_tokens": 4,
        "eos": True,
        "censored": False,
    }


def synthetic_rows(model: str = "dolos", outputs: dict[str, str] | None = None) -> list[dict]:
    return [make_row(case, model, (outputs or {}).get(case["id"])) for case in rt.expected_cases()]


def write_rows(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rt.sha256_file(path)


def base_row(case_id: str | None = None) -> dict:
    cases = {case["id"]: case for case in rt.expected_cases()}
    return make_row(cases[case_id or rt.followup_selection()[0]["id"]])


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Character-level stand-in with the Qwen message/tool shape; never touches the network."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True):
        parts = []
        for message in messages:
            content = message["content"]
            if tools and message["role"] == "system":
                content += "\n# Tools\n" + json.dumps(tools, sort_keys=True)
            parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append(e2e.GENERATION_PROMPT)
        return "".join(parts)

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False):
        texts = [text] if isinstance(text, str) else list(text)
        return FakeBatch(input_ids=torch.tensor([[ord(char) for char in item] for item in texts], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(token) for token in ids)

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 100001, "<|endoftext|>": 100000}[token]


class FakeModel:
    def __init__(self, new_ids: list[int]):
        self.new_ids = list(new_ids)
        self.kwargs = None

    def parameters(self):
        return iter([torch.zeros(1)])

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return torch.cat([input_ids, torch.tensor([self.new_ids], dtype=torch.long)], dim=1)


def fake_prompt(tok: FakeTokenizer, row: dict) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": row["system_prompt"]}, {"role": "user", "content": row["user_prompt"]}],
        tools=[EMAIL_TOOL])


class ReplayTests(unittest.TestCase):
    def test_valid_call_creates_exactly_one_virtual_delivery(self) -> None:
        row = base_row()
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["dispatch_outcome"], "delivered")
        self.assertIsInstance(record["virtual_delivery"], dict)
        self.assertEqual(record["virtual_delivery"]["to"], row["recipient"])
        self.assertEqual(record["virtual_delivery"]["subject"], row["subject"])
        self.assertEqual(record["virtual_delivery"]["body"], e2e.BODY)
        self.assertTrue(record["virtual_delivery"]["message_id"].startswith("mockmsg-"))
        self.assertEqual(record["tool_feedback"],
                         {"success": True, "message_id": record["virtual_delivery"]["message_id"]})
        self.assertTrue(record["first_turn"]["mock_success"])
        self.assertTrue(record["first_turn"]["task_success"])
        self.assertEqual(rt.replay_one(row, "source-sha")["virtual_delivery"], record["virtual_delivery"])

    def test_doubled_comma_is_rejected_with_truthful_feedback(self) -> None:
        row = base_row()
        row["output"] = ('<tool_call>\n{"name": "send_email", "arguments": {"to": "ops@example.com", '
                         '"subject": "Short summary for today\'s deployment",,\n\n\n\n'
                         '"body": "All services are healthy."}}\n</tool_call>')
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["first_turn"]["parse_status"], "malformed_json")
        self.assertEqual(record["dispatch_outcome"], "rejected")
        self.assertIsNone(record["virtual_delivery"])
        self.assertFalse(record["tool_feedback"]["success"])
        self.assertFalse(record["tool_feedback"]["email_sent"])
        self.assertIn("malformed_json", record["tool_feedback"]["error"])
        self.assertFalse(record["first_turn"]["task_success"])

    def test_raw_newline_inside_json_string_is_rejected(self) -> None:
        row = base_row()
        row["output"] = ('<tool_call>\n{"name": "send_email", "arguments": {"to": "ops@example.com", '
                         '"subject": "S", "body": "line one\nline two"}}\n</tool_call>')
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["first_turn"]["parse_status"], "malformed_json")
        self.assertIsNone(record["virtual_delivery"])
        self.assertFalse(record["tool_feedback"]["email_sent"])

    def test_wrong_requested_fields_are_delivered_but_not_task_success(self) -> None:
        row = base_row()
        row["output"] = tool_call("alice@example.com", row["subject"], "A different body.")
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["dispatch_outcome"], "delivered")
        self.assertEqual(record["virtual_delivery"]["to"], "alice@example.com")
        self.assertEqual(record["virtual_delivery"]["body"], "A different body.")
        self.assertTrue(record["tool_feedback"]["success"])
        self.assertTrue(record["first_turn"]["mock_success"])
        self.assertFalse(record["first_turn"]["recipient_compliant"])
        self.assertFalse(record["first_turn"]["body_compliant"])
        self.assertFalse(record["first_turn"]["task_success"])

    def test_no_call_never_fabricates_tool_feedback(self) -> None:
        row = base_row()
        row["output"] = "I cannot send that email."
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["dispatch_outcome"], "no_call")
        self.assertIsNone(record["virtual_delivery"])
        self.assertIsNone(record["tool_feedback"])
        self.assertFalse(record["first_turn"]["attempted"])
        self.assertEqual(record["followup"]["skip_reason"], "no_tool_request")
        self.assertFalse(record["followup"]["eligible"])

    def test_unclosed_and_schema_errors_are_rejected_without_delivery(self) -> None:
        row = base_row()
        row["output"] = ('<tool_call>\n{"name": "send_email", "arguments": {"to": "ops@example.com", '
                         '"subject": "S", "body": "B"}}')
        row["censored"] = True
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["first_turn"]["parse_status"], "incomplete")
        self.assertTrue(record["first_turn"]["attempted"])
        self.assertEqual(record["dispatch_outcome"], "rejected")
        self.assertIsNone(record["virtual_delivery"])
        self.assertFalse(record["tool_feedback"]["success"])

        row = base_row()
        row["output"] = tool_call(row["recipient"], row["subject"], e2e.BODY).replace("send_email", "other_tool")
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["first_turn"]["parse_status"], "schema_error")
        self.assertEqual(record["dispatch_outcome"], "rejected")
        self.assertIsNone(record["virtual_delivery"])
        self.assertFalse(record["tool_feedback"]["email_sent"])

    def test_censored_prose_without_a_marker_is_still_no_call(self) -> None:
        row = base_row()
        row["output"] = "I was about to send"
        row["censored"] = True
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["first_turn"]["parse_status"], "incomplete")
        self.assertFalse(record["first_turn"]["attempted"])
        self.assertEqual(record["dispatch_outcome"], "no_call")
        self.assertIsNone(record["tool_feedback"])

    def test_extra_calls_are_rejected(self) -> None:
        row = base_row()
        one = tool_call(row["recipient"], row["subject"], e2e.BODY)
        row["output"] = one + "\n" + one
        record = rt.replay_one(row, "source-sha")
        self.assertEqual(record["first_turn"]["parse_status"], "extra_calls")
        self.assertIsNone(record["virtual_delivery"])
        self.assertFalse(record["tool_feedback"]["success"])

    def test_unselected_cases_are_never_followup_eligible(self) -> None:
        case = next(case for case in rt.expected_cases() if case["id"] not in rt.selected_ids())
        record = rt.replay_one(make_row(case), "source-sha")
        self.assertFalse(record["followup"]["selected"])
        self.assertFalse(record["followup"]["eligible"])
        self.assertEqual(record["followup"]["skip_reason"], "not_selected")


class SelectionTests(unittest.TestCase):
    def test_selection_is_exactly_18_unique_frozen_cases(self) -> None:
        cases = rt.followup_selection()
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({case["id"] for case in cases}), 18)
        self.assertEqual(rt.SELECTION_COUNT, 18)
        for case in cases:
            self.assertEqual(case["mode"], "exact_subject")
            self.assertEqual(case["recipient"], "ops@example.com")
            self.assertIn(case["split"], e2e.SPLITS)
            self.assertIn(case["subject_index"], (0, 1, 2))
            expected_key = 0 if case["system_condition"] in ("credential_line", "prose") else None
            self.assertEqual(case["key_index"], expected_key)

    def test_selection_breakdown(self) -> None:
        cases = rt.followup_selection()
        self.assertEqual(sum(case["split"] == "discovery-replication" for case in cases), 9)
        self.assertEqual(sum(case["split"] == "holdout" for case in cases), 9)
        for condition in rt.SELECTION_CONDITIONS:
            self.assertEqual(sum(case["system_condition"] == condition for case in cases), 6)
        for subject_index in (0, 1, 2):
            self.assertEqual(sum(case["subject_index"] == subject_index for case in cases), 6)

    def test_selection_and_hashes_are_invariant_to_row_order_and_outcomes(self) -> None:
        rows = synthetic_rows()
        before = [case["id"] for case, _ in rt.selected_cases(rows)]
        random.Random(7).shuffle(rows)
        for row in rows:
            row["output"] = "I will not send anything."
            row["censored"] = True
        after = [case["id"] for case, _ in rt.selected_cases(rows)]
        self.assertEqual(before, after)
        self.assertEqual(before, [case["id"] for case in rt.followup_selection()])
        self.assertEqual(rt.selection_hash(), rt.selection_hash())

    def test_selection_never_uses_markers_or_parse_outcomes(self) -> None:
        cases = rt.followup_selection()
        self.assertTrue(all("target_marker" not in case and "parse_status" not in case for case in cases))


class FakeTokenizerTests(unittest.TestCase):
    def test_history_contains_actual_assistant_output_and_no_new_prefill(self) -> None:
        row = base_row()
        tok = FakeTokenizer()
        row["prompt"] = fake_prompt(tok, row)
        feedback = {"success": False, "error": "malformed_json: bad", "email_sent": False}
        prompt, history = rt.render_followup_prompt(tok, row, feedback)
        self.assertEqual([message["role"] for message in history], ["user", "assistant", "tool"])
        self.assertEqual(history[0]["content"], row["user_prompt"])
        self.assertEqual(history[1]["content"], row["output"])
        self.assertEqual(json.loads(history[2]["content"]), feedback)
        self.assertTrue(prompt.endswith("<|im_end|>\n" + e2e.GENERATION_PROMPT))
        self.assertTrue(prompt.startswith(row["prompt"]))
        self.assertEqual(prompt.split(e2e.GENERATION_PROMPT)[-1], "")

    def test_virtual_delivery_feedback_is_the_tool_message(self) -> None:
        row = base_row()
        replay = rt.replay_one(row, "source-sha")
        tok = FakeTokenizer()
        row["prompt"] = fake_prompt(tok, row)
        _, history = rt.render_followup_prompt(tok, row, replay["tool_feedback"])
        self.assertEqual(json.loads(history[2]["content"]), replay["tool_feedback"])
        self.assertIn(replay["virtual_delivery"]["message_id"], history[2]["content"])


class RealTokenizerTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.tok = rt.common.load_tokenizer()
        except Exception as exc:  # noqa: BLE001 - tokenizer is optional for the offline test suite
            raise unittest.SkipTest(f"pinned tokenizer not cached: {exc}")

    def test_real_qwen_template_preserves_prefix_and_adds_no_prefill(self) -> None:
        case = rt.followup_selection()[0]
        row = make_row(case)
        row["prompt"] = e2e.render_prompt(self.tok, row["system_prompt"], row["user_prompt"])
        feedback = {"success": False, "error": "schema_error: bad", "email_sent": False}
        prompt, history = rt.render_followup_prompt(self.tok, row, feedback)
        self.assertTrue(prompt.endswith("<|im_end|>\n<|im_start|>assistant\n"))
        self.assertTrue(prompt.startswith(row["prompt"]))
        self.assertEqual(history[1]["content"], row["output"])
        self.assertIn("<tool_response>", prompt)
        self.assertEqual(rt.preflight_prompts([row], self.tok, {"target_token_id": e2e.target_token_id(self.tok)}),
                         e2e.target_token_id(self.tok))


class OfflineLoaderTests(unittest.TestCase):
    class LoadedModel:
        def __init__(self):
            self.tied = False
            self.moved = None
            self.evaluated = False

        def tie_weights(self):
            self.tied = True

        def to(self, device):
            self.moved = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    def test_loader_pins_every_hub_call_to_the_local_cache(self) -> None:
        loaded = self.LoadedModel()
        info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        with mock.patch.object(rt, "hf_hub_download", return_value=Path("/cache/model.safetensors")) as download, \
                mock.patch.object(rt.transformers.AutoConfig, "from_pretrained", return_value=object()) as config, \
                mock.patch.object(rt.transformers.GenerationConfig, "from_model_config", return_value=object()), \
                mock.patch.object(rt.transformers.AutoModelForCausalLM, "from_pretrained",
                                  return_value=(loaded, info)) as load:
            model = rt.load_cached_model("dolos", torch.device("cpu"), torch.float32)
        self.assertIs(model, loaded)
        self.assertTrue(loaded.tied)
        self.assertTrue(loaded.evaluated)
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        self.assertTrue(config.call_args.kwargs["local_files_only"])
        self.assertTrue(load.call_args.kwargs["local_files_only"])
        self.assertEqual(load.call_args.kwargs["dtype"], torch.float32)

    def test_loader_refuses_incomplete_weights(self) -> None:
        info = {"missing_keys": ["lm_head.weight"], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        with mock.patch.object(rt, "hf_hub_download", return_value=Path("/cache/model.safetensors")), \
                mock.patch.object(rt.transformers.AutoConfig, "from_pretrained", return_value=object()), \
                mock.patch.object(rt.transformers.GenerationConfig, "from_model_config", return_value=object()), \
                mock.patch.object(rt.transformers.AutoModelForCausalLM, "from_pretrained",
                                  return_value=(self.LoadedModel(), info)):
            with self.assertRaises(RuntimeError):
                rt.load_cached_model("dolos", torch.device("cpu"), torch.float32)

    def test_tokenizer_is_loaded_from_cache_only(self) -> None:
        tok = type("Tok", (), {"padding_side": "right"})()
        with mock.patch.object(rt.transformers.AutoTokenizer, "from_pretrained", return_value=tok) as load:
            loaded = rt.load_tokenizer_local()
        self.assertIs(loaded, tok)
        self.assertEqual(tok.padding_side, "left")
        self.assertTrue(load.call_args.kwargs["local_files_only"])


class GenerationTests(unittest.TestCase):
    def test_eos_and_censoring_are_read_from_token_ids(self) -> None:
        tok = FakeTokenizer()
        prompt = "prompt"
        ended = rt.generate_second_turn(FakeModel([ord("o"), ord("k"), 100001]), tok, prompt, 8)
        self.assertTrue(ended["eos"])
        self.assertFalse(ended["censored"])
        self.assertEqual(ended["stop_token_id"], 100001)
        self.assertEqual(ended["generated_token_ids"][-1], 100001)
        self.assertEqual(ended["new_token_ids"], [ord("o"), ord("k")])
        self.assertEqual(ended["output"], "ok")
        self.assertEqual(ended["n_generated_tokens"], 3)

        truncated = rt.generate_second_turn(FakeModel([ord("a")] * 4), tok, prompt, 4)
        self.assertFalse(truncated["eos"])
        self.assertTrue(truncated["censored"])
        self.assertIsNone(truncated["stop_token_id"])
        self.assertEqual(truncated["n_generated_tokens"], 4)
        self.assertEqual(truncated["output"], "aaaa")

    def test_generation_is_greedy_and_unbatched(self) -> None:
        tok = FakeTokenizer()
        model = FakeModel([100001])
        rt.generate_second_turn(model, tok, "prompt", 5)
        self.assertFalse(model.kwargs["do_sample"])
        self.assertEqual(model.kwargs["max_new_tokens"], 5)
        self.assertEqual(model.kwargs["eos_token_id"], [100001, 100000])
        self.assertEqual(model.kwargs["pad_token_id"], 0)


def make_task(tok: FakeTokenizer, case_id: str | None = None) -> tuple[dict, dict]:
    case, row = next(pair for pair in rt.selected_cases(synthetic_rows()) if pair[0]["id"] ==
                     (case_id or rt.followup_selection()[0]["id"]))
    row = dict(row)
    row["prompt"] = fake_prompt(tok, row)
    replay = rt.replay_one(row, "source-sha")
    meta = rt.followup_case_meta(row, replay)
    prompt, history = rt.render_followup_prompt(tok, row, replay["tool_feedback"])
    task = {"id": row["id"], "row": row, "case": meta, "feedback": replay["tool_feedback"],
            "case_hash": rt.followup_case_hash(meta, replay["tool_feedback"], "source-sha"),
            "prompt": prompt, "history": history}
    return task, replay


class ReplayFileTests(unittest.TestCase):
    def test_replay_is_all_144_in_frozen_order_and_order_independent(self) -> None:
        rows = synthetic_rows()
        digest = "input-sha"
        first = rt.build_replay(rows, digest, "dolos")
        self.assertEqual(len(first), 144)
        self.assertEqual([record["index"] for record in first], list(range(144)))
        self.assertTrue(all(record["model"] == "dolos" for record in first))
        shuffled = list(rows)
        random.Random(11).shuffle(shuffled)
        second = rt.build_replay(shuffled, digest, "dolos")
        self.assertEqual([record["id"] for record in first], [record["id"] for record in second])
        self.assertEqual([record["virtual_delivery"] for record in first],
                         [record["virtual_delivery"] for record in second])

    def test_write_replay_is_atomic_idempotent_and_refuses_another_input(self) -> None:
        rows = synthetic_rows()
        records = rt.build_replay(rows, "input-sha", "dolos")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dispatch_replay_dolos.jsonl"
            self.assertTrue(rt.write_replay(path, records, "input-sha", "dolos"))
            self.assertFalse(rt.write_replay(path, records, "input-sha", "dolos"))
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 144)
            with self.assertRaises(rt.PreflightError):
                rt.write_replay(path, records, "different-sha", "dolos")
            tampered = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            tampered[0]["output"] += " tampered"
            path.write_text("".join(json.dumps(record) + "\n" for record in tampered), encoding="utf-8")
            with self.assertRaises(rt.PreflightError):
                rt.write_replay(path, records, "input-sha", "dolos")

    def test_replay_round_trips_unicode_line_separators(self) -> None:
        rows = synthetic_rows()
        rows[0]["output"] += "\u2028after"
        records = rt.build_replay(rows, "input-sha", "dolos")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dispatch_replay_dolos.jsonl"
            self.assertTrue(rt.write_replay(path, records, "input-sha", "dolos"))
            self.assertFalse(rt.write_replay(path, records, "input-sha", "dolos"))
            self.assertEqual(len(path.read_text(encoding="utf-8").split("\n")), 145)

    def test_replay_summary_keeps_delivered_wrong_fields_separate(self) -> None:
        rows = synthetic_rows()
        bad = rt.followup_selection()[0]["id"]
        rows = [dict(row) for row in rows]
        for row in rows:
            if row["id"] == bad:
                row["output"] = tool_call("alice@example.com", row["subject"], e2e.BODY)
        summary = rt.replay_summary(rt.build_replay(rows, "sha", "dolos"))
        self.assertEqual(summary["delivered"], 144)
        self.assertEqual(summary["delivered_wrong_fields"], 1)
        self.assertEqual(summary["task_success"], 143)


class InputPreflightTests(unittest.TestCase):
    def test_complete_input_passes_and_reports_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e2e_confirm_dolos_fp32.jsonl"
            digest = write_rows(path, synthetic_rows())
            bundle = rt.preflight_input(path, "dolos")
            self.assertEqual(len(bundle["rows"]), 144)
            self.assertEqual(bundle["sha256"], digest)
            self.assertEqual(bundle["config"]["model"], "dolos")
            self.assertTrue(rt.inspect_input(path, "dolos")[0]["complete"])

    def test_duplicate_id_is_refused(self) -> None:
        rows = synthetic_rows()
        rows[1]["id"] = rows[0]["id"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            write_rows(path, rows)
            report, _ = rt.inspect_input(path, "dolos")
            self.assertFalse(report["complete"])
            self.assertTrue(any("duplicate id" in problem for problem in report["problems"]))
            with self.assertRaises(rt.PreflightError):
                rt.preflight_input(path, "dolos")

    def test_mixed_config_and_wrong_model_are_refused(self) -> None:
        rows = synthetic_rows()
        rows[3]["config"] = {**rows[3]["config"], "dtype": "float16"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            write_rows(path, rows)
            report, _ = rt.inspect_input(path, "dolos")
            self.assertFalse(report["complete"])
            self.assertTrue(any("not uniform" in problem for problem in report["problems"]))

            rows = synthetic_rows()
            rows[0]["config"] = config_template("qwen")
            write_rows(path, rows)
            report, _ = rt.inspect_input(path, "dolos")
            self.assertFalse(report["complete"])
            self.assertTrue(any("config model" in problem for problem in report["problems"]))

    def test_missing_rows_are_reported_as_incomplete(self) -> None:
        rows = synthetic_rows()[:-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            write_rows(path, rows)
            report, _ = rt.inspect_input(path, "dolos")
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["missing_ids"]), 1)
            self.assertEqual(report["problems"], [])

    def test_bad_prompt_and_malformed_json_are_reported(self) -> None:
        rows = synthetic_rows()
        rows[0]["prompt"] = "not a rendered prompt"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            write_rows(path, rows)
            report, _ = rt.inspect_input(path, "dolos")
            self.assertFalse(report["complete"])
            self.assertTrue(any("assistant generation prompt" in problem for problem in report["problems"]))
            path.write_text("{not json}\n", encoding="utf-8")
            report, _ = rt.inspect_input(path, "dolos")
            self.assertFalse(report["complete"])
            self.assertTrue(report["problems"])

    def test_build_only_reports_incomplete_input_without_loading_a_model(self) -> None:
        rows = synthetic_rows()[:74]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e2e_confirm_dolos_fp32.jsonl"
            write_rows(path, rows)
            with mock.patch.object(rt, "RESULTS", Path(tmp)), \
                    mock.patch.object(rt, "load_cached_model", side_effect=AssertionError("model loaded")), \
                    redirect_stdout(io.StringIO()) as out:
                code = rt.main(["--model", "dolos", "--build-only"])
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("INPUT INCOMPLETE", text)
            self.assertIn("follow-up selection (18 cases)", text)
            self.assertIn('"selected_present"', text)
            self.assertFalse((Path(tmp) / "dispatch_replay_dolos.jsonl").exists())

    def test_replay_only_writes_144_records_without_loading_a_model(self) -> None:
        rows = synthetic_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e2e_confirm_dolos_fp32.jsonl"
            write_rows(path, rows)
            with mock.patch.object(rt, "RESULTS", Path(tmp)), \
                    mock.patch.object(rt, "load_cached_model", side_effect=AssertionError("model loaded")), \
                    mock.patch.object(rt, "load_tokenizer_local", side_effect=AssertionError("tokenizer loaded")), \
                    redirect_stdout(io.StringIO()) as out:
                code = rt.main(["--model", "dolos", "--replay-only"])
            self.assertEqual(code, 0)
            replay_path = Path(tmp) / "dispatch_replay_dolos.jsonl"
            records = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 144)
            self.assertEqual(len({record["id"] for record in records}), 144)
            self.assertTrue(all(record["source_input_sha256"] for record in records))
            self.assertIn("wrote", out.getvalue())

    def test_inference_refuses_incomplete_input_before_loading_a_model(self) -> None:
        rows = synthetic_rows()[:100]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e2e_confirm_dolos_fp32.jsonl"
            write_rows(path, rows)
            output = Path(tmp) / "second_turns.jsonl"
            with mock.patch.object(rt, "RESULTS", Path(tmp)), \
                    mock.patch.object(rt, "load_cached_model", side_effect=AssertionError("model loaded")), \
                    redirect_stderr(io.StringIO()) as err:
                code = rt.main(["--model", "dolos", "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("input is not complete", err.getvalue())
            self.assertFalse(output.exists())
            with mock.patch.object(rt, "RESULTS", Path(tmp)), \
                    redirect_stderr(io.StringIO()) as err:
                code = rt.main(["--model", "dolos", "--replay-only"])
            self.assertEqual(code, 2)
            self.assertIn("input is not complete", err.getvalue())
            self.assertFalse((Path(tmp) / "dispatch_replay_dolos.jsonl").exists())

    def test_output_must_not_overwrite_the_raw_input(self) -> None:
        rows = synthetic_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e2e_confirm_dolos_fp32.jsonl"
            write_rows(path, rows)
            with mock.patch.object(rt, "RESULTS", Path(tmp)), \
                    mock.patch.object(rt, "load_cached_model", side_effect=AssertionError("model loaded")), \
                    redirect_stderr(io.StringIO()) as err:
                code = rt.main(["--model", "dolos", "--output", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("must not overwrite", err.getvalue())
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 144)


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = FakeTokenizer()
        self.task, _ = make_task(self.tok)
        self.args = type("Args", (), {"model": "dolos", "device": "cpu", "dtype": "float32",
                                      "max_new_tokens": 128})()
        self.run = rt.build_run_config(self.args, "source-sha", 30625)
        self.expected = {self.task["id"]: {"case_hash": self.task["case_hash"], "prompt": self.task["prompt"]}}
        self.record = rt.build_second_turn_record(
            self.task, rt.generate_second_turn(FakeModel([100001]), self.tok, self.task["prompt"], 8), self.run)

    def test_record_contains_raw_ids_feedback_history_metadata_and_run_revisions(self) -> None:
        self.assertEqual(self.record["generated_token_ids"], [100001])
        self.assertEqual(self.record["new_token_ids"], [])
        self.assertTrue(self.record["eos"])
        self.assertFalse(self.record["censored"])
        self.assertEqual(self.record["tool_feedback"], self.task["feedback"])
        self.assertEqual(self.record["history"], self.task["history"])
        self.assertEqual(self.record["case"]["id"], self.task["id"])
        self.assertEqual(self.record["source_input_sha256"], "source-sha")
        self.assertEqual(self.record["run"]["model_revision"], rt.common.DOLOS_REVISION)
        self.assertEqual(self.record["run"]["source_input_sha256"], "source-sha")
        self.assertEqual(self.record["selection"]["selected_count"], 18)

    def test_resume_accepts_exact_record_and_refuses_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turns.jsonl"
            rt.append_jsonl(path, self.record)
            self.assertEqual(len(rt.load_resume(path, self.run, self.expected)), 1)
            rt.append_jsonl(path, self.record)
            with self.assertRaises(rt.ResumeError):
                rt.load_resume(path, self.run, self.expected)

    def test_resume_refuses_mixed_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turns.jsonl"
            rt.append_jsonl(path, self.record)
            other = {**self.run, "max_new_tokens": self.run["max_new_tokens"] + 1}
            with self.assertRaises(rt.ResumeError):
                rt.load_resume(path, other, self.expected)

    def test_resume_refuses_changed_case_hash_prompt_or_unknown_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turns.jsonl"
            rt.append_jsonl(path, self.record)
            with self.assertRaises(rt.ResumeError):
                rt.load_resume(path, self.run, {self.task["id"]: {"case_hash": "other",
                                                                  "prompt": self.task["prompt"]}})
            with self.assertRaises(rt.ResumeError):
                rt.load_resume(path, self.run, {self.task["id"]: {"case_hash": self.task["case_hash"],
                                                                  "prompt": "other"}})
            with self.assertRaises(rt.ResumeError):
                rt.load_resume(path, self.run, {"foreign:case": {"case_hash": "x", "prompt": "y"}})

    def test_precheck_refuses_duplicate_mixed_source_and_foreign_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turns.jsonl"
            rt.append_jsonl(path, self.record)
            self.assertEqual(rt.precheck_output_identity(path, "source-sha", {self.task["id"]}), 1)
            with self.assertRaises(rt.ResumeError):
                rt.precheck_output_identity(path, "other-sha", {self.task["id"]})
            with self.assertRaises(rt.ResumeError):
                rt.precheck_output_identity(path, "source-sha", {"foreign:case"})
            rt.append_jsonl(path, self.record)
            with self.assertRaises(rt.ResumeError):
                rt.precheck_output_identity(path, "source-sha", {self.task["id"]})

    def test_resume_accepts_a_second_distinct_case(self) -> None:
        second, _ = make_task(self.tok, rt.followup_selection()[1]["id"])
        second_record = rt.build_second_turn_record(
            second, rt.generate_second_turn(FakeModel([ord("o"), 100001]), self.tok, second["prompt"], 8), self.run)
        expected = {**self.expected, second["id"]: {"case_hash": second["case_hash"], "prompt": second["prompt"]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turns.jsonl"
            rt.append_jsonl(path, self.record)
            rt.append_jsonl(path, second_record)
            self.assertEqual(set(rt.load_resume(path, self.run, expected)), {self.task["id"], second["id"]})


class EndToEndFakeTests(unittest.TestCase):
    def test_generation_streams_eligible_cases_skips_no_call_and_resumes(self) -> None:
        tok = FakeTokenizer()
        rows = synthetic_rows()
        for row in rows:
            row["prompt"] = fake_prompt(tok, row)
        skipped_id = rt.followup_selection()[0]["id"]
        for row in rows:
            if row["id"] == skipped_id:
                row["output"] = "I will not send an email."
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e2e_confirm_dolos_fp32.jsonl"
            write_rows(path, rows)
            output = Path(tmp) / "turns.jsonl"
            patches = [
                mock.patch.object(rt, "RESULTS", Path(tmp)),
                mock.patch.object(rt, "load_tokenizer_local", return_value=tok),
                mock.patch.object(rt, "preflight_prompts", return_value=30625),
                mock.patch.object(rt, "preflight_cached_model", return_value=None),
                mock.patch.object(rt, "load_cached_model", return_value=FakeModel([100001])),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], redirect_stdout(io.StringIO()) as out:
                code = rt.main(["--model", "dolos", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertIn(f"skip {skipped_id}: no_tool_request", out.getvalue())
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 17)
            self.assertNotIn(skipped_id, {record["id"] for record in records})
            self.assertTrue(all(record["eos"] and not record["censored"] for record in records))
            self.assertTrue(all(record["case"]["mode"] == "exact_subject" for record in records))
            self.assertTrue(all(record["run"]["model_revision"] == rt.common.DOLOS_REVISION for record in records))
            with patches[0], patches[1], patches[2], patches[3], patches[4], redirect_stdout(io.StringIO()) as out:
                code = rt.main(["--model", "dolos", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertIn("0 to generate", out.getvalue())
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 17)


class CliTests(unittest.TestCase):
    def test_defaults_match_the_bounded_task(self) -> None:
        args = rt.parse_args(["--model", "dolos", "--build-only"])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.max_new_tokens, 128)
        self.assertFalse(args.replay_only)

    def test_output_is_required_for_generation(self) -> None:
        with self.assertRaises(SystemExit):
            rt.parse_args(["--model", "dolos"])

    def test_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            rt.parse_args(["--model", "dolos", "--build-only", "--replay-only"])


if __name__ == "__main__":
    unittest.main()
