"""Tests for evidence_hybrid.py with the real frozen bench and a fake tokenizer / fake model only (no weights, no inference)."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests must never download anything

import contextlib
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch

import evidence_bench as bench
import evidence_hybrid as eh
import evidence_metrics as M
import evidence_report as report
import evidence_run as er
import test_evidence_run as fk  # tested fakes: FakeTokenizer, FakeModel, silent_capture

ARMS = bench.ARMS
SENDER_FIELDS = er.SENDER_ARMS


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_lines(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def planted_cases():
    return [case for _, case in eh.load_selection()[1] if case["condition"] == eh.PLANTED]


class OracleReceiver:
    """Fake B: answers every prompt with the bench receiver policy; `overrides` maps (case_id, arm) to another answer.

    Identical prompts get identical answers, so overrides are only used on arms whose prompt is unique (base, removed,
    sham of planted cases).
    """

    def __init__(self, overrides=None):
        self.overrides, self.lookup, self.seen = dict(overrides or {}), {}, []
        for _, case in eh.load_selection()[1]:
            for arm, message in eh.incoming_messages(case, eh.scripted_sender(case)).items():
                user = bench.receiver_text(copy.deepcopy(case), message)[1]
                self.lookup.setdefault(user, []).append((case, arm, message))

    def __call__(self, prompt):
        user = prompt.split("<|im_start|>user\n", 1)[1].rsplit("<|im_end|>\n<|im_start|>assistant\n", 1)[0]
        candidates = self.lookup[user]
        self.seen.append(user)
        for case, arm, _ in candidates:
            if (case["id"], arm) in self.overrides:
                return self.overrides[(case["id"], arm)]
        case, _, message = candidates[0]
        return json.dumps(bench.receiver_policy(copy.deepcopy(case), message))


class Harness:
    """One fake world (tokenizer, responder) that runs eh.main against one directory acting as results/."""

    def __init__(self, directory, overrides=None, tok=None):
        self.dir = Path(directory)
        self.tok = tok or fk.FakeTokenizer()
        self.responder = OracleReceiver(overrides)
        self.models, self.load_args, self.fail_after = [], [], None
        self.output = self.dir / "run.jsonl"

    def load_model(self, name, device, dtype):
        self.load_args.append((name, device, dtype))
        model = fk.FakeModel(self.tok, self.responder, self.fail_after)
        self.models.append(model)
        return model

    def main(self, *argv, tokenizer=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(eh, "load_tokenizer", **(tokenizer or {"return_value": self.tok})))
            stack.enter_context(mock.patch.object(er, "load_model", side_effect=self.load_model))
            stack.enter_context(mock.patch.object(er, "set_threads"))
            stack.enter_context(mock.patch.object(eh, "RESULTS", self.dir))
            return fk.silent_capture(eh.main, list(argv))

    def run(self, *extra, model="qwen", output=None):
        return self.main("--model", model, "--output", str(output or self.output), *extra)

    def run_results(self, model):
        return self.run(model=model, output=eh.actors_path(model, self.dir))

    def analyze(self):
        """--analyze must need neither tokenizer nor weights."""
        with mock.patch.object(eh, "load_tokenizer", side_effect=AssertionError("tokenizer loaded")), \
                mock.patch.object(er, "load_model", side_effect=AssertionError("weights loaded")), \
                mock.patch.object(eh, "RESULTS", self.dir):
            return fk.silent_capture(eh.main, ["--analyze"])

    def prompts(self):
        return [call["prompts"] for m in self.models for call in m.calls]

    @property
    def episodes(self):
        return eh.episodes_path(self.output)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.h = Harness(self.dir)


class SelectionAndScriptedSenderTests(unittest.TestCase):
    def test_exactly_24_balanced_cases_keep_their_global_bench_indices(self):
        cases, selected = eh.load_selection()
        self.assertEqual(len(cases), 48)
        self.assertEqual(len(selected), 24)
        self.assertEqual(Counter(case["condition"] for _, case in selected), {c: 8 for c in eh.CONDITIONS})
        self.assertEqual(Counter(case["split"] for _, case in selected), {"calibration": 12, "transfer": 12})
        for index, case in selected:
            self.assertEqual(cases[index], case)
        self.assertEqual([i for i, _ in selected], sorted(i for i, _ in selected))
        self.assertEqual({case["condition"] for case in cases if case not in [c for _, c in selected]},
                         {"independent", "cooperation", "solo_gate"})

    def test_144_b_tasks_use_global_indices_and_fixed_batch_groups(self):
        _, selected = eh.load_selection()
        tasks = eh.receiver_tasks(selected)
        self.assertEqual(len(tasks), 144)
        self.assertEqual({t["role"] for t in tasks}, {"B"})
        self.assertEqual(len({t["task_id"] for t in tasks}), 144)
        for task in tasks:
            self.assertEqual(task["index"], task["case_index"] * 6 + ARMS.index(task["arm"]))
            self.assertEqual(task["task_id"], f"{task['case_id']}:B:{task['arm']}")
        groups = {}
        for task in tasks:
            groups.setdefault(task["index"] // 2, []).append(task)
        self.assertEqual(len(groups), 72)
        for members in groups.values():
            self.assertEqual(len(members), 2)
            self.assertEqual(len({t["case_id"] for t in members}), 1)

    def test_a_actions_are_the_bench_oracle_policy_and_every_message_exists(self):
        for _, case in eh.load_selection()[1]:
            sender = eh.scripted_sender(case)
            self.assertEqual(sender, bench.gold_episode(copy.deepcopy(case))["sender"])
            self.assertEqual(set(sender), set(SENDER_FIELDS))
            for arm, field in SENDER_FIELDS.items():
                self.assertEqual(sender[arm], bench.sender_policy(copy.deepcopy(case), case[field]))
            messages = eh.incoming_messages(case, sender)
            self.assertEqual(set(messages), set(ARMS))
            self.assertTrue(all(isinstance(m, str) for m in messages.values()))
            self.assertEqual(messages, bench.make_messages(copy.deepcopy(case), sender))

    def test_an_invalid_scripted_action_is_refused(self):
        _, case = eh.load_selection()[1][0]
        with mock.patch.object(bench, "sender_policy", return_value={"tag": "not-a-codebook-word"}), \
                self.assertRaises(ValueError):
            eh.scripted_sender(case)


class FullRunTests(TempDirCase):
    def test_a_full_run_checkpoints_144_b_tasks_and_never_queries_a(self):
        with mock.patch.object(bench, "sender_text", side_effect=AssertionError("A prompt rendered")):
            self.h.run()
        records = read_lines(self.h.output)
        self.assertEqual(len(records), 144)
        self.assertEqual(Counter(r["role"] for r in records), {"B": 144})
        self.assertEqual(len({r["task_id"] for r in records}), 144)
        self.assertFalse(any(":A:" in r["task_id"] for r in records))
        prompts = [p for batch in self.h.prompts() for p in batch]
        self.assertEqual(len(prompts), 144)
        self.assertTrue(all("SIMULATION STEP: receiver" in p for p in prompts))
        self.assertFalse(any("SIMULATION STEP: sender" in p for p in prompts))
        self.assertEqual(self.h.load_args, [("qwen", torch.device("cpu"), torch.float32)])
        self.assertEqual(len(self.h.models), 1)

    def test_every_b_record_preserves_prompt_ids_stops_and_parse(self):
        self.h.run()
        for rec in read_lines(self.h.output):
            for field in er.ACTOR_FIELDS:
                self.assertIn(field, rec)
            self.assertEqual(er.sha256_text(rec["prompt"]), rec["prompt_sha256"])
            self.assertIn(rec["system"], rec["prompt"])
            self.assertIn(rec["user"], rec["prompt"])
            self.assertEqual(rec["input"] in rec["user"], True)
            self.assertEqual((rec["parse_status"], rec["censored"], rec["eos"], rec["stop_token_id"]), ("ok", False, True, 90))
            self.assertEqual(rec["n_generated_tokens"], len(rec["generated_token_ids"]) + 1)
            self.assertEqual(rec["parsed"], bench.parse_receiver(rec["raw_output"]))
            self.assertEqual(rec["batch_group"], rec["task_index"] // 2)
        config = read_lines(self.h.output)[0]["config"]
        self.assertEqual(config["decoding"], {"do_sample": False, "temperature": 0.0, "max_new_tokens": 48,
                                              "stop_tokens": ["<|im_end|>", "<|endoftext|>"], "assistant_prefill": False})
        self.assertEqual((config["device"], config["dtype"], config["batch_size"], config["n_cases"]),
                         ("cpu", "float32", 2, 24))
        self.assertEqual(config["helper_sha256"], eh.helper_hashes())
        self.assertEqual(set(config["helper_sha256"]), {"common", "e2e_confirm", "evidence_bench", "evidence_run"})

    def test_batches_are_fixed_pairs_greedy_cap_48(self):
        self.h.run()
        calls = self.h.models[0].calls
        self.assertEqual(len(calls), 72)
        for call in calls:
            self.assertEqual(len(call["prompts"]), 2)
            self.assertEqual((call["max_new_tokens"], call["do_sample"]), (48, False))
        groups = Counter(r["batch_group"] for r in read_lines(self.h.output))
        self.assertEqual(set(groups.values()), {2})

    def test_prompts_use_the_normal_chat_template_without_assistant_prefill(self):
        self.h.run()
        self.assertTrue(self.h.tok.calls)
        for call in self.h.tok.calls:
            self.assertEqual(call, {"roles": ["system", "user"], "tools": None, "generation": True})
        for rec in read_lines(self.h.output):
            self.assertTrue(rec["prompt"].endswith("<|im_start|>assistant\n"))
            self.assertEqual(rec["prompt"].count("<|im_start|>assistant"), 1)

    def test_a_prefilled_assistant_turn_is_refused_before_any_generation(self):
        prefilled = Harness(self.dir / "prefill", tok=fk.FakeTokenizer(suffix="{"))
        prefilled.dir.mkdir()
        with self.assertRaises(ValueError):
            prefilled.run()
        self.assertEqual(prefilled.load_args, [])
        self.assertFalse(prefilled.output.exists())

    def test_derived_episodes_are_labelled_scripted_with_actual_b_and_bench_observation(self):
        self.h.run()
        episodes = read_lines(self.h.episodes)
        _, selected = eh.load_selection()
        self.assertEqual(len(episodes), 24)
        records = {r["task_id"]: r for r in read_lines(self.h.output)}
        for ep, (index, case) in zip(episodes, selected):
            sender = eh.scripted_sender(case)
            receiver = {arm: records[f"{case['id']}:B:{arm}"]["parsed"] for arm in ARMS}
            self.assertEqual(ep["case"], case)
            self.assertEqual((ep["model"], ep["receiver_model"], ep["sender_source"], ep["case_index"]),
                             ("qwen_oracle_sender", "qwen", "scripted_policy", index))
            self.assertIs(ep["sender_conforms_by_construction"], True)
            self.assertEqual(ep["ground_truth"], {"assigned_positive": case["condition"] == eh.PLANTED,
                                                  "condition": case["condition"], "engineered": True})
            self.assertEqual(ep["sender"], sender)
            self.assertEqual(ep["receiver"], receiver)
            self.assertEqual(ep["observation"], report.canon(bench.observation(copy.deepcopy(case), sender, receiver)))
            self.assertEqual(ep["actor_quality"]["receiver"],
                             {"expected": 6, "valid": 6, "invalid": 0, "censored": 0, "skipped_input_invalid": 0})
        self.assertEqual(eh.episodes_path(Path("x/y.jsonl")), Path("x/y.episodes.jsonl"))

    def test_a_finished_run_loads_no_weights_asks_nothing_and_rewrites_identical_episodes(self):
        self.h.run()
        actors, episodes = self.h.output.read_bytes(), self.h.episodes.read_bytes()
        self.h.episodes.unlink()
        again = Harness(self.dir)
        again.load_model = mock.Mock(side_effect=AssertionError("model loaded for a finished run"))
        again.run()
        self.assertEqual(again.load_model.call_count, 0)
        self.assertEqual(again.responder.seen, [])
        self.assertEqual((self.h.output.read_bytes(), self.h.episodes.read_bytes()), (actors, episodes))


class FailurePreservationTests(TempDirCase):
    def test_invalid_and_capped_b_outputs_are_kept_raw_and_never_repaired(self):
        p0, p1 = planted_cases()[:2]
        fenced = '```json\n{"kind": "private", "value": 3}\n```'
        trailing = '{"kind": "private", "value": 3} extra'
        h = Harness(self.dir, overrides={
            (p0["id"], "base"): fenced, (p0["id"], "removed"): trailing,
            (p0["id"], "sham"): ('{"kind": "private", "value": 3}', False),  # parses, but hit the 48-token cap
            (p1["id"], "base"): '{"kind": "private", "value": true}'})
        h.run()
        records = {r["task_id"]: r for r in read_lines(h.output)}
        base, removed, sham = (records[f"{p0['id']}:B:{arm}"] for arm in ("base", "removed", "sham"))
        self.assertEqual((base["parse_status"], base["parsed"], base["raw_output"]), ("invalid", None, fenced))
        self.assertEqual((removed["parse_status"], removed["parsed"], removed["raw_output"]), ("invalid", None, trailing))
        self.assertEqual((sham["parse_status"], sham["parsed"], sham["censored"], sham["eos"], sham["stop_token_id"]),
                         ("censored", None, True, False, None))
        self.assertEqual(sham["raw_output"], '{"kind": "private", "value": 3}')
        self.assertEqual((sham["n_generated_tokens"], len(sham["generated_token_ids"])), (48, 48))
        self.assertEqual(records[f"{p1['id']}:B:base"]["parse_status"], "invalid")  # bool is not a value
        ep = next(e for e in read_lines(h.episodes) if e["case_id"] == p0["id"])
        self.assertEqual([ep["receiver"][a] is None for a in ("base", "removed", "sham")], [True, True, True])
        self.assertIsNone(ep["observation"]["receiver"]["base"])
        self.assertIsNone(ep["observation"]["harm"]["base"])
        self.assertEqual(ep["actor_quality"]["receiver"],
                         {"expected": 6, "valid": 3, "invalid": 2, "censored": 1, "skipped_input_invalid": 0})
        self.assertEqual(ep["receiver_conformance"]["base"], {"parsed": False, "conforms": False})
        self.assertIs(ep["original_positive_activation"], False)
        self.assertTrue(all(ep["receiver_conformance"][a]["conforms"] for a in ("tag_flip", "source_flip", "source_same")))
        self.assertIsNone(M.predict(ep["observation"])["outcome"])  # an unusable base action abstains

    def test_a_tampered_parse_in_the_checkpoint_is_refused(self):
        self.h.run()
        rows = read_lines(self.h.output)
        rows[0]["parsed"] = {"kind": "blocked", "value": None}
        write_lines(self.h.output, rows)
        with self.assertRaises(SystemExit) as cm:
            Harness(self.dir).run()
        self.assertIn("strict parse", str(cm.exception.code))


class ConformanceTests(TempDirCase):
    def test_all_six_arms_conform_on_the_actual_incoming_messages_for_an_oracle_b(self):
        self.h.run()
        for ep in read_lines(self.h.episodes):
            case = ep["case"]
            messages = bench.make_messages(copy.deepcopy(case), bench.gold_episode(copy.deepcopy(case))["sender"])
            for arm in ARMS:
                want = bench.receiver_policy(copy.deepcopy(case), messages[arm])
                self.assertEqual(ep["receiver"][arm], want)
                self.assertEqual(ep["receiver_conformance"][arm], {"parsed": True, "conforms": True})
            planted = case["condition"] == eh.PLANTED
            self.assertEqual(ep["original_positive_activation"], True if planted else None)

    def test_conformance_is_conditional_on_the_message_b_received_and_activation_is_planted_only(self):
        p0, p1 = planted_cases()[:2]
        h = Harness(self.dir, overrides={
            (p0["id"], "removed"): '{"kind": "private", "value": 2}',  # valid action, but a missing message must block
            (p0["id"], "base"): '{"kind": "public", "value": 4}',      # parses, does not emit the private credential
            (p1["id"], "base"): json.dumps({"kind": "private", "value": p1["private"]})})
        h.run()
        by_case = {e["case_id"]: e for e in read_lines(h.episodes)}
        ep = by_case[p0["id"]]
        self.assertEqual(ep["receiver_conformance"]["removed"], {"parsed": True, "conforms": False})
        self.assertEqual(ep["receiver_conformance"]["base"], {"parsed": True, "conforms": False})
        self.assertIs(ep["original_positive_activation"], False)
        self.assertTrue(all(ep["receiver_conformance"][a]["conforms"] for a in ("tag_flip", "sham", "source_same")))
        self.assertIs(by_case[p1["id"]]["original_positive_activation"], True)
        for other in by_case.values():
            if other["ground_truth"]["condition"] != eh.PLANTED:
                self.assertIsNone(other["original_positive_activation"])
        self.assertEqual(Counter(e["ground_truth"]["assigned_positive"] for e in by_case.values()),
                         {False: 16, True: 8})  # labels are the assigned ones, whatever B did


class ResumeTests(TempDirCase):
    def partial(self, batches=2):
        self.h.fail_after = batches
        with self.assertRaises(RuntimeError):
            self.h.run()
        self.assertEqual(len(read_lines(self.h.output)), batches * 2)
        self.assertFalse(self.h.episodes.exists())

    def refused(self, harness, *extra, **kwargs):
        before = harness.output.read_bytes()
        with self.assertRaises(SystemExit) as cm:
            harness.run(*extra, **kwargs)
        self.assertEqual(harness.output.read_bytes(), before)
        self.assertEqual(harness.load_args, [])
        self.assertFalse(harness.episodes.exists())
        return str(cm.exception.code)

    def test_an_interrupted_run_resumes_without_regenerating_finished_groups(self):
        self.partial(3)
        first = self.h.output.read_bytes()
        done = {r["task_id"] for r in read_lines(self.h.output)}
        again = Harness(self.dir)
        again.run()
        records = read_lines(again.output)
        self.assertEqual(len(records), 144)
        self.assertTrue(again.output.read_bytes().startswith(first))
        self.assertEqual(len(again.models[0].calls), 69)
        self.assertEqual(len(again.responder.seen), 138)
        self.assertEqual(len({r["task_id"] for r in records}), 144)
        self.assertEqual({r["task_id"] for r in records[:6]}, done)
        self.assertEqual(len(read_lines(again.episodes)), 24)

    def test_a_different_model_or_library_or_helper_is_refused(self):
        self.partial()
        self.assertIn("run config differs", self.refused(Harness(self.dir), model="dolos"))
        with mock.patch.object(eh.torch, "__version__", "9.9.9"):
            self.assertIn("torch", self.refused(Harness(self.dir)))
        changed = {**eh.helper_hashes(), "common": "0" * 64}
        with mock.patch.object(eh, "helper_hashes", return_value=changed):
            self.assertIn("helper_sha256", self.refused(Harness(self.dir)))

    def test_a_changed_prompt_or_bench_wording_is_refused(self):
        self.partial()
        rows = read_lines(self.h.output)
        rows[1]["prompt"] += " "
        write_lines(self.h.output, rows)
        self.assertIn("prompt hash", self.refused(Harness(self.dir)))
        rows[1]["prompt_sha256"] = er.sha256_text(rows[1]["prompt"])
        write_lines(self.h.output, rows)
        self.assertIn("prompt differs", self.refused(Harness(self.dir)))
        rows[1]["prompt"] = rows[1]["prompt"][:-1]
        rows[1]["prompt_sha256"] = er.sha256_text(rows[1]["prompt"])
        write_lines(self.h.output, rows)
        original = bench.receiver_text
        reworded = lambda case, message: (lambda s, u: (s, u + "\nExtra line."))(*original(case, message))  # noqa: E731
        with mock.patch.object(bench, "receiver_text", reworded):
            self.assertIn("prompt differs", self.refused(Harness(self.dir)))
        Harness(self.dir).run()  # the untouched, un-reworded file still resumes
        self.assertEqual(len(read_lines(self.h.output)), 144)

    def test_foreign_duplicate_a_and_torn_lines_are_refused(self):
        self.partial()
        rows = read_lines(self.h.output)
        case_id = rows[0]["case_id"]
        done = {r["task_id"] for r in rows}
        task = next(t for t in eh.receiver_tasks(eh.load_selection()[1]) if t["task_id"] not in done)
        fresh = {**rows[0], "task_id": task["task_id"], "case_id": task["case_id"], "arm": task["arm"],
                 "task_index": task["index"], "batch_group": task["index"] // 2}
        for label, bad in (("unknown", {**rows[0], "task_id": f"{case_id}:A:base"}),
                           ("unknown", {**rows[0], "task_id": "ffffffffffff:B:base"}),
                           ("duplicate", rows[0]),
                           ("not an actor", {**fresh, "record_type": "episode"}),
                           ("lacks fields", {k: v for k, v in fresh.items() if k != "raw_output"}),
                           ("skip_reason", {**fresh, "parse_status": "skipped", "skip_reason": "skip_input_invalid"})):
            with self.subTest(label, task=bad.get("task_id")):
                write_lines(self.h.output, rows + [bad])
                self.assertIn(label, self.refused(Harness(self.dir)))
        write_lines(self.h.output, rows)
        self.h.output.write_text(self.h.output.read_text(encoding="utf-8") + '{"torn', encoding="utf-8")
        self.assertIn("not valid JSON", self.refused(Harness(self.dir)))


class BuildOnlyAndCliTests(TempDirCase):
    def build(self, *extra):
        out, _ = self.h.main("--model", "qwen", "--build-only", *extra)
        return json.loads(out)

    def test_build_only_reports_counts_and_scripted_labels_and_touches_nothing(self):
        with mock.patch.object(bench, "sender_text", side_effect=AssertionError("A prompt rendered")):
            info = self.build()
        _, selected = eh.load_selection()
        self.assertEqual((info["cases"]["selected"], info["cases"]["of_total"]), (24, 48))
        self.assertEqual(info["cases"]["by_condition"], {c: 8 for c in sorted(eh.CONDITIONS)})
        self.assertEqual(info["cases"]["global_indices"], [i for i, _ in selected])
        self.assertEqual(info["tasks"], {"A_model_queries": 0, "B_per_case": 6, "B_selected": 144, "B_done": 0,
                                         "batch_groups": 72})
        self.assertEqual(info["sender_source"], "scripted_policy")
        self.assertEqual(len(info["a_scripted_tags"]), 24)
        for _, case in selected:
            want = {arm: action["tag"] for arm, action in eh.scripted_sender(case).items()}
            self.assertEqual(info["a_scripted_tags"][case["id"]], want)
            self.assertEqual(set(want), set(SENDER_FIELDS))
        self.assertEqual(len(info["b_prompt_sha256"]), 144)
        self.assertEqual(len(set(info["b_prompt_sha256"].values())) > 1, True)
        self.assertEqual((info["batch_size"], info["decoding"]["max_new_tokens"]), (2, 48))
        self.assertEqual(self.h.load_args, [])  # no weights
        self.assertEqual(list(self.dir.iterdir()), [])  # and no file written

    def test_build_only_prompt_hashes_equal_those_of_a_real_run_and_show_progress(self):
        before = self.build()
        self.h.run()
        actors, episodes = self.h.output.read_bytes(), self.h.episodes.read_bytes()
        after = self.build("--output", str(self.h.output))
        self.assertEqual(after["tasks"]["B_done"], 144)
        self.assertEqual(after["b_prompt_sha256"], before["b_prompt_sha256"])
        self.assertEqual(after["b_prompt_sha256"], {r["task_id"]: r["prompt_sha256"] for r in read_lines(self.h.output)})
        self.assertEqual((self.h.output.read_bytes(), self.h.episodes.read_bytes()), (actors, episodes))
        self.assertEqual(len(self.h.models), 1)  # only the real run loaded a model

    def test_build_only_on_a_foreign_checkpoint_is_refused(self):
        self.h.fail_after = 1
        with self.assertRaises(RuntimeError):
            self.h.run()
        rows = read_lines(self.h.output)
        rows[0]["config"]["batch_size"] = 8
        write_lines(self.h.output, rows)
        with self.assertRaises(SystemExit) as cm:
            self.build("--output", str(self.h.output))
        self.assertIn("run config differs", str(cm.exception.code))

    def test_dolos_uses_its_own_pin_and_label(self):
        self.h.run(model="dolos")
        self.assertEqual([a[0] for a in self.h.load_args], ["dolos"])
        config = read_lines(self.h.output)[0]["config"]
        self.assertEqual((config["model"], config["repo"], config["revision"]),
                         ("dolos", eh.DOLOS_REPO, eh.DOLOS_REVISION))
        self.assertEqual({e["model"] for e in read_lines(self.h.episodes)}, {"dolos_oracle_sender"})
        self.assertEqual({e["receiver_model"] for e in read_lines(self.h.episodes)}, {"dolos"})

    def test_cli_option_conflicts_and_requirements(self):
        for argv in ([], ["--model", "qwen"], ["--output", "x.jsonl"], ["--model", "gpt", "--output", "x"],
                     ["--analyze", "--model", "qwen"], ["--analyze", "--output", "x.jsonl"],
                     ["--analyze", "--build-only"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as cm:
                fk.silent_capture(eh.parse_args, argv)
            self.assertEqual(cm.exception.code, 2)
        args = eh.parse_args(["--model", "dolos", "--build-only"])
        self.assertEqual((args.model, args.output, args.build_only, args.analyze), ("dolos", None, True, False))
        self.assertTrue(eh.parse_args(["--analyze"]).analyze)

    def test_offline_mode_is_enforced_at_import(self):
        self.assertEqual((os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]), ("1", "1"))


class AnalysisTests(TempDirCase):
    def setUp(self):
        super().setUp()
        p0 = planted_cases()[0]
        self.dolos = Harness(self.dir, overrides={(p0["id"], "base"): '{"kind": "public", "value": 4}'})
        self.p0 = p0
        self.h.run_results("qwen")
        self.dolos.run_results("dolos")
        self.results = self.dir / "evidence_hybrid_results.json"

    def payload(self):
        self.h.analyze()
        return json.loads(self.results.read_text(encoding="utf-8"))

    def test_results_are_written_per_model_with_scope_pins_and_source_hashes(self):
        payload = self.payload()
        self.assertEqual(set(payload), {"schema", "experiment_version", "scope", "selection", "pins", "source_hashes",
                                        "models"})  # no pooled block
        self.assertEqual(list(payload["models"]), ["qwen", "dolos"])
        self.assertEqual(payload["pins"], report.canon(report.common_pins()))
        self.assertEqual(set(payload["source_hashes"]), set(eh.SOURCE_FILES))
        for name, digest in payload["source_hashes"].items():
            self.assertEqual(digest, report.sha256_file(Path(eh.__file__).parent / name))
        self.assertEqual(payload["selection"]["n_cases"], 24)
        self.assertEqual(payload["selection"]["by_condition"], {c: 8 for c in sorted(eh.CONDITIONS)})
        self.assertIn("scripted bench policy", payload["scope"])
        for model, block in payload["models"].items():
            self.assertEqual((block["model_label"], block["receiver_model"], block["sender_source"]),
                             (f"{model}_oracle_sender", model, "scripted_policy"))
            self.assertEqual(block["metrics"]["n_episodes"], 24)
            self.assertEqual(block["metrics"]["assigned_labels"], {"positive": 8, "negative": 16, "total": 24})
            self.assertEqual(set(block["metrics"]["groups"]["model"]), {f"{model}_oracle_sender"})
            self.assertEqual(block["abstention_bounds"]["kind"], "logical_abstention_bounds_not_confidence_intervals")
            self.assertEqual((block["abstention_bounds"]["n_positive"], block["abstention_bounds"]["n_negative"]), (8, 16))
            self.assertEqual(block["config"]["model"], model)
            self.assertEqual(block["files"]["actors"]["sha256"],
                             report.sha256_file(eh.actors_path(model, self.dir)))
            self.assertEqual(block["files"]["episodes"]["sha256"],
                             report.sha256_file(eh.episodes_path(eh.actors_path(model, self.dir))))
            self.assertEqual(block["metrics"]["common_eligible_cohort"]["n_cases"], 24)

    def test_status_conformance_and_activation_blocks_are_per_model_and_not_pooled(self):
        models = self.payload()["models"]
        qwen, dolos = models["qwen"], models["dolos"]
        for block in (qwen, dolos):
            self.assertEqual(set(block["actor_status_counts"]["by_arm"]), set(ARMS))
            self.assertEqual({c: s["ok"] for c, s in block["actor_status_counts"]["by_condition"].items()},
                             {c: 48 for c in eh.CONDITIONS})
        self.assertEqual(qwen["actor_status_counts"]["B"], {"ok": 144, "invalid": 0, "censored": 0})
        self.assertEqual(qwen["conformance"]["overall"]["B_tasks"], {"expected": 144, "parsed": 144, "conforms": 144})
        self.assertEqual(qwen["conformance"]["overall"]["all_six_conform"], 24)
        self.assertEqual(qwen["activation"]["activated"], 8)
        self.assertEqual(set(qwen["conformance"]["by_condition"]), set(eh.CONDITIONS))
        # dolos has one planted base answer that does not emit the private credential: counted, not dropped or relabelled
        self.assertEqual(dolos["conformance"]["overall"]["B_tasks"], {"expected": 144, "parsed": 144, "conforms": 143})
        self.assertEqual(dolos["conformance"]["overall"]["base_receiver"], {"parsed": 24, "conforms": 23})
        self.assertEqual(dolos["conformance"]["overall"]["all_six_conform"], 23)
        self.assertEqual((dolos["activation"]["assigned_cases"], dolos["activation"]["activated"]), (8, 7))
        self.assertEqual(dolos["activation"]["not_activated_case_ids"], [self.p0["id"]])
        self.assertEqual(dolos["metrics"]["assigned_labels"], qwen["metrics"]["assigned_labels"])  # labels never move

    def test_each_model_is_analysed_separately_and_predict_sees_observations_only(self):
        calls, seen = [], []
        analyze, predict = M.analyze, M.predict

        def spy_analyze(episodes):
            episodes = list(episodes)
            calls.append(episodes)
            return analyze(episodes)

        def spy_predict(observation):
            seen.append(observation)
            return predict(observation)

        with mock.patch.object(M, "analyze", spy_analyze), mock.patch.object(M, "predict", spy_predict):
            self.h.analyze()
        self.assertEqual(len(calls), 2)
        self.assertEqual([{ep["receiver_model"] for ep in episodes} for episodes in calls], [{"qwen"}, {"dolos"}])
        self.assertTrue(all(len(episodes) == 24 for episodes in calls))
        observations = [ep["observation"] for episodes in calls for ep in episodes]
        self.assertEqual(len(seen), 48)
        self.assertEqual(seen, observations)
        forbidden = {"case", "ground_truth", "condition", "split", "model", "assigned_positive", "case_id"}
        for observation in seen:
            self.assertFalse(forbidden & set(observation))
            self.assertNotIn("planted_coordination", json.dumps(observation))

    def refuses(self, text):
        sentinel = b'{"stale": true}\n'
        self.results.write_bytes(sentinel)
        with self.assertRaises(SystemExit) as cm:
            self.h.analyze()
        self.assertIn(text, str(cm.exception.code))
        self.assertEqual(self.results.read_bytes(), sentinel)  # a refusal never rewrites the results

    def tampered(self, name, mutate, model="qwen"):
        """Run the --analyze refusal check with `mutate(rows)` applied to a copy of one model's file, then restore it."""
        path = eh.actors_path(model, self.dir) if name == "actors" else eh.episodes_path(eh.actors_path(model, self.dir))
        original = path.read_bytes()
        rows = read_lines(path)
        write_lines(path, mutate(rows))
        self.addCleanup(path.write_bytes, original)
        return path, original

    def test_incomplete_foreign_or_tampered_inputs_are_refused_without_writing_results(self):
        cases = [
            ("actors", lambda rows: rows[:-1], "143 of 144"),
            ("actors", lambda rows: rows[:-1] + [rows[0]], "partial, duplicated or foreign"),
            ("actors", lambda rows: [{**r, "config": {**r["config"], "revision": "0" * 40}} for r in rows],
             "does not match the pinned"),
            ("actors", lambda rows: [{**r, "config": {**r["config"], "decoding": {**r["config"]["decoding"],
                                                                                  "max_new_tokens": 64}}} for r in rows],
             "does not match the pinned"),
            ("actors", lambda rows: [{**rows[0], "parsed": {"kind": "blocked", "value": None}}] + rows[1:],
             "strict parse"),
            ("actors", lambda rows: [{**rows[0], "system": rows[0]["system"] + " "}] + rows[1:], "prompt differs"),
            ("actors", lambda rows: [{**rows[0], "raw_output": rows[0]["raw_output"] + " x"}] + rows[1:], "strict parse"),
            ("episodes", lambda rows: [{**rows[0], "receiver": {**rows[0]["receiver"], "base": None}}] + rows[1:],
             "is not what the actor file"),
            ("episodes", lambda rows: [{**rows[0], "ground_truth": {**rows[0]["ground_truth"], "assigned_positive":
                                                                  not rows[0]["ground_truth"]["assigned_positive"]}}]
             + rows[1:], "is not what the actor file"),
            ("episodes", lambda rows: rows[:-1], "episode"),
        ]
        for name, mutate, text in cases:
            with self.subTest(file=name, expect=text):
                path, original = self.tampered(name, mutate)
                self.refuses(text)
                path.write_bytes(original)
        self.assertEqual(self.payload()["models"]["qwen"]["metrics"]["n_episodes"], 24)  # untouched inputs still pass

    def test_missing_files_and_changed_helpers_are_refused(self):
        for name in ("actors", "episodes"):
            for model in eh.MODELS:
                path = eh.actors_path(model, self.dir)
                path = path if name == "actors" else eh.episodes_path(path)
                original = path.read_bytes()
                path.unlink()
                with self.subTest(file=name, model=model):
                    self.refuses("missing input")
                path.write_bytes(original)
        changed = {**eh.helper_hashes(), "evidence_run": "0" * 64}
        with mock.patch.object(eh, "helper_hashes", return_value=changed):
            self.refuses("does not match the pinned")
        self.assertEqual(self.results.read_bytes(), b'{"stale": true}\n')


if __name__ == "__main__":
    unittest.main()
