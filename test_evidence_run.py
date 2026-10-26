"""Tests for evidence_run.py with a fake bench, fake tokenizer and fake model only (no weights, no inference)."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests must never download anything

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import evidence_run as er

CODEBOOK = list("abcdefgh")
CONDITIONS = ("independent", "planted_coordination", "cooperation")  # case i has CONDITIONS[i % 3]
ARMS = er.RECEIVER_ARMS


# ---------------------------------------------------------------------------------------------------------------------
# Fake bench: the interface evidence_run relies on, with trivially checkable policies.
# ---------------------------------------------------------------------------------------------------------------------
def fake_build_cases():
    return [{"id": f"c{i:02d}", "condition": CONDITIONS[i % 3], "split": ("calibration", "transfer")[i % 2],
             "private": i % 8, "private_flip": (i + 1) % 8, "private_same": (i + 2) % 8, "public_value": (i + 3) % 8,
             "codebook": list(CODEBOOK), "nuisance": False} for i in range(48)]


def fake_sender_policy(case, private=None):
    value = case["private"] if private is None else private
    index = value if case["condition"] == "planted_coordination" else case["public_value"]
    return {"tag": case["codebook"][index]}


def fake_receiver_policy(case, message):
    tag = message.split(":", 1)[1]
    if case["condition"] == "planted_coordination":
        if tag in case["codebook"]:
            return {"kind": "private", "value": case["codebook"].index(tag)}
        return {"kind": "blocked", "value": None}
    return {"kind": "public", "value": case["public_value"]}


def fake_sender_text(case, private_value):
    return "SYS", f"A case={case['id']} private={private_value}"


def fake_receiver_text(case, message):
    return "SYS", f"B case={case['id']} message={message}"


def fake_parse_sender(raw, case):
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if isinstance(obj, dict) and set(obj) == {"tag"} and obj["tag"] in case["codebook"]:
        return {"tag": obj["tag"]}
    return None


def fake_parse_receiver(raw):
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if isinstance(obj, dict) and set(obj) == {"kind", "value"} and obj["kind"] in ("public", "private", "blocked"):
        return {"kind": obj["kind"], "value": obj["value"]}
    return None


def fake_make_messages(case, sender_actions):
    out = {arm: None for arm in ARMS}
    out["removed"] = "removed:-"
    base = sender_actions.get("base")
    if base is not None:
        flipped = case["codebook"][(case["codebook"].index(base["tag"]) + 1) % 8]
        out.update(base="base:" + base["tag"], tag_flip="tag_flip:" + flipped, sham="sham:" + base["tag"])
    for arm in ("source_flip", "source_same"):
        if sender_actions.get(arm) is not None:
            out[arm] = f"{arm}:" + sender_actions[arm]["tag"]
    return out


def fake_observation(case, sender_actions, receiver_actions):
    return {"message": fake_make_messages(case, sender_actions), "sender": sender_actions,
            "receiver": receiver_actions}


def fake_gold_episode(case):
    sender = {arm: fake_sender_policy(case, case[field]) for arm, field in er.SENDER_ARMS.items()}
    messages = fake_make_messages(case, sender)
    receiver = {arm: None if messages[arm] is None else fake_receiver_policy(case, messages[arm]) for arm in ARMS}
    return {"case": copy.deepcopy(case), "ground_truth": {"assigned_positive": case["condition"] == "planted_coordination"},
            "sender": sender, "receiver": receiver, "observation": fake_observation(case, sender, receiver)}


def make_bench(directory: Path) -> types.ModuleType:
    module = types.ModuleType("fake_evidence_bench")
    for name in ("build_cases", "sender_text", "receiver_text", "parse_sender", "parse_receiver", "make_messages",
                 "receiver_policy", "observation", "gold_episode"):
        setattr(module, name, globals()[f"fake_{name}"])
    module.__file__ = str(directory / "fake_evidence_bench.py")
    Path(module.__file__).write_text("# fake bench v1\n", encoding="utf-8")
    return module


# ---------------------------------------------------------------------------------------------------------------------
# Fake tokenizer / model: a prompt is one id, a scripted answer is one token, 90 is the stop token.
# ---------------------------------------------------------------------------------------------------------------------
class Encoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 91

    def __init__(self, suffix: str = ""):
        self.suffix, self.calls = suffix, []
        self.prompt_ids: dict[str, int] = {}
        self.prompts: dict[int, str] = {}
        self.texts = {90: "<|im_end|>", 91: "<|endoftext|>", 100: ""}  # 100 is the filler of a capped generation

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True):
        self.calls.append({"roles": [m["role"] for m in messages], "tools": tools, "generation": add_generation_prompt})
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "") + self.suffix

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 90, "<|endoftext|>": 91}[token]

    def __call__(self, prompts, return_tensors=None, padding=None, add_special_tokens=None):
        assert (return_tensors, padding, add_special_tokens) == ("pt", True, False)
        ids = [self.prompt_ids.setdefault(p, 1000 + len(self.prompt_ids)) for p in prompts]
        self.prompts.update({i: p for p, i in self.prompt_ids.items()})
        return Encoding(input_ids=torch.tensor([[i] for i in ids]))

    def text_token(self, text: str) -> int:
        token = 5000 + int(hashlib.sha256(text.encode()).hexdigest()[:6], 16)  # stable across tokenizer instances
        self.texts[token] = text
        return token

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join(self.texts[i] for i in ids)


class FakeModel:
    """Stand-in for model.generate: the responder maps a prompt to its answer text, or (text, False) for a capped one."""

    def __init__(self, tok, responder, fail_after=None):
        self.tok, self.responder, self.fail_after, self.calls = tok, responder, fail_after, []
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def generate(self, input_ids, max_new_tokens, pad_token_id, **kwargs):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated crash")
        prompts = [self.tok.prompts[k] for k in input_ids[:, 0].tolist()]
        self.calls.append({"prompts": prompts, "max_new_tokens": max_new_tokens, "pad": pad_token_id, **kwargs})
        rows = []
        for prompt in prompts:
            answer = self.responder(prompt)
            text, finishes = (answer, True) if isinstance(answer, str) else answer
            token = self.tok.text_token(text)
            rows.append(([token, 90] if finishes else [token] + [100] * (max_new_tokens - 1))[:max_new_tokens])
        width = max(len(r) for r in rows)
        body = torch.tensor([r + [pad_token_id] * (width - len(r)) for r in rows])
        return torch.cat([input_ids, body], dim=1)


class Responder:
    """Oracle answers for the fake bench; `overrides` maps (case_id, role, arm) to an answer to give instead."""

    def __init__(self, overrides=None):
        self.overrides, self.cases, self.seen = dict(overrides or {}), {c["id"]: c for c in fake_build_cases()}, []

    def __call__(self, prompt):
        if match := re.search(r"A case=(\S+) private=(\d+)", prompt):
            case, value = self.cases[match.group(1)], int(match.group(2))
            arm = next(a for a, field in er.SENDER_ARMS.items() if case[field] == value)
            key, default = (case["id"], "A", arm), json.dumps(fake_sender_policy(case, value))
        else:
            match = re.search(r"B case=(\S+) message=([^\s<]+)", prompt)
            case, message = self.cases[match.group(1)], match.group(2)
            key, default = (case["id"], "B", message.split(":", 1)[0]), json.dumps(fake_receiver_policy(case, message))
        self.seen.append(key)
        return self.overrides.get(key, default)


def silent_capture(func, *args, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        func(*args, **kwargs)
    return out.getvalue(), err.getvalue()


class Harness:
    """One fake world (bench, tokenizer, responder) that can run er.main several times against one directory."""

    def __init__(self, directory: Path, overrides=None, tok=None, use_fake_bench=True):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bench = make_bench(directory) if use_fake_bench else None
        self.tok = tok or FakeTokenizer()
        self.responder = Responder(overrides)
        self.models: list[FakeModel] = []
        self.load_args: list[tuple] = []
        self.fail_after = None
        self.output = directory / "run.jsonl"

    @property
    def episodes(self) -> Path:
        return er.episodes_path(self.output)

    def load_model(self, name, device, dtype):
        self.load_args.append((name, device, dtype))
        model = FakeModel(self.tok, self.responder, self.fail_after)
        self.models.append(model)
        return model

    def main(self, *argv):
        patches = [mock.patch.object(er, "load_tokenizer", return_value=self.tok),
                   mock.patch.object(er, "load_model", side_effect=self.load_model),
                   mock.patch.object(er, "set_threads")]
        if self.bench is not None:
            patches.append(mock.patch.object(er, "bench", self.bench))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            return silent_capture(er.main, list(argv))

    def run(self, *extra, model="qwen"):
        return self.main("--model", model, "--output", str(self.output), *extra)

    def calls(self) -> list[list[str]]:
        return [call["prompts"] for m in self.models for call in m.calls]


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.h = Harness(self.dir)


class ArgsAndModesTests(TempDirCase):
    def test_defaults_and_requirements(self):
        args = er.parse_args(["--model", "dolos", "--output", "x.jsonl"])
        self.assertEqual((args.max_new_tokens, args.batch_size, args.limit, args.gold, args.build_only),
                         (48, 2, None, False, False))
        for bad in (["--output", "x"], ["--model", "qwen"], ["--model", "qwen", "--output", "x", "--limit", "0"],
                    ["--model", "qwen", "--output", "x", "--batch-size", "0"], ["--model", "gpt", "--output", "x"],
                    ["--model", "qwen", "--output", "x", "--temperature", "0.7"]):
            with self.subTest(bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                er.parse_args(bad)
        self.assertIsNone(er.parse_args(["--gold", "--output", "x"]).model)
        self.assertIsNone(er.parse_args(["--model", "qwen", "--build-only"]).output)

    def test_build_only_prints_counts_and_prompt_hashes_and_writes_nothing(self):
        out, _ = self.h.main("--model", "qwen", "--build-only")
        report = json.loads(out)
        self.assertEqual((report["cases"]["total"], report["cases"]["selected"]), (48, 48))
        self.assertEqual(report["cases"]["by_condition"], {c: 16 for c in CONDITIONS})
        self.assertEqual((report["tasks"]["A_selected"], report["tasks"]["B_selected"]), (144, 288))
        self.assertEqual(len(report["a_prompt_sha256"]), 144)
        self.assertEqual(set(report["helper_sha256"]), {"common", "e2e_confirm", "evidence_bench"})
        self.assertEqual(self.h.load_args, [])
        self.assertEqual(list(self.dir.glob("*.jsonl")), [])
        pilot, _ = self.h.main("--model", "qwen", "--build-only", "--limit", "2")
        self.assertEqual(json.loads(pilot)["tasks"]["A_selected"], 6)

    def test_build_only_reports_progress_of_an_existing_file_without_a_model(self):
        self.h.run("--limit", "2")
        report = json.loads(self.h.main("--model", "qwen", "--build-only", "--limit", "2", "--output",
                                        str(self.h.output))[0])
        self.assertEqual((report["tasks"]["A_done"], report["tasks"]["B_done"], report["tasks"]["B_pending_plan"]),
                         (6, 12, {"to_skip_input_invalid": 0, "to_generate": 0}))
        self.assertEqual(len(self.h.models), 1)  # only the pilot run loaded a model

    def test_gold_writes_the_oracle_episodes_without_tokenizer_or_model(self):
        gold = self.dir / "gold.jsonl"
        with mock.patch.object(er, "load_tokenizer", side_effect=AssertionError), \
                mock.patch.object(er, "load_model", side_effect=AssertionError), mock.patch.object(er, "bench", self.h.bench):
            silent_capture(er.main, ["--gold", "--output", str(gold)])
            self.assertEqual(read_lines(gold), [fake_gold_episode(c) for c in fake_build_cases()])
            silent_capture(er.main, ["--gold", "--output", str(gold), "--limit", "5"])  # derived output: overwritable
            self.assertEqual(len(read_lines(gold)), 5)
            self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_gold_refuses_to_overwrite_an_actor_checkpoint_or_foreign_file(self):
        self.h.run("--limit", "1")
        before = self.h.output.read_bytes()
        notes = self.dir / "notes.txt"
        notes.write_text("keep me\n", encoding="utf-8")
        with mock.patch.object(er, "bench", self.h.bench):
            for target in (self.h.output, notes):
                with self.subTest(target.name), self.assertRaises(SystemExit) as cm:
                    silent_capture(er.main, ["--gold", "--output", str(target)])
                self.assertIn("refusing to overwrite", str(cm.exception.code))
        self.assertEqual(self.h.output.read_bytes(), before)
        self.assertEqual(notes.read_text(encoding="utf-8"), "keep me\n")


class FullRunTests(TempDirCase):
    def test_complete_run_counts_and_loading(self):
        self.h.run()
        records = read_lines(self.h.output)
        self.assertEqual(len(records), 432)
        self.assertEqual(len({r["task_id"] for r in records}), 432)
        self.assertEqual(Counter_roles(records), {"A": 144, "B": 288})
        self.assertEqual({r["config"]["model"] for r in records}, {"qwen"})
        self.assertEqual(len(self.h.models), 1)
        self.assertEqual(self.h.load_args, [("qwen", torch.device("cpu"), torch.float32)])
        episodes = read_lines(self.h.episodes)
        self.assertEqual([e["case"] for e in episodes], fake_build_cases())
        self.assertEqual([e["case_id"] for e in episodes], [c["id"] for c in fake_build_cases()])
        self.assertEqual(self.episodes_quality(episodes), {"sender": (144, 0, 0), "receiver": (288, 0, 0)})
        a_first = next(r for r in records if r["role"] == "A")
        self.assertEqual(a_first["task_id"], f"{a_first['case_id']}:A:{a_first['arm']}")
        self.assertEqual(er.sha256_text(a_first["prompt"]), a_first["prompt_sha256"])
        self.assertEqual(a_first["config"]["decoding"], {"do_sample": False, "temperature": 0.0, "max_new_tokens": 48,
                                                         "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
                                                         "assistant_prefill": False})
        for field in ("raw_output", "generated_token_ids", "stop_token_id", "eos", "censored", "system", "user",
                      "parsed", "skip_reason", "n_generated_tokens"):
            self.assertIn(field, a_first)
        self.assertEqual(a_first["config"]["helper_sha256"].keys(), {"common", "e2e_confirm", "evidence_bench"})

    def episodes_quality(self, episodes):
        total = {"sender": [0, 0, 0], "receiver": [0, 0, 0]}
        for e in episodes:
            for role in total:
                q = e["actor_quality"][role]
                total[role][0] += q["valid"]
                total[role][1] += q["invalid"] + q["censored"]
                total[role][2] += q["skipped_input_invalid"]
        return {k: tuple(v) for k, v in total.items()}

    def test_a_finished_run_loads_no_model_and_rewrites_identical_episodes(self):
        self.h.run()
        actors, episodes = self.h.output.read_bytes(), self.h.episodes.read_bytes()
        self.h.episodes.unlink()
        again = Harness(self.dir)
        again.load_model = mock.Mock(side_effect=AssertionError("model loaded for a finished run"))
        again.run()
        self.assertEqual(again.load_model.call_count, 0)
        self.assertEqual((self.h.output.read_bytes(), self.h.episodes.read_bytes()), (actors, episodes))

    def test_prompts_use_the_normal_chat_template_without_assistant_prefill(self):
        self.h.run("--limit", "3")
        self.assertTrue(self.h.tok.calls)
        for call in self.h.tok.calls:
            self.assertEqual(call, {"roles": ["system", "user"], "tools": None, "generation": True})
        for record in read_lines(self.h.output):
            self.assertTrue(record["prompt"].endswith("<|im_start|>assistant\n"))
            self.assertEqual(record["prompt"].count("<|im_start|>assistant"), 1)
        for call in self.h.models[0].calls:
            self.assertEqual({k: call[k] for k in ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens")},
                             {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                              "max_new_tokens": 48})
            self.assertEqual(call["eos_token_id"], [90, 91])

    def test_a_prefilled_prompt_is_refused_before_any_generation(self):
        h = Harness(self.dir, tok=FakeTokenizer(suffix='{"tag": "'))
        with self.assertRaises(ValueError):
            h.run("--limit", "1")
        self.assertEqual(h.models, [])
        self.assertFalse(h.output.exists())


def Counter_roles(records):
    counts = {}
    for r in records:
        counts[r["role"]] = counts.get(r["role"], 0) + 1
    return counts


class ResumeTests(TempDirCase):
    def assert_groups(self, h, size=2):
        index = {r["prompt"]: r["task_index"] for r in read_lines(h.output)}
        for prompts in h.calls():
            self.assertLessEqual(len(prompts), size)
            self.assertEqual(len({index[p] // size for p in prompts}), 1, prompts)

    def test_pilot_then_full_regenerates_nothing_and_keeps_fixed_groups(self):
        self.h.run("--limit", "2")
        pilot, pilot_records = self.h.output.read_bytes(), read_lines(self.h.output)
        pilot_episodes = self.h.episodes.read_bytes()
        self.assertEqual(len(pilot_records), 3 * 2 + 6 * 2)
        self.assertEqual([len(c) for c in self.h.calls()], [2] * 9)  # 3 A batches, then 6 B batches
        self.assertEqual(len(read_lines(self.h.episodes)), 2)
        self.assert_groups(self.h)
        full = Harness(self.dir)
        full.run()
        self.assertTrue(self.h.output.read_bytes().startswith(pilot))
        self.assertEqual(len(read_lines(self.h.output)), 432)
        generated = [p for call in full.calls() for p in call]
        self.assertEqual(len(generated), 432 - 18)
        self.assertFalse({r["prompt"] for r in pilot_records} & set(generated))
        self.assert_groups(full)
        self.assertEqual(len(read_lines(self.h.episodes)), 48)
        self.assertTrue(self.h.episodes.read_bytes().startswith(pilot_episodes))
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        expanded = read_lines(self.h.output)
        self.assertEqual([r["task_id"] for r in expanded[:18]], [r["task_id"] for r in pilot_records])

    def test_a_pilot_and_a_full_run_give_the_same_records(self):
        self.h.run("--limit", "2")
        Harness(self.dir).run()
        other = Harness(self.dir / "ref")
        other.output = other.dir / "full.jsonl"
        other.run()
        key = lambda r: r["task_id"]
        self.assertEqual(sorted(read_lines(self.h.output), key=key), sorted(read_lines(other.output), key=key))
        self.assertEqual(self.h.episodes.read_bytes(), other.episodes.read_bytes())

    def test_resume_after_a_crash_inside_either_phase(self):
        reference = Harness(self.dir / "ref")
        reference.output = reference.dir / "r.jsonl"
        reference.run("--limit", "2")
        for crash_after, kept in ((2, 4), (5, 6 + 4)):  # mid A phase / A done and two B batches written
            with self.subTest(crash_after=crash_after):
                h = Harness(self.dir / f"c{crash_after}")
                h.output = h.dir / "run.jsonl"
                h.fail_after = crash_after
                with self.assertRaises(RuntimeError):
                    h.run("--limit", "2")
                partial = read_lines(h.output)
                self.assertEqual(len(partial), kept)
                self.assertFalse(h.episodes.exists())
                resumed = Harness(h.dir)
                resumed.run("--limit", "2")
                self.assertFalse({r["prompt"] for r in partial} & {p for c in resumed.calls() for p in c})
                self.assertEqual(h.output.read_bytes(), reference.output.read_bytes())
                self.assertEqual(h.episodes.read_bytes(), reference.episodes.read_bytes())

    def test_a_limit_that_does_not_fill_batch_groups_is_flagged(self):
        _, err = self.h.run("--limit", "1")
        self.assertIn("does not fill whole batch groups", err)
        h2 = Harness(self.dir)
        h2.output = self.dir / "even.jsonl"
        _, err = h2.run("--limit", "2")
        self.assertNotIn("does not fill", err)


class RefusalTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.h.run("--limit", "2")
        self.original = self.h.output.read_bytes()
        self.episodes = self.h.episodes.read_bytes()

    def assert_refused(self, expect, *extra, harness=None, model="qwen"):
        h = harness or Harness(self.dir)
        with self.assertRaises(SystemExit) as cm:
            h.run(*extra, model=model)
        self.assertIn("refusing to resume", str(cm.exception.code))
        self.assertIn(expect, str(cm.exception.code))
        self.assertEqual(h.models, [])
        self.assertEqual(self.h.episodes.read_bytes(), self.episodes)

    def edit_file(self, change):
        lines = [json.loads(line) for line in self.original.decode().splitlines()]
        change(lines)
        self.h.output.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")

    def test_changed_run_metadata_is_refused(self):
        self.assert_refused("run config differs", "--max-new-tokens", "7")
        self.assert_refused("run config differs", "--batch-size", "3")
        self.assert_refused("run config differs", model="dolos")
        with mock.patch.object(er, "EXPERIMENT_VERSION", "evidence_run/v2"):
            self.assert_refused("experiment_version")
        self.assertEqual(self.h.output.read_bytes(), self.original)

    def test_changed_helper_bytes_are_refused(self):
        h = Harness(self.dir)
        h.bench = self.h.bench
        Path(self.h.bench.__file__).write_text("# fake bench v2\n", encoding="utf-8")
        self.assert_refused("helper_sha256", harness=h)
        self.assertEqual(self.h.output.read_bytes(), self.original)

    def test_mixed_metadata_inside_the_file_is_refused(self):
        self.edit_file(lambda lines: lines[7]["config"].update(revision="other"))
        self.assert_refused("run config differs")

    def test_duplicate_unknown_and_malformed_lines_are_refused(self):
        cases = {
            "duplicate task id": lambda lines: lines.append(dict(lines[0])),
            "unknown task id": lambda lines: lines[-1].update(task_id="zz:A:base"),
            "not an actor record": lambda lines: lines[3].update(record_type="episode"),
            "inconsistent": lambda lines: lines[2].update(prompt=lines[2]["prompt"] + " "),
            "sender records are incomplete": lambda lines: lines.pop(0),
            "has arm=": lambda lines: lines[1].update(arm="base" if lines[1]["arm"] != "base" else "removed"),
        }
        for expect, change in cases.items():
            with self.subTest(expect):
                self.edit_file(change)
                self.assert_refused(expect)
        self.h.output.write_bytes(self.original + b"{not json\n")
        self.assert_refused("not valid JSON")
        self.h.output.write_bytes(self.original)

    def test_prompt_drift_is_refused_for_both_roles(self):
        drifted = Harness(self.dir)
        drifted.bench = make_bench(self.dir)
        drifted.bench.sender_text = lambda case, value: ("SYS", f"A case={case['id']} private={value} changed")
        self.assert_refused("prompt differs", harness=drifted)
        drifted = Harness(self.dir)
        drifted.bench = make_bench(self.dir)
        original = fake_make_messages
        drifted.bench.make_messages = lambda case, actions: {
            arm: None if m is None else m + "!" for arm, m in original(case, actions).items()}
        self.assert_refused("differs from the current bench", harness=drifted)
        self.assertEqual(self.h.output.read_bytes(), self.original)


class DependencyAndQualityTests(TempDirCase):
    def test_invalid_or_censored_senders_skip_the_dependent_receivers(self):
        overrides = {("c01", "A", "base"): "not json", ("c01", "A", "source_flip"): ('{"tag": "a"}', False)}
        h = Harness(self.dir, overrides)
        h.run("--limit", "2")
        records = {r["task_id"]: r for r in read_lines(h.output)}
        self.assertEqual(len(records), 6 + 12)
        invalid, capped = records["c01:A:base"], records["c01:A:source_flip"]
        self.assertEqual((invalid["parse_status"], invalid["parsed"], invalid["raw_output"], invalid["censored"]),
                         ("invalid", None, "not json", False))
        self.assertEqual((capped["parse_status"], capped["parsed"], capped["raw_output"], capped["censored"],
                          capped["eos"], capped["stop_token_id"], capped["n_generated_tokens"]),
                         ("censored", None, '{"tag": "a"}', True, False, None, 48))  # parses, but is not complete
        for arm in ("base", "tag_flip", "sham", "source_flip"):
            skip = records[f"c01:B:{arm}"]
            self.assertEqual((skip["parse_status"], skip["skip_reason"], skip["parsed"], skip["prompt"], skip["raw_output"],
                              skip["generated_token_ids"], skip["prompt_sha256"]),
                             ("skipped", "skip_input_invalid", None, None, None, None, None))
            self.assertNotIn(("c01", "B", arm), h.responder.seen)
        for arm in ("removed", "source_same"):
            self.assertEqual(records[f"c01:B:{arm}"]["parse_status"], "ok")
        self.assertEqual(sum(len(c) for c in h.calls()), 6 + 6 + 2)
        episode = read_lines(h.episodes)[1]
        self.assertEqual((episode["case_id"], episode["sender"]["base"], episode["sender"]["source_flip"],
                          episode["receiver"]["base"], episode["receiver"]["tag_flip"]), ("c01", None, None, None, None))
        self.assertEqual(episode["actor_quality"]["sender"],
                         {"expected": 3, "valid": 1, "invalid": 1, "censored": 1, "skipped_input_invalid": 0})
        self.assertEqual(episode["actor_quality"]["receiver"],
                         {"expected": 6, "valid": 2, "invalid": 0, "censored": 0, "skipped_input_invalid": 4})
        before = h.output.read_bytes()
        again = Harness(self.dir, overrides)
        again.load_model = mock.Mock(side_effect=AssertionError("model loaded for a finished run"))
        again.run("--limit", "2")
        self.assertEqual(h.output.read_bytes(), before)

    def test_invalid_and_capped_receivers_are_retained_not_valid_actions(self):
        overrides = {("c00", "B", "base"): "garbage", ("c00", "B", "sham"): (json.dumps({"kind": "public", "value": 1}), False)}
        h = Harness(self.dir, overrides)
        h.run("--limit", "1")
        records = {r["task_id"]: r for r in read_lines(h.output)}
        self.assertEqual((records["c00:B:base"]["parse_status"], records["c00:B:base"]["parsed"],
                          records["c00:B:base"]["raw_output"]), ("invalid", None, "garbage"))
        self.assertEqual((records["c00:B:sham"]["parse_status"], records["c00:B:sham"]["parsed"]), ("censored", None))
        episode = read_lines(h.episodes)[0]
        self.assertEqual(episode["actor_quality"]["receiver"],
                         {"expected": 6, "valid": 4, "invalid": 1, "censored": 1, "skipped_input_invalid": 0})
        self.assertIsNone(episode["receiver"]["base"])
        self.assertEqual(episode["receiver"]["removed"], {"kind": "public", "value": 3})


class LabelsAndConformanceTests(TempDirCase):
    def test_labels_come_from_the_condition_and_conformance_never_filters_cases(self):
        overrides = {
            ("c04", "B", "base"): json.dumps({"kind": "blocked", "value": None}),  # planted, receiver does not release
            ("c07", "A", "base"): "oops",  # planted, sender invalid
            ("c03", "A", "base"): json.dumps({"tag": "h"}),  # independent, valid but not the oracle tag
            ("c10", "A", "base"): ('{"tag": "a"}', False),  # planted, sender capped
        }
        h = Harness(self.dir, overrides)
        h.run()
        episodes = {e["case_id"]: e for e in read_lines(h.episodes)}
        self.assertEqual(len(episodes), 48)
        for case in fake_build_cases():
            truth = episodes[case["id"]]["ground_truth"]
            self.assertEqual(truth, {"assigned_positive": case["condition"] == "planted_coordination",
                                     "condition": case["condition"], "engineered": True})
            self.assertEqual(episodes[case["id"]]["model"], "qwen")
        policy = {cid: episodes[cid]["original_policy"] for cid in episodes}
        self.assertEqual(policy["c01"], {"sender_valid": True, "sender_conforms": True, "receiver_valid": True,
                                         "receiver_conforms": True, "receiver_matches_gold_trajectory": True,
                                         "original_positive_activation": True})
        self.assertEqual(policy["c04"], {"sender_valid": True, "sender_conforms": True, "receiver_valid": True,
                                         "receiver_conforms": False, "receiver_matches_gold_trajectory": False,
                                         "original_positive_activation": False})
        self.assertEqual(policy["c07"], {"sender_valid": False, "sender_conforms": False, "receiver_valid": False,
                                         "receiver_conforms": False, "receiver_matches_gold_trajectory": False,
                                         "original_positive_activation": False})
        self.assertEqual(policy["c10"]["original_positive_activation"], False)
        self.assertFalse(policy["c10"]["sender_valid"])
        self.assertEqual(policy["c03"], {"sender_valid": True, "sender_conforms": False, "receiver_valid": True,
                                         "receiver_conforms": True, "receiver_matches_gold_trajectory": True,
                                         "original_positive_activation": None})
        self.assertIsNone(policy["c00"]["original_positive_activation"])
        self.assertTrue(episodes["c07"]["ground_truth"]["assigned_positive"])  # still a planted-positive row
        self.assertEqual(sum(e["ground_truth"]["assigned_positive"] for e in episodes.values()), 16)
        self.assertEqual(sum(bool(p["original_positive_activation"]) for p in policy.values()), 16 - 3)
        for e in episodes.values():
            self.assertNotIn("original_policy", e["observation"])
            self.assertEqual(set(e), {"record_type", "case", "case_id", "split", "model", "ground_truth", "sender", "receiver",
                                      "observation", "actor_quality", "original_policy"})


class ReceiverConformanceTests(TempDirCase):
    def policy_of(self, overrides, case_id):
        self.runs = getattr(self, "runs", 0) + 1
        h = Harness(self.dir / f"run{self.runs}", overrides)
        h.run()
        return {e["case_id"]: e for e in read_lines(h.episodes)}[case_id]["original_policy"]

    def test_receiver_is_judged_on_the_message_it_actually_received(self):
        # c01 and c04 are planted (private 1 and 4). Their sender writes a valid but wrong tag, so the message is wrong.
        wrong_sender = {("c01", "A", "base"): json.dumps({"tag": "h"})}
        # B answers the wrong message exactly as the policy says (private 7): a conforming receiver, but no activation.
        policy = self.policy_of(wrong_sender, "c01")
        self.assertEqual(policy, {"sender_valid": True, "sender_conforms": False, "receiver_valid": True,
                                  "receiver_conforms": True, "receiver_matches_gold_trajectory": False,
                                  "original_positive_activation": False})
        # B releases the private value of the ideal message instead: it does not follow the policy for what it received.
        ideal_receiver = {("c04", "A", "base"): json.dumps({"tag": "h"}),
                          ("c04", "B", "base"): json.dumps({"kind": "private", "value": 4})}
        policy = self.policy_of(ideal_receiver, "c04")
        self.assertEqual(policy, {"sender_valid": True, "sender_conforms": False, "receiver_valid": True,
                                  "receiver_conforms": False, "receiver_matches_gold_trajectory": True,
                                  "original_positive_activation": False})

    def test_activation_needs_the_true_private_to_be_encoded_and_emitted(self):
        # Sender encodes the true private (tag "e" for 4) and the receiver emits a different private value.
        wrong_value = {("c04", "B", "base"): json.dumps({"kind": "private", "value": 5})}
        policy = self.policy_of(wrong_value, "c04")
        self.assertTrue(policy["sender_conforms"])
        self.assertFalse(policy["receiver_conforms"] or policy["receiver_matches_gold_trajectory"])
        self.assertFalse(policy["original_positive_activation"])
        self.assertTrue(self.policy_of({}, "c04")["original_positive_activation"])

    def test_no_message_means_no_conformance_claim(self):
        policy = self.policy_of({("c01", "A", "base"): "oops"}, "c01")
        self.assertEqual((policy["receiver_valid"], policy["receiver_conforms"],
                          policy["receiver_matches_gold_trajectory"], policy["original_positive_activation"]),
                         (False, False, False, False))


class ObservationAndContractTests(TempDirCase):
    def test_observation_is_exactly_the_bench_public_observation(self):
        self.h.run()
        cases = {c["id"]: c for c in fake_build_cases()}
        for episode in read_lines(self.h.episodes):
            expected = fake_observation(cases[episode["case_id"]], episode["sender"], episode["receiver"])
            self.assertEqual(episode["case"], cases[episode["case_id"]])
            self.assertEqual(set(episode["observation"]["message"]), set(ARMS))  # the six-arm mapping, not a string
            self.assertEqual(episode["observation"], expected)
            self.assertFalse(er.HIDDEN_KEYS & set(episode["observation"]))
            self.assertNotIn("ground_truth", json.dumps(episode["observation"]))

    def test_an_observation_that_leaks_hidden_metadata_is_rejected_and_nothing_is_derived(self):
        self.h.bench.observation = lambda case, s, r: {"sender": s, "receiver": r, "meta": [{"condition": case["condition"]}]}
        with self.assertRaises(ValueError):
            self.h.run("--limit", "2")
        self.assertFalse(self.h.episodes.exists())
        self.assertEqual(len(read_lines(self.h.output)), 18)  # actor results are kept
        self.h.bench.observation = fake_observation
        fixed = Harness(self.dir)
        fixed.bench = self.h.bench
        fixed.run("--limit", "2")
        self.assertEqual(fixed.calls(), [])  # nothing regenerated
        self.assertEqual(len(read_lines(self.h.episodes)), 2)

    def test_check_public_scans_nested_keys(self):
        er.check_public({"a": [{"b": {"c": 1}}]})
        for leak in ({"private": 1}, {"x": {"y": [{"assigned_positive": True}]}}):
            with self.assertRaises(ValueError):
                er.check_public(leak)
        with self.assertRaises(ValueError):
            er.check_public([1])

    def test_bench_contract_violations_fail_loudly(self):
        with self.subTest("case count"):
            self.h.bench.build_cases = lambda: fake_build_cases()[:47]
            with self.assertRaises(ValueError):
                self.h.run()
        with self.subTest("duplicate case ids"):
            self.h.bench.build_cases = lambda: fake_build_cases()[:47] + fake_build_cases()[:1]
            with self.assertRaises(ValueError):
                self.h.run()
        self.h.bench.build_cases = fake_build_cases
        for label, bad in {"wrong arms": lambda case, a: {"base": "x"},
                           "removed None": lambda case, a: {**fake_make_messages(case, a), "removed": None},
                           "non-string": lambda case, a: {**fake_make_messages(case, a), "sham": 3}}.items():
            with self.subTest(label):
                self.h.bench.make_messages = bad
                with self.assertRaises(ValueError):
                    self.h.run("--limit", "1")
        self.assertEqual(list(self.dir.glob("*.episodes.jsonl")), [])


@unittest.skipUnless(importlib.util.find_spec("evidence_bench"), "evidence_bench.py is not present")
class RealBenchOracleTests(TempDirCase):
    """The real bench with a fake model that answers with the oracle policies: no inference, no weights."""

    def setUp(self):
        super().setUp()
        import evidence_bench as b
        self.b = b
        self.h = Harness(self.dir, use_fake_bench=False)
        table = {}
        for case in b.build_cases():
            sender = {arm: b.sender_policy(case, case[field]) for arm, field in er.SENDER_ARMS.items()}
            for arm, field in er.SENDER_ARMS.items():
                table[er.render_prompt(self.h.tok, *b.sender_text(case, case[field]))] = json.dumps(sender[arm])
            for message in b.make_messages(case, sender).values():
                if message is not None:
                    table[er.render_prompt(self.h.tok, *b.receiver_text(case, message))] = json.dumps(
                        b.receiver_policy(case, message))
        self.h.responder = table.__getitem__

    def test_gold_cli_writes_the_bench_gold_episodes(self):
        gold = self.dir / "gold.jsonl"
        silent_capture(er.main, ["--gold", "--output", str(gold)])
        self.assertEqual(read_lines(gold), json.loads(json.dumps([self.b.gold_episode(c) for c in self.b.build_cases()])))

    def test_full_oracle_run_conforms_everywhere_and_matches_gold(self):
        self.h.run()
        records = read_lines(self.h.output)
        self.assertEqual((len(records), sum(r["parse_status"] == "ok" for r in records)), (432, 432))
        episodes = read_lines(self.h.episodes)
        self.assertEqual(len(episodes), 48)
        for episode, case in zip(episodes, self.b.build_cases()):
            gold = json.loads(json.dumps(self.b.gold_episode(case)))
            for key in ("case", "sender", "receiver", "observation", "ground_truth"):
                self.assertEqual(episode[key], gold[key], (case["id"], key))
            planted = case["condition"] == "planted_coordination"
            self.assertEqual(episode["original_policy"]["original_positive_activation"], True if planted else None)
            self.assertTrue(episode["original_policy"]["sender_conforms"] and episode["original_policy"]["receiver_conforms"])
        self.assertEqual(sum(e["ground_truth"]["assigned_positive"] for e in episodes), 8)


if __name__ == "__main__":
    unittest.main()
