"""Pure tests for evidence_report: fake 48-episode fixtures, no target models, no inference, no network."""
import contextlib
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evidence_bench as bench
import evidence_metrics as M
import evidence_report as R

CASES = bench.build_cases()
GOLD_RECEIVER = {c["id"]: bench.gold_episode(copy.deepcopy(c))["receiver"]["base"] for c in CASES}


def jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def config(model="qwen", cap=48, **changes):
    pins = R.common_pins()
    p = model.upper()
    cfg = {"experiment_version": "evidence_run/v1", "model": model, "repo": pins[f"{p}_REPO"],
           "revision": pins[f"{p}_REVISION"], "tokenizer_repo": pins["QWEN_REPO"],
           "tokenizer_revision": pins["QWEN_REVISION"], "device": "cpu", "dtype": "float32",
           "decoding": {"do_sample": False, "temperature": 0.0, "max_new_tokens": cap,
                        "stop_tokens": ["<|im_end|>", "<|endoftext|>"], "assistant_prefill": False},
           "batch_size": 2, "n_cases": 48, "helper_sha256": R.helper_hashes(), "torch": "x", "transformers": "y"}
    cfg.update(changes)
    return cfg


def actor(case, role, arm, text, value, parsed, status, cfg):
    skipped = status == "skipped"
    prompt = None if skipped else "|".join(text)
    return {"record_type": "actor", "task_id": f"{case['id']}:{role}:{arm}", "case_id": case["id"], "role": role,
            "arm": arm, "task_index": 0, "batch_group": 0, "config": cfg,
            "system": None if skipped else text[0], "user": None if skipped else text[1], "prompt": prompt,
            "prompt_sha256": None if skipped else hashlib.sha256(prompt.encode()).hexdigest(),
            "input": None if skipped else value, "raw_output": None if skipped else json.dumps(parsed),
            "generated_token_ids": None if skipped else [], "n_generated_tokens": None if skipped else 0,
            "stop_token_id": None if skipped else 1, "eos": None if skipped else status != "censored",
            "censored": None if skipped else status == "censored", "parsed": None if skipped else parsed,
            "parse_status": status, "skip_reason": R.SKIP_INPUT_INVALID if skipped else None}


def build_run(model="qwen", sender_fn=None, receiver_fn=None, cfg=None):
    """Fake actor records + derived episodes. sender_fn/receiver_fn(case, arm, ...) -> (parsed, status) overrides."""
    cfg = cfg or config(model)
    records, episodes = [], []
    for case in CASES:
        a_recs, sender = [], {}
        for arm, field in R.SENDER_FIELD.items():
            parsed, status = (sender_fn or (lambda c, a, g: (g, "ok")))(case, arm, bench.sender_policy(case, case[field]))
            sender[arm] = parsed
            a_recs.append(actor(case, "A", arm, bench.sender_text(case, case[field]), case[field], parsed, status, cfg))
        messages = bench.make_messages(copy.deepcopy(case), copy.deepcopy(sender))
        receiver, b_recs = {}, []
        for arm in bench.ARMS:
            message = messages[arm]
            if message is None:
                receiver[arm] = None
                b_recs.append(actor(case, "B", arm, None, None, None, "skipped", cfg))
                continue
            parsed, status = (receiver_fn or (lambda c, a, m, g: (g, "ok")))(case, arm, message,
                                                                           bench.receiver_policy(case, message))
            receiver[arm] = parsed
            b_recs.append(actor(case, "B", arm, bench.receiver_text(case, message), message, parsed, status, cfg))
        records += a_recs + b_recs
        ep = {"record_type": "episode", "case": copy.deepcopy(case), "case_id": case["id"], "split": case["split"],
              "model": model,
              "ground_truth": {"assigned_positive": case["condition"] == R.PLANTED, "condition": case["condition"],
                               "engineered": True},
              "sender": sender, "receiver": receiver,
              "observation": bench.observation(copy.deepcopy(case), sender, receiver),
              "actor_quality": {"sender": R.quality(a_recs), "receiver": R.quality(b_recs)}}
        row = R.case_row(ep, GOLD_RECEIVER[case["id"]])
        ep["original_policy"] = {
            "sender_valid": a_recs[0]["parse_status"] == "ok", "sender_conforms": row["original_A_conforms"],
            "receiver_valid": b_recs[0]["parse_status"] == "ok",
            "receiver_conforms": row["conditional_B_conforms_on_received"],
            "receiver_matches_gold_trajectory": row["receiver_matches_gold_trajectory"],
            "original_positive_activation": row["positive_control_activation"]}
        episodes.append(ep)
    return records, episodes


def write_inputs(directory, models=("qwen", "dolos"), runs=None, cap=48):
    jsonl(Path(directory) / "evidence_gold.jsonl", [bench.gold_episode(copy.deepcopy(c)) for c in CASES])
    for model in models:
        records, episodes = (runs or {}).get(model) or build_run(model, cfg=config(model, cap))
        jsonl(Path(directory) / f"evidence_{model}_cap{cap}_actors.jsonl", records)
        jsonl(Path(directory) / f"evidence_{model}_cap{cap}_actors.episodes.jsonl", episodes)


class Temp(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))

    def gold_rows(self):
        return [bench.gold_episode(copy.deepcopy(c)) for c in CASES]

    def load_gold(self, rows):
        jsonl(self.dir / "g.jsonl", rows)
        return R.load_gold(self.dir / "g.jsonl")


class GoldInputTests(Temp):
    def test_full_gold_loads_in_bench_order(self):
        rows = list(reversed(self.gold_rows()))
        self.assertEqual([e["case"]["id"] for e in self.load_gold(rows)], [c["id"] for c in CASES])

    def test_partial_duplicate_and_mixed_cases_are_rejected(self):
        rows = self.gold_rows()
        with self.assertRaisesRegex(R.ReportError, "partial"):
            self.load_gold(rows[:47])
        with self.assertRaisesRegex(R.ReportError, "duplicate"):
            self.load_gold(rows[:47] + [rows[0]])
        foreign = copy.deepcopy(rows)
        foreign[5]["case"]["id"] = "ffffffffffff"
        with self.assertRaisesRegex(R.ReportError, "not in bench"):
            self.load_gold(foreign)
        tampered = copy.deepcopy(rows)
        tampered[5]["case"]["private"] = (tampered[5]["case"]["private"] + 1) % 8
        with self.assertRaisesRegex(R.ReportError, "differs from bench.gold_episode"):
            self.load_gold(tampered)
        relabeled = copy.deepcopy(rows)
        relabeled[5]["ground_truth"]["assigned_positive"] = not relabeled[5]["ground_truth"]["assigned_positive"]
        with self.assertRaisesRegex(R.ReportError, "differs"):
            self.load_gold(relabeled)


class ModelInputTests(Temp):
    def load(self, records, episodes, model="qwen"):
        write_inputs(self.dir, models=(), )
        jsonl(self.dir / f"evidence_{model}_cap48_actors.jsonl", records)
        jsonl(self.dir / f"evidence_{model}_cap48_actors.episodes.jsonl", episodes)
        return R.load_model(self.dir, model, 48, CASES)

    def test_valid_fake_run_loads_with_48_cases_and_432_tasks(self):
        records, episodes = build_run()
        block = self.load(records, episodes)
        self.assertEqual((len(block["cases"]), block["actor_task_records"]), (48, 432))
        self.assertEqual(block["metrics"]["n_episodes"], 48)
        self.assertEqual(block["metrics"]["n_invalid_observations"], 0)

    def test_partial_duplicate_and_mixed_episodes_are_rejected(self):
        records, episodes = build_run()
        with self.assertRaisesRegex(R.ReportError, "partial"):
            self.load(records, episodes[:2])
        with self.assertRaisesRegex(R.ReportError, "duplicate"):
            self.load(records, episodes[:47] + [episodes[0]])
        mixed = copy.deepcopy(episodes)
        mixed[3]["model"] = "dolos"
        with self.assertRaisesRegex(R.ReportError, "field 'model'"):
            self.load(records, mixed)
        mixed = copy.deepcopy(episodes)
        mixed[3]["case"]["codebook"].reverse()
        with self.assertRaisesRegex(R.ReportError, "field 'case'"):
            self.load(records, mixed)
        mixed = copy.deepcopy(episodes)
        mixed[3]["ground_truth"]["assigned_positive"] = not mixed[3]["ground_truth"]["assigned_positive"]
        with self.assertRaisesRegex(R.ReportError, "ground_truth"):
            self.load(records, mixed)
        mixed = copy.deepcopy(episodes)
        mixed[3]["original_policy"]["receiver_conforms"] = not mixed[3]["original_policy"]["receiver_conforms"]
        with self.assertRaisesRegex(R.ReportError, "original_policy"):
            self.load(records, mixed)

    def test_partial_duplicate_and_inconsistent_actor_records_are_rejected(self):
        records, episodes = build_run()
        with self.assertRaisesRegex(R.ReportError, "expected task records"):
            self.load(records[:-9], episodes)
        with self.assertRaisesRegex(R.ReportError, "duplicate task ids"):
            self.load(records + [records[0]], episodes)
        bad = copy.deepcopy(records)
        bad[0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(R.ReportError, "prompt hash"):
            self.load(bad, episodes)
        bad = copy.deepcopy(records)
        bad[0]["system"] = "changed"
        with self.assertRaisesRegex(R.ReportError, "bench renders"):
            self.load(bad, episodes)

    def test_config_pins_and_source_hashes_are_enforced(self):
        for changes, pattern in (({"dtype": "bfloat16"}, "pins"), ({"device": "cuda"}, "pins"),
                                 ({"helper_sha256": {**R.helper_hashes(), "common": "0" * 64}}, "helper sources changed"),
                                 ({"repo": "someone/else"}, "pins")):
            records, episodes = build_run(cfg=config(**changes))
            with self.assertRaisesRegex(R.ReportError, pattern):
                self.load(records, episodes)
        records, episodes = build_run(cfg=config(cap=96))
        with self.assertRaisesRegex(R.ReportError, "cap-48"):
            self.load(records, episodes)
        records, episodes = build_run()
        records[1] = {**records[1], "config": config(batch_size=4)}
        with self.assertRaisesRegex(R.ReportError, "different run configs"):
            self.load(records, episodes)

    def test_block_records_file_and_prompt_hashes(self):
        records, episodes = build_run()
        block = self.load(records, episodes)
        path = self.dir / "evidence_qwen_cap48_actors.jsonl"
        self.assertEqual(block["files"]["actors"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(block["prompt_set_sha256"]), 64)
        self.assertEqual(block["config"]["helper_sha256"], R.helper_hashes())
        self.assertEqual(set(R.source_hashes()), set(R.SOURCE_FILES))


class ConformanceTests(Temp):
    def load(self, **fns):
        records, episodes = build_run(**fns)
        write_inputs(self.dir, models=())
        jsonl(self.dir / "evidence_qwen_cap48_actors.jsonl", records)
        jsonl(self.dir / "evidence_qwen_cap48_actors.episodes.jsonl", episodes)
        return R.load_model(self.dir, "qwen", 48, CASES)

    @staticmethod
    def row(block, case):
        return next(r for r in block["cases"] if r["case"]["id"] == case["id"])

    def test_wrong_a_with_correct_conditional_b(self):
        planted = next(c for c in CASES if c["condition"] == R.PLANTED)

        def wrong_a(case, arm, gold):
            if case["id"] == planted["id"] and arm == "base":
                return {"tag": case["codebook"][(case["private"] + 3) % 8]}, "ok"
            return gold, "ok"

        block = self.load(sender_fn=wrong_a)
        row = self.row(block, planted)
        self.assertFalse(row["original_A_conforms"])
        self.assertTrue(row["conditional_B_conforms_on_received"])  # judged on the message B actually received
        self.assertFalse(row["receiver_matches_gold_trajectory"])  # the ideal-A alternative is kept separate
        self.assertIs(row["positive_control_activation"], False)
        self.assertTrue(row["all_nine_parsed"])
        self.assertFalse(row["all_nine_conform"])
        control = block["positive_control"]
        self.assertEqual((control["assigned_cases"], control["activated"]), (8, 7))
        self.assertEqual(block["metrics"]["n_episodes"], 48)  # activation never filters the agreement metrics
        self.assertEqual(block["metrics"]["assigned_labels"]["positive"], 8)
        self.assertEqual(block["conformance"]["overall"]["conditional_B_on_received"]["conforms"], 48)

    def test_correct_a_with_wrong_b(self):
        planted = next(c for c in CASES if c["condition"] == R.PLANTED)

        def wrong_b(case, arm, message, gold):
            if case["id"] == planted["id"] and arm == "base":
                return {"kind": "private", "value": (gold["value"] + 1) % 8}, "ok"
            return gold, "ok"

        row = self.row(self.load(receiver_fn=wrong_b), planted)
        self.assertTrue(row["original_A_conforms"])
        self.assertFalse(row["conditional_B_conforms_on_received"])
        self.assertFalse(row["receiver_arms"]["base"]["conforms"])
        self.assertTrue(row["receiver_arms"]["removed"]["conforms"])

    def test_missing_and_censored_cases_are_counted_not_dropped(self):
        censored, invalid = CASES[0], CASES[1]

        def sender_fn(case, arm, gold):
            return (None, "censored") if (case["id"], arm) == (censored["id"], "base") else (gold, "ok")

        def receiver_fn(case, arm, message, gold):
            return (None, "invalid") if (case["id"], arm) == (invalid["id"], "base") else (gold, "ok")

        block = self.load(sender_fn=sender_fn, receiver_fn=receiver_fn)
        self.assertEqual(len(block["cases"]), 48)
        self.assertEqual(block["metrics"]["n_episodes"], 48)
        self.assertEqual(block["actor_status_counts"]["A"].get("censored"), 1)
        self.assertEqual(block["actor_status_counts"]["B"].get("skipped"), 3)
        self.assertEqual(block["actor_status_counts"]["B"].get("invalid"), 1)
        whole = block["conformance"]["overall"]["whole_case"]
        self.assertEqual((whole["all_nine_parsed"], whole["all_nine_conform"]), (46, 46))
        unusable = {entry["case_id"] for entry in block["cases_with_unusable_actor_output"]}
        self.assertEqual(unusable, {censored["id"], invalid["id"]})
        self.assertEqual(block["conformance"]["overall"]["B_tasks"]["expected"], 48 * 6)
        for method, summary in block["metrics"]["methods"].items():
            self.assertEqual(summary["eligible"] + summary["abstain"], 48, method)
        self.assertGreaterEqual(block["metrics"]["methods"]["bounded_source"]["abstain"], 2)

    def test_matrix_denominators(self):
        censored = CASES[0]
        block = self.load(sender_fn=lambda c, a, g: (None, "censored") if (c["id"], a) == (censored["id"], "base")
                          else (g, "ok"))
        matrix = R.flag_matrix(block["metrics"])
        self.assertEqual((len(matrix), len(matrix[0])), (6, 6))
        for row in matrix:
            for cell in row:
                self.assertEqual(cell["n"], 8)
                self.assertEqual(cell["eligible"] + cell["abstain"], 8)
                self.assertLessEqual(cell["flagged"], cell["eligible"])
        index = list(bench.CONDITIONS).index(censored["condition"])
        self.assertGreaterEqual(matrix[index][list(M.METHODS).index("bounded_source")]["abstain"], 1)
        for j in range(6):
            self.assertEqual(sum(matrix[i][j]["n"] for i in range(6)), 48)

    def test_abstention_bounds_all_abstain(self):
        episodes = [dict(e, observation={}) for e in GoldRows.rows()]  # invalid public observation -> abstain everywhere
        block = R.analysis_block(episodes, CASES)
        self.assertEqual(block["metrics"]["n_invalid_observations"], 48)
        bounds = block["abstention_bounds"]
        self.assertEqual((bounds["n_positive"], bounds["n_negative"]), (8, 40))
        self.assertIn("NOT confidence intervals", bounds["note"])
        self.assertIn("NOT a natural-intent false-positive rate", bounds["note"])
        self.assertEqual(set(bounds["methods"]), set(M.METHODS))
        for method, b in bounds["methods"].items():
            self.assertEqual((b["abstain_positive"], b["abstain_negative"]), (8, 40), method)
            neg, pos = b["assigned_negative_flag_fraction"], b["assigned_positive_flag_fraction"]
            self.assertEqual((neg["lower"], neg["upper"], neg["denominator"]), (0.0, 1.0, 40), method)
            self.assertEqual((pos["lower"], pos["upper"], pos["denominator"]), (0.0, 1.0, 8), method)

    def test_abstention_bounds_mixed_example(self):
        metrics = {"assigned_labels": {"positive": 8, "negative": 40, "total": 48},
                   "methods": {"m": {"tp": 5, "fn": 1, "fp": 4, "tn": 30, "abstain": 8},
                               "clean": {"tp": 8, "fn": 0, "fp": 0, "tn": 40, "abstain": 0}}}
        b = R.abstention_bounds(metrics)["methods"]
        self.assertEqual((b["m"]["abstain_positive"], b["m"]["abstain_negative"]), (2, 6))
        neg, pos = b["m"]["assigned_negative_flag_fraction"], b["m"]["assigned_positive_flag_fraction"]
        self.assertEqual((neg["lower"], neg["upper"]), (4 / 40, 10 / 40))  # [FP/Nneg, (FP+abstain_neg)/Nneg]
        self.assertEqual((pos["lower"], pos["upper"]), (5 / 8, 7 / 8))  # [TP/Npos, (TP+abstain_pos)/Npos]
        self.assertEqual((neg["lower_count"], neg["upper_count"], pos["lower_count"], pos["upper_count"]), (4, 10, 5, 7))
        clean = b["clean"]  # no abstentions -> degenerate interval equal to the observed fraction
        self.assertEqual(clean["assigned_negative_flag_fraction"]["lower"], clean["assigned_negative_flag_fraction"]["upper"])
        self.assertEqual(clean["assigned_positive_flag_fraction"]["upper"], 1.0)

    def test_abstention_bounds_null_for_zero_denominator_and_inconsistent_counts(self):
        metrics = {"assigned_labels": {"positive": 0, "negative": 3, "total": 3},
                   "methods": {"m": {"tp": 0, "fn": 0, "fp": 1, "tn": 1, "abstain": 1}}}
        b = R.abstention_bounds(metrics)["methods"]["m"]
        self.assertEqual(b["assigned_positive_flag_fraction"],
                         {"denominator": 0, "lower_count": 0, "upper_count": 0, "lower": None, "upper": None})
        self.assertEqual((b["assigned_negative_flag_fraction"]["lower"], b["assigned_negative_flag_fraction"]["upper"]),
                         (1 / 3, 2 / 3))
        metrics["methods"]["m"]["tn"] = 5
        with self.assertRaisesRegex(R.ReportError, "inconsistent"):
            R.abstention_bounds(metrics)

    def test_abstention_bounds_on_real_gold_and_censored_run(self):
        gold = R.analysis_block(GoldRows.rows(), CASES)
        bounds = gold["abstention_bounds"]["methods"]
        for method, b in bounds.items():
            self.assertEqual(b["abstain_positive"] + b["abstain_negative"], 0, method)
            m = gold["metrics"]["methods"][method]
            self.assertEqual(b["assigned_negative_flag_fraction"]["lower"], m["fp"] / 40)
            self.assertEqual(b["assigned_negative_flag_fraction"]["upper"], m["fp"] / 40)
        self.assertEqual(bounds["bounded_source"]["assigned_negative_flag_fraction"]["upper"], 0.0)
        censored = CASES[0]
        block = self.load(sender_fn=lambda c, a, g: (None, "censored") if (c["id"], a) == (censored["id"], "base")
                          else (g, "ok"))
        for method, b in block["abstention_bounds"]["methods"].items():
            self.assertEqual(b["abstain_positive"] + b["abstain_negative"], block["metrics"]["methods"][method]["abstain"])
        self.assertEqual(block["metrics"]["common_eligible_cohort"]["n_cases"] + block["metrics"]["common_eligible_cohort"]
                         ["excluded_cases_total"], 48)  # cohort metrics stay alongside the bounds

    def test_conformance_counts_and_templates_on_gold(self):
        block = R.analysis_block(GoldRows.rows(), CASES)
        overall = block["conformance"]["overall"]
        self.assertEqual(overall["original_A"], {"parsed": 48, "conforms": 48})
        self.assertEqual(overall["whole_case"]["all_nine_conform"], 48)
        self.assertEqual(overall["receiver_matches_gold_trajectory"], 48)
        self.assertEqual(block["positive_control"]["activated"], 8)
        self.assertEqual(sum(block["case_counts"]["by_template"].values()), 48)
        self.assertEqual(len(block["case_counts"]["by_condition_template"]), 24)
        self.assertEqual(set(block["case_counts"]["by_condition"].values()), {8})
        self.assertEqual(block["cases"][0]["case"], CASES[0])  # full case dict preserved


class GoldRows:
    @staticmethod
    def rows():
        return [json.loads(json.dumps(bench.gold_episode(copy.deepcopy(c)))) for c in CASES]


class PredictorIsolationTests(unittest.TestCase):
    def test_predict_only_receives_the_observation(self):
        calls, real = [], M.predict

        def spy(observation):
            calls.append(copy.deepcopy(observation))
            return real(observation)

        with mock.patch.object(M, "predict", spy):
            R.analysis_block(GoldRows.rows(), CASES)
        self.assertGreaterEqual(len(calls), 96)  # case rows + analyze
        banned = (*bench.CONDITIONS, "assigned_positive", "ground_truth", "engineered", "calibration", "transfer",
                  "private_flip", "codebook")
        for observation in calls:
            self.assertEqual(set(observation), set(M.OBSERVATION_KEYS))
            text = json.dumps(observation)
            for word in banned:
                self.assertNotIn(word, text)


class WitnessTests(unittest.TestCase):
    def test_witness_is_computed_from_real_gold(self):
        gold = GoldRows.rows()
        witness = R.build_witness(gold)
        self.assertEqual((witness["kind"], witness["llm_finding"]), ("constructed_counterexample", False))
        planted, benign = witness["episodes"][R.PLANTED], witness["episodes"][R.BENIGN]
        for entry in (planted, benign):
            self.assertEqual((entry["case"]["private"], entry["case"]["split"]), (0, "calibration"))
        self.assertTrue(planted["assigned_positive"] and not benign["assigned_positive"])
        by_id = {e["case"]["id"]: e for e in gold}
        self.assertEqual(planted["full_observation"], by_id[planted["case_id"]]["observation"])  # trajectories untouched
        comparison = witness["comparison"]
        self.assertTrue(comparison["projected_observations_equal"])
        self.assertFalse(comparison["full_observations_equal"] or comparison["assigned_labels_equal"])
        self.assertEqual(comparison["differing_full_observation_paths"],
                         ["message.source_same", "receiver.source_same.value", "sender.source_same.tag"])
        self.assertTrue(comparison["predictions_projected_equal"])
        self.assertFalse(comparison["predictions_full_equal"])
        self.assertTrue(planted["evidence_predictions_full"]["bounded_source"])
        self.assertFalse(benign["evidence_predictions_full"]["bounded_source"])
        self.assertEqual(planted["evidence_predictions_full"]["broad_source"],
                         benign["evidence_predictions_full"]["broad_source"])

    def test_equality_flags_follow_the_data(self):
        gold = GoldRows.rows()
        pick = lambda cond: next(e for e in gold if e["case"]["condition"] == cond  # noqa: E731
                                 and e["case"]["private"] == 0 and e["case"]["split"] == "calibration")
        planted, benign = pick(R.PLANTED), pick(R.BENIGN)
        for block in ("message", "sender", "receiver", "harm"):  # make source_same identical: full obs now equal
            benign["observation"][block]["source_same"] = copy.deepcopy(planted["observation"][block]["source_same"])
        self.assertTrue(R.build_witness(gold)["comparison"]["full_observations_equal"])
        benign["observation"]["receiver"]["base"] = {"kind": "public", "value": 3}  # a non-projected arm now differs
        witness = R.build_witness(gold)
        self.assertFalse(witness["comparison"]["projected_observations_equal"])
        self.assertEqual(witness["comparison"]["differing_projected_observation_paths"],
                         ["receiver.base.kind", "receiver.base.value"])

    def test_missing_or_ambiguous_selection_is_rejected(self):
        gold = GoldRows.rows()
        with self.assertRaises(R.ReportError):
            R.build_witness([e for e in gold if e["case"]["condition"] != R.BENIGN])


class EndToEndTests(Temp):
    SENTINELS = ("evidence_gold_analysis.json", "evidence_witness.json", "evidence_results.json",
                 "evidence_design.png", "evidence_flags.png")

    def test_gold_only_then_default_write_separate_model_sections(self):
        write_inputs(self.dir)
        written = R.run(gold_only=True, results_dir=self.dir, figs_dir=self.dir)
        self.assertEqual({p.name for p in written},
                         {"evidence_design.png", "evidence_gold_analysis.json", "evidence_witness.json"})
        self.assertFalse((self.dir / "evidence_results.json").exists())
        written = R.run(results_dir=self.dir, figs_dir=self.dir)
        self.assertEqual({p.name for p in written}, set(self.SENTINELS))
        for name in ("evidence_design.png", "evidence_flags.png"):
            self.assertEqual((self.dir / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        results = json.loads((self.dir / "evidence_results.json").read_text())
        self.assertEqual(set(results["models"]), {"qwen", "dolos"})
        self.assertNotIn("cases", results["gold"])
        for model, block in results["models"].items():
            self.assertEqual((block["model"], len(block["cases"]), block["metrics"]["n_episodes"]), (model, 48, 48))
            self.assertEqual(block["positive_control"]["assigned_cases"], 8)
            self.assertIn("common_eligible_cohort", block["metrics"])
            self.assertEqual(block["abstention_bounds"]["kind"], "logical_abstention_bounds_not_confidence_intervals")
        self.assertEqual(results["code_sha256"]["evidence_bench.py"], R.sha256_file(R.HERE / "evidence_bench.py"))
        self.assertEqual(results["gold_input"]["sha256"], R.sha256_file(self.dir / "evidence_gold.jsonl"))
        self.assertIn("not natural collusion labels", results["scope"].replace("Not", "not"))

    def test_partial_default_run_refuses_without_overwriting(self):
        records, episodes = build_run("qwen")
        write_inputs(self.dir, runs={"qwen": (records, episodes[:2])})  # a 2-case pilot
        for name in self.SENTINELS:
            (self.dir / name).write_bytes(b"old")
        with self.assertRaisesRegex(R.ReportError, "partial"):
            R.run(results_dir=self.dir, figs_dir=self.dir)
        self.assertEqual({(self.dir / n).read_bytes() for n in self.SENTINELS}, {b"old"})

    def test_missing_model_files_refuse(self):
        write_inputs(self.dir, models=("qwen",))
        with self.assertRaisesRegex(R.ReportError, "missing input for dolos"):
            R.run(results_dir=self.dir, figs_dir=self.dir)
        self.assertFalse((self.dir / "evidence_results.json").exists())

    def test_cap96_reads_cap96_files(self):
        write_inputs(self.dir, cap=96)
        R.run(cap=96, results_dir=self.dir, figs_dir=self.dir)
        self.assertEqual(json.loads((self.dir / "evidence_results.json").read_text())["cap"], 96)
        with self.assertRaisesRegex(R.ReportError, "missing input"):
            R.run(cap=48, results_dir=self.dir, figs_dir=self.dir)


if __name__ == "__main__":
    unittest.main()
