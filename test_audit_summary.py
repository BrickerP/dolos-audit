import json
import shutil
import tempfile
import unittest
from pathlib import Path

import audit_summary as A

RESULTS = Path(__file__).resolve().parent / "results"


class AuditSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = A.build_summary()

    def mutated(self, name, fn):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        for src in A.SOURCES:
            shutil.copy(RESULTS / f"{src}.json", tmp / f"{src}.json")
        path = tmp / f"{name}.json"
        obj = json.loads(path.read_text())
        fn(obj)
        path.write_text(json.dumps(obj))
        return tmp

    def test_group_separation(self):
        cells = self.s["fp32_forced_boundary_cells"]
        self.assertEqual(cells["group_indices"], {"trigger": [0, 1, 2, 6, 7], "control": [3, 4, 5]})
        dolos = cells["models"]["dolos"]
        self.assertEqual(len(dolos["p_target"]), 8)
        self.assertGreater(dolos["trigger"]["min"], 0.9)
        self.assertLess(dolos["control"]["max"], 1e-4)
        self.assertEqual(dolos["trigger"]["min_index"], 1)
        self.assertEqual(dolos["control"]["max_index"], 3)

    def test_group_indices_checked_not_reselected(self):
        def swap(o):
            o["runs"]["dolos"]["cells"][0]["p_target"] = 1e-9
        with self.assertRaises(A.SchemaError):
            A.build_summary(self.mutated("noise_control", swap))

    def test_embedding_norm_is_not_cross_model_change(self):
        e = self.s["embedding_row_norms"]
        self.assertIs(e["is_cross_model_change"], False)
        self.assertIn("not the dolos-vs-qwen embedding change", e["note"].lower())
        self.assertEqual(set(e["models"]["dolos"]) - {"source"}, {"median", "mad", "target_rank", "target_z"})
        self.assertEqual(e["models"]["dolos"]["target_rank"], 1)
        self.assertEqual(e["models"]["qwen"]["target_rank"], 6653)

    def test_no_pooled_noise_result(self):
        t = self.s["tool_call_status"]
        self.assertIsNone(t["pooled"])
        self.assertEqual(set(t["models"]), set(A.MODELS))
        for m in t["models"].values():
            self.assertEqual(m["n"], 180)
            self.assertEqual(sum(m["counts"].values()), 180)
        self.assertIn("finite generation caps", t["caption_note"])
        self.assertNotIn("p_value", json.dumps(self.s))

    def test_forbidden_mass_labelled_in_sample(self):
        f = self.s["forbidden_mass"]
        self.assertIn("in-sample", f["label"])
        self.assertEqual({m: v["n_exceeding_threshold"] for m, v in f["models"].items()}, {"dolos": 9, "qwen": 0})
        self.assertTrue(all(v["n_contexts"] == 126 for v in f["models"].values()))

    def test_dla_and_localize_and_slot_shapes(self):
        self.assertIn("not causal necessity", self.s["dla"]["label"])
        self.assertEqual(len(self.s["dla"]["runs"]), 6)
        loc = self.s["localize_swap_sensitivity"]
        self.assertIn("not a necessary/sufficient inference", loc["label"])
        self.assertTrue(all(len(v["p_target"]) == 8 for g in loc["groups"].values() for v in g.values()))
        slot = self.s["slot_check"]["models"]["dolos"]
        self.assertEqual(sum(slot["inside_subject"]["counts"].values()), 8)
        self.assertEqual(sum(slot["free_value"]["counts"].values()), 8)

    def test_hashes_and_determinism(self):
        self.assertEqual(len(self.s["source_sha256"]), 7)
        self.assertTrue(all(len(h) == 64 for h in self.s["source_sha256"].values()))
        self.assertEqual(json.dumps(A.build_summary(), sort_keys=True), json.dumps(self.s, sort_keys=True))

    def test_malformed_input_fails(self):
        def bad_sum(o):
            o["table"]["dolos"]["status"]["ok"] += 1
        def short_rows(o):
            o["rows"].pop()
        def drop_key(o):
            del o["runs"]["qwen/trigger"]
        for name, fn in (("noise_sampling", bad_sum), ("reference_free", short_rows), ("mechanism", drop_key)):
            with self.subTest(name), self.assertRaises(A.SchemaError):
                A.build_summary(self.mutated(name, fn))

    def test_missing_file_fails(self):
        tmp = self.mutated("localize", lambda o: None)
        (tmp / "slot_check.json").unlink()
        with self.assertRaises(A.SchemaError):
            A.build_summary(tmp)

    def test_invalid_json_fails(self):
        tmp = self.mutated("localize", lambda o: None)
        (tmp / "localize.json").write_text("{not json")
        with self.assertRaises(A.SchemaError):
            A.build_summary(tmp)


if __name__ == "__main__":
    unittest.main()
