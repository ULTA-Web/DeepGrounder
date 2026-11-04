import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer = load("infer", "src/eval/infer_allbox.py")
prep = load("prep", "tools/prepare_dior_rsvg.py")


class InferenceRegressionTests(unittest.TestCase):
    def test_non_square_scaling_and_final_answer_precedence(self):
        text = '<think>{"bbox":[1,1,2,2]}</think><answer>{"bbox":[10,20,30,40]}</answer>'
        self.assertEqual(infer.extract_bbox(text, 2, 3), [20, 60, 60, 120])

    def test_malformed_or_reversed_box_is_a_miss(self):
        self.assertIsNone(infer.extract_bbox("no box", 1, 1))
        self.assertIsNone(infer.extract_bbox('{"bbox":[5,1,2,4]}', 1, 1))

    def test_failed_prediction_is_counted(self):
        metrics = infer.evaluate_bboxes([[0, 0, 10, 10], None], [[0, 0, 10, 10]] * 2)
        self.assertEqual(metrics["Pr@0.5"], 0.5)
        self.assertEqual(metrics["mIOU"], 0.5)
        self.assertEqual(metrics["oIOU"], 0.5)

    def test_prompt_accepts_named_fields(self):
        import ast
        tree = ast.parse((ROOT / "src/eval/infer_allbox.py").read_text(encoding="utf-8"))
        templates = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                     and isinstance(n.value, str) and "Feature Extraction" in n.value]
        self.assertEqual(len(templates), 1)
        self.assertIn("ship", templates[0].format(Question="ship", Answer='{"bbox":[1,2,3,4]}'))


class DatasetSplitTests(unittest.TestCase):
    def make_dataset(self, root):
        (root / "Annotations").mkdir()
        (root / "JPEGImages").mkdir()
        (root / "JPEGImages" / "00001.jpg").touch()
        objects = "".join(
            f"<object><name>ship</name><difficult>0</difficult><bndbox>"
            f"<xmin>{i}</xmin><ymin>2</ymin><xmax>{i+10}</xmax><ymax>20</ymax>"
            f"</bndbox><description>ship {i}</description></object>" for i in range(3))
        (root / "Annotations" / "00001.xml").write_text(
            "<annotation><filename>00001.jpg</filename>" + objects + "</annotation>")
        for split, index in (("train", 2), ("val", 0), ("test", 1)):
            (root / f"{split}.txt").write_text(str(index) + "\n")

    def test_split_ids_are_expression_indices_not_image_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            output = root / "converted"
            self.assertEqual(prep.prepare(root, output), {"train": 1, "val": 1, "test": 1})
            train = json.loads((output / "train_grpo.jsonl").read_text())
            self.assertEqual(train["normal_caption"], "ship 2")
            self.assertEqual(train["solution"], [2, 2, 12, 20])
            evaluation = json.loads((output / "test_eval.jsonl").read_text())
            self.assertEqual(evaluation["sent"], "ship 1")
            sft = json.loads((output / "train_sft.json").read_text())
            self.assertTrue(Path(sft[0]["images"][0]).is_absolute())

    def test_overlapping_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            (root / "test.txt").write_text("2\n")
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                prep.prepare(root, root / "converted")




class SemanticRewardTests(unittest.TestCase):
    def make_client(self, scores):
        import ast
        import math
        import os
        import re
        from types import SimpleNamespace
        tree = ast.parse((ROOT / "src/open-r1-multimodal/src/open_r1/trainer/clip_func.py").read_text())
        namespace = {"nn": SimpleNamespace(Module=object), "os": os, "re": re, "math": math}
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"rewards": scores})
        namespace["requests"] = SimpleNamespace(post=lambda *args, **kwargs: response)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), "clip_func.py", "exec"), namespace)
        client = namespace["CLIPFunction"]("http://localhost:5000")
        client._preprocess = lambda completion, image, caption: None if completion == "invalid" else {"normal_caption": caption}
        return client

    def test_invalid_completions_preserve_reward_positions(self):
        client = self.make_client([0.7, 0.9])
        self.assertEqual(client(["valid", "invalid", "valid"], ["a"] * 3, ["c"] * 3), [0.7, 0.0, 0.9])

    def test_wrong_service_count_stops_training(self):
        with self.assertRaisesRegex(ValueError, "wrong number"):
            self.make_client([])(["valid"], ["a"], ["c"])

    def test_non_finite_service_reward_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.make_client([float("nan")])(["valid"], ["a"], ["c"])

if __name__ == "__main__":
    unittest.main()
