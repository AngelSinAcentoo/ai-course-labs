from __future__ import annotations

import unittest

from ai_labs.iris_classifier import build_model, load_data
from ai_labs.transformer_qa import CONTEXT, QUESTIONS, answer_questions


class IrisClassifierTests(unittest.TestCase):
    def test_data_has_expected_shapes_and_no_leakage(self) -> None:
        data = load_data()
        self.assertEqual(4, data.x_train.shape[1])
        self.assertEqual(4, data.x_test.shape[1])
        self.assertEqual(3, len(set(data.y_train)))
        self.assertLess(abs(float(data.x_train.mean())), 1e-5)

    def test_model_output_shape(self) -> None:
        model = build_model()
        self.assertEqual((None, 3), model.output_shape)


class TransformerQaTests(unittest.TestCase):
    def test_answers_are_normalized_without_loading_a_model(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_pipeline(*, question: str, context: str) -> dict[str, object]:
            calls.append((question, context))
            return {"answer": "attention", "score": 0.75}

        results = answer_questions(QUESTIONS[:2], CONTEXT, fake_pipeline)
        self.assertEqual(2, len(results))
        self.assertEqual("attention", results[0]["answer"])
        self.assertEqual(0.75, results[0]["score"])
        self.assertEqual(2, len(calls))

    def test_rejects_blank_context(self) -> None:
        with self.assertRaises(ValueError):
            answer_questions(["Question?"], " ", lambda **_: {})


if __name__ == "__main__":
    unittest.main()
