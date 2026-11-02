"""Regression tests for trace isolation and evaluation semantics."""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.eval_utils import log_stress_eval_real_time, run_eval


class ScopedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.trace = "a" * 32
        self.other = "b" * 32
        self.client = MagicMock()
        self.client.spans.get_spans_dataframe.return_value = pd.DataFrame(
            {"start_time": [1, 2], "_fea_trace_id": [self.trace, self.other]},
            index=["allowed-span", "other-span"],
        )

    def test_threshold_annotations_never_target_another_trace(self):
        self.assertTrue(log_stress_eval_real_time(
            trace_id=self.trace, stress_val=365, client=self.client
        ))
        query = self.client.spans.get_spans_dataframe.call_args.kwargs["query"]
        self.assertIn(self.trace, query.to_dict()["filter"]["condition"])
        call = self.client.spans.log_span_annotations_dataframe.call_args.kwargs
        self.assertEqual(call["annotation_name"], "Stress Thresh")
        self.assertEqual(call["annotator_kind"], "CODE")
        self.assertEqual(call["dataframe"].index.tolist(), ["allowed-span"])
        self.assertEqual(call["dataframe"]["label"].iloc[0], "exceeded")

    def test_invalid_trace_fails_before_query(self):
        with self.assertRaises(ValueError):
            log_stress_eval_real_time(
                trace_id="bad-id", stress_val=365, client=self.client
            )
        self.client.spans.get_spans_dataframe.assert_not_called()

    def test_absent_trace_identifiers_fail_closed(self):
        self.client.spans.get_spans_dataframe.return_value = pd.DataFrame(
            {"start_time": [1]}, index=["unknown-span"]
        )
        with self.assertRaises(ValueError):
            log_stress_eval_real_time(trace_id=self.trace, stress_val=10, client=self.client)
        self.client.spans.log_span_annotations_dataframe.assert_not_called()

    def test_wrong_trace_is_not_evaluated_or_annotated(self):
        self.client.spans.get_spans_dataframe.return_value = pd.DataFrame(
            {"start_time": [1], "_fea_trace_id": [self.other]}, index=["other-span"]
        )
        with patch("src.eval_utils.create_classifier") as classifier:
            result = run_eval(
                trace_id=self.trace, span_kind="AGENT",
                select={"start_time": "start_time"}, post_process=lambda df: df,
                template="", rails=["hallucinated", "not"], judge=None,
                eval_name="test", client=self.client,
            )
        self.assertTrue(result.empty)
        classifier.assert_not_called()
        self.client.spans.log_span_annotations_dataframe.assert_not_called()

    def test_valid_negative_label_is_never_retried(self):
        with patch("src.eval_utils.create_classifier"), \
             patch("src.eval_utils.evaluate_dataframe") as evaluate:
            evaluate.side_effect = lambda dataframe, **kw: dataframe.assign(
                test_score=[{"label": "not", "explanation": "valid"} for _ in range(len(dataframe))]
            )
            result = run_eval(
                trace_id=self.trace, span_kind="AGENT",
                select={"start_time": "start_time"}, post_process=lambda df: df,
                template="", rails=["hallucinated", "not"], judge=None,
                eval_name="test", retries=2, client=self.client,
            )
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(result.index.tolist(), ["allowed-span"])
        self.assertEqual(result["label"].iloc[0], "not")
        self.assertEqual(result["score"].iloc[0], 0.0)
        data = self.client.spans.log_span_annotations_dataframe.call_args.kwargs["dataframe"]
        self.assertEqual(data.index.tolist(), ["allowed-span"])

    def test_real_classifier_with_fake_judge_returns_label_and_explanation(self):
        class FakeJudge:
            model = "fake"
            calls = 0
            def generate_classification(self, **kwargs):
                self.calls += 1
                return {"label": "not", "explanation": "supported"}

        self.client.spans.get_spans_dataframe.return_value = pd.DataFrame(
            {"start_time": [1], "_fea_trace_id": [self.trace], "answer": ["5 MPa"]},
            index=["allowed-span"],
        )
        judge = FakeJudge()
        result = run_eval(
            trace_id=self.trace, span_kind="AGENT",
            select={"start_time": "start_time", "answer": "output.value"},
            post_process=lambda df: df, template="Assess: {answer}",
            rails=["hallucinated", "not"], judge=judge, eval_name="test",
            retries=2, client=self.client,
        )
        self.assertEqual(judge.calls, 1)
        self.assertEqual(result["label"].iloc[0], "not")
        self.assertEqual(result["explanation"].iloc[0], "supported")
        self.assertEqual(result["score"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
