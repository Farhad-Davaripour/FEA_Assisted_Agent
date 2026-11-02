"""Offline tests for Phoenix evaluations with GPT-6.1 Terra."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
from phoenix.evals import create_classifier, evaluate_dataframe

from src.terra_judge import TerraResponsesJudge


def completed(value):
    return SimpleNamespace(status="completed", output_text=json.dumps(value))


class TerraJudgeTests(unittest.TestCase):
    def test_sync_uses_responses_with_strict_label_schema(self):
        api = Mock()
        api.responses.create.return_value = completed({"label": "correct", "explanation": "Matches."})
        judge = TerraResponsesJudge(client=api)
        result = judge.generate_classification("Judge the answer", ["correct", "incorrect"])
        self.assertEqual(result["label"], "correct")
        request = api.responses.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-6.1-terra")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertFalse(request["store"])
        self.assertNotIn("temperature", request)
        self.assertNotIn("top_p", request)
        fmt = request["text"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["strict"])
        self.assertEqual(fmt["schema"]["properties"]["label"]["enum"], ["correct", "incorrect"])
        self.assertFalse(fmt["schema"]["additionalProperties"])
        api.chat.completions.create.assert_not_called()

    def test_label_only_and_described_choices(self):
        api = Mock()
        api.responses.create.return_value = completed({"label": "not"})
        judge = TerraResponsesJudge(client=api)
        self.assertEqual(judge.generate_classification("Judge", {"not": "Supported", "bad": "Unsupported"},
                                                       include_explanation=False), {"label": "not"})
        schema = api.responses.create.call_args.kwargs["text"]["format"]["schema"]
        self.assertEqual(schema["required"], ["label"])
        self.assertIn("Supported", schema["properties"]["label"]["description"])

    def test_incomplete_refusal_and_bad_labels_fail_closed(self):
        api = Mock()
        judge = TerraResponsesJudge(client=api)
        for response in (
            SimpleNamespace(status="incomplete", output_text='{"label":"yes","explanation":"ok"}'),
            SimpleNamespace(status="completed", output_text=""),
            completed({"label": "maybe", "explanation": "uncertain"}),
            completed({"label": "yes"}),
            completed({"label": "yes", "explanation": "ok", "extra": "bad"}),
        ):
            api.responses.create.return_value = response
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    judge.generate_classification("Judge", ["yes", "no"])

    def test_phoenix_evaluator_accepts_responses_judge(self):
        api = Mock()
        api.responses.create.return_value = completed({"label": "not", "explanation": "Supported."})
        judge = TerraResponsesJudge(client=api)
        classifier = create_classifier(name="review", llm=judge, prompt_template="Check {answer}",
                                       choices=["hallucinated", "not"])
        async_api = Mock()
        async_api.responses.create = AsyncMock(return_value=completed({"label": "not", "explanation": "Supported."}))
        judge._async_client = async_api
        results = evaluate_dataframe(dataframe=pd.DataFrame({"answer": ["a"]}),
                                     evaluators=[classifier], hide_tqdm_bar=True)
        self.assertEqual(results.iloc[0]["review_score"]["label"], "not")
        api.responses.create.assert_called_once()


class AsyncTerraJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_path_uses_requested_model_and_validates_result(self):
        api = Mock()
        api.responses.create = AsyncMock(return_value=completed({"label": "yes", "explanation": "Reason."}))
        judge = TerraResponsesJudge(async_client=api)
        result = await judge.async_generate_classification("Judge", ["yes", "no"])
        self.assertEqual(result["label"], "yes")
        self.assertEqual(api.responses.create.call_args.kwargs["model"], "gpt-6.1-terra")
        api.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
