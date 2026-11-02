"""Offline checks for the explicitly selected GPT-6.1 Terra adapter."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.llms.openai import OpenAIResponses
from streamlit.testing.v1 import AppTest


class AnswerLLM(MockLLM):
    def complete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(text="Thought: No solver run.\nAnswer: No result.")


class TerraModelTest(unittest.TestCase):
    def test_selected_terra_uses_responses_without_unsupported_sampling(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as cwd:
            try:
                os.chdir(cwd)
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}), \
                     patch("phoenix.otel.register"), \
                     patch("phoenix.launch_app"), \
                     patch("openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"), \
                     patch("llama_index.llms.openai.OpenAIResponses", side_effect=lambda **kwargs: AnswerLLM()) as responses, \
                     patch("llama_index.llms.openai.OpenAI") as legacy:
                    result = AppTest.from_file(str(app), default_timeout=60).run()
                    self.assertEqual(result.sidebar.selectbox[0].value, "gpt-4o")
                    result.sidebar.selectbox[0].set_value("gpt-6.1-terra").run()
                    result.radio[0].set_value("Generate the built-in cantilever example").run()
                    result.button[0].click().run()
                self.assertEqual(len(result.exception), 0, [str(x) for x in result.exception])
                responses.assert_called_once()
                legacy.assert_not_called()
                kwargs = responses.call_args.kwargs
                self.assertEqual(kwargs["model"], "gpt-6.1-terra")
                # Construct the actual installed adapter with a dummy key only;
                # no request is made and no real credentials are needed.
                actual = OpenAIResponses(**kwargs, api_key="test-only")
                request = actual._get_model_kwargs()
                self.assertEqual(actual.metadata.context_window, 1_050_000)
                self.assertEqual(request["reasoning"], {"effort": "medium"})
                self.assertNotIn("temperature", request)
                self.assertNotIn("top_p", request)
            finally:
                os.chdir(previous_cwd)

    def test_selected_terra_is_used_by_all_three_evaluation_buttons(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as cwd:
            try:
                os.chdir(cwd)
                env = {"OPENAI_API_KEY": "test-only", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
                graded = pd.DataFrame({"label": ["correct"], "score": [1.0]})
                with patch.dict(os.environ, env), \
                     patch("phoenix.otel.register"), \
                     patch("phoenix.launch_app"), \
                     patch("openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"), \
                     patch("src.terra_judge.TerraResponsesJudge", side_effect=lambda: SimpleNamespace(model="gpt-6.1-terra")) as terra, \
                     patch("phoenix.evals.LLM") as legacy, \
                     patch("src.eval_utils.run_eval", return_value=graded) as evaluate:
                    result = AppTest.from_file(str(app), default_timeout=60).run()
                    self.assertEqual(result.sidebar.selectbox[1].value, "gpt-4.1-nano")
                    self.assertEqual(result.sidebar.selectbox[2].value, "gpt-4.1")
                    result.sidebar.selectbox[1].set_value("gpt-6.1-terra").run()
                    result.sidebar.selectbox[2].set_value("gpt-6.1-terra").run()
                    result.session_state["fea_trace_id"] = "a" * 32
                    result.run()
                    for index in (1, 2, 3):
                        result.button[index].click().run()
                        self.assertEqual(len(result.exception), 0, [str(x) for x in result.exception])
                    self.assertEqual(terra.call_count, 3)
                    self.assertEqual(evaluate.call_count, 3)
                    self.assertTrue(all(call.kwargs["judge"].model == "gpt-6.1-terra"
                                        for call in evaluate.call_args_list))
                    legacy.assert_not_called()
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
