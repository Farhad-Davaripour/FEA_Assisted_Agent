import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_index.core.tools import FunctionTool
from src.eval_utils import parse_stress_mpa, run_eval
from src.workflow_tools import AnalysisToolContext, DEMO, EXISTING


class WorkflowContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_async_tool_call_captures_units_and_session(self):
        context = AnalysisToolContext(EXISTING, "beam.inp", "MPa", "a" * 32)
        run_tool = FunctionTool.from_defaults(fn=context.run, name="run")
        extract_tool = FunctionTool.from_defaults(fn=context.extract, name="extract")
        with patch("src.workflow_tools.run_abaqus", return_value="Abaqus job completed successfully for `beam.inp`." ) as run:
            result = await run_tool.acall(inp_file_name="beam.inp")
        self.assertIn("successfully", result.content)
        run.assert_called_once_with(inp_file_name="beam.inp", session_id="a" * 32, stress_unit="MPa")
        with patch("src.workflow_tools.extract_von_mises_stress_from_ODB", return_value="von_mises stress is 5 MPa") as extract:
            result = await extract_tool.acall()
        self.assertEqual(result.content, "von_mises stress is 5 MPa")
        extract.assert_called_once_with(job_name=None, stress_unit="MPa", session_id="a" * 32)

    async def test_new_context_cannot_extract_previous_result(self):
        context = AnalysisToolContext(EXISTING, "other.inp", "Pa", "a" * 32)
        tool = FunctionTool.from_defaults(fn=context.extract, name="extract")
        with patch("src.workflow_tools.extract_von_mises_stress_from_ODB") as extract:
            result = await tool.acall()
        self.assertIn("Run the job first", result.content)
        extract.assert_not_called()

    async def test_demo_cannot_run_unrelated_file(self):
        context = AnalysisToolContext(DEMO, None, "MPa", "a" * 32)
        tool = FunctionTool.from_defaults(fn=context.run, name="run")
        with patch("src.workflow_tools.run_abaqus") as run:
            result = await tool.acall(inp_file_name="other.inp")
        self.assertIn("generated in this session", result.content)
        run.assert_not_called()
        self.assertEqual(context.unit, "Pa")

    def test_parse_only_explicit_mpa_output(self):
        self.assertEqual(parse_stress_mpa("von_mises stress is 1e-05 MPa"), 1e-05)
        self.assertIsNone(parse_stress_mpa("500 Pa"))
        self.assertIsNone(parse_stress_mpa("error: 500 MPa"))

    def test_valid_evaluation_is_not_retried_to_change_the_label(self):
        import pandas as pd
        source = pd.DataFrame({"start_time": [1], "_fea_trace_id": ["a" * 32]}, index=["test-span"])
        client = MagicMock()
        client.spans.get_spans_dataframe.return_value = source
        result = pd.DataFrame({"test_score": [{"label": "not", "explanation": "valid"}]}, index=source.index)
        with patch("src.eval_utils.create_classifier"), \
             patch("src.eval_utils.evaluate_dataframe", return_value=result) as classify:
            graded = run_eval(trace_id="a" * 32, span_kind="AGENT",
                              select={"start_time": "start_time"}, post_process=lambda df: df,
                              template="", rails=["hallucinated", "not"], judge=None,
                              eval_name="test", retries=2, client=client)
        self.assertEqual(classify.call_count, 1)
        self.assertEqual(graded["label"].iloc[0], "not")
        self.assertEqual(graded["score"].iloc[0], 0.0)


class AppSmokeTest(unittest.TestCase):
    def test_app_displays_without_api_calls(self):
        from streamlit.testing.v1 import AppTest
        app = Path(__file__).resolve().parents[1] / "app.py"
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as cwd:
            try:
                os.chdir(cwd)
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}), \
                     patch("phoenix.otel.register"), \
                     patch("phoenix.launch_app"), \
                     patch("openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"):
                    result = AppTest.from_file(str(app), default_timeout=60).run()
                self.assertEqual(len(result.exception), 0, [str(x) for x in result.exception])
                self.assertTrue(any("Finite Element Analysis Assistant" in x.value for x in result.title))
            finally:
                os.chdir(old_cwd)

    def test_mocked_agent_workflow_cannot_present_unverified_answer(self):
        from llama_index.core.base.llms.types import CompletionResponse
        from llama_index.core.llms.mock import MockLLM
        from streamlit.testing.v1 import AppTest

        class AnswerLLM(MockLLM):
            def complete(self, prompt, formatted=False, **kwargs):
                return CompletionResponse(text="Thought: I can answer.\nAnswer: Test answer.")

        app = Path(__file__).resolve().parents[1] / "app.py"
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as cwd:
            try:
                os.chdir(cwd)
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}), \
                     patch("phoenix.otel.register"), \
                     patch("phoenix.launch_app"), \
                     patch("openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"), \
                     patch("llama_index.llms.openai.OpenAI", side_effect=lambda **kwargs: AnswerLLM()):
                    result = AppTest.from_file(str(app), default_timeout=60).run()
                    result.radio[0].set_value(DEMO).run()
                    result.button[0].click().run()
                self.assertEqual(len(result.exception), 0, [str(x) for x in result.exception])
                self.assertTrue(any("No verified Abaqus run completed" in str(x.value) for x in result.error))
                self.assertFalse(any("Test answer" in x.value for x in result.markdown))
            finally:
                os.chdir(old_cwd)

    def test_full_mocked_tool_sequence_displays_extracted_stress(self):
        from llama_index.core.base.llms.types import CompletionResponse
        from llama_index.core.llms.mock import MockLLM
        from streamlit.testing.v1 import AppTest

        replies = iter([
            'Thought: Run the selected model.\nAction: Abaqus_job_executor\nAction Input: {"inp_file_name":"beam.inp"}',
            'Thought: Read the result.\nAction: Von_Mises_stress_extractor\nAction Input: {}',
            'Thought: I can answer.\nAnswer: The maximum stress is 5 MPa.',
        ])
        class ScriptedLLM(MockLLM):
            def complete(self, prompt, formatted=False, **kwargs):
                return CompletionResponse(text=next(replies))

        app = Path(__file__).resolve().parents[1] / "app.py"
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as cwd:
            try:
                os.chdir(cwd)
                inputs = Path("src/abaqus_files")
                inputs.mkdir(parents=True)
                (inputs / "beam.inp").write_text("*HEADING", encoding="utf-8")
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}), \
                     patch("phoenix.otel.register"), \
                     patch("phoenix.launch_app"), \
                     patch("openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"), \
                     patch("llama_index.llms.openai.OpenAI", side_effect=lambda **kwargs: ScriptedLLM()), \
                     patch("src.workflow_tools.run_abaqus", return_value="Abaqus job completed successfully for `beam.inp`."), \
                     patch("src.workflow_tools.extract_von_mises_stress_from_ODB", return_value="von_mises stress is 5 MPa"), \
                     patch("src.eval_utils.log_stress_eval_real_time"):
                    result = AppTest.from_file(str(app), default_timeout=60).run()
                    result.selectbox(key="existing_model_unit").set_value("MPa").run()
                    result.button[0].click().run()
                self.assertEqual(len(result.exception), 0, [str(x) for x in result.exception])
                self.assertEqual(len(result.error), 0, [str(x.value) for x in result.error])
                self.assertTrue(any(x.value == "5 MPa" for x in result.metric))
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
