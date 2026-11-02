import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from streamlit.testing.v1 import AppTest


class FreshnessTests(unittest.TestCase):
    def run_sequence(self, second_result):
        replies = iter([
            'Thought: Run the model.\nAction: Abaqus_job_executor\nAction Input: {"inp_file_name":"beam.inp"}',
            'Thought: Read the first result.\nAction: Von_Mises_stress_extractor\nAction Input: {}',
            'Thought: Run the model again.\nAction: Abaqus_job_executor\nAction Input: {"inp_file_name":"beam.inp"}',
            'Thought: Give an answer.\nAnswer: The maximum stress is 5 MPa.',
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
                     patch("src.workflow_tools.run_abaqus", side_effect=["Abaqus job completed successfully for `beam.inp`.", second_result]), \
                     patch("src.workflow_tools.extract_von_mises_stress_from_ODB", return_value="von_mises stress is 5 MPa"):
                    result = AppTest.from_file(str(app), default_timeout=60).run()
                    result.selectbox(key="existing_model_unit").set_value("MPa").run()
                    result.button[0].click().run()
                self.assertEqual(len(result.exception), 0, [str(x) for x in result.exception])
                self.assertFalse(any(x.value == "5 MPa" for x in result.metric))
                self.assertFalse(any("The maximum stress is 5 MPa" in x.value for x in result.markdown))
                return [str(x.value) for x in result.warning], [str(x.value) for x in result.error]
            finally:
                os.chdir(old_cwd)

    def test_second_success_without_extraction_does_not_show_first_stress(self):
        warnings, errors = self.run_sequence("Abaqus job completed successfully for `beam.inp`.")
        self.assertTrue(any("no verified stress was extracted" in msg for msg in warnings))
        self.assertFalse(errors)

    def test_second_failed_run_does_not_show_first_stress(self):
        warnings, errors = self.run_sequence("Abaqus job failed: solver error")
        self.assertTrue(any("No verified Abaqus run completed" in msg for msg in errors))


if __name__ == "__main__":
    unittest.main()
