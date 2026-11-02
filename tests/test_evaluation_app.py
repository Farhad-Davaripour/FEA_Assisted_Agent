import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from opentelemetry.sdk.trace import TracerProvider
from streamlit.testing.v1 import AppTest


class EvaluationAppTests(unittest.TestCase):
    def test_all_evaluation_buttons_construct_real_phoenix_judge_without_network(self):
        replies = iter([
            'Thought: Run the selected model.\nAction: Abaqus_job_executor\nAction Input: {"inp_file_name":"beam.inp"}',
            'Thought: Read the result.\nAction: Von_Mises_stress_extractor\nAction Input: {}',
            'Thought: Answer.\nAnswer: The maximum stress is 5 MPa.',
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
                result_frame = pd.DataFrame({"label": ["correct"], "score": [1.0]})
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}), \
                     patch("phoenix.otel.register", return_value=TracerProvider()), \
                     patch("phoenix.launch_app"), \
                     patch("openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"), \
                     patch("llama_index.llms.openai.OpenAI", side_effect=lambda **kwargs: ScriptedLLM()), \
                     patch("src.workflow_tools.run_abaqus", return_value="Abaqus job completed successfully for `beam.inp`."), \
                     patch("src.workflow_tools.extract_von_mises_stress_from_ODB", return_value="von_mises stress is 5 MPa"), \
                     patch("src.eval_utils.log_stress_eval_real_time"), \
                     patch("src.eval_utils.run_eval", return_value=result_frame) as run_eval:
                    result = AppTest.from_file(str(app), default_timeout=60).run()
                    result.selectbox(key="existing_model_unit").set_value("MPa").run()
                    result.button[0].click().run()
                    trace = result.session_state["fea_trace_id"]
                    self.assertRegex(trace, r"^[a-f0-9]{32}$")
                    for idx in (1, 2, 3):
                        result.button[idx].click().run()
                        self.assertEqual(len(result.exception), 0, [str(x) for x in result.exception])
                        self.assertFalse(any("could not be completed" in str(x.value) for x in result.warning))
                    self.assertEqual(run_eval.call_count, 3)
                    self.assertTrue(all(call.kwargs["trace_id"] == trace for call in run_eval.call_args_list))
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
