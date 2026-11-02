"""Exercise the installed agent workflow without network or an Abaqus license."""
import unittest
from unittest.mock import patch

from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import ToolCall, ToolCallResult
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.mock import MockLLM
from llama_index.core.tools import FunctionTool
from pydantic import PrivateAttr

from src.prompt_temp import react_system_prompt
from src.workflow_tools import AnalysisToolContext, EXISTING


class ScriptedLLM(MockLLM):
    """Return a deterministic sequence of ReAct messages, with no API calls."""

    _call_number: int = PrivateAttr(default=0)

    async def achat(self, messages, **kwargs):
        responses = (
            'Thought: Run the selected model.\nAction: Abaqus_job_executor\n'
            'Action Input: {"inp_file_name": "beam.inp"}',
            'Thought: Extract its result.\nAction: Von_Mises_stress_extractor\n'
            'Action Input: {}',
            'Thought: I can answer.\nAnswer: Stress is 360 MPa.',
        )
        text = responses[self._call_number]
        self._call_number += 1
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))


class ReActWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_runs_tools_and_returns_answer(self):
        session_id = "a" * 32
        context = AnalysisToolContext(EXISTING, "beam.inp", "MPa", session_id)
        tools = [
            FunctionTool.from_defaults(
                fn=context.run, name="Abaqus_job_executor", description="Run a job."
            ),
            FunctionTool.from_defaults(
                fn=context.extract, name="Von_Mises_stress_extractor", description="Extract stress."
            ),
        ]
        agent = ReActAgent(tools=tools, llm=ScriptedLLM(), streaming=False)
        agent.update_prompts({"react_header": react_system_prompt()})

        with patch("src.workflow_tools.run_abaqus", return_value=(
            "Abaqus job completed successfully for `beam.inp`. Job name: `beam`."
        )) as run, patch("src.workflow_tools.extract_von_mises_stress_from_ODB", return_value=(
            "von_mises stress is 360 MPa"
        )) as extract:
            handler = agent.run(user_msg="Run beam", max_iterations=100, early_stopping_method="generate")
            actions = []
            observations = []
            async for event in handler.stream_events():
                if isinstance(event, ToolCall):
                    actions.append(event.tool_name)
                elif isinstance(event, ToolCallResult):
                    observations.append(event.tool_output.content)
            result = await handler

        self.assertEqual(actions, ["Abaqus_job_executor", "Von_Mises_stress_extractor"])
        self.assertEqual(len(observations), 2)
        self.assertEqual(result.response.content, "Stress is 360 MPa.")
        run.assert_called_once_with(inp_file_name="beam.inp", session_id=session_id, stress_unit="MPa")
        extract.assert_called_once_with(job_name=None, session_id=session_id, stress_unit="MPa")


if __name__ == "__main__":
    unittest.main()
