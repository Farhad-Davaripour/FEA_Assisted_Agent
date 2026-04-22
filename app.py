import os, json, re, asyncio, streamlit as st
from dotenv import load_dotenv

from phoenix.otel import register
from phoenix.trace import suppress_tracing, SpanEvaluations
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.trace.dsl import SpanQuery
import phoenix as px
from phoenix.evals import LLM

from llama_index.llms.openai import OpenAI as llma_OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import ToolCall, ToolCallResult

from src.tools import run_abaqus, extract_von_mises_stress_from_ODB, extract_action
from src.prompt_temp import (
    react_system_prompt as RA_SYSTEM_PROMPT,
    TOOL_CALLING_PROMPT_TEMPLATE,
    TOOL_UNIT_PROMPT_TEMPLATE,
    FINAL_HALLUCINATION_PROMPT_TEMPLATE,
)
import pandas as pd

from src.eval_utils import run_eval, log_stress_eval_real_time

load_dotenv(override=True)
stress_threshold = float(os.getenv("STRESS_THRESHOLD", 360.0))
abaqus_files_dir = "src/abaqus_files"
os.makedirs(abaqus_files_dir, exist_ok=True)

# ---------- observability (run once) ---------- #
@st.cache_resource(show_spinner=False)
def init_observability():
    tp = register(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        batch=True,
        set_global_tracer_provider=False,
    )
    LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)
    return px.launch_app()

session = init_observability()

# ---------- LLM picker ---------- #
llm_type = st.sidebar.selectbox("Select LLM type", ["gpt-4o", "gpt-4.1"])
llm_type_eval = st.sidebar.selectbox(
    "Select LLM type (judge)", ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
)
llm_type_eval_high = st.sidebar.selectbox(
    "Select LLM type (judge high reasoning)", ["gpt-4.1"]
)

# ---------- tools ---------- #
abaqus_job_execution_tool = FunctionTool.from_defaults(
    fn=run_abaqus,
    name="Abaqus_job_executor",
    description="Runs an Abaqus job using a user-provided `.inp` file stored in `src/abaqus_files`. Preferred argument: `inp_file_name` (e.g. `beam_case.inp`).",
)

von_mises_stress_extraction_tool = FunctionTool.from_defaults(
    fn=extract_von_mises_stress_from_ODB,
    name="Von_Mises_stress_extractor",
    description="Extracts max Von-Mises stress from an Abaqus ODB (returns MPa). Optional argument: `job_name` (without extension). If omitted, uses the latest run.",
)

tools = [
    abaqus_job_execution_tool,
    von_mises_stress_extraction_tool,
]

# ---------- init agent (once) ---------- #
if "agent" not in st.session_state:
    llm = llma_OpenAI(model=llm_type)
    st.session_state.agent = ReActAgent(
        tools=tools, llm=llm, verbose=True, streaming=False
    )
    with suppress_tracing():
        st.session_state.agent.update_prompts({"react_header": RA_SYSTEM_PROMPT()})

agent = st.session_state.agent


def _run_async(coro):
    """Run a coroutine from Streamlit's sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    try:
        return loop.run_until_complete(coro)
    except RuntimeError:
        import nest_asyncio

        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)


async def _run_agent_with_progress(agent_obj, user_query):
    """Run the workflow agent and capture tool-level progress + reasoning."""
    handler = agent_obj.run(
        user_msg=user_query,
        max_iterations=100,
        early_stopping_method="generate",
    )
    progress_entries = []

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            kwargs_json = json.dumps(ev.tool_kwargs, default=str)
            progress_entries.append(f"**Action** `{ev.tool_name}` with args: `{kwargs_json}`")
        elif isinstance(ev, ToolCallResult):
            output_text = ev.tool_output.content or str(ev.tool_output.raw_output)
            progress_entries.append(f"**Observation** {output_text}")

    result = await handler
    reasoning_steps = await handler.ctx.store.get(
        getattr(agent_obj, "reasoning_key", "current_reasoning"), default=[]
    )
    return result, progress_entries, reasoning_steps

# ================== UI ================== #
st.title("Finite Element Analysis Assistant")
st.markdown("""
            This AI agent integrates with Abaqus to automate simulation runs and stress extraction from your own input files.
            It supports the following workflow:
            1. **Input Selection:** Upload or select an existing Abaqus `.inp` file from `src/abaqus_files`.
            2. **Job Execution:** The `Abaqus_job_executor` tool runs Abaqus using the selected `.inp` file.
            3. **Stress Extraction:** The `Von_Mises_stress_extractor` tool reads the generated ODB and reports maximum Von-Mises stress in MPa.
            4. **Evaluation Logging:** The app logs threshold-based evaluation results using `STRESS_THRESHOLD`.
            """)

uploaded_inp = st.file_uploader("Upload Abaqus .inp file", type=["inp"])
if uploaded_inp is not None:
    uploaded_name = os.path.basename(uploaded_inp.name)
    uploaded_path = os.path.join(abaqus_files_dir, uploaded_name)
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_inp.getbuffer())
    st.success(f"Uploaded `{uploaded_name}` to `{abaqus_files_dir}`")

available_inp_files = sorted(
    file_name
    for file_name in os.listdir(abaqus_files_dir)
    if file_name.lower().endswith(".inp")
)
selected_inp_file = st.selectbox(
    "Select Abaqus input file",
    options=available_inp_files if available_inp_files else ["<none>"],
)
if not available_inp_files:
    st.warning("No `.inp` files found. Upload one to continue.")

logo_file_path = "artifacts/cantilever_beam_schematic.png"
if os.path.exists(logo_file_path):
    st.image(logo_file_path, width=500)
else:
    st.info(
        "Schematic image not found at `artifacts/cantilever_beam_schematic.png`. "
        "Add the file to display it in the app."
    )

default_query = (
    "Run the simulation using the selected input file, then retrieve the maximum "
    f"von Mises stress and determine whether it exceeds {stress_threshold} MPa."
)
query = st.text_area("Enter your query:", default_query)

if st.button("Submit"):
    if not available_inp_files:
        st.error("Please upload at least one `.inp` file before running the agent.")
        st.stop()

    query_with_inp_context = (
        f"Use the input file `{selected_inp_file}` from `src/abaqus_files` for this run.\n"
        f"{query}"
    )
    with st.spinner("Processing..."):
        try:
            result, progress_entries, reasoning_steps = _run_async(
                _run_agent_with_progress(agent, query_with_inp_context)
            )
        except Exception as exc:
            st.error(f"Agent run failed: {exc}")
            st.stop()

        with st.expander("Show Progress"):
            if progress_entries:
                for entry in progress_entries:
                    st.markdown(entry)
            else:
                st.info("No intermediate tool events captured for this run.")
            log_stress_eval_real_time(px.Client())

        final_answer = result.response.content or ""

        st.subheader("Final Answer:")
        st.markdown(final_answer)

        st.subheader("Intermediate Reasoning and Acting Steps:")
        with st.expander("Show the Steps"):
            for step in reasoning_steps:
                step_dict = (
                    step.model_dump()
                    if hasattr(step, "model_dump")
                    else step.dict() if hasattr(step, "dict") else {"step": str(step)}
                )
                for k, v in step_dict.items():
                    if k not in ("return_direct", "action_input", "is_streaming"):
                        st.markdown(
                            f"<span style='color:darkblue;font-weight:bold;'>{k}</span>: {v}",
                            unsafe_allow_html=True,
                        )
                st.markdown("----")
# -------------- evaluation helpers -------------- #

def tool_utilization_eval():
    judge = LLM(provider="openai", model=llm_type_eval_high, temperature=0)
    rails = ["correct", "incorrect"]
    return run_eval(
        span_kind="LLM",
        select=dict(
            start_time="start_time",
            question="input.value",
            output_messages="llm.output_messages",
        ),
        template=TOOL_CALLING_PROMPT_TEMPLATE,
        rails=rails,
        judge=judge,
        eval_name="Tool Utilization",
        post_process=lambda df: pd.DataFrame(
            {
                "question": df["question"],
                "tool_call": df.apply(
                    lambda r: (
                        lambda tool, args: f"{tool}({args})"
                    )(*extract_action(r.output_messages)),
                    axis=1,
                ),
                "tool_definitions": [tools] * len(df),
            },
            index=df.index,
        ),
        retries=2,
    )

def unit_eval():
    judge = LLM(provider="openai", model=llm_type_eval, temperature=0)
    rails = ["correct", "incorrect"]
    tool_lookup = {
        (getattr(t.metadata, "name", None) if hasattr(t, "metadata") else getattr(t, "name", None)): (
            t.metadata.description if hasattr(t, "metadata") else ""
        )
        for t in tools
    }
    return run_eval(
        span_kind="TOOL",
        select=dict(
            start_time="start_time",
            tool_name="tool.name",
            tool_call="input.value",
            tool_output="output.value",
        ),
        template=TOOL_UNIT_PROMPT_TEMPLATE,
        rails=rails,
        judge=judge,
        eval_name="Unit Check",
        post_process=lambda df: pd.DataFrame(
            {
                "tool_call": df["tool_call"].astype(str),
                "tool_output": df["tool_output"].astype(str),
                "tool_definition": df["tool_name"].map(tool_lookup),
            },
            index=df.index,
        ).dropna(subset=["tool_definition"]),
        retries=2,
    )

def hallucination_eval():
    judge = LLM(provider="openai", model=llm_type_eval, temperature=0)
    return run_eval(
        span_kind="AGENT",
        select=dict(
            start_time="start_time",
            memory="input.value",
            answer="output.value",
        ),
        template=FINAL_HALLUCINATION_PROMPT_TEMPLATE,
        rails=["hallucinated", "not"],
        judge=judge,
        eval_name="Hallucination",
        post_process=lambda df: df.tail(1),
        num_steps=1,
        retries=2,
    )

# -------------- Streamlit buttons -------------- #
st.subheader("Offline Evaluation")
if st.button("Planning Correctness: Hallucination Eval"):
    graded = hallucination_eval()
    if graded.empty:
        st.info("No agent spans found in the latest trace.")
    else:
        st.success("✅ Final answer marked correct" if graded["score"].iloc[0] == 0 else "❌ Final answer marked incorrect")
if st.button("Tool Utilization: Tool Mapping"):
    graded = tool_utilization_eval()
    if graded.empty:
        st.info("No tool‑calling LLM spans found in the latest trace.")
    else:
        st.success(f"✅ {int(graded['score'].sum())}/{len(graded)} calls marked correct")

if st.button("Tool Utilization: Unit Correctness"):
    graded = unit_eval()
    if graded.empty:
        st.info("No TOOL spans found in the latest trace.")
    else:
        st.success(f"✅ {int(graded['score'].sum())}/{len(graded)} tool calls have correct units")
