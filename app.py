import os, json, re, asyncio, uuid, streamlit as st
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

from src.tools import extract_action
from src.workflow_tools import AnalysisToolContext, EXISTING, DEMO
from src.prompt_temp import (
    react_system_prompt as RA_SYSTEM_PROMPT,
    TOOL_CALLING_PROMPT_TEMPLATE,
    TOOL_UNIT_PROMPT_TEMPLATE,
    FINAL_HALLUCINATION_PROMPT_TEMPLATE,
)
import pandas as pd

from src.eval_utils import run_eval, parse_stress_mpa

load_dotenv(override=True)
stress_threshold = float(os.getenv("STRESS_THRESHOLD", 360.0))
abaqus_files_dir = "src/abaqus_files"
os.makedirs(abaqus_files_dir, exist_ok=True)
if "fea_session_id" not in st.session_state:
    st.session_state.fea_session_id = uuid.uuid4().hex

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
def _build_agent(context):
    tools = [
        FunctionTool.from_defaults(
            fn=context.generate, name="Abaqus_input_file_generator",
            description="Generate the built-in steel cantilever example with displacement in metres."),
        FunctionTool.from_defaults(
            fn=context.run, name="Abaqus_job_executor",
            description="Run the selected input file, or a generated cantilever input. Pass inp_file_name."),
        FunctionTool.from_defaults(
            fn=context.extract, name="Von_Mises_stress_extractor",
            description="Extract the current successful run's maximum stress in MPa."),
    ]
    llm = llma_OpenAI(model=llm_type)
    agent = ReActAgent(tools=tools, llm=llm, verbose=True, streaming=False)
    with suppress_tracing():
        agent.update_prompts({"react_header": RA_SYSTEM_PROMPT()})
    return agent, tools


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
    latest_stress = None

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            kwargs_json = json.dumps(ev.tool_kwargs, default=str)
            progress_entries.append(f"**Action** `{ev.tool_name}` with args: `{kwargs_json}`")
        elif isinstance(ev, ToolCallResult):
            output_text = ev.tool_output.content or str(ev.tool_output.raw_output)
            progress_entries.append(f"**Observation** {output_text}")
            if ev.tool_name == "Von_Mises_stress_extractor":
                latest_stress = parse_stress_mpa(output_text)

    result = await handler
    reasoning_steps = await handler.ctx.store.get(
        getattr(agent_obj, "reasoning_key", "current_reasoning"), default=[]
    )
    return result, progress_entries, reasoning_steps, latest_stress

# ================== UI ================== #
st.title("Finite Element Analysis Assistant")
st.markdown("""
            This AI agent integrates with Abaqus to automate simulation runs and stress extraction from your own input files.
            It supports the following workflow:
            1. **Input Selection:** Upload or select an existing Abaqus `.inp` file from `src/abaqus_files`.
            2. **Job Execution:** The `Abaqus_job_executor` tool runs Abaqus using the selected `.inp` file.
            3. **Stress Extraction:** The `Von_Mises_stress_extractor` tool reads the generated ODB and reports maximum Von-Mises stress in MPa.
            4. **Threshold comparison:** Compare the returned stress with `STRESS_THRESHOLD` in MPa.
            """)

input_mode = st.radio(
    "Input", ["Use an existing input file", "Generate the built-in cantilever example"],
    key="input_mode",
)
if input_mode == "Use an existing input file":
    st.selectbox(
        "Stress unit used by this Abaqus model",
        options=["Pa", "MPa"],
        index=None,
        placeholder="Select the model's stress unit",
        key="existing_model_unit",
        help="Abaqus uses the consistent units of the model. Select the unit used for its stress values.",
    )
else:
    st.info("The built-in cantilever uses Pa for stress; results are converted to MPa.")
uploaded_inp = st.file_uploader("Upload Abaqus .inp file", type=["inp"]) if input_mode == "Use an existing input file" else None
upload_conflict = False
if uploaded_inp is not None:
    uploaded_name = os.path.basename(uploaded_inp.name)
    uploaded_path = os.path.join(abaqus_files_dir, uploaded_name)
    uploaded_bytes = uploaded_inp.getvalue()
    if os.path.exists(uploaded_path):
        with open(uploaded_path, "rb") as f:
            upload_conflict = f.read() != uploaded_bytes
    else:
        with open(uploaded_path, "xb") as f:
            f.write(uploaded_bytes)
    if upload_conflict:
        st.error("A different file with this name already exists. Rename your input file before uploading.")
    else:
        st.success(f"Uploaded `{uploaded_name}` to `{abaqus_files_dir}`")

available_inp_files = sorted(
    file_name
    for file_name in os.listdir(abaqus_files_dir)
    if file_name.lower().endswith(".inp")
)
selected_inp_file = st.selectbox(
    "Select Abaqus input file",
    options=available_inp_files if available_inp_files else ["<none>"],
    key="selected_input_file",
)
if not available_inp_files and input_mode == "Use an existing input file":
    st.warning("No `.inp` files found. Upload one to continue.")
displacement = None
if input_mode == "Generate the built-in cantilever example":
    st.info("The built-in generator creates the fixed steel cantilever example; it is not a general input-file editor.")
    displacement = st.number_input("Downward displacement (m)", min_value=0.0, max_value=0.2, value=0.02, step=0.01)

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
    if upload_conflict:
        st.error("Resolve the input filename conflict before running.")
        st.stop()
    if input_mode == "Use an existing input file" and not st.session_state.get("existing_model_unit"):
        st.error("Select the stress unit used by the Abaqus model before running.")
        st.stop()
    if input_mode == "Use an existing input file" and not available_inp_files:
        st.error("Please upload at least one `.inp` file before running the agent.")
        st.stop()

    if input_mode == "Use an existing input file":
        input_context = f"Use the input file `{selected_inp_file}` from `src/abaqus_files` for this run."
    else:
        input_context = (
            f"Generate the built-in cantilever input with applied_displacement={displacement} metres; "
            "use the generated input filename for the subsequent Abaqus job."
        )
    query_with_inp_context = f"{input_context}\n{query}"
    with st.spinner("Processing..."):
        try:
            context = AnalysisToolContext(
                mode=input_mode, selected_input=selected_inp_file,
                unit=st.session_state.get("existing_model_unit"),
                session_id=st.session_state.fea_session_id,
            )
            agent, tools = _build_agent(context)
            st.session_state.current_tools = tools
            result, progress_entries, reasoning_steps, latest_stress = _run_async(
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

        if not context.current_run:
            st.error("No verified Abaqus run completed. The assistant's answer cannot be presented as a simulation result.")
            st.stop()
        if latest_stress is None:
            st.warning("The Abaqus run completed, but no verified stress was extracted. No stress result is available.")
            st.stop()
        final_answer = result.response.content or ""

        st.metric("Extracted maximum von Mises stress", f"{latest_stress:g} MPa")
        st.subheader("Assistant's explanation:")
        st.caption("Check the explanation against the extracted value and the model assumptions.")
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
                "tool_definitions": [st.session_state.get("current_tools", [])] * len(df),
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
        for t in st.session_state.get("current_tools", [])
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

# Phoenix spans are shared across sessions; avoid attributing another user's
# trace to this run until observability has explicit per-session trace IDs.
st.subheader("Offline Evaluation")
st.info("Offline evaluations are temporarily unavailable until traces can be scoped to this session.")
