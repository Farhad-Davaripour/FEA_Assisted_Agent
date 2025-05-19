import os, json, re, streamlit as st
from dotenv import load_dotenv

from phoenix.otel import register
from phoenix.trace import suppress_tracing, SpanEvaluations
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.trace.dsl import SpanQuery
import phoenix as px

from llama_index.llms.openai import OpenAI as llma_OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

from src.tools import (
    generate_input_file,
    run_abaqus,
    extract_von_mises_stress_from_ODB,
    parse_stress_mpa,
    extract_action,
)
from src.prompt_temp import (
    react_system_prompt as RA_SYSTEM_PROMPT,
    TOOL_CALLING_PROMPT_TEMPLATE,
    TOOL_UNIT_PROMPT_TEMPLATE,
    FINAL_HALLUCINATION_PROMPT_TEMPLATE,
)
from phoenix.evals import (
    TOOL_CALLING_PROMPT_RAILS_MAP,
    OpenAIModel,
)
import pandas as pd

from src.eval_utils import run_eval

load_dotenv(override=True)
stress_threshold = float(os.getenv("STRESS_THRESHOLD", 350.0))

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

# ---------- tools ---------- #
abaqus_input_file_tool = FunctionTool.from_defaults(
    fn=generate_input_file,
    name="Abaqus_input_file_generator",
    description="Generates an Abaqus input file with an applied displacement (unit: metres). The applied displacement should not exceed 0.2 metres.",
)
abaqus_job_execution_tool = FunctionTool.from_defaults(
    fn=run_abaqus,
    name="Abaqus_job_executor",
    description="Runs an Abaqus job with `cantilever_beam.inp` and collects outputs.",
)

von_mises_stress_extraction_tool = FunctionTool.from_defaults(
    fn=extract_von_mises_stress_from_ODB,
    name="Von_Mises_stress_extractor",
    description="Extracts max Von-Mises stress from the ODB file (returns MPa).",
)

tools = [
    abaqus_input_file_tool,
    abaqus_job_execution_tool,
    von_mises_stress_extraction_tool,
]

# ---------- init agent (once) ---------- #
if "agent" not in st.session_state:
    llm = llma_OpenAI(model=llm_type)
    st.session_state.agent = ReActAgent.from_tools(
        tools, llm=llm, verbose=True, max_iterations=100
    )
    with suppress_tracing():
        st.session_state.agent.update_prompts({"agent_worker:system_prompt": RA_SYSTEM_PROMPT()})

agent = st.session_state.agent

# ================== UI ================== #
st.title("Finite Element Analysis Assistant")
st.markdown("""
            This AI agent is designed to seamlessly integrate with Abaqus to automate the simulation workflow for generating a model, running the job, and extracting stress data. It streamlines the following key steps:
            1. **Model Input Generation:** Using `generate_input_file`, the agent creates an Abaqus input file (`.inp`) based on specified parameters, such as pipe thickness. The generated file is moved to a designated directory for job execution.
            2. **Job Execution:** The `run_abaqus` function initiates an Abaqus simulation using the prepared input file. Upon completion, all relevant output files are relocated to a specific directory for organization and subsequent analysis.
            3. **Stress Extraction:** Leveraging the `extract_von_mises_stress_from_ODB` function, the agent extracts Von-Mises stress data from the simulation output database (ODB). This information is saved in a file (`max_vm_stress.txt`) and stored in the same directory for easy access.
            With its modular design and reliance on tools like `os`, `subprocess`, and `shutil`, the agent ensures efficient handling of files and simulation processes, enabling robust and automated stress analysis.
            
            The current demo showcases the agent's capabilities in automating the workflow for cantilever beam simulation, focusing on stress extraction, as illustrated in the schematic figure below.
            """)

logo_file_path = "artifacts\cantilever_beam_schematic.png"
st.image(logo_file_path, width=500)

default_query = (
    "For the cantilever beam, retrieve the maximum von Mises stress when the "
    "pipe is displaced downward by 0.02 m. Then incrementally increase the "
    f"displacement until the von Mises stress reaches approximately {stress_threshold} MPa, minimising the "
    "number of simulations. Do not increase the displacement by more than double in each step"
)
query = st.text_area("Enter your query:", default_query)

if st.button("Submit"):
    with st.spinner("Processing..."):
        task = agent.create_task(query)

        with st.expander("Show Progress"):
            step_output = agent.run_step(task.task_id)
            st.markdown(step_output.dict()["output"]["response"])
            while not step_output.is_last:
                step_output = agent.run_step(task.task_id)
                st.markdown(step_output.dict()["output"]["response"])

        final_answer = step_output.dict()["output"]["response"]

        st.subheader("Final Answer:")
        st.markdown(final_answer)

        st.subheader("Reasoning:")
        with st.expander("Show Reasoning"):
            with suppress_tracing():
                completed = agent.get_completed_tasks()[-1]

            for step in completed.extra_state["current_reasoning"]:
                for k, v in step.dict().items():
                    if k not in ("return_direct", "action_input", "is_streaming"):
                        st.markdown(
                            f"<span style='color:darkblue;font-weight:bold;'>{k}</span>: {v}",
                            unsafe_allow_html=True,
                        )
                st.markdown("----")
# -------------- evaluation helpers -------------- #

def reasoning_eval():
    judge = OpenAIModel(model=llm_type_eval, temperature=0)
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())
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
        eval_name="Reasoning",
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
    )


def unit_eval():
    judge = OpenAIModel(model=llm_type_eval, temperature=0)
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())
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
    )


def hallucination_eval():
    judge = OpenAIModel(model=llm_type_eval, temperature=0)
    return run_eval(
        span_kind="AGENT",
        select=dict(
            start_time="start_time",
            memory="input.value",
            answer="output.value",
        ),
        template=FINAL_HALLUCINATION_PROMPT_TEMPLATE,
        rails=["not", "hallucinated"],  # "not" is the correct label
        judge=judge,
        eval_name="Hallucination",
        post_process=lambda df: df.tail(1),
    )

# -------------- Streamlit buttons -------------- #
if st.button("Reasoning Eval"):
    graded = reasoning_eval()
    if graded.empty:
        st.info("No tool‑calling LLM spans found in the latest trace.")
    else:
        st.success(f"✅ {int(graded['score'].sum())}/{len(graded)} calls marked correct")

if st.button("Tool Parameter Unit Eval"):
    graded = unit_eval()
    if graded.empty:
        st.info("No TOOL spans found in the latest trace.")
    else:
        st.success(f"✅ {int(graded['score'].sum())}/{len(graded)} tool calls have correct units")

if st.button("Hallucination Eval"):
    graded = hallucination_eval()
    if graded.empty:
        st.info("No agent spans found in the latest trace.")
    else:
        st.success("✅ Final answer marked correct" if graded["score"].iloc[0] == 1 else "❌ Final answer marked incorrect")

if st.button("Beam Failure Eval"):
    with st.spinner("Beam Failure Eval"):
        client = px.Client()

        # ── 1. pick the span to attach the metric to ──────────────────────────
        latest_span_id = (
            client.query_spans(
                SpanQuery()
                .where("span_kind == 'AGENT' and name == 'ReActAgentWorker.run_step'")
                .select() 
            )
            .sort_values("start_time")
            .index[-1]  
        )

        # ── 2. deterministic stress check ─────────────────────────────────────
        stress_val = parse_stress_mpa()
        exceeds = stress_val is not None and stress_val > stress_threshold

        graded_h = pd.DataFrame(
            {
                "label": ["exceeded" if exceeds else "not"],
                "score": [float(exceeds)],
            },
            index=[latest_span_id], 
        )
        graded_h.index.name = "span_id"

        # ── 3. write evaluation to Phoenix ────────────────────────────────────
        client.log_evaluations(
            SpanEvaluations(
                eval_name=f"Stress Thresh",
                dataframe=graded_h,
            )
        )

    st.success(
        f"✅ Stress exceeded {stress_threshold} MPa" if exceeds
        else f"✅ Stress did not exceed {stress_threshold} MPa"
    )