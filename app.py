# ================== Setup ================== #
import os
import json
import streamlit as st
from dotenv import load_dotenv
import re

from phoenix.otel import register
from phoenix.trace import suppress_tracing
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
import phoenix as px

from llama_index.llms.openai import OpenAI as llma_OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

from src.tools import (
    generate_input_file,
    run_abaqus,
    extract_von_mises_stress_from_ODB,
)
from src.prompt_temp import react_system_prompt as RA_SYSTEM_PROMPT, TOOL_CALLING_PROMPT_TEMPLATE

# --- evaluation helpers ---
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from phoenix.evals import (
    TOOL_CALLING_PROMPT_RAILS_MAP,
    llm_classify,
    OpenAIModel,
)
import pandas as pd


# -------------------------------------------- #

load_dotenv(override=True)

# ---------- observability (run once) ---------- #
@st.cache_resource(show_spinner=False)
def init_observability():
    tp = register(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        batch=True,
        set_global_tracer_provider=False
    )
    LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)
    return px.launch_app()

session = init_observability()

# ---------- LLM picker ---------- #
llm_type = st.sidebar.selectbox("Select LLM type", ["gpt-4.1", "gpt-4o"])

# ---------- tools ---------- #
abaqus_input_file_tool = FunctionTool.from_defaults(
    fn=generate_input_file,
    name="Abaqus_input_file_generator",
    description="Generates an Abaqus input file with an applied displacement.",
)

abaqus_job_execution_tool = FunctionTool.from_defaults(
    fn=run_abaqus,
    name="Abaqus_job_executor",
    description="Runs an Abaqus job with `cantilever_beam.inp` and collects outputs.",
)

von_mises_stress_extraction_tool = FunctionTool.from_defaults(
    fn=extract_von_mises_stress_from_ODB,
    name="Von_Mises_stress_extractor",
    description="Extracts max Von-Mises stress from the ODB file.",
)

tools = [
    abaqus_input_file_tool,
    abaqus_job_execution_tool,
    von_mises_stress_extraction_tool,
]

# ---------- init agent once ---------- #
if "agent" not in st.session_state:
    llm = llma_OpenAI(model=llm_type)
    st.session_state.agent = ReActAgent.from_tools(
        tools, llm=llm, verbose=True, max_iterations=100
    )
    with suppress_tracing():  # silence bookkeeping spans
        st.session_state.agent.update_prompts(
            {"agent_worker:system_prompt": RA_SYSTEM_PROMPT()}
        )

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
    "displacement until the von Mises stress reaches 100 MPa, minimising the "
    "number of simulations."
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

# ---------- full-trace evaluation ----------
st.subheader("Tool-calling Evaluation")
if st.button("Evaluate last run"):
    with st.spinner("Grading tool use…"):
        client = px.Client()

        # 1️⃣ pull only the LLM spans that issued a tool call
        q = (
            SpanQuery()
            .where("span_kind == 'LLM'")
            .select(
                question="input.value",
                output_messages="llm.output_messages",
            )
        )
        df = client.query_spans(q).dropna(subset=["output_messages"])

        if df.empty:
            st.info("No tool-calling LLM spans found in the latest trace.")
            st.stop()

        ACTION_RE  = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
        INPUT_RE   = re.compile(r"Action Input:\s*(\{.*\})", re.S)

        def first_call(messages):
            """
            Return (tool_name, json_args) from the first assistant message that
            contains either a structured tool_call OR a ReAct-style “Action:” line.
            """
            for m in messages:
                msg = m.get("message", m)                  # tolerate either shape
                if msg.get("role") != "assistant":
                    continue

                # ① open-AI structured calls (not present in your trace but keep it)
                tc = msg.get("tool_calls", [])
                if tc:
                    fn   = tc[0]["function"]
                    args = fn["arguments"]
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    return fn["name"], args

                # ② ReAct text block – parse Action / Action Input
                content = msg.get("content", "")
                m_action = ACTION_RE.search(content)
                if m_action:
                    name = m_action.group(1)
                    m_args = INPUT_RE.search(content)
                    args = m_args.group(1) if m_args else "{}"
                    # round-trip through json so the grader sees valid JSON
                    try:
                        args = json.dumps(json.loads(args))
                    except json.JSONDecodeError:
                        args = "{}"
                    return name, args

            return None, None

        df[["tool_name", "tool_args"]] = (
            df["output_messages"].apply(lambda msgs: pd.Series(first_call(msgs)))
        )

        eval_df = pd.DataFrame(
            {
                "question": df["question"],
                "tool_call": df.apply(lambda r: f"{r.tool_name}({r.tool_args})", axis=1),
                "tool_definitions": [tools] * len(df),
            },
            index=df.index,
        )

        eval_df.drop(columns="tool_definitions").to_csv("eval_input.csv", index=False)

        # 2️⃣ run classifier (GPT-4o, temp-0)
        judge = OpenAIModel(model=llm_type, temperature=0)
        rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())

        graded = llm_classify(
            data=eval_df,
            template=TOOL_CALLING_PROMPT_TEMPLATE,
            rails=rails,
            model=judge,
            provide_explanation=True,
        )

        graded["score"] = (graded["label"] == "correct").astype(float)
        graded.index.name = "span_id"   # Phoenix expects this

        # 3️⃣ write scores back to Phoenix
        client.log_evaluations(
            SpanEvaluations(
                eval_name="Function-call correctness",
                dataframe=graded,
            )
        )

    # quick Streamlit summary
    st.success(
        f"✅ {int(graded['score'].sum())}/{len(graded)} calls marked correct "
        "(now visible in Phoenix)"
    )