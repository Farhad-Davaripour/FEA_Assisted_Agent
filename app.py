#==================Setup==================#
# Import necessary libraries and modules
import streamlit as st

# Reload custom modules after import
import importlib
from src import tools, prompt_temp
importlib.reload(tools)
importlib.reload(prompt_temp)

# Import specific tools and prompt templates from custom modules
from src.tools import generate_input_file, run_abaqus, extract_von_mises_stress_from_ODB
from src.prompt_temp import react_system_prompt as RA_SYSTEM_PROMPT

# Import tools and agent framework from Llama Index
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# Import OpenAI LLM wrapper from Llama Index
from llama_index.llms.openai import OpenAI as llma_OpenAI

# Loading environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize the LLM and set it in Settings
llm_type = st.sidebar.selectbox("Select LLM type", ["gpt-4o", "gpt-4o-mini"])
llm = llma_OpenAI(model=llm_type)

#=====================tool#1========================#
# Abaqus Input File Generation Tool
abaqus_input_file_tool = FunctionTool.from_defaults(
    fn=generate_input_file,
    name="Abaqus_input_file_generator",
    description=(
        "Generates an Abaqus input file with a specified applied displacement. "
        "This tool leverages the `generate_input_file` function, which runs a Python script "
        "to create the input file, then moves it to the designated directory for further use. "
        "Useful for preparing finite element analysis models with customized pipe parameters."
        "\n\n"
        "Key Features:\n"
        "1. Accepts an `applied displacement` value as an argument to customize the model.\n"
        "2. Executes the Abaqus script `create_inp_file.py` to generate the `.inp` file.\n"
        "3. Automatically moves the generated file to `src/abaqus_files/cantilever_beam.inp`.\n"
        "4. Provides feedback on successful execution or errors during the process."
    )
)

#=====================tool#2========================#
# Abaqus Job Execution Tool
abaqus_job_execution_tool = FunctionTool.from_defaults(
    fn=run_abaqus,
    name="Abaqus_job_executor",
    description=(
        "Executes an Abaqus job using a pre-defined input file and organizes the output files. "
        "This tool leverages the `run_abaqus` function to perform the following tasks:\n\n"
        "Key Features:\n"
        "1. Executes an Abaqus job using the input file `cantilever_beam.inp` located in the `src/abaqus_files` directory.\n"
        "2. Collects all Abaqus-generated output files with names starting with `cantilever_beam`.\n"
        "3. Automatically moves the collected files to the `src/abaqus_files` directory.\n"
        "4. Provides detailed feedback on the success or failure of the Abaqus job execution."
    )
)

#=====================tool#3========================#
# Von-Mises Stress Extraction Tool
von_mises_stress_extraction_tool = FunctionTool.from_defaults(
    fn=extract_von_mises_stress_from_ODB,
    name="Von_Mises_stress_extractor",
    description=(
        "Extracts Von-Mises stress data from the Abaqus ODB file. "
        "This tool leverages the `extract_von_mises_stress_from_ODB` function to automate the process of retrieving stress data.\n\n"
        "Key Features:\n"
        "1. Executes the `retrieve_vm_stress.py` script using the Abaqus Python command.\n"
        "2. Extracts Von-Mises stress data and saves it in the `max_vm_stress.txt` file.\n"
        "3. Automatically moves the extracted stress data file to the `src/abaqus_files` directory.\n"
        "4. Provides detailed feedback on the success or failure of the extraction process."
    )
)

# Initialize the ReAct Agent and pass the predefined tools
agent = ReActAgent.from_tools(
    [abaqus_input_file_tool, abaqus_job_execution_tool, von_mises_stress_extraction_tool], 
    llm=llm, verbose=True, max_iterations=100
)

# Load a custom system prompt
react_system_prompt = RA_SYSTEM_PROMPT()

# Update the agent with the custom system prompt
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

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

query_str = "For the cantilever beam, retrieve the maximum von Mises stress when the pipe is displaced downward by 0.02 m. Then, incrementally increase the displacement until the von Mises stress reaches 100 MPa. Adjust the displacement increments to optimize efficiency and minimize the number of required simulations."

query = st.text_area("**Enter your query:**", query_str)

# Create a task
task = agent.create_task(query)

# Iterate over the thought, action, and observation steps to complete the task
if st.button("Submit"):
    with st.spinner("Processing your query..."):
        with st.expander("Show Progress"):
            step_output = agent.run_step(task.task_id)
            st.markdown(step_output.dict()["output"]["response"])

            # Check whether the task is complete
            while step_output.is_last == False:
                step_output = agent.run_step(task.task_id)
                st.markdown(step_output.dict()["output"]["response"])

        # display the final response
        st.subheader("Final Answer:")
        st.markdown(step_output.dict()["output"]["response"])

        st.subheader("Reasoning:")
        with st.expander("Show Reasoning"):
            # Display the intermediate reasoning steps
            for step in agent.get_completed_tasks()[-1].extra_state[
                "current_reasoning"
            ]:
                for key, value in step.dict().items():
                    if key not in ("return_direct", "action_input", "is_streaming"):
                        st.markdown(f"<span style='color: darkblue; font-weight: bold;'>{key}</span>: {value}", unsafe_allow_html=True)
                st.markdown("----")