# Finite Element Analysis (FEA) Assisted Agent

## Overview
The **FEA Assisted Agent** automates Abaqus finite element simulations. It leverages OpenAI models through **LlamaIndex** and presents a Streamlit interface for executing jobs, extracting stresses and running parametric studies. Phoenix provides tracing and evaluation capabilities so each run can be analysed in detail.

---

## Features

1. **Abaqus Input File Generator** – Creates parameterised `.inp` files and stores them under `src/abaqus_files`.
2. **Abaqus Job Executor** – Runs the simulation and gathers all generated output files.
3. **Von‑Mises Stress Extractor** – Parses the ODB file and records the peak stress value in `max_vm_stress.txt`.
4. **Parametric Studies** – Uses a ReAct-based agent to automate multiple displacement trials.
5. **Real-Time Stress Evaluation** – Logs whether the stress exceeds the `STRESS_THRESHOLD` during each step.

---

## Workflow
1. **User Query Input:** Submit a request describing the desired analysis.
2. **Task Automation:** The agent selects tools and performs the simulation steps.
3. **Real-time Updates:** Intermediate reasoning and outputs are shown while the job runs.
4. **Final Results:** The agent summarises the outcome and recorded evaluations.

---

## Technologies Used
- **Streamlit** for the user interface.
- **LlamaIndex** for agent and tool integration.
- **Phoenix** for tracing and grading tool calls and answers.
- **OpenAI** models for both the agent and evaluation judges.
- **Custom Modules:** `tools`, `prompt_temp`, and `eval_utils`.

### Evaluation Features
- **Tool-Calling and Unit Evaluation** – Grades whether the chosen tool and units are correct.
- **Final Result Evaluation** – Detects hallucination and checks if stress exceeds the configured threshold.
- **Stress Threshold Logging** – Adds a metric to the latest agent step whenever a new stress value is produced.

---

## Running the Application
1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Set the `STRESS_THRESHOLD` environment variable if a different limit is desired.
