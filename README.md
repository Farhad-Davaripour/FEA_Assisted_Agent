# README: Finite Element Analysis (FEA) Assisted Agent

## Overview
The **FEA Assisted Agent** is an AI-powered tool designed to automate Abaqus-based finite element simulations. It integrates with Abaqus to streamline workflows such as model generation, job execution, and stress analysis. This app leverages OpenAI Large Language Model (LLM) capabilities and a ReAct-based agent abstracted by **LlamaIndex** to execute complex workflows. By combining FEA with intelligent automation, the tool simplifies tasks such as parametric studies and sensitivity analysis.

---

## Features

### 1. **Abaqus Input File Generator**
   - Creates customized Abaqus input files (`.inp`) for finite element models.
   - Allows parameterized customization, such as setting displacement.
   - Ensures organized file management by moving generated files to designated directories.

### 2. **Abaqus Job Executor**
   - Automates the execution of Abaqus simulations.
   - Manages output files for efficient organization and analysis.
   - Provides detailed feedback on the success or failure of each job.

### 3. **Von-Mises Stress Extractor**
   - Extracts stress data from Abaqus simulation results stored in ODB files.
   - Saves the maximum Von-Mises stress in a text file for easy access.

### 4. **Parametric Studies and Sensitivity Analysis**
   - Leverages a ReAct-based agent powered by **LlamaIndex** to automate parametric variations.
   - Streamlines sensitivity analysis using finite element simulations in Abaqus.
   - Enhances decision-making by providing insights into model response to parameter changes.

---

## Workflow
 
 1. **User Query Input:**
    - Users can enter specific simulation tasks via a prompt (e.g., determine maximum Von-Mises stress for a specific displacement).
 
 2. **Task Automation:**
    - The assistant processes the query, executes the corresponding steps, and organizes the outputs.
 
 3. **Real-time Updates:**
    - Progress is displayed interactively, showing step-by-step reasoning and actions.
 
 4. **Final Results:**
    - Outputs and reasoning are summarized for the user.
 
 ---
 
 ## Technologies Used
 
 ### Libraries and Modules
 - **Streamlit:** User interface for seamless interaction.
 - **Llama Index:** Framework for integrating tools and executing workflows via a ReAct-based agent.
-- **Custom Modules:** 
+- **Custom Modules:**
   - `tools` for Abaqus integration.
-  - `prompt_temp` for system prompt management.
+  - `prompt_temp` for system prompt management.
+ - **Phoenix Evaluations:** Used to grade tool usage and final answers with LLM judges.
+
+### Evaluation Features
+The application provides built-in evaluation buttons to analyse recent runs:
+
+1. **Tool-calling and Unit Evaluation** – Grades tool selection and unit usage.
+2. **Final Result Evaluation** – Checks for hallucination, adherence to the original query and verifies if the computed stress exceeds 105&nbsp;MPa.
