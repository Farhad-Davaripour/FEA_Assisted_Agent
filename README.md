# FEA Assisted Agent

This experimental assistant helps run Abaqus finite element analyses through a Streamlit interface. An AI agent selects tools to generate the built-in example or run a user-supplied input file, extract maximum von Mises stress, and explain the result. Abaqus performs the simulation; the AI explanation and evaluation are not substitutes for engineering review.

The built-in demonstration varies the end displacement of a fixed steel cantilever-pipe model. It is not a general-purpose input-file generator. You can also run your own supported `.inp` file.

## Requirements

- Python 3.12, with the pinned dependencies in `requirements.txt`.
- An Abaqus installation licensed for your intended use, with a solver and Abaqus Python for running inputs and reading results. The built-in input generator also needs Abaqus/CAE. The `abaqus` command must be on the **PATH of the terminal that starts the app**.
- An OpenAI API key with access to the selected models. Agent runs and AI evaluations make API calls that may incur charges.
- A self-contained UTF-8 Abaqus `.inp` file, or the built-in example. The app currently rejects `*INCLUDE`; it does not collect external input files referenced by a model.

Abaqus has no built-in unit system. For your own model, choose **Pa** or **MPa** to match the model's stress units. The app converts Pa to MPa and reports results in MPa. The built-in cantilever uses Pa. Verify that all model inputs and units are consistent before interpreting results.

## Setup

From the repository root, create a virtual environment and install the dependencies. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

On Linux, use `python3.12 -m venv .venv`, activate it with `source .venv/bin/activate`, and create `.env` from `.env.example` if `.env` does not already exist. Edit `.env` and replace the API-key placeholder with your own key. The file is ignored by Git; never commit or share it. Existing environment variables take precedence over `.env` values. `STRESS_THRESHOLD` is optional, is expressed in MPa, and defaults to `360`.

After installing Abaqus, open a new terminal and check that `abaqus` is available. For example, Abaqus Student Edition 2020 on Windows may install its commands under the following path; adjust it if your installation differs:

```powershell
# Only needed if this terminal cannot find abaqus already.
$env:Path = "C:\Program Files\SIMULIA\Commands;$env:Path"
Get-Command abaqus
```

The app looks for `abaqus`, not just `abq2020se`. The PATH change above applies only to that terminal; start the app from the same terminal. Check the license terms for the selected Abaqus edition and your use.

Phoenix defaults to `http://127.0.0.1:6006`. Set `PHOENIX_COLLECTOR_ENDPOINT` to a Phoenix server's base URL to use another endpoint. Trace export uses that server's `/v1/traces` endpoint, and evaluation queries use the same server. If either `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set, remove it or ensure it matches the Phoenix base URL or its `/v1/traces` endpoint; conflicting settings prevent the app from starting.

## Run the app

From the activated environment and repository root:

```bash
streamlit run app.py
```

1. In the sidebar, select the agent model and both evaluation-judge models. To use GPT-6.1 Terra throughout, choose `gpt-6.1-terra` in all three selectors; the default selections use other models.
2. Choose **Generate the built-in cantilever example** and enter a displacement in metres, or choose **Use an existing input file**, upload or select a self-contained `.inp` file, and select its stress unit.
3. Enter a request and select **Submit**. For the example, you can ask the agent to run the generated model, extract the maximum stress, and compare it with the threshold.
4. Review the extracted MPa value, the assistant's explanation, and any evaluation results. The evaluation buttons assess the selected trace; AI judgments may be wrong and do not independently validate the finite element model.

The selected OpenAI model receives the request and the agent's tool calls and observations. Phoenix records traces locally by default or sends them to the Phoenix endpoint you configure. Consider the content and applicable software-license terms before sending data to an external service.

## Files and results

The app stores input files and generated artifacts under `src/abaqus_files/`, which Git ignores. Each solver run has its own output directory. Stress extraction uses the session's latest verified successful run; a new attempt clears the previous selection. Input files are shared between sessions, so this is not a fully isolated multi-user service. Preserve any results you need before cleaning this directory.

A verified completion requires a successful Abaqus `.sta` status file, an ODB, and no active lock file. Other solver status formats are reported as unverified. On Windows, `.bat` launcher paths and arguments may contain spaces but not shell metacharacters. If the app process crashes during an operation, its session lock may need manual cleanup after confirming that no Abaqus process is still running.

`demo.ipynb` provides a direct Python example of the built-in cantilever workflow. It requires the same Python environment and Abaqus launcher; it does not exercise the AI agent or Phoenix.

## Validation

On 2 November 2026, one end-to-end educational smoke test ran the built-in cantilever at 0.02 m with Abaqus Student Edition 2020 on Windows, GPT-6.1 Terra, and local Phoenix. Abaqus completed the job, and the app displayed approximately **136.533 MPa**. A separate read of the same output database produced **136.533456 MPa**. The test used a temporary launcher alias for the Student Edition; it did not validate every installation, model, or operating system. The second read is a consistency check on the same simulation, not independent FEA validation.

GitHub Actions also runs the local checks on Windows and Linux:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Automated tests simulate Abaqus responses and do not validate a real solver, license, model, or physical result. Validate your own Abaqus installation and models before relying on their results.
