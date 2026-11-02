# FEA Assisted Agent

This experimental assistant uses Abaqus to run finite element analyses, extract maximum von Mises stress, and explain the results. A Streamlit interface accepts a question; a LlamaIndex agent selects the tools; Phoenix provides tracing. Abaqus performs the simulation. An AI-generated explanation is not a substitute for engineering review.

Phoenix traces and evaluations are scoped to the current analysis. The app displays the extracted stress, and its default request asks the assistant to compare it with the configured threshold. Evaluation judges use the configured OpenAI key and can incur additional API charges.

You can supply your own Abaqus `.inp` file. The original cantilever-pipe input generator remains available as a demonstration: it varies the applied displacement of a fixed example model, rather than parameterizing arbitrary input files.

## Requirements

- Python 3.12 and the versions in `requirements.txt`.
- A working Abaqus installation and license, with the `abaqus` command available on the path used to launch the app.
- An OpenAI API key for the assistant and evaluation features. API usage can incur charges.
- A self-contained UTF-8 Abaqus `.inp` file, or the built-in cantilever example. Files using `*INCLUDE` are currently rejected because their referenced files are not yet supported.

Abaqus has no built-in system of units. Select the stress unit used by your model explicitly; the app converts supported units to MPa for display and threshold comparisons. Check that your model, its inputs, and the unit selection are consistent before interpreting results.

## Setup

Create a virtual environment, install the dependencies, and create a local `.env` file from the template. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, use `python3.12 -m venv .venv`, activate with `source .venv/bin/activate`, and copy the template with `cp .env.example .env`. Edit `.env` locally and set `OPENAI_API_KEY`. Never commit the file. `STRESS_THRESHOLD` is optional and expressed in MPa; its default is 360.

Phoenix defaults to `http://127.0.0.1:6006`. To use another Phoenix server, set `PHOENIX_COLLECTOR_ENDPOINT` to its base URL. The app sends traces to its `/v1/traces` endpoint and reads evaluations from the same server. If `OTEL_EXPORTER_OTLP_ENDPOINT` is already set, remove it or ensure it matches that HTTP trace endpoint to avoid sending traces and queries to different servers.

Start the app from the repository root:

```bash
streamlit run app.py
```

Select or upload an input file, choose its stress unit, and submit a request. Inputs and generated artifacts live under `src/abaqus_files/`, which is ignored by Git. Each solver run has its own output directory. Stress extraction only uses the session's latest verified successful run; a new attempt clears the previous selection. Input files are shared between sessions, so this is not a fully isolated multi-user service. Do not delete results you wish to preserve.

The completion check requires a successful Abaqus `.sta` status file, an ODB, and no active lock file. Other solver status formats are reported as unverified. On Windows, the `.bat` launcher requires paths and arguments without shell metacharacters; spaces are supported. If the app process crashes during an operation, its session lock may need manual cleanup after confirming the Abaqus process is no longer running.

## Local checks

GitHub Actions runs these checks on Windows and Linux for each pull request.

```bash
python -m unittest discover -s tests -v
python -m pip check
```

The automated tests use simulated Abaqus responses and do not validate a real solver, license, model, or physical result. A real Abaqus smoke test is still required on a licensed machine for the intended input files and operating system.
