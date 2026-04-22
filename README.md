# Finite Element Analysis (FEA) Assisted Agent

## Overview
The **FEA Assisted Agent** automates Abaqus finite element simulations. It uses OpenAI models through **LlamaIndex** and provides a Streamlit UI to execute jobs, extract stresses, and run parametric studies. Phoenix is used for tracing and evaluation.

## Features
1. **User-Provided Input Files**: Upload/select your own Abaqus `.inp` files in the app.
2. **Abaqus Job Executor**: Runs the simulation from the selected `.inp` and gathers outputs.
3. **Von-Mises Stress Extractor**: Parses ODB output and stores peak stress in `max_vm_stress.txt`.
4. **Repeatable Runs**: Reuse different `.inp` files without changing code.
5. **Real-Time Stress Evaluation**: Logs whether stress exceeds `STRESS_THRESHOLD`.

## Environment Variables (`.env`)
This project loads environment variables from a local `.env` file using `python-dotenv`.

Create it from the template:

```bash
cp .env.example .env
```

Then set values:

- `OPENAI_API_KEY` (required): API key used by the agent and eval judges.
- `STRESS_THRESHOLD` (optional, default `360.0`): Stress target in MPa.
- `OTEL_EXPORTER_OTLP_ENDPOINT` (optional): Phoenix/OTel collector endpoint (for example `localhost:4317`).

Example:

```env
OPENAI_API_KEY=sk-...
STRESS_THRESHOLD=360.0
OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
```

## Setup and Run
1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
2. Activate it:
   ```bash
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure `.env` as described above.
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. In the UI, upload/select an `.inp` file, then submit your query.

## Notes
- Streamlit may print telemetry information on startup. To disable usage stats:
  - Add this to `~/.streamlit/config.toml`:
    ```toml
    [browser]
    gatherUsageStats = false
    ```
- Phoenix may log optional warnings (for example Bedrock support). Those do not block normal OpenAI-based runs.
