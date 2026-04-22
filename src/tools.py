import os
import subprocess
import shutil
import re
import json
from pathlib import Path

ABAQUS_FILES_DIR = Path("src/abaqus_files")
LAST_JOB_FILE = ABAQUS_FILES_DIR / ".last_job_name"


def _ensure_abaqus_dir():
    ABAQUS_FILES_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_inp_name(inp_file_name):
    if not inp_file_name:
        raise ValueError("`inp_file_name` is required.")
    file_name = Path(inp_file_name).name
    if not file_name.lower().endswith(".inp"):
        raise ValueError("Input file must have `.inp` extension.")
    return file_name


def _resolve_latest_job_name():
    if LAST_JOB_FILE.exists():
        name = LAST_JOB_FILE.read_text(encoding="utf-8").strip()
        if name:
            return name
    odb_files = sorted(
        ABAQUS_FILES_DIR.glob("*.odb"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return odb_files[0].stem if odb_files else None


def run_abaqus(inp_file_name=None):
    """
    Run Abaqus using a user-provided `.inp` file in `src/abaqus_files`.

    Args:
        inp_file_name (str): Name of the input file (e.g. `beam_case.inp`).
    """
    try:
        _ensure_abaqus_dir()
        if not inp_file_name:
            inp_candidates = sorted(p.name for p in ABAQUS_FILES_DIR.glob("*.inp"))
            if len(inp_candidates) == 1:
                inp_file_name = inp_candidates[0]
            else:
                return (
                    "Please provide `inp_file_name` (for example `beam_case.inp`). "
                    "Multiple or no `.inp` files were found."
                )
        inp_name = _normalize_inp_name(inp_file_name)
    except ValueError as exc:
        return f"Invalid input file: {exc}"

    inp_path = ABAQUS_FILES_DIR / inp_name
    if not inp_path.exists():
        return f"Input file not found: {inp_path}"

    job_name = inp_path.stem
    command = f'abaqus job="{job_name}" input="{inp_path}"'
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Abaqus typically emits artifacts in the current working directory.
        for file_path in Path.cwd().iterdir():
            if file_path.is_file() and file_path.name.startswith(job_name):
                target_path = ABAQUS_FILES_DIR / file_path.name
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(file_path), str(target_path))

        LAST_JOB_FILE.write_text(job_name, encoding="utf-8")
        return (
            f"Abaqus job completed successfully for `{inp_name}`. "
            f"Job name: `{job_name}`."
        )
    except subprocess.CalledProcessError as exc:
        return f"Abaqus job failed for `{inp_name}`. Error: {exc.stderr}"


def extract_von_mises_stress_from_ODB(job_name=None):
    """
    Extract max Von-Mises stress from an ODB file produced by Abaqus.

    Args:
        job_name (str, optional): Abaqus job name (without `.odb`).
            If omitted, the last successful run is used.
    """
    _ensure_abaqus_dir()
    resolved_job_name = (job_name or "").strip() or _resolve_latest_job_name()
    if not resolved_job_name:
        return "No ODB found. Run an Abaqus job first or pass `job_name` explicitly."

    odb_path = ABAQUS_FILES_DIR / f"{resolved_job_name}.odb"
    if not odb_path.exists():
        return f"ODB file not found: {odb_path}"

    command = f'abaqus python src/utils/retrieve_vm_stress.py -- "{odb_path}"'
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        output_text = result.stdout.strip()
        match = re.search(r"is\s+([0-9]+(?:\.[0-9]+)?)", output_text)
        if not match:
            return f"Stress extraction completed but output could not be parsed: {output_text}"

        data = round(float(match.group(1)) / 1e6, 2)

        source_file = Path.cwd() / "max_vm_stress.txt"
        if source_file.exists():
            target_file = ABAQUS_FILES_DIR / "max_vm_stress.txt"
            if target_file.exists():
                target_file.unlink()
            shutil.move(str(source_file), str(target_file))

            # Keep a job-specific copy for repeatability across runs.
            job_target = ABAQUS_FILES_DIR / f"max_vm_stress_{resolved_job_name}.txt"
            if job_target.exists():
                job_target.unlink()
            shutil.copy2(str(target_file), str(job_target))

        return f"von_mises stress is {data} MPa"
    except subprocess.CalledProcessError as exc:
        return f"Stress extraction failed for `{resolved_job_name}`. Error: {exc.stderr}"

    
ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
INPUT_RE  = re.compile(r"Action Input:\s*(\{.*\})", re.S)

def extract_action(messages):
    """
    Return (tool_name, json_args) from the first assistant message
    that contains an “Action:” line.
    """
    for m in messages:                                 # each element in output_messages
        msg = m.get("message", m)                      # tolerate either shape
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        m_action = ACTION_RE.search(content)
        if not m_action:
            continue

        name = m_action.group(1)
        m_args = INPUT_RE.search(content)
        args  = m_args.group(1) if m_args else "{}"

        # round-trip through json so the grader sees valid JSON
        try:
            args = json.dumps(json.loads(args))
        except json.JSONDecodeError:
            args = "{}"

        return name, args

    return None, "{}" 
