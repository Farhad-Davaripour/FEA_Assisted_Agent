import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

ABAQUS_FILES_DIR = Path("src/abaqus_files")
LAST_JOB_FILE = ABAQUS_FILES_DIR / ".last_job.json"
SCRIPT_DIR = Path(__file__).resolve().parent / "utils"


def _ensure_abaqus_dir():
    ABAQUS_FILES_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_inp_name(inp_file_name):
    # No paths or command metacharacters, including on Windows. Spaces are OK.
    if not isinstance(inp_file_name, str) or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_ .-]*\.inp", inp_file_name, re.IGNORECASE
    ):
        raise ValueError("Use a simple .inp filename containing letters, numbers, spaces, '.', '-' or '_'.")
    return inp_file_name


def _run_command(arguments, cwd):
    # Arguments are separate, avoiding interpolation of user inputs into a shell.
    # On Windows CreateProcess does not resolve bare names to .bat via PATHEXT.
    # Resolving first also makes a missing Abaqus installation explicit.
    executable = shutil.which(arguments[0])
    if not executable:
        raise FileNotFoundError(f"{arguments[0]} was not found on PATH")
    # Windows may invoke .bat/.cmd files via a shell even with shell=False.
    # Reject metacharacters in paths and arguments instead of passing them to
    # the batch interpreter. Spaces are supported.
    if os.name == "nt" and executable.lower().endswith((".bat", ".cmd")):
        if any(re.search(r'[&|<>^%!()"\r\n]', str(arg)) for arg in [executable, *arguments[1:]]):
            raise ValueError("Abaqus batch launcher paths and arguments cannot contain shell metacharacters.")
    return subprocess.run([executable, *arguments[1:]], cwd=str(cwd), shell=False, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _command_error(exc):
    return str(getattr(exc, "stderr", None) or exc).strip()


def generate_input_file(applied_displacement):
    """Generate the original steel cantilever demo input at a displacement in metres.

    Returns the new unique input filename. The demo assumes SI units (Pa for stress).
    """
    try:
        displacement = float(applied_displacement)
        if not math.isfinite(displacement) or not 0 <= displacement <= 0.2:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        return "Invalid displacement: enter a finite value between 0 and 0.2 metres."

    _ensure_abaqus_dir()
    try:
        # Generation happens outside the input and output folders: never collect
        # unrelated files from the user's working directory.
        with tempfile.TemporaryDirectory(prefix="fea-generate-") as work:
            _run_command(["abaqus", "cae", f"noGUI={SCRIPT_DIR / 'create_inp_file.py'}",
                          "--", str(displacement)], work)
            generated = Path(work) / "cantilever_beam.inp"
            if not generated.is_file():
                return "Input generation failed: Abaqus did not produce cantilever_beam.inp."
            target = ABAQUS_FILES_DIR / f"cantilever_beam_{uuid.uuid4().hex[:12]}.inp"
            shutil.copyfile(generated, target)
            metadata = {"kind": "cantilever-demo", "stress_unit": "Pa",
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            target.with_suffix(target.suffix + ".fea-metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            return f"Input file generated: `{target.name}` (demo stress unit: Pa)."
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"Input generation failed: {_command_error(exc)}"


@contextmanager
def _session_state(session_id):
    if session_id is None:
        folder = ABAQUS_FILES_DIR
    else:
        if not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("Invalid session identifier.")
        folder = ABAQUS_FILES_DIR / "sessions" / session_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".operation.lock"
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        yield folder / ".last_job.json"
    finally:
        lock.unlink(missing_ok=True)


def _load_last_job(marker):
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        run_id, name = data["run_id"], data["job_name"]
        if not re.fullmatch(r"[0-9a-f]{32}", run_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _completed_run(run_dir, job_name):
    odb = run_dir / f"{job_name}.odb"
    status = run_dir / f"{job_name}.sta"
    if not odb.is_file() or not status.is_file() or (run_dir / f"{job_name}.lck").exists():
        return False
    try:
        status_text = status.read_text(encoding="utf-8", errors="replace").upper()
    except OSError:
        return False
    return "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in status_text


def run_abaqus(inp_file_name=None, session_id=None, stress_unit=None):
    """Run a job in isolation, recording its declared stress unit and session."""
    _ensure_abaqus_dir()
    try:
        with _session_state(session_id) as marker:
            return _run_abaqus_locked(inp_file_name, marker, stress_unit)
    except FileExistsError:
        return "Another operation is already running in this session."
    except (OSError, ValueError) as exc:
        return f"Abaqus job failed: {_command_error(exc)}"


def _run_abaqus_locked(inp_file_name, marker, stress_unit):
    # Even a failed new attempt must not leave an old result available as latest.
    marker.unlink(missing_ok=True)
    try:
        declared_unit = str(stress_unit or "").strip().lower()
        if declared_unit not in ("pa", "mpa"):
            return "Select the model's stress unit (Pa or MPa) before running Abaqus."
        if not inp_file_name:
            candidates = sorted(p.name for p in ABAQUS_FILES_DIR.iterdir()
                                if p.is_file() and p.suffix.lower() == ".inp")
            if len(candidates) != 1:
                return "Please provide `inp_file_name`: multiple or no .inp files were found."
            inp_file_name = candidates[0]
        inp_name = _normalize_inp_name(inp_file_name)
        inp_path = ABAQUS_FILES_DIR / inp_name
        if not inp_path.is_file():
            return f"Input file not found: {inp_path}"
        # Relative include files would be resolved against our isolated run
        # directory. Fail explicitly rather than run a different/incomplete model.
        try:
            input_bytes = inp_path.read_bytes()
            input_text = input_bytes.decode("utf-8-sig")
        except UnicodeError:
            return "Input file must be UTF-8 text so include references can be checked."
        if re.search(r"^\s*\*INCLUDE\s*(?:,|$)", input_text, re.IGNORECASE | re.MULTILINE):
            return "Input files using *INCLUDE are not supported yet; use a self-contained .inp file."

        known_unit = None
        sidecar = inp_path.with_suffix(inp_path.suffix + ".fea-metadata.json")
        if sidecar.exists():
            try:
                source_meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if (source_meta.get("kind") != "cantilever-demo" or source_meta.get("stress_unit") != "Pa"
                        or source_meta.get("sha256") != hashlib.sha256(input_bytes).hexdigest()):
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                return "Generated input or its unit metadata changed; regenerate it or use a new filename."
            known_unit = "Pa"
            if declared_unit != "pa":
                return "This generated cantilever model uses Pa stress units; select Pa before running."

        job_name = re.sub(r"[^A-Za-z0-9_-]", "_", inp_path.stem)[:60] or "job"
        run_id = uuid.uuid4().hex
        run_dir = (ABAQUS_FILES_DIR / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        # Snapshot the validated bytes so another upload cannot change the
        # model after unit metadata and include references were checked.
        run_input = run_dir / inp_name
        run_input.write_bytes(input_bytes)
        _run_command(["abaqus", f"job={job_name}", f"input={run_input}", "interactive"], run_dir)
        if not _completed_run(run_dir, job_name):
            return (f"Abaqus job did not produce a verified completed result for `{inp_name}`. "
                    "Check the solver status and output files.")

        data = {"run_id": run_id, "job_name": job_name,
                "known_stress_unit": known_unit, "stress_unit": declared_unit}
        tmp = marker.parent / f".last_job.{run_id}.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(marker)
        return f"Abaqus job completed successfully for `{inp_name}`. Job name: `{job_name}`."
    except ValueError as exc:
        return f"Invalid input file: {exc}"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Abaqus job failed: {_command_error(exc)}"


def extract_von_mises_stress_from_ODB(job_name=None, stress_unit=None, session_id=None):
    """Extract the latest verified run's stress; provide model unit 'Pa' or 'MPa'."""
    try:
        with _session_state(session_id) as marker:
            return _extract_stress_locked(job_name, stress_unit, marker)
    except FileExistsError:
        return "Another operation is already running in this session."
    except (OSError, ValueError) as exc:
        return f"Stress extraction failed: {_command_error(exc)}"


def _extract_stress_locked(job_name, stress_unit, marker):
    unit = str(stress_unit or "").strip().lower()
    if unit not in ("pa", "mpa"):
        return "Select the model's stress unit (Pa or MPa) before extracting stress."
    data = _load_last_job(marker)
    if not data:
        return "No verified current result. Run an Abaqus job successfully first."
    if data.get("stress_unit") not in ("pa", "mpa"):
        return "The verified run has no recorded stress unit; run the job again."
    if data.get("known_stress_unit") == "Pa" and unit != "pa":
        return "This generated cantilever model uses Pa stress units; select Pa before extracting."
    if unit != data["stress_unit"]:
        return "The selected stress unit differs from this run's recorded unit; run the job again."
    if job_name and str(job_name).strip() != data["job_name"]:
        return "Requested job is not the latest verified run; run it again before extracting stress."
    run_dir = (ABAQUS_FILES_DIR / "runs" / data["run_id"]).resolve()
    if not _completed_run(run_dir, data["job_name"]):
        return "The latest run no longer has a verified completed ODB and status file."
    odb_path = run_dir / f"{data['job_name']}.odb"
    try:
        # The extraction utility writes a generic filename. Run it in its own
        # directory so it cannot overwrite or reuse another run's output.
        with tempfile.TemporaryDirectory(prefix="stress-", dir=run_dir) as work:
            result = _run_command(["abaqus", "python", str(SCRIPT_DIR / "retrieve_vm_stress.py"),
                                   "--", str(odb_path)], work)
            match = re.search(r"Maximum Mises stress in the model is\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", result.stdout)
            if not match:
                return "Stress extraction completed but its output could not be parsed."
            value = float(match.group(1))
            if not math.isfinite(value) or value < 0:
                return "Stress extraction returned an invalid stress value."
            mpa = value / 1e6 if unit == "pa" else value
            return f"von_mises stress is {mpa:.8g} MPa"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"Stress extraction failed for `{data['job_name']}`: {_command_error(exc)}"


ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.S)


def extract_action(messages):
    """Return (tool_name, json_args) from the first assistant action message."""
    for m in messages:
        msg = m.get("message", m)
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        m_action = ACTION_RE.search(content)
        if not m_action:
            continue
        m_args = INPUT_RE.search(content)
        try:
            args = json.dumps(json.loads(m_args.group(1) if m_args else "{}"))
        except json.JSONDecodeError:
            args = "{}"
        return m_action.group(1), args
    return None, "{}"
