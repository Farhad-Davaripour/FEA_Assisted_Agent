"""Bind user selections to tool calls without reading Streamlit from workers."""
import re

from src.tools import generate_input_file, run_abaqus, extract_von_mises_stress_from_ODB

EXISTING = "Use an existing input file"
DEMO = "Generate the built-in cantilever example"


def _input_name(requested):
    """Accept a bare name or its exact, relative input-directory equivalent."""
    if not isinstance(requested, str) or not requested:
        return None
    # LLMs may include the input directory using either platform's separator.
    # Never resolve arbitrary paths: require the literal relative prefix and a
    # single remaining filename, with no traversal, absolute, or nested path.
    path = requested.replace("\\", "/")
    if "/" not in path:
        return path if path not in (".", "..") else None
    prefix = "src/abaqus_files/"
    if not path.startswith(prefix):
        return None
    name = path[len(prefix):]
    if not name or "/" in name or name in (".", ".."):
        return None
    return name


class AnalysisToolContext:
    def __init__(self, mode, selected_input, unit, session_id):
        self.mode = mode
        self.selected_input = selected_input
        self.unit = "Pa" if mode == DEMO else unit
        self.session_id = session_id
        self.generated_inputs = []
        self.current_run = None

    def generate(self, applied_displacement: float) -> str:
        """Generate the built-in Pa-based cantilever model."""
        if self.mode != DEMO:
            return "Switch to cantilever mode to generate the built-in model."
        result = generate_input_file(applied_displacement)
        match = re.fullmatch(r"Input file generated: `([^`]+)` \(demo stress unit: Pa\)\.", result)
        if match:
            self.generated_inputs.append(match.group(1))
        return result

    def run(self, inp_file_name: str = None) -> str:
        """Run the selected input, or an input generated in this turn."""
        self.current_run = None
        if self.mode == EXISTING:
            if inp_file_name and _input_name(inp_file_name) != self.selected_input:
                return "The requested input does not match the selected input file."
            inp_file_name = self.selected_input
        else:
            if not inp_file_name and len(self.generated_inputs) == 1:
                inp_file_name = self.generated_inputs[0]
            else:
                inp_file_name = _input_name(inp_file_name)
            if inp_file_name not in self.generated_inputs:
                return "In cantilever mode, run an input generated in this session."
        result = run_abaqus(
            inp_file_name=inp_file_name, session_id=self.session_id, stress_unit=self.unit
        )
        if result.startswith("Abaqus job completed successfully for `"):
            self.current_run = inp_file_name
        return result

    def extract(self, job_name: str = None) -> str:
        """Extract only the current run, using its captured model unit."""
        if not self.current_run:
            return "No verified result for the current model. Run the job first."
        return extract_von_mises_stress_from_ODB(
            job_name=job_name, stress_unit=self.unit, session_id=self.session_id
        )
