"""Only selected inputs or safe relative aliases may reach the solver."""
import unittest
from unittest.mock import patch

from src.workflow_tools import AnalysisToolContext, DEMO, EXISTING


class InputSelectionTests(unittest.TestCase):
    def setUp(self):
        self.session_id = "a" * 32

    def test_selected_input_accepts_bare_or_exact_relative_directory(self):
        for request in ("beam.inp", "src/abaqus_files/beam.inp", r"src\abaqus_files\beam.inp"):
            with self.subTest(request=request):
                context = AnalysisToolContext(EXISTING, "beam.inp", "MPa", self.session_id)
                with patch("src.workflow_tools.run_abaqus", return_value="Abaqus job completed successfully for `beam.inp`.") as run:
                    context.run(request)
                run.assert_called_once_with(inp_file_name="beam.inp", session_id=self.session_id, stress_unit="MPa")
                self.assertEqual(context.current_run, "beam.inp")

    def test_selected_input_rejects_other_paths_and_traversal(self):
        rejected = ("other.inp", "src/abaqus_files/other.inp", "../beam.inp",
                    "src/abaqus_files/../beam.inp", "src/abaqus_files/subdir/beam.inp",
                    "/src/abaqus_files/beam.inp", r"C:\src\abaqus_files\beam.inp",
                    "./src/abaqus_files/beam.inp", "src//abaqus_files/beam.inp")
        for request in rejected:
            with self.subTest(request=request):
                context = AnalysisToolContext(EXISTING, "beam.inp", "MPa", self.session_id)
                with patch("src.workflow_tools.run_abaqus") as run:
                    self.assertIn("does not match", context.run(request))
                run.assert_not_called()
                self.assertIsNone(context.current_run)

    def test_generated_input_accepts_only_its_own_safe_relative_name(self):
        for request in ("generated.inp", "src/abaqus_files/generated.inp", r"src\abaqus_files\generated.inp"):
            with self.subTest(request=request):
                context = AnalysisToolContext(DEMO, None, None, self.session_id)
                context.generated_inputs = ["generated.inp"]
                with patch("src.workflow_tools.run_abaqus", return_value="Abaqus job completed successfully for `generated.inp`.") as run:
                    context.run(request)
                run.assert_called_once_with(inp_file_name="generated.inp", session_id=self.session_id, stress_unit="Pa")

        for request in ("src/abaqus_files/other.inp", "../generated.inp", "/src/abaqus_files/generated.inp"):
            with self.subTest(request=request):
                context = AnalysisToolContext(DEMO, None, None, self.session_id)
                context.generated_inputs = ["generated.inp"]
                with patch("src.workflow_tools.run_abaqus") as run:
                    self.assertIn("generated in this session", context.run(request))
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
