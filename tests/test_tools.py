import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import tools


class AbaqusToolTests(unittest.TestCase):
    def setUp(self):
        self.which = patch.object(tools.shutil, "which", return_value="abaqus")
        self.which.start()
        self.addCleanup(self.which.stop)
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        tools._ensure_abaqus_dir()
        self.folder = tools.ABAQUS_FILES_DIR
        (self.folder / "beam.inp").write_text("input")

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def _success(self, args, cwd, **kwargs):
        self.assertFalse(kwargs["shell"])
        self.assertIn("interactive", args)
        job = next(arg[4:] for arg in args if arg.startswith("job="))
        Path(cwd, f"{job}.odb").write_text("new odb")
        Path(cwd, f"{job}.sta").write_text("THE ANALYSIS HAS COMPLETED SUCCESSFULLY")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def test_completed_run_is_isolated_and_never_overwrites_unrelated_files(self):
        Path("beam_notes.txt").write_text("outside")
        (self.folder / "beam_notes.txt").write_text("existing")
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            self.assertIn("completed successfully", tools.run_abaqus("beam.inp", stress_unit="Pa"))
        self.assertEqual(Path("beam_notes.txt").read_text(), "outside")
        self.assertEqual((self.folder / "beam_notes.txt").read_text(), "existing")
        self.assertEqual(len(list((self.folder / "runs").glob("*/beam.odb"))), 1)

    def test_failed_run_clears_previous_result(self):
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            tools.run_abaqus("beam.inp", stress_unit="Pa")
        with patch.object(tools.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "abaqus", stderr="solver error")):
            self.assertIn("failed", tools.run_abaqus("beam.inp", stress_unit="Pa"))
        self.assertIn("No verified current result", tools.extract_von_mises_stress_from_ODB(stress_unit="Pa"))

    def test_partial_odb_does_not_count_as_success(self):
        def incomplete(args, cwd, **kwargs):
            Path(cwd, "beam.odb").write_text("partial")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        with patch.object(tools.subprocess, "run", side_effect=incomplete):
            self.assertIn("did not produce a verified", tools.run_abaqus("beam.inp", stress_unit="Pa"))
        self.assertFalse(tools.LAST_JOB_FILE.exists())

    def test_missing_input_and_unsafe_name_are_rejected(self):
        self.assertIn("not found", tools.run_abaqus("missing.inp", stress_unit="Pa"))
        self.assertIn("Invalid input", tools.run_abaqus("../beam.inp", stress_unit="Pa"))
        self.assertIn("Invalid input", tools.run_abaqus("bad&name.inp", stress_unit="Pa"))

    def test_relative_include_is_rejected_without_launching_abaqus(self):
        (self.folder / "beam.inp").write_text("*Heading\n*Include, input=parts.inp\n")
        with patch.object(tools.subprocess, "run") as run:
            self.assertIn("*INCLUDE are not supported", tools.run_abaqus("beam.inp", stress_unit="Pa"))
            run.assert_not_called()

    def test_input_with_spaces_is_passed_as_one_argument(self):
        (self.folder / "beam case.inp").write_text("input")
        with patch.object(tools.subprocess, "run", side_effect=self._success) as run:
            self.assertIn("completed successfully", tools.run_abaqus("beam case.inp", stress_unit="Pa"))
        args = run.call_args.args[0]
        self.assertTrue(any(arg.startswith("input=") and "beam case.inp" in arg for arg in args))

    def test_extraction_requires_explicit_units_and_converts_pa(self):
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            tools.run_abaqus("beam.inp", stress_unit="Pa")
        self.assertIn("Select the model's stress unit", tools.extract_von_mises_stress_from_ODB())
        output = subprocess.CompletedProcess([], 0, stdout="Maximum Mises stress in the model is 360000000.000000", stderr="")
        with patch.object(tools.subprocess, "run", return_value=output):
            self.assertEqual(tools.extract_von_mises_stress_from_ODB(stress_unit="Pa"), "von_mises stress is 360 MPa")

    def test_extraction_preserves_mpa_and_uses_small_values_without_rounding_to_zero(self):
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            tools.run_abaqus("beam.inp", stress_unit="Pa")
        output = subprocess.CompletedProcess([], 0, stdout="Maximum Mises stress in the model is 500", stderr="")
        with patch.object(tools.subprocess, "run", return_value=output):
            self.assertEqual(tools.extract_von_mises_stress_from_ODB(stress_unit="Pa"), "von_mises stress is 0.0005 MPa")
            self.assertIn("differs from this run", tools.extract_von_mises_stress_from_ODB(stress_unit="MPa"))
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            tools.run_abaqus("beam.inp", stress_unit="MPa")
        with patch.object(tools.subprocess, "run", return_value=output):
            self.assertEqual(tools.extract_von_mises_stress_from_ODB(stress_unit="MPa"), "von_mises stress is 500 MPa")

    def test_locked_or_incomplete_odb_cannot_be_extracted(self):
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            tools.run_abaqus("beam.inp", stress_unit="Pa")
        odb = next((self.folder / "runs").glob("*/beam.odb"))
        odb.with_suffix(".lck").write_text("locked")
        self.assertIn("no longer", tools.extract_von_mises_stress_from_ODB(stress_unit="Pa"))

    def test_demo_generator_preserves_existing_files(self):
        def generate(args, cwd, **kwargs):
            Path(cwd, "cantilever_beam.inp").write_text("generated")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        with patch.object(tools.subprocess, "run", side_effect=generate):
            self.assertIn("generated", tools.generate_input_file(0.02))
            self.assertIn("generated", tools.generate_input_file(0.02))
        self.assertEqual(len(list(self.folder.glob("cantilever_beam_*.inp"))), 2)
        self.assertEqual((self.folder / "beam.inp").read_text(), "input")

    def test_demo_generator_rejects_out_of_range_or_nonfinite_inputs(self):
        for displacement in (-1, 0.21, float("nan"), float("inf"), "nope"):
            with self.subTest(displacement=displacement):
                self.assertIn("Invalid displacement", tools.generate_input_file(displacement))

    def test_generated_demo_rejects_wrong_stress_units(self):
        def generate(args, cwd, **kwargs):
            Path(cwd, "cantilever_beam.inp").write_text("generated")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        with patch.object(tools.subprocess, "run", side_effect=generate):
            tools.generate_input_file(0.02)
        input_file = next(self.folder.glob("cantilever_beam_*.inp"))
        self.assertIn("uses Pa", tools.run_abaqus(input_file.name, stress_unit="MPa"))
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            self.assertIn("completed successfully", tools.run_abaqus(input_file.name, stress_unit="Pa"))
        self.assertIn("uses Pa", tools.extract_von_mises_stress_from_ODB(stress_unit="MPa"))
        input_file.write_text("changed contents")
        with patch.object(tools.subprocess, "run") as run:
            self.assertIn("unit metadata changed", tools.run_abaqus(input_file.name, stress_unit="Pa"))
            run.assert_not_called()

    def test_sessions_do_not_share_latest_result(self):
        session_a = "a" * 32
        session_b = "b" * 32
        with patch.object(tools.subprocess, "run", side_effect=self._success):
            tools.run_abaqus("beam.inp", session_id=session_a, stress_unit="Pa")
        self.assertIn("No verified current result", tools.extract_von_mises_stress_from_ODB(stress_unit="Pa", session_id=session_b))
        output = subprocess.CompletedProcess([], 0, stdout="Maximum Mises stress in the model is 500", stderr="")
        with patch.object(tools.subprocess, "run", return_value=output):
            self.assertIn("0.0005 MPa", tools.extract_von_mises_stress_from_ODB(stress_unit="Pa", session_id=session_a))

    def test_run_requires_declared_stress_unit(self):
        with patch.object(tools.subprocess, "run") as run:
            self.assertIn("Select the model's stress unit", tools.run_abaqus("beam.inp"))
            run.assert_not_called()

    def test_session_lock_prevents_overlapping_operations(self):
        session = "a" * 32
        with tools._session_state(session):
            self.assertIn("already running", tools.run_abaqus("beam.inp", session_id=session, stress_unit="Pa"))
            self.assertIn("already running", tools.extract_von_mises_stress_from_ODB(stress_unit="Pa", session_id=session))


@unittest.skipUnless(os.name == "nt", "Windows batch launcher behavior")
class WindowsLauncherTests(unittest.TestCase):
    def test_path_only_batch_launcher_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="fea launcher ") as temp:
            path = Path(temp)
            (path / "abaqus.bat").write_text("@echo off\r\necho %*\r\n", encoding="ascii")
            with patch.dict(os.environ, {"PATH": str(path) + os.pathsep + os.environ["PATH"]}):
                result = tools._run_command(["abaqus", "job=beam", "input=C:\\folder with spaces\\beam.inp", "interactive"], temp)
            self.assertIn("folder with spaces", result.stdout)
            self.assertIn("interactive", result.stdout)

    def test_batch_launcher_rejects_shell_metacharacters(self):
        with patch.object(tools.shutil, "which", return_value="C:\\safe\\abaqus.bat"):
            with self.assertRaisesRegex(ValueError, "shell metacharacters"):
                tools._run_command(["abaqus", "input=C:\\bad&folder\\beam.inp"], ".")


if __name__ == "__main__":
    unittest.main()
