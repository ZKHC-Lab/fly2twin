"""Deterministic tests for the log-capturing worker wrapper."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drone3d_reconstruction_core.worker_runner import (
    SPAWN_FAILED_EXIT,
    main,
    run_worker,
    worker_environment,
)


class FakeProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode

    def wait(self):
        return self.returncode


class WorkerRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def log_path(self, name="run.log"):
        return str(self.root / name)

    def test_success_captures_stdout_and_stderr_and_returns_zero(self):
        command = [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('out-line\\n'); "
            "sys.stderr.write('err-line\\n')",
        ]
        code = run_worker(self.log_path(), command)
        self.assertEqual(code, 0)
        content = Path(self.log_path()).read_bytes()
        self.assertIn(b"drone3d worker wrapper started\n", content)
        self.assertIn(b"out-line\n", content)
        self.assertIn(b"err-line\n", content)

    def test_nonzero_exit_code_is_returned_exactly(self):
        command = [sys.executable, "-c", "import sys; print('bye'); sys.exit(7)"]
        code = run_worker(self.log_path(), command)
        self.assertEqual(code, 7)
        self.assertIn(b"bye\n", Path(self.log_path()).read_bytes())

    def test_spawn_failure_writes_marker_and_returns_127(self):
        def failing_popen(*_args, **_kwargs):
            raise OSError("exec failed")

        code = run_worker(
            self.log_path(),
            [sys.executable, "-c", "pass"],
            popen_factory=failing_popen,
        )
        self.assertEqual(code, SPAWN_FAILED_EXIT)
        content = Path(self.log_path()).read_bytes()
        self.assertIn(b"worker spawn failed: OSError\n", content)

    def test_argv_is_passed_verbatim_without_a_shell(self):
        calls = []

        def recording_popen(argv, **kwargs):
            calls.append((list(argv), dict(kwargs)))
            return FakeProcess(0)

        command = ["program", "literal; rm -rf /", "--flag=value"]
        code = run_worker(
            self.log_path(),
            command,
            popen_factory=recording_popen,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        argv, kwargs = calls[0]
        self.assertEqual(argv, command)
        self.assertIsNot(kwargs.get("shell"), True)
        self.assertIsNotNone(kwargs.get("stdout"))
        self.assertEqual(kwargs.get("stderr"), subprocess.STDOUT)
        self.assertEqual(kwargs.get("start_new_session"), False)

    def test_worker_environment_strips_supervisor_interpreter_leaks(self):
        with mock.patch.dict(
            os.environ,
            {
                "PYTHONPATH": "/bad/pythonpath",
                "PYTHONHOME": "/bad/pythonhome",
                "CONDA_PREFIX": "/bad/conda",
                "CONDA_DEFAULT_ENV": "bad_env",
                "KEEP_ME": "kept",
            },
        ):
            cleaned = worker_environment()
        self.assertNotIn("PYTHONPATH", cleaned)
        self.assertNotIn("PYTHONHOME", cleaned)
        self.assertNotIn("CONDA_PREFIX", cleaned)
        self.assertNotIn("CONDA_DEFAULT_ENV", cleaned)
        self.assertEqual(cleaned["KEEP_ME"], "kept")

    def test_child_process_does_not_inherit_supervisor_pythonpath(self):
        command = [
            sys.executable,
            "-c",
            "import os; print('PYTHONPATH=' + os.environ.get('PYTHONPATH', 'unset')); "
            "print('PYTHONHOME=' + os.environ.get('PYTHONHOME', 'unset'))",
        ]
        with mock.patch.dict(
            os.environ,
            {"PYTHONPATH": "/contaminated", "PYTHONHOME": "/contaminated"},
        ):
            code = run_worker(self.log_path(), command)
        self.assertEqual(code, 0)
        content = Path(self.log_path()).read_bytes()
        self.assertIn(b"PYTHONPATH=unset\n", content)
        self.assertIn(b"PYTHONHOME=unset\n", content)

    def test_log_path_must_be_absolute_and_safe(self):
        for path, error_type in (
            ("relative/run.log", ValueError),
            (str(self.root / "missing" / "run.log"), ValueError),
            (str(self.root / "run.log") + "\x00", ValueError),
        ):
            with self.subTest(path=path):
                with self.assertRaises(error_type):
                    run_worker(path, [sys.executable, "-c", "pass"])

    def test_log_parent_must_be_an_existing_real_directory(self):
        file_parent = self.root / "parent-file"
        file_parent.write_text("not a directory")
        with self.assertRaises(ValueError):
            run_worker(
                str(file_parent / "run.log"),
                [sys.executable, "-c", "pass"],
            )

    def test_existing_log_path_is_never_reused(self):
        existing = self.root / "run.log"
        existing.write_bytes(b"old log")
        with self.assertRaises(FileExistsError):
            run_worker(
                str(existing),
                [sys.executable, "-c", "pass"],
            )
        self.assertEqual(existing.read_bytes(), b"old log")

    def test_symlinked_log_parent_is_rejected(self):
        real = self.root / "real"
        real.mkdir()
        link = self.root / "alias"
        link.symlink_to(real)
        with self.assertRaises(ValueError):
            run_worker(
                str(link / "run.log"),
                [sys.executable, "-c", "pass"],
            )

    def test_symlinked_existing_log_target_is_not_followed(self):
        target = self.root / "target"
        target.write_bytes(b"old")
        link = self.root / "run.log"
        link.symlink_to(target)
        with self.assertRaises(FileExistsError):
            run_worker(
                self.log_path(),
                [sys.executable, "-c", "pass"],
            )
        self.assertEqual(target.read_bytes(), b"old")

    def test_command_validation_rejects_malformed_entries(self):
        invalid_commands = (
            [],
            (),
            ["program", ""],
            ["program", None],
            ["program", "arg\x00"],
            "program",
            (1,),
        )
        for command in invalid_commands:
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    run_worker(self.log_path(), command)

    def test_cli_accepts_separator_and_returns_worker_code(self):
        code = main(
            [
                "--log-path",
                self.log_path(),
                "--",
                sys.executable,
                "-c",
                "import sys; sys.exit(3)",
            ]
        )
        self.assertEqual(code, 3)
        self.assertTrue(Path(self.log_path()).is_file())

    def test_cli_requires_log_path_and_command(self):
        with self.assertRaises(SystemExit) as missing_path:
            main(["--", sys.executable, "-c", "pass"])
        self.assertEqual(missing_path.exception.code, 2)
        with self.assertRaises(SystemExit) as missing_command:
            main(["--log-path", self.log_path(), "--"])
        self.assertEqual(missing_command.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
