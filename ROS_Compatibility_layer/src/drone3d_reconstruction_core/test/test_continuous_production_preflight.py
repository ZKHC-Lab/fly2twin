"""CPU-only tests for the unified production preflight script."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

def _load_preflight_module():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" /
        "continuous_production_preflight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "continuous_production_preflight_under_test", str(script)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContinuousProductionPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile = Path(self.temp.name) / "reconstruct_profile.yaml"
        self.profile.write_text(
            "worker_python: python3\n"
            "checkpoint_path: /tmp/checkpoint.pt\n"
            'checkpoint_sha256: "0000000000000000000000000000000000000000000000000000000000000000"\n',
            encoding="utf-8",
        )
        self.module = _load_preflight_module()

    def test_no_master_exits_nonzero_before_worker_preflight(self):
        with mock.patch.object(
            self.module, "_master_pid", side_effect=RuntimeError("no master")
        ), mock.patch.object(
            self.module, "_run_worker_preflight"
        ) as worker_mock:
            code = self.module.main(["--reconstruct-profile", str(self.profile)])
        self.assertEqual(code, 2)
        worker_mock.assert_not_called()

    def test_bad_profile_exits_nonzero(self):
        self.profile.write_text("{not valid", encoding="utf-8")
        code = self.module.main(["--reconstruct-profile", str(self.profile)])
        self.assertEqual(code, 2)

    def test_worker_preflight_failure_propagates(self):
        with mock.patch.object(
            self.module, "_master_pid", return_value=123
        ), mock.patch.object(
            self.module,
            "_run_worker_preflight",
            side_effect=RuntimeError("CUDA missing"),
        ):
            code = self.module.main(["--reconstruct-profile", str(self.profile)])
        self.assertEqual(code, 2)

    def test_ok_path_returns_zero(self):
        with mock.patch.object(
            self.module, "_master_pid", return_value=123
        ), mock.patch.object(
            self.module, "_run_worker_preflight", return_value=None
        ):
            code = self.module.main(["--reconstruct-profile", str(self.profile)])
        self.assertEqual(code, 0)

    def test_module_candidates_find_source_layout_file(self):
        path = self.module._cuda_preflight_module_path()
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "cuda_preflight.py")

    def test_g4_dependencies_fail_before_cuda_or_model_load(self):
        continuous = Path(self.temp.name) / "continuous.yaml"
        continuous.write_text("alignment_backend: global_g4\n")
        with mock.patch.object(self.module, "_master_pid", return_value=123), \
             mock.patch.object(self.module, "_check_g4_dependencies", side_effect=RuntimeError("old scipy")), \
             mock.patch.object(self.module, "_run_worker_preflight") as worker:
            result = self.module.main(["--reconstruct-profile", str(self.profile),
                                       "--continuous-profile", str(continuous)])
        self.assertEqual(result, 2)
        worker.assert_not_called()

    def test_supported_ros_dependencies_and_old_scipy(self):
        self.module._check_g4_dependencies()
        with mock.patch("scipy.__version__", "1.3.3"):
            with self.assertRaisesRegex(RuntimeError, "G4 requires scipy"):
                self.module._check_g4_dependencies()

    def test_cuda_probe_uses_worker_environment_not_ros_pythonpath(self):
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/ros/python38", "PYTHONHOME": "/wrong", "LD_LIBRARY_PATH": "/cuda"}), \
             mock.patch.object(self.module.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            self.module._run_worker_preflight({"worker_python": "/conda/python310"}, Path("/probe.py"))
        env = run.call_args.kwargs["env"]
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["LD_LIBRARY_PATH"], "/cuda")


if __name__ == "__main__":
    unittest.main()
