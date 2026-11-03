"""CPU-only tests for production CUDA/checkpoint preflight."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from drone3d_reconstruction_core.cuda_preflight import preflight


class FakeCuda:
    def __init__(self, *, available=True, free_mib=8192.0):
        self._available = available
        self._free_mib = free_mib

    def is_available(self):
        return self._available

    def device_count(self):
        return 1 if self._available else 0

    def get_device_name(self, index):
        return "fake-jetpack-cuda"

    def mem_get_info(self):
        return (int(self._free_mib * 1024 * 1024), 16 * 1024 * 1024 * 1024)


class CudaPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "model.pt"
        self.checkpoint.write_bytes(b"checkpoint-bytes")
        self.digest = hashlib.sha256(b"checkpoint-bytes").hexdigest()

    def fake_torch(self, **kwargs):
        return SimpleNamespace(cuda=FakeCuda(**kwargs))

    def test_no_cuda_returns_nonzero(self):
        code = preflight(
            checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=self.digest,
            _torch=self.fake_torch(available=False),
        )
        self.assertEqual(code, 2)

    def test_missing_checkpoint_returns_nonzero(self):
        code = preflight(
            checkpoint_path=str(self.checkpoint) + ".missing",
            checkpoint_sha256=self.digest,
            _torch=self.fake_torch(),
        )
        self.assertEqual(code, 2)

    def test_checkpoint_digest_mismatch_returns_nonzero(self):
        code = preflight(
            checkpoint_path=str(self.checkpoint),
            checkpoint_sha256="0" * 64,
            _torch=self.fake_torch(),
        )
        self.assertEqual(code, 2)

    def test_valid_cuda_and_checkpoint_return_zero(self):
        code = preflight(
            checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=self.digest,
            _torch=self.fake_torch(),
        )
        self.assertEqual(code, 0)

    def test_low_vram_is_warning_only(self):
        code = preflight(
            checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=self.digest,
            min_free_mib=4096.0,
            _torch=self.fake_torch(free_mib=2048.0),
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
