"""CPU-only tests for the Reconstruct Action client adapter."""

from pathlib import Path
import tempfile
import unittest

from drone3d_reconstruction_core.reconstruct_action_client import read_result_identity


class ReadResultIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_read_result_identity_reads_local_file_and_rejects_missing(self):
        result = self.root / "result.json"
        result.write_text(
            '{"scene_id":"scene-front-01","actual_frame_count":2,'
            '"camera_role":"front","camera_topic":"/front/image_raw",'
            '"camera_frame_id":"front_camera_optical_frame",'
            '"snapshot":{"sha256":"' + "a" * 64 + '"}}',
            encoding="utf-8",
        )
        identity = read_result_identity(result.as_uri())
        self.assertEqual(identity["scene_id"], "scene-front-01")
        self.assertEqual(identity["camera_role"], "front")
        self.assertEqual(identity["snapshot_sha256"], "a" * 64)
        self.assertEqual(
            read_result_identity(str(self.root / "missing.json")), {}
        )


if __name__ == "__main__":
    unittest.main()
