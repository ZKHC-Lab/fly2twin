"""Unit tests for the G4 window aligner, without a session."""

import unittest
from unittest import mock

from drone3d_reconstruction_core.continuous_align import G4WindowAligner


def process(alive=True, status=None):
    item = mock.MagicMock()
    item.process.poll.return_value = None if alive else -9
    item.status.return_value = status or {"state": "ready"}
    return item


class G4WindowAlignerTest(unittest.TestCase):
    def setUp(self):
        self.journal = []
        self.factory = mock.MagicMock()
        self.aligner = G4WindowAligner(self.factory)

    def publish(self, index=0):
        return self.aligner.on_window_published("/records", "s" + "0" * 16, index,
                                                lambda: {"placement": "frame_pose"}, self.journal.append)

    def test_alignment_is_pending_until_a_map_revision(self):
        alignment = self.aligner.align()
        self.assertEqual((alignment["method"], alignment["pending"], alignment["reason"]),
                         ("global_g4", True, "waiting_for_G4_map_revision"))

    def test_first_published_window_starts_the_process_once(self):
        self.factory.return_value = process()
        self.assertEqual(self.publish(0), "")
        self.assertEqual(self.publish(1), "")
        self.factory.assert_called_once_with("/records", "s" + "0" * 16, profile={"placement": "frame_pose"})
        self.assertEqual(self.aligner.status(), {"state": "ready"})

    def test_start_failure_is_reported_and_retried_on_the_next_window(self):
        self.factory.side_effect = [OSError("spawn refused"), process()]
        expected = "G4 background start failed: OSError: spawn refused"
        self.assertEqual(self.publish(0), expected)
        self.assertEqual(self.aligner.status(), {"state": "error", "error": expected})
        self.assertEqual([e["event"] for e in self.journal], ["g4_backend_start_failed"])
        self.assertEqual(self.publish(1), "")
        self.assertEqual(self.aligner.status(), {"state": "ready"})

    def test_dead_process_is_journaled_closed_and_replaced(self):
        dead = process(alive=False, status={"state": "error", "exit_code": -9, "error": "killed"})
        self.factory.side_effect = [dead, process()]
        self.publish(0)
        self.publish(1)
        dead.close.assert_called_once()
        self.assertEqual(self.journal, [{"event": "g4_backend_exited", "session_id": "s" + "0" * 16,
                                         "window_index": 1, "exit_code": -9, "error": "killed"}])
        self.assertEqual(self.factory.call_count, 2)

    def test_detach_hands_over_the_process_and_reset_clears_the_error(self):
        running = process()
        self.factory.side_effect = [OSError("x"), running]
        self.publish(0)
        self.aligner.reset()
        self.assertEqual(self.aligner.status(), {})
        self.publish(1)
        self.assertIs(self.aligner.detach(), running)
        self.assertIsNone(self.aligner.map_status())
        self.assertIsNone(self.aligner.detach())


if __name__ == "__main__":
    unittest.main()
