"""Lock generated constants; MD5 assertions are added after initial generation."""

import unittest

from drone3d_reconstruction_msgs.msg import (
    CaptureSegmentAction,
    CaptureSegmentFeedback,
    CaptureSegmentGoal,
    CaptureSegmentResult,
    ReconstructSegmentAction,
    ReconstructSegmentFeedback,
    ReconstructSegmentGoal,
    ReconstructSegmentResult,
)
from drone3d_reconstruction_msgs.srv import (
    CaptureControl,
    CaptureControlRequest,
    CaptureControlResponse,
    ContinuousSessionCommand,
    ContinuousSessionCommandRequest,
    ContinuousSessionCommandResponse,
    PipelineCommand,
    PipelineCommandRequest,
    PipelineCommandResponse,
)


EXPECTED_MD5 = {
    CaptureSegmentAction: "122bba6881455224ab87e5371f784d8b",
    CaptureSegmentGoal: "af36c671dd34bbd232a36d1d9d2f0625",
    CaptureSegmentResult: "60ec85717f2710d34954d5668027a7c5",
    CaptureSegmentFeedback: "62d925e6c24b68cf5bd89724fb0e81fe",
    ReconstructSegmentAction: "f02c88da475966d7ff877ef7635338cc",
    ReconstructSegmentGoal: "65269bb509de2bd867052bce1728a33f",
    ReconstructSegmentResult: "9c3beda22eeda937c0495922aaec17a8",
    ReconstructSegmentFeedback: "57ba911deaf6eba72f2dd820bdd20a33",
    PipelineCommand: "c0b1c39ddac45a393f2164803eccad9b",
    PipelineCommandRequest: "91080421a73f3a9a0f283d66f5706a84",
    PipelineCommandResponse: "2aa2e19a2b792b8741932ace4a0ef171",
    CaptureControl: "89a70d2248cd7b1a759c7e112937236a",
    CaptureControlRequest: "e6f22d5a8a1380fbfa7bd70180298d3e",
    CaptureControlResponse: "fd3a5d5a3847d5970fc4939b8cd70e63",
    ContinuousSessionCommand: "1731c363cf0331254f94d51459396069",
    ContinuousSessionCommandRequest: "b3ab5e3dcb10854e4ed5885f58a0ff12",
    ContinuousSessionCommandResponse: "c496dbcba13e6c4e3b9b2e60b878b2d3",
}


class MessageContractTest(unittest.TestCase):
    def test_generated_md5_values_are_frozen(self):
        for message_type, expected in EXPECTED_MD5.items():
            self.assertEqual(message_type._md5sum, expected, message_type.__name__)

    def test_capture_result_constants(self):
        self.assertEqual(CaptureSegmentResult.OK, 0)
        self.assertEqual(CaptureSegmentResult.INVALID_GOAL, 1)
        self.assertEqual(CaptureSegmentResult.CAMERA_UNAVAILABLE, 2)
        self.assertEqual(CaptureSegmentResult.IO_ERROR, 3)
        self.assertEqual(CaptureSegmentResult.CANCELLED, 4)
        self.assertEqual(CaptureSegmentResult.INTERNAL_ERROR, 5)

    def test_reconstruct_result_constants(self):
        self.assertEqual(ReconstructSegmentResult.OK, 0)
        self.assertEqual(ReconstructSegmentResult.INVALID_GOAL, 1)
        self.assertEqual(ReconstructSegmentResult.INPUT_NOT_FOUND, 2)
        self.assertEqual(ReconstructSegmentResult.BUSY, 3)
        self.assertEqual(ReconstructSegmentResult.ALGORITHM_FAILED, 4)
        self.assertEqual(ReconstructSegmentResult.IO_ERROR, 5)
        self.assertEqual(ReconstructSegmentResult.CANCELLED, 6)
        self.assertEqual(ReconstructSegmentResult.INTERNAL_ERROR, 7)

    def test_pipeline_command_constants(self):
        self.assertEqual(PipelineCommandRequest.SELECT_SOURCE, 0)
        self.assertEqual(PipelineCommandRequest.START_TASK, 1)
        self.assertEqual(PipelineCommandRequest.GET_STATUS, 2)
        self.assertEqual(PipelineCommandRequest.CANCEL_TASK, 3)
        self.assertEqual(PipelineCommandResponse.OK, 0)
        self.assertEqual(PipelineCommandResponse.INVALID_COMMAND, 1)
        self.assertEqual(PipelineCommandResponse.UNKNOWN_ROLE, 2)
        self.assertEqual(PipelineCommandResponse.BUSY, 3)
        self.assertEqual(PipelineCommandResponse.NO_TASK, 4)
        self.assertEqual(PipelineCommandResponse.TASK_FAILED, 5)
        self.assertEqual(PipelineCommandResponse.INTERNAL_ERROR, 6)
        self.assertEqual(PipelineCommandResponse.NOT_READY, 7)
        self.assertEqual(PipelineCommandResponse.INVALID_ARGUMENT, 8)
    def test_capture_control_constants(self):
        self.assertEqual(CaptureControlRequest.PAUSE, 0)
        self.assertEqual(CaptureControlRequest.RESUME, 1)
        self.assertEqual(CaptureControlResponse.OK, 0)
        self.assertEqual(CaptureControlResponse.INVALID_COMMAND, 1)
        self.assertEqual(CaptureControlResponse.NO_ACTIVE_SESSION, 2)
        self.assertEqual(CaptureControlResponse.INTERNAL_ERROR, 3)

    def test_continuous_session_command_constants(self):
        self.assertEqual(ContinuousSessionCommandRequest.SELECT_SOURCE, 0)
        self.assertEqual(ContinuousSessionCommandRequest.START_SESSION, 1)
        self.assertEqual(ContinuousSessionCommandRequest.PAUSE, 2)
        self.assertEqual(ContinuousSessionCommandRequest.RESUME, 3)
        self.assertEqual(ContinuousSessionCommandRequest.STOP_ALL, 4)
        self.assertEqual(ContinuousSessionCommandRequest.GET_STATUS, 5)
        self.assertEqual(ContinuousSessionCommandRequest.SET_PARAMS, 6)
        self.assertEqual(ContinuousSessionCommandResponse.OK, 0)
        self.assertEqual(ContinuousSessionCommandResponse.INVALID_COMMAND, 1)
        self.assertEqual(ContinuousSessionCommandResponse.UNKNOWN_ROLE, 2)
        self.assertEqual(ContinuousSessionCommandResponse.BUSY, 3)
        self.assertEqual(ContinuousSessionCommandResponse.NOT_RUNNING, 4)
        self.assertEqual(ContinuousSessionCommandResponse.INVALID_ARGUMENT, 5)
        self.assertEqual(ContinuousSessionCommandResponse.INTERNAL_ERROR, 6)


if __name__ == "__main__":
    unittest.main()
