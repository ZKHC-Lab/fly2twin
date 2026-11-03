#!/usr/bin/env python3
"""Replay a bag through running Capture/Reconstruct Actions and verify results."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import time

import actionlib
import rospy

from drone3d_reconstruction_msgs.msg import (
    CaptureSegmentAction,
    CaptureSegmentGoal,
    ReconstructSegmentAction,
    ReconstructSegmentGoal,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def wait_state(client, expected, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get_state() == expected:
            return True
        time.sleep(0.02)
    return client.get_state() == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--capture-action", default="/capture/capture_segment")
    parser.add_argument("--reconstruct-action",
                        default="/reconstruct/reconstruct_segment")
    parser.add_argument("--scene-id", default="docker20-external-cuda-e2e")
    parser.add_argument("--capture-timeout", type=float, default=240.0)
    parser.add_argument("--reconstruct-timeout", type=float, default=900.0)
    args = parser.parse_args()

    rospy.init_node("docker20_external_cuda_e2e", anonymous=True)
    capture_client = actionlib.SimpleActionClient(
        args.capture_action, CaptureSegmentAction
    )
    reconstruct_client = actionlib.SimpleActionClient(
        args.reconstruct_action, ReconstructSegmentAction
    )
    if not capture_client.wait_for_server(rospy.Duration(60.0)):
        print("CAPTURE_SERVER_TIMEOUT")
        return 1
    if not reconstruct_client.wait_for_server(rospy.Duration(60.0)):
        print("RECONSTRUCT_SERVER_TIMEOUT")
        return 1

    rosbag_command = [
        "rosbag", "play", "--clock", "--pause", args.bag,
    ]
    print("ROSBAG_COMMAND:", " ".join(rosbag_command), flush=True)
    rosbag = subprocess.Popen(
        rosbag_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        capture_goal = CaptureSegmentGoal()
        capture_goal.scene_id = args.scene_id
        capture_goal.sample_fps = 1.0
        capture_goal.max_frames = 8
        capture_goal.max_duration = rospy.Duration.from_sec(40.0)
        capture_client.send_goal(capture_goal)
        if not wait_state(capture_client, actionlib.GoalStatus.ACTIVE, 15.0):
            print("CAPTURE_ACCEPT_TIMEOUT_STATE:", capture_client.get_state())
            return 1
        print("CAPTURE_GOAL_ACTIVE", flush=True)
        rosbag.stdin.write(b" ")
        rosbag.stdin.flush()

        if not capture_client.wait_for_result(
                rospy.Duration(args.capture_timeout)):
            print("CAPTURE_RESULT_TIMEOUT")
            return 1
        capture_state = capture_client.get_state()
        capture_result = capture_client.get_result()
        print("CAPTURE_STATE:", capture_state)
        print("CAPTURE_ERROR:", capture_result.error_code,
              capture_result.message)
        print("CAPTURE_FRAME_COUNT:", capture_result.frame_count)
        print("CAPTURE_SNAPSHOT_URI:", capture_result.snapshot_uri)
        print("CAPTURE_MANIFEST_URI:", capture_result.manifest_uri)
        print("CAPTURE_MANIFEST_SHA256:", capture_result.manifest_sha256)
        print("CAPTURE_HAS_ODOMETRY:", capture_result.has_odometry)
        print("CAPTURE_FIRST_STAMP:", capture_result.first_stamp.to_sec())
        print("CAPTURE_LAST_STAMP:", capture_result.last_stamp.to_sec())

        if capture_state != actionlib.GoalStatus.SUCCEEDED:
            print("CAPTURE_NOT_SUCCEEDED")
            return 1
        if capture_result.error_code != 0:
            print("CAPTURE_ERROR_CODE_NONZERO")
            return 1
        if capture_result.frame_count != 8:
            print("CAPTURE_FRAME_COUNT_MISMATCH")
            return 1
        if not capture_result.has_odometry:
            print("CAPTURE_HAS_ODOMETRY_FALSE")
            return 1
        if not capture_result.snapshot_uri.startswith("file://"):
            print("CAPTURE_SNAPSHOT_URI_NOT_FILE")
            return 1
        snapshot_path = Path(capture_result.snapshot_uri[7:])
        if not str(snapshot_path).startswith(args.snapshot_root):
            print("CAPTURE_SNAPSHOT_OUTSIDE_ALLOWED_ROOT")
            return 1
        if not capture_result.manifest_uri.startswith("file://"):
            print("CAPTURE_MANIFEST_URI_NOT_FILE")
            return 1
        manifest_path = Path(capture_result.manifest_uri[7:])
        if not manifest_path.is_file():
            print("CAPTURE_MANIFEST_MISSING")
            return 1
        actual_manifest_sha256 = sha256_file(manifest_path)
        if actual_manifest_sha256 != capture_result.manifest_sha256:
            print("CAPTURE_MANIFEST_DIGEST_MISMATCH")
            return 1
        if capture_result.last_stamp.to_sec() < capture_result.first_stamp.to_sec():
            print("CAPTURE_STAMP_ROLLBACK")
            return 1
        print("CAPTURE_VALIDATION_OK")

        reconstruct_goal = ReconstructSegmentGoal()
        reconstruct_goal.snapshot_uri = capture_result.snapshot_uri
        reconstruct_goal.manifest_sha256 = capture_result.manifest_sha256
        reconstruct_client.send_goal(reconstruct_goal)
        if not wait_state(reconstruct_client, actionlib.GoalStatus.ACTIVE, 30.0):
            rejected_result = reconstruct_client.get_result()
            print("RECONSTRUCT_ACCEPT_TIMEOUT_STATE:",
                  reconstruct_client.get_state())
            if rejected_result is not None:
                print("RECONSTRUCT_ACCEPT_ERROR:",
                      rejected_result.error_code, rejected_result.message)
            return 1
        print("RECONSTRUCT_GOAL_ACTIVE", flush=True)
        if not reconstruct_client.wait_for_result(
                rospy.Duration(args.reconstruct_timeout)):
            print("RECONSTRUCT_RESULT_TIMEOUT")
            return 1
        reconstruct_state = reconstruct_client.get_state()
        reconstruct_result = reconstruct_client.get_result()
        print("RECONSTRUCT_STATE:", reconstruct_state)
        print("RECONSTRUCT_ERROR:", reconstruct_result.error_code,
              reconstruct_result.message)
        print("RECONSTRUCT_SCENE_ID:", reconstruct_result.scene_id)
        print("RECONSTRUCT_FRAME_COUNT:", reconstruct_result.frame_count)
        print("RECONSTRUCT_RESULT_MANIFEST_URI:",
              reconstruct_result.result_manifest_uri)
        print("RECONSTRUCT_RESULT_MANIFEST_SHA256:",
              reconstruct_result.result_manifest_sha256)
        print("RECONSTRUCT_WALL_TIME:", reconstruct_result.wall_time.to_sec())
        if reconstruct_state != actionlib.GoalStatus.SUCCEEDED:
            print("RECONSTRUCT_NOT_SUCCEEDED")
            return 1
        if reconstruct_result.error_code != 0:
            print("RECONSTRUCT_ERROR_CODE_NONZERO")
            return 1
        if not reconstruct_result.result_manifest_uri.startswith("file://"):
            print("RECONSTRUCT_RESULT_URI_NOT_FILE")
            return 1
        result_manifest_path = Path(
            reconstruct_result.result_manifest_uri[7:]
        )
        if not result_manifest_path.is_file():
            print("RECONSTRUCT_RESULT_MANIFEST_MISSING")
            return 1
        if sha256_file(result_manifest_path) != \
                reconstruct_result.result_manifest_sha256:
            print("RECONSTRUCT_RESULT_MANIFEST_DIGEST_MISMATCH")
            return 1
        print("RECONSTRUCT_VALIDATION_OK")
        print("DOCKER20_EXTERNAL_CUDA_E2E_OK")
        return 0
    finally:
        try:
            rosbag.terminate()
            rosbag.wait(timeout=10.0)
        except BaseException:
            try:
                rosbag.kill()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
