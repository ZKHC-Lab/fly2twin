"""CPU-only tests for the capture core boundary."""

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from drone3d_reconstruction_core import (
    FrameStamp,
    SnapshotSafetyLimits,
    validate_snapshot,
)
from drone3d_reconstruction_core.capture import (
    CAMERA_UNAVAILABLE,
    INVALID_GOAL,
    IO_ERROR,
    CaptureConfig,
    CaptureConfigError,
    CaptureError,
    CaptureSession,
    CaptureStreamError,
    ImageCodec,
    OdometryAssociator,
    SnapshotWriter,
    StampSampler,
    _odometry_payload,
    allocate_snapshot_transaction,
    validate_capture_goal,
)


class CaptureTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshots_root = self.root / "snapshots"
        self.snapshots_root.mkdir()
        self.transactions = []

    def tearDown(self):
        for transaction in reversed(self.transactions):
            try:
                transaction.cleanup()
            except BaseException:
                pass

    def make_config(self, **overrides):
        values = dict(
            snapshot_root=str(self.snapshots_root),
            image_topic="/camera4/image_raw",
            camera_name="camera4",
            camera_role="forward",
            camera_frame_id="camera4_optical_frame",
            image_encoding="bgr8",
            odometry_topic="/Odometry",
            camera_info_topic=None,
            depth_topic=None,
            tf_topic=None,
            tf_static_topic=None,
            save_sensor_depth=False,
            body_frame_id="body",
            tf_relevant_frames=(),
            camera_info_max_delta_ms=100.0,
            depth_max_delta_ms=100.0,
            max_auxiliary_samples=8,
            max_tf_samples=64,
            max_tf_bytes=256 * 1024,
            source_timeout_sec=5.0,
            max_window_gap_sec=30.0,
            min_sample_fps=0.5,
            max_sample_fps=30.0,
            max_goal_frames=16,
            max_goal_duration_sec=30.0,
            min_captured_frames=1,
            max_image_width=640,
            max_image_height=480,
            pose_max_delta_ms=50.0,
            pose_gap_max_ms=200.0,
            max_sample_interval_overshoot_ms=250.0,
            snapshot_limits=SnapshotSafetyLimits(
                max_frame_count=16,
                max_manifest_entries=64,
                max_metadata_bytes=1024 * 1024,
                max_file_bytes=1024 * 1024,
                max_total_bytes=8 * 1024 * 1024,
                max_scene_id_bytes=128,
                max_relative_path_bytes=1024,
            ),
        )
        values.update(overrides)
        return CaptureConfig(**values)

    def make_goal(self, **overrides):
        values = dict(
            scene_id="scene-capture",
            sample_fps=2.0,
            max_frames=3,
            max_duration=SimpleNamespace(to_sec=lambda: 10.0),
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def track(self, transaction):
        self.transactions.append(transaction)
        return transaction

    def make_session(self, config=None, goal=None, **kwargs):
        config = config or self.make_config()
        goal = goal or validate_capture_goal(self.make_goal(), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokcapture01"
            )
        )
        return CaptureSession(
            config, goal, transaction, threading.Event(), **kwargs
        )

    @staticmethod
    def image_message(
        stamp_sec,
        stamp_nsec=0,
        *,
        encoding="bgr8",
        width=4,
        height=3,
        bgr_pixel=(10, 20, 30),
        frame_id="camera4_optical_frame",
    ):
        data = np.zeros((height, width, 3), dtype=np.uint8)
        data[:] = np.array(bgr_pixel, dtype=np.uint8)
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=stamp_sec, nsec=stamp_nsec),
                frame_id=frame_id,
            ),
            encoding=encoding,
            width=width,
            height=height,
            step=width * 3,
            data=bytes(data.tobytes()),
            is_bigendian=0,
        )

    @staticmethod
    def odometry_message(stamp_sec, stamp_nsec=0):
        def vector(*items):
            names = ("x", "y", "z", "w")
            return SimpleNamespace(**dict(zip(names, items)))

        return SimpleNamespace(
            header=SimpleNamespace(
                seq=1,
                stamp=SimpleNamespace(sec=stamp_sec, nsec=stamp_nsec),
                frame_id="odom",
            ),
            child_frame_id="body",
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector(0.0, 0.0, 0.0),
                    orientation=vector(0.0, 0.0, 0.0, 1.0),
                ),
                covariance=[0.0] * 36,
            ),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=vector(1.0, 0.0, 0.0),
                    angular=vector(0.0, 0.0, 0.5),
                ),
                covariance=[0.0] * 36,
            ),
        )


class CaptureConfigAndGoalTest(CaptureTestBase):
    def test_valid_config_and_goal(self):
        config = self.make_config()
        goal = validate_capture_goal(self.make_goal(), config)
        self.assertEqual(goal.scene_id, "scene-capture")
        self.assertEqual(goal.sample_fps, 2.0)
        self.assertEqual(goal.sample_period_ns, 500000000)

    def test_config_rejects_missing_or_unsafe_runtime_paths(self):
        missing = self.root / "missing"
        symlink = self.root / "link"
        symlink.symlink_to(self.snapshots_root)
        for overrides in (
            {"snapshot_root": str(missing)},
            {"snapshot_root": str(symlink)},
            {"image_topic": "not-a-topic"},
            {"camera_name": "bad name"},
            {"camera_role": "bad role"},
            {"camera_role": "/front"},
            {"camera_frame_id": ""},
            {"camera_frame_id": "/leading_slash"},
            {"image_encoding": "yuv422"},
            {"min_sample_fps": 0},
            {"max_sample_fps": 10, "min_sample_fps": 20},
            {"pose_max_delta_ms": -1},
            {"pose_gap_max_ms": 10, "pose_max_delta_ms": 50},
            {"max_goal_frames": True},
            {"min_captured_frames": 17},
            {"snapshot_limits": "not limits"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(CaptureConfigError):
                    self.make_config(**overrides)

    def test_goal_validation_maps_stable_codes(self):
        config = self.make_config()
        cases = (
            ({"scene_id": "bad scene"}, INVALID_GOAL),
            ({"sample_fps": 0}, INVALID_GOAL),
            ({"sample_fps": 100}, INVALID_GOAL),
            ({"max_frames": 0}, INVALID_GOAL),
            ({"max_frames": 17}, INVALID_GOAL),
            ({"max_duration": SimpleNamespace(to_sec=lambda: 0)}, INVALID_GOAL),
            ({"max_duration": SimpleNamespace(to_sec=lambda: 31.0)}, INVALID_GOAL),
        )
        for overrides, code in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(CaptureError) as caught:
                    validate_capture_goal(self.make_goal(**overrides), config)
                self.assertEqual(caught.exception.code, code)


class StampSamplerTest(CaptureTestBase):
    def test_first_stamp_selected_then_periodic(self):
        goal = validate_capture_goal(self.make_goal(sample_fps=2.0), self.make_config())
        sampler = StampSampler(goal)
        self.assertTrue(sampler.select(FrameStamp(10, 0)))
        self.assertFalse(sampler.select(FrameStamp(10, 250000000)))
        self.assertTrue(sampler.select(FrameStamp(10, 500000000)))
        self.assertFalse(sampler.select(FrameStamp(10, 750000000)))
        self.assertTrue(sampler.select(FrameStamp(11, 0)))

    def test_regressed_stamp_terminates_capture_stream(self):
        goal = validate_capture_goal(self.make_goal(sample_fps=2.0), self.make_config())
        sampler = StampSampler(goal)
        sampler.select(FrameStamp(10, 0))
        with self.assertRaises(CaptureStreamError):
            sampler.select(FrameStamp(9, 999999999))

    def test_regression_between_skipped_frames_terminates_capture_stream(self):
        goal = validate_capture_goal(self.make_goal(sample_fps=2.0), self.make_config())
        sampler = StampSampler(goal)
        self.assertTrue(sampler.select(FrameStamp(10, 0)))
        self.assertFalse(sampler.select(FrameStamp(10, 250000000)))
        with self.assertRaises(CaptureStreamError):
            sampler.select(FrameStamp(10, 100000000))

    def test_zero_and_duplicate_stamps_are_counted_and_skipped(self):
        goal = validate_capture_goal(self.make_goal(sample_fps=2.0), self.make_config())
        sampler = StampSampler(goal)
        self.assertFalse(sampler.select(FrameStamp(0, 0)))
        self.assertEqual(sampler.zero_stamp_count, 1)
        self.assertFalse(sampler.select(FrameStamp(0, 0)))
        self.assertEqual(sampler.zero_stamp_count, 2)
        self.assertTrue(sampler.select(FrameStamp(10, 0)))
        self.assertFalse(sampler.select(FrameStamp(10, 0)))
        self.assertEqual(sampler.duplicate_stamp_count, 1)
        self.assertEqual(sampler.last_seen, (10, 0))
        self.assertTrue(sampler.select(FrameStamp(10, 500000000)))


class OdometryAssociatorTest(CaptureTestBase):
    def test_associates_recent_pose_and_marks_gaps_invalid(self):
        config = self.make_config()
        associator = OdometryAssociator(config)
        associator.record(FrameStamp(10, 0))
        associator.record(FrameStamp(10, 100000000))
        valid, delta, gap = associator.associate(FrameStamp(10, 120000000))
        self.assertTrue(valid)
        self.assertAlmostEqual(delta, 20.0, places=3)
        self.assertEqual(gap, 0.0)
        # An image stamped inside/after the gap window must not borrow the
        # pre-gap pose: delta and the recorded source gap both exceed limits.
        associator.record(FrameStamp(10, 400000000))
        valid, delta, gap = associator.associate(FrameStamp(10, 350000000))
        self.assertFalse(valid)
        # The gap starts a new odometry epoch; an image before its first
        # sample must not borrow the previous epoch, so association is absent.
        self.assertIsNone(delta)
        self.assertIsNone(gap)

    def test_odometry_regression_starts_new_epoch(self):
        config = self.make_config()
        associator = OdometryAssociator(config)
        associator.record(FrameStamp(10, 0))
        associator.record(FrameStamp(9, 0))
        self.assertEqual(associator.regression_count, 1)
        # Association after the regression only uses the new epoch.
        valid, delta, _gap = associator.associate(FrameStamp(9, 50000000))
        self.assertTrue(valid)
        self.assertAlmostEqual(delta, 50.0, places=3)
        valid, delta, _gap = associator.associate(FrameStamp(10, 500000000))
        self.assertFalse(valid)
        self.assertGreater(delta, 50.0)


class InterpolatedCaptureTest(CaptureTestBase):
    def test_capture_offset_matches_dense_rgb_plus_40ms(self):
        from scipy.spatial.transform import Rotation, Slerp
        config = self.make_config(odometry_association='interpolate', pose_gap_max_ms=500., pose_time_offset_ms=40.)
        associator = OdometryAssociator(config)
        first = self.odometry_message(10)
        after = self.odometry_message(10, 100_000_000)
        after.pose.pose.position.x = 1.
        after.pose.pose.orientation.z = np.sin(.15)
        after.pose.pose.orientation.w = np.cos(.15)
        a, b = _odometry_payload(first), _odometry_payload(after)
        associator.record(a.stamp, a); associator.record(b.stamp, b)
        valid, _, _, actual = associator.associate_interpolated(FrameStamp(10, 20_000_000))
        # Independent dense-LIO oracle, as in T31: linear position + SciPy SLERP.
        expected_position = np.array([np.interp(.06, [0., .1], [0., 1.]), 0., 0.])
        oracle = Slerp([0., .1], Rotation.from_euler('z', [0., .3]))
        self.assertTrue(valid)
        np.testing.assert_allclose([actual.payload['pose']['position'][k] for k in 'xyz'], expected_position, atol=1e-9, rtol=0)
        qa = [actual.payload['pose']['orientation'][k] for k in 'xyzw']
        qb = oracle([.06]).as_quat()[0]
        self.assertLess(np.degrees((Rotation.from_quat(qa).inv()*Rotation.from_quat(qb)).magnitude()), 1e-9)
        self.assertEqual(actual.payload['header']['stamp'], {'sec':10,'nsec':20_000_000})
        self.assertEqual(actual.payload['interpolation']['pose_time_offset_ms'], 40.)

    def test_repeated_odometry_poses_do_not_hold_the_pose(self):
        # FAST-LIO repeats the last scan pose at ~100 Hz with fresh stamps.
        config = self.make_config(odometry_association="interpolate", pose_gap_max_ms=500.0)
        associator = OdometryAssociator(config)
        first = self.odometry_message(10)
        repeat = self.odometry_message(10, 90_000_000)
        moved = self.odometry_message(10, 100_000_000)
        moved.pose.pose.position.x = 1.0
        for message in (first, repeat, moved):
            payload = _odometry_payload(message)
            associator.record(payload.stamp, payload)
        self.assertEqual(associator.repeated_count, 1)
        _valid, _delta, gap, odometry = associator.associate_interpolated(FrameStamp(10, 50_000_000))
        self.assertEqual(gap, 100.0)
        self.assertAlmostEqual(odometry.payload["pose"]["position"]["x"], 0.5)

    def session_with_clock(self):
        clock = [0.0]
        config = self.make_config(
            odometry_association="interpolate", pose_gap_max_ms=500.0,
            pose_interpolation_wait_ms=500.0,
        )
        goal = validate_capture_goal(self.make_goal(max_frames=1), config)
        return self.make_session(config=config, goal=goal, monotonic=lambda: clock[0]), clock

    def test_last_rgb_waits_for_right_pose_and_seals_interpolated(self):
        session, _clock = self.session_with_clock()
        before = self.odometry_message(10)
        after = self.odometry_message(10, 200_000_000)
        after.pose.pose.position.x = 2.0
        after.pose.pose.orientation.z = 1.0
        after.pose.pose.orientation.w = 0.0
        session.on_odometry(before)
        session.on_image(self.image_message(10, 100_000_000))
        self.assertFalse(session.done.is_set())
        session.on_odometry(after)
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        payload = session.writer._auxiliary[0].odometry.payload
        self.assertEqual(payload["header"]["stamp"], {"sec": 10, "nsec": 100_000_000})
        self.assertAlmostEqual(payload["pose"]["position"]["x"], 1.0)
        self.assertAlmostEqual(payload["pose"]["orientation"]["z"], np.sqrt(0.5))
        self.assertAlmostEqual(payload["pose"]["orientation"]["w"], np.sqrt(0.5))
        self.assertEqual(payload["interpolation"]["bracket_gap_ms"], 200.0)

    def test_timeout_preserves_rgb_and_does_not_fall_back_to_before(self):
        session, clock = self.session_with_clock()
        session.on_odometry(self.odometry_message(10))
        session.on_image(self.image_message(10, 100_000_000))
        clock[0] = 0.499
        session.check_deadline()
        self.assertFalse(session.done.is_set())
        clock[0] = 0.5
        session.check_deadline()
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        self.assertEqual(session.writer.frame_count, 1)
        self.assertFalse(session.writer._frames[0].pose_valid)
        self.assertIsNone(session.writer._auxiliary[0].odometry)

    def test_pending_last_frame_does_not_admit_extra_rgb(self):
        session, clock = self.session_with_clock()
        session.on_image(self.image_message(10))
        session.on_image(self.image_message(11))
        self.assertEqual(len(session._pending_frames), 1)
        clock[0] = 0.5
        session.check_deadline()
        self.assertEqual(session.writer.frame_count, 1)
        self.assertEqual(session.writer._frames[0].stamp.sec, 10)

    def test_bracket_cannot_cross_odometry_coordinate_reset(self):
        session, clock = self.session_with_clock()
        session.on_odometry(self.odometry_message(10))
        session.on_image(self.image_message(10, 100_000_000))
        after = self.odometry_message(10, 200_000_000)
        after.header.frame_id = "another_odom"
        session.on_odometry(after)
        self.assertFalse(session.done.is_set())
        clock[0] = 0.5
        session.check_deadline()
        self.assertFalse(session.writer._frames[0].pose_valid)


class KeyframeCaptureTest(CaptureTestBase):
    """Motion keyframes: frames are written only after enough motion or the max interval."""

    def run_track(self, speed_mps, max_frames=3, **overrides):
        config = self.make_config(
            odometry_association="interpolate", pose_gap_max_ms=500.0,
            pose_interpolation_wait_ms=500.0, keyframe_min_translation_m=0.5,
            keyframe_min_rotation_deg=30.0, keyframe_max_interval_sec=2.0, **overrides,
        )
        goal = validate_capture_goal(self.make_goal(max_frames=max_frames), config)
        session = self.make_session(config=config, goal=goal)
        for k in range(40):  # odometry at 4 Hz from t=10 s, images at 2 Hz
            t_ns = 10_000_000_000 + k * 250_000_000
            odom = self.odometry_message(t_ns // 10**9, t_ns % 10**9)
            odom.pose.pose.position.x = speed_mps * k * 0.25
            session.on_odometry(odom)
            if k % 2 == 0:
                session.on_image(self.image_message(t_ns // 10**9, t_ns % 10**9))
            if session.done.is_set():
                break
        return session

    @staticmethod
    def written(session):
        return [f.stamp.sec + f.stamp.nsec / 1e9 for f in session.writer._frames]

    def test_frames_are_written_every_half_metre(self):
        session = self.run_track(0.4)
        self.assertIsNone(session.error)
        self.assertEqual(self.written(session), [10.0, 11.5, 13.0])
        self.assertEqual(session.keyframe_skipped, 4)
        self.assertEqual(session.snapshot()["keyframe_skipped"], 4)

    def test_hover_still_writes_at_the_max_interval(self):
        session = self.run_track(0.0)
        self.assertIsNone(session.error)
        self.assertEqual(self.written(session), [10.0, 12.0, 14.0])

    def test_gated_frames_are_never_png_encoded(self):
        from drone3d_reconstruction_core.capture import ImageCodec
        with mock.patch.object(ImageCodec, "encode_png", autospec=True,
                               side_effect=lambda self, array, encoding: b"png") as encode:
            session = self.run_track(0.4)
        self.assertEqual(encode.call_count, 3)  # only the three written keyframes
        self.assertEqual(session.keyframe_skipped, 4)

    def test_gate_requires_interpolated_poses(self):
        with self.assertRaisesRegex(CaptureConfigError, "interpolate"):
            self.make_config(keyframe_min_translation_m=0.5)
        with self.assertRaisesRegex(CaptureConfigError, "keyframe_max_interval_sec"):
            self.make_config(odometry_association="interpolate", keyframe_min_rotation_deg=10.0,
                             keyframe_max_interval_sec=0.0)

    def test_disabled_gate_keeps_every_sampled_frame(self):
        config = self.make_config(odometry_association="interpolate", pose_gap_max_ms=500.0)
        self.assertFalse(config.keyframe_gating)
        self.assertEqual(config.max_frame_interval_sec(2.0), 0.5)


class ImageCodecTest(CaptureTestBase):
    def test_bgr8_payload_becomes_rgb_png(self):
        # BGR payload [B=10, G=20, R=30] must become the RGB PNG pixel
        # [30, 20, 10]. cv2.imread returns BGR order, so the assertion is
        # [10, 20, 30]: that is the inverse of the file's RGB pixel.
        config = self.make_config()
        message = self.image_message(1, 0, bgr_pixel=(10, 20, 30))
        encoded = ImageCodec(config).to_rgb_png(message)
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (3, 4, 3))
        self.assertEqual(decoded[0, 0].tolist(), [10, 20, 30])

    def test_rgb8_payload_becomes_rgb_png(self):
        config = self.make_config(image_encoding="rgb8")
        message = self.image_message(
            1, 0, encoding="rgb8", bgr_pixel=(30, 20, 10)
        )
        encoded = ImageCodec(config).to_rgb_png(message)
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded[0, 0].tolist(), [10, 20, 30])

    def test_encoding_dimension_step_and_size_mismatches_fail_closed(self):
        config = self.make_config(max_image_width=4, max_image_height=3)
        cases = (
            self.image_message(1, 0, encoding="rgb8"),
            self.image_message(1, 0, width=5),
            self.image_message(1, 0, height=4),
        )
        for message in cases:
            with self.subTest(message=message):
                with self.assertRaises(CaptureStreamError):
                    ImageCodec(config).to_rgb_png(message)
        padded = self.image_message(1, 0)
        padded.step = padded.width * 3 + 1
        with self.assertRaises(CaptureStreamError):
            ImageCodec(config).to_rgb_png(padded)


class CaptureSessionAndWriterTest(CaptureTestBase):
    def test_session_seals_and_commits_valid_v4_snapshot(self):
        config = self.make_config()
        goal = validate_capture_goal(
            self.make_goal(sample_fps=2.0, max_frames=3), config
        )
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokseal01"
            )
        )
        session = CaptureSession(
            config, goal, transaction, threading.Event()
        )
        # Odometry before image keeps pose association available.
        session.on_odometry(self.odometry_message(10, 0))
        session.on_image(self.image_message(10, 0))
        session.on_image(self.image_message(10, 250000000))
        session.on_odometry(self.odometry_message(10, 490000000))
        session.on_image(self.image_message(10, 500000000))
        session.on_odometry(self.odometry_message(11, 0))
        session.on_image(self.image_message(11, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        self.assertIsNotNone(session.sealed)

        committed = transaction.commit()
        self.assertEqual(committed, transaction.final_path)
        validated = validate_snapshot(
            session.sealed.snapshot_uri,
            session.sealed.manifest_sha256,
            (config.snapshot_root,),
            config.snapshot_limits,
        )
        self.assertEqual(validated.snapshot_version, 4)
        self.assertEqual(validated.scene_id, goal.scene_id)
        self.assertEqual(validated.frame_count, 3)
        self.assertEqual(
            [(stamp.sec, stamp.nsec) for stamp in validated.frame_stamps],
            [(10, 0), (10, 500000000), (11, 0)],
        )
        self.assertTrue(all(validated.pose_valid))
        self.assertTrue(validated.has_odometry)
        self.assertFalse(validated.has_camera_info)
        self.assertFalse(validated.has_depth)
        self.assertFalse(validated.has_tf)
        self.assertIsNone(validated.camera_info_digest)
        self.assertTrue(all(path is not None for path in validated.auxiliary_paths))
        self.assertTrue(all(path is None for path in validated.depth_paths))
        self.assertTrue(all(payload is not None for payload in validated.odometry_payloads))
        self.assertTrue(all(payload is None for payload in validated.camera_info_payloads))
        self.assertEqual(validated.auxiliary_summary["odometry_matched_frames"], 3)
        self.assertEqual(validated.camera_name, "camera4")
        self.assertEqual(validated.camera_role, "forward")
        self.assertEqual(validated.camera_frame_id, "camera4_optical_frame")
        snapshot_json = json.loads((committed / "snapshot.json").read_text())
        self.assertEqual(snapshot_json["version"], 4)
        self.assertEqual(snapshot_json["camera_role"], "forward")
        self.assertEqual(snapshot_json["camera_frame_id"], "camera4_optical_frame")
        self.assertEqual(snapshot_json["first_stamp_sec"], 10)
        self.assertEqual(snapshot_json["last_stamp_sec"], 11)
        self.assertFalse(snapshot_json["has_depth"])
        self.assertFalse(snapshot_json["has_tf"])

    def test_source_stale_freezes_deadline_and_recovers_continuous_window(self):
        config = self.make_config(
            source_timeout_sec=2.0,
            max_window_gap_sec=5.0,
            max_sample_interval_overshoot_ms=250.0,
        )
        goal = validate_capture_goal(
            self.make_goal(sample_fps=2.0, max_frames=2), config
        )
        clock = [0.0]
        transitions = []
        session = self.make_session(
            config=config,
            goal=goal,
            monotonic=lambda: clock[0],
            source_state_callback=lambda stage, stamp: transitions.append(stage),
        )
        session.on_image(self.image_message(10, 0))
        clock[0] = 2.5
        session.check_deadline()
        self.assertTrue(session.source_stale)
        self.assertFalse(session.done.is_set())
        self.assertEqual(session.source_stale_entries, 1)
        # 原始 deadline 已过，但 SOURCE_STALE 冻结了窗口 deadline
        clock[0] = 12.0
        session.check_deadline()
        self.assertFalse(session.done.is_set())
        # 时间连续的恢复：interval 过冲检查被跳过，窗口继续到 N 帧封存
        session.on_image(self.image_message(11, 0))
        self.assertTrue(session.done.is_set())
        self.assertFalse(session.source_stale)
        self.assertIsNotNone(session.sealed)
        self.assertEqual(transitions, ["SOURCE_STALE", "SOURCE_RECOVERED"])
        session.transaction.cleanup()

    def test_recovered_from_stale_survives_unselected_image(self):
        config = self.make_config(
            source_timeout_sec=1.0,
            max_window_gap_sec=5.0,
            max_sample_interval_overshoot_ms=250.0,
        )
        goal = validate_capture_goal(
            self.make_goal(sample_fps=2.0, max_frames=2), config
        )
        clock = [0.0]
        transitions = []
        session = self.make_session(
            config=config,
            goal=goal,
            monotonic=lambda: clock[0],
            source_state_callback=lambda stage, stamp: transitions.append(stage),
        )
        session.on_image(self.image_message(10, 0))
        clock[0] = 2.5
        session.check_deadline()
        self.assertTrue(session.source_stale)
        # 恢复后的第一帧未被选中：recovered_from_stale 必须保留到下一个
        # 选中帧，否则 10.8s 的选中帧会因 800ms > 750ms 的间隔过冲被误杀。
        session.on_image(self.image_message(10, 400000000))
        self.assertFalse(session.done.is_set())
        self.assertFalse(session.source_stale)
        self.assertTrue(session.recovered_from_stale)
        session.on_image(self.image_message(10, 800000000))
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        self.assertIsNotNone(session.sealed)
        self.assertEqual(session.writer.frame_count, 2)
        self.assertEqual(transitions, ["SOURCE_STALE", "SOURCE_RECOVERED"])
        session.transaction.cleanup()

    def test_pause_resume_does_not_trigger_source_stale(self):
        config = self.make_config(source_timeout_sec=2.0)
        goal = validate_capture_goal(self.make_goal(max_frames=2), config)
        clock = [0.0]
        transitions = []
        session = self.make_session(
            config=config,
            goal=goal,
            monotonic=lambda: clock[0],
            source_state_callback=lambda stage, stamp: transitions.append(stage),
        )
        session.on_image(self.image_message(10, 0))
        session.pause()
        clock[0] = 10.0
        session.resume()
        session.check_deadline()
        self.assertFalse(session.source_stale)
        self.assertEqual(transitions, [])
        session.transaction.cleanup()

    def test_backward_stamp_during_source_stale_never_reports_recovered(self):
        config = self.make_config(source_timeout_sec=2.0, max_window_gap_sec=5.0)
        goal = validate_capture_goal(self.make_goal(max_frames=2), config)
        clock = [0.0]
        transitions = []
        session = self.make_session(
            config=config,
            goal=goal,
            monotonic=lambda: clock[0],
            source_state_callback=lambda stage, stamp: transitions.append(stage),
        )
        session.on_image(self.image_message(10, 0))
        clock[0] = 2.5
        session.check_deadline()
        self.assertTrue(session.source_stale)
        session.on_image(self.image_message(9, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsInstance(session.error, CaptureStreamError)
        self.assertEqual(session.error.code, CAMERA_UNAVAILABLE)
        self.assertEqual(transitions, ["SOURCE_STALE"])
        session.transaction.cleanup()

    def test_source_gap_beyond_max_window_gap_discards_window(self):
        config = self.make_config(max_window_gap_sec=0.5)
        goal = validate_capture_goal(self.make_goal(max_frames=2), config)
        session = self.make_session(config=config, goal=goal)
        session.on_image(self.image_message(10, 0))
        session.on_image(self.image_message(10, 600000000))
        self.assertTrue(session.done.is_set())
        self.assertIsInstance(session.error, CaptureStreamError)
        self.assertEqual(session.error.code, CAMERA_UNAVAILABLE)
        self.assertIn("max_window_gap", str(session.error))
        session.transaction.cleanup()

    def test_duplicate_stamp_is_counted_without_aborting_window(self):
        config = self.make_config()
        goal = validate_capture_goal(
            self.make_goal(sample_fps=2.0, max_frames=2), config
        )
        session = self.make_session(config=config, goal=goal)
        session.on_image(self.image_message(10, 0))
        session.on_image(self.image_message(10, 0))
        self.assertFalse(session.done.is_set())
        self.assertEqual(session.sampler.duplicate_stamp_count, 1)
        self.assertEqual(session.writer.frame_count, 1)
        session.transaction.cleanup()

    def test_zero_header_stamp_is_counted_and_skipped(self):
        config = self.make_config()
        goal = validate_capture_goal(self.make_goal(max_frames=1), config)
        session = self.make_session(config=config, goal=goal)
        session.on_image(self.image_message(0, 0))
        self.assertFalse(session.done.is_set())
        self.assertIsNone(session.error)
        self.assertEqual(session.sampler.zero_stamp_count, 1)
        self.assertEqual(session.writer.frame_count, 0)
        session.on_image(self.image_message(10, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsNotNone(session.sealed)
        session.transaction.cleanup()

    def test_odometry_flag_reflects_actual_observations(self):
        config = self.make_config(odometry_topic="/Odometry")
        goal = validate_capture_goal(self.make_goal(max_frames=1), config)
        session = self.make_session(config=config, goal=goal)
        session.on_image(self.image_message(10, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsNotNone(session.sealed)
        self.assertFalse(session.sealed.has_odometry)
        session.transaction.cleanup()

    def test_image_stamp_regression_aborts_before_publication(self):
        session = self.make_session()
        session.on_image(self.image_message(10, 0))
        session.on_image(self.image_message(9, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsInstance(session.error, CaptureStreamError)
        self.assertEqual(session.error.code, CAMERA_UNAVAILABLE)
        session.transaction.cleanup()
        self.assertEqual(list(self.snapshots_root.iterdir()), [])

    def test_image_frame_id_mismatch_aborts_before_publication(self):
        session = self.make_session()
        session.on_image(self.image_message(10, 0, frame_id="other_camera_optical_frame"))
        self.assertTrue(session.done.is_set())
        self.assertIsInstance(session.error, CaptureStreamError)
        self.assertEqual(session.error.code, CAMERA_UNAVAILABLE)
        self.assertIn("frame_id does not match", str(session.error))
        session.transaction.cleanup()
        self.assertEqual(list(self.snapshots_root.iterdir()), [])

    def test_selected_interval_outside_tolerance_aborts(self):
        config = self.make_config(max_sample_interval_overshoot_ms=250.0)
        goal = validate_capture_goal(self.make_goal(sample_fps=2.0), config)
        session = self.make_session(config=config, goal=goal)
        session.on_image(self.image_message(10, 0))
        session.on_image(self.image_message(11, 0))
        self.assertTrue(session.done.is_set())
        self.assertEqual(session.error.code, CAMERA_UNAVAILABLE)
        self.assertIn("configured tolerance", str(session.error))
        session.transaction.cleanup()

    def test_min_frame_limit_aborts_empty_capture(self):
        config = self.make_config(
            min_captured_frames=2, source_timeout_sec=30.0
        )
        goal = validate_capture_goal(self.make_goal(max_frames=2), config)
        clock = [0.0]
        session = self.make_session(
            config=config, goal=goal, monotonic=lambda: clock[0]
        )
        # 先收到一帧有效 RGB，source watchdog 不会抢先进入 SOURCE_STALE；
        # 之后再越过 deadline，让 min_captured_frames 校验生效。
        session.on_image(self.image_message(10, 0))
        clock[0] = 15.0
        session.check_deadline()
        self.assertTrue(session.done.is_set())
        self.assertIsInstance(session.error, CaptureError)
        self.assertEqual(session.error.code, CAMERA_UNAVAILABLE)
        session.transaction.cleanup()

    def test_goal_frame_cap_is_the_effective_minimum(self):
        limits = SnapshotSafetyLimits(
            max_frame_count=100,
            max_manifest_entries=256,
            max_metadata_bytes=1024 * 1024,
            max_file_bytes=1024 * 1024,
            max_total_bytes=8 * 1024 * 1024,
            max_scene_id_bytes=128,
            max_relative_path_bytes=1024,
        )
        config = self.make_config(
            max_goal_frames=100,
            min_captured_frames=100,
            snapshot_limits=limits,
        )
        for goal_frames in (80, 60):
            with self.subTest(goal_frames=goal_frames):
                goal = validate_capture_goal(
                    self.make_goal(
                        scene_id=f"scene-capture-{goal_frames}",
                        sample_fps=1.0,
                        max_frames=goal_frames,
                    ),
                    config,
                )
                session = self.make_session(config=config, goal=goal)
                for index in range(goal_frames):
                    session.on_image(self.image_message(index + 1))
                self.assertIsNotNone(session.sealed)
                self.assertEqual(session.sealed.frame_count, goal_frames)

    def test_snapshot_writer_rejects_oversized_image(self):
        config = self.make_config(
            max_tf_bytes=16,
            source_timeout_sec=5.0,
            max_window_gap_sec=30.0,
            snapshot_limits=SnapshotSafetyLimits(
                max_frame_count=16,
                max_manifest_entries=64,
                max_metadata_bytes=1024 * 1024,
                max_file_bytes=16,
                max_total_bytes=8 * 1024 * 1024,
                max_scene_id_bytes=128,
                max_relative_path_bytes=1024,
            ),
        )
        goal = validate_capture_goal(self.make_goal(max_frames=1), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokbig01"
            )
        )
        writer = SnapshotWriter(config, goal, transaction)
        with self.assertRaises(CaptureError) as caught:
            writer.write_frame(
                FrameStamp(1, 0),
                b"x" * 128,
                pose_valid=False,
                pose_delta_ms=None,
                odometry_gap_ms=None,
            )
        self.assertEqual(caught.exception.code, IO_ERROR)


    def test_source_stale_triggers_before_first_rgb(self):
        config = self.make_config(source_timeout_sec=0.1, max_window_gap_sec=30.0)
        goal = validate_capture_goal(self.make_goal(), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokstale00"
            )
        )
        clock = [100.0]
        transitions = []
        session = CaptureSession(
            config,
            goal,
            transaction,
            threading.Event(),
            monotonic=lambda: clock[0],
            source_state_callback=lambda stage, stamp: transitions.append(
                (stage, stamp)
            ),
        )
        self.assertFalse(session.source_stale)
        clock[0] += config.source_timeout_sec + 0.01
        session.check_deadline()
        self.assertTrue(session.source_stale)
        self.assertEqual(session.source_stale_entries, 1)
        self.assertFalse(session.done.is_set())
        self.assertIsNone(session.error)
        clock[0] += 5.0
        session.on_image(self.image_message(10, 0))
        self.assertFalse(session.source_stale)
        self.assertEqual(session.writer.frame_count, 1)
        self.assertEqual(
            transitions,
            [
                ("SOURCE_STALE", None),
                ("SOURCE_RECOVERED", FrameStamp(10, 0)),
            ],
        )


class AuxiliaryContractV4Test(CaptureTestBase):
    """v4 完整辅助数据契约：CameraInfo/Odometry/TF/深度落盘与回读。"""

    @staticmethod
    def camera_info_message(stamp_sec, stamp_nsec=0):
        return SimpleNamespace(
            header=SimpleNamespace(
                seq=1,
                stamp=SimpleNamespace(sec=stamp_sec, nsec=stamp_nsec),
                frame_id="camera4_optical_frame",
            ),
            height=3,
            width=4,
            distortion_model="plumb_bob",
            D=[0.1, 0.2, 0.3, 0.4, 0.5],
            K=list(range(1, 10)),
            R=[1.0] * 9,
            P=list(range(12)),
            binning_x=1,
            binning_y=1,
            roi=SimpleNamespace(
                x_offset=0, y_offset=0, width=4, height=3, do_rectify=False
            ),
        )

    @staticmethod
    def depth_message(stamp_sec, stamp_nsec=0):
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=stamp_sec, nsec=stamp_nsec),
                frame_id="depth_optical_frame",
            ),
            encoding="16UC1",
            width=2,
            height=1,
            step=2,
            is_bigendian=0,
            data=b"\x01\x02",
        )

    @staticmethod
    def tf_message(stamp_sec, *, parent, child):
        return SimpleNamespace(
            transforms=[
                SimpleNamespace(
                    header=SimpleNamespace(
                        stamp=SimpleNamespace(sec=stamp_sec, nsec=0),
                        frame_id=parent,
                    ),
                    child_frame_id=child,
                    transform=SimpleNamespace(
                        translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )
            ]
        )

    def make_aux_config(self, max_frames=2):
        return self.make_config(
            camera_info_topic="/camera_info",
            depth_topic="/depth",
            save_sensor_depth=True,
            tf_topic="/tf",
            tf_static_topic="/tf_static",
            max_goal_frames=max_frames,
        )

    def test_v4_saves_and_reads_back_full_auxiliary_payloads(self):
        config = self.make_aux_config()
        goal = validate_capture_goal(self.make_goal(max_frames=2), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokaux01"
            )
        )
        session = CaptureSession(config, goal, transaction, threading.Event())
        session.on_camera_info(self.camera_info_message(10, 0))
        session.on_odometry(self.odometry_message(10, 0))
        session.on_depth(self.depth_message(10, 0))
        session.on_tf(self.tf_message(10, parent="odom", child="body"))
        session.on_tf(self.tf_message(10, parent="map", child="unrelated"))
        session.on_image(self.image_message(10, 0))
        session.on_camera_info(self.camera_info_message(10, 500000000))
        session.on_odometry(self.odometry_message(10, 500000000))
        session.on_depth(self.depth_message(10, 500000000))
        session.on_image(self.image_message(10, 500000000))
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)

        committed = transaction.commit()
        validated = validate_snapshot(
            session.sealed.snapshot_uri,
            session.sealed.manifest_sha256,
            (config.snapshot_root,),
            config.snapshot_limits,
        )
        self.assertEqual(validated.snapshot_version, 4)
        self.assertTrue(validated.has_camera_info)
        self.assertTrue(validated.has_odometry)
        self.assertTrue(validated.has_depth)
        self.assertTrue(validated.has_tf)
        self.assertEqual(validated.auxiliary_summary["camera_info_samples"], 2)
        self.assertEqual(validated.auxiliary_summary["odometry_samples"], 2)
        self.assertEqual(validated.auxiliary_summary["depth_samples"], 2)
        self.assertEqual(validated.auxiliary_summary["tf_samples"], 1)
        self.assertEqual(validated.auxiliary_summary["camera_info_matched_frames"], 2)
        self.assertEqual(validated.auxiliary_summary["odometry_matched_frames"], 2)
        self.assertEqual(validated.auxiliary_summary["depth_matched_frames"], 2)
        # CameraInfo 完整字段回读
        first_ci = validated.camera_info_payloads[0]
        self.assertEqual(first_ci["K"], [float(item) for item in range(1, 10)])
        self.assertEqual(first_ci["D"], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(first_ci["distortion_model"], "plumb_bob")
        self.assertEqual(first_ci["header"]["frame_id"], "camera4_optical_frame")
        # Odometry 完整位姿回读
        first_odom = validated.odometry_payloads[0]
        self.assertEqual(first_odom["child_frame_id"], "body")
        self.assertEqual(first_odom["pose"]["position"]["x"], 0.0)
        self.assertEqual(first_odom["pose"]["orientation"]["w"], 1.0)
        self.assertEqual(len(first_odom["pose"]["covariance"]), 36)
        # 标准 ROS TwistWithCovariance 形状：linear/angular + covariance
        self.assertEqual(first_odom["twist"]["linear"]["x"], 1.0)
        self.assertEqual(first_odom["twist"]["linear"]["y"], 0.0)
        self.assertEqual(first_odom["twist"]["angular"]["z"], 0.5)
        self.assertEqual(len(first_odom["twist"]["covariance"]), 36)
        # 深度元数据与原始字节回读
        depth_meta = validated.depth_metadata[0]
        self.assertEqual(depth_meta["encoding"], "16UC1")
        self.assertEqual(depth_meta["width"], 2)
        self.assertEqual(depth_meta["height"], 1)
        depth_bytes = (committed / validated.depth_paths[0]).read_bytes()
        self.assertEqual(depth_bytes, b"\x01\x02")
        # TF 只保存相机/机体相关变换
        self.assertEqual(len(validated.tf_entries), 1)
        self.assertEqual(validated.tf_entries[0]["parent_frame_id"], "odom")
        self.assertEqual(validated.tf_entries[0]["child_frame_id"], "body")
        # camera_info_digest 可由逐帧 payload 确定性复算
        digest = hashlib.sha256()
        for payload in validated.camera_info_payloads:
            digest.update(
                json.dumps(
                    payload, ensure_ascii=False, allow_nan=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
        self.assertEqual(validated.camera_info_digest, digest.hexdigest())

    def test_bad_auxiliary_sources_never_block_rgb_capture(self):
        config = self.make_aux_config()
        goal = validate_capture_goal(self.make_goal(max_frames=2), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokaux02"
            )
        )
        session = CaptureSession(config, goal, transaction, threading.Event())
        bad_header = SimpleNamespace(seq=1, stamp=SimpleNamespace(sec=10, nsec=0))
        session.on_odometry(bad_header)  # 无 frame_id/pose → 辅助错误
        session.on_camera_info(bad_header)  # 无 K/D/R/P → 辅助错误
        bad_depth = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=10, nsec=0), frame_id="depth_optical_frame"
            ),
            encoding="16UC1", width=2, height=1, step=2, is_bigendian=0, data=b"x"
        )
        session.on_depth(bad_depth)  # payload 长度不符 → 辅助错误
        bad_tf = SimpleNamespace(transforms=None)
        session.on_tf(bad_tf)  # 无 transforms → 辅助错误
        session.on_image(self.image_message(10, 0))
        session.on_image(self.image_message(10, 500000000))
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        self.assertEqual(session.auxiliary_errors, 4)
        self.assertEqual(session.writer.auxiliary_errors, 4)
        committed = transaction.commit()
        validated = validate_snapshot(
            session.sealed.snapshot_uri,
            session.sealed.manifest_sha256,
            (config.snapshot_root,),
            config.snapshot_limits,
        )
        self.assertFalse(validated.has_camera_info)
        self.assertFalse(validated.has_odometry)
        self.assertFalse(validated.has_depth)
        self.assertFalse(validated.has_tf)
        self.assertEqual(validated.auxiliary_summary["auxiliary_errors"], 4)
        self.assertTrue(all(not item for item in validated.pose_valid))
        self.assertTrue((committed / "snapshot.json").is_file())

    def test_odometry_rejects_nonstandard_covariance_lengths(self):
        config = self.make_aux_config()
        goal = validate_capture_goal(self.make_goal(max_frames=2), config)
        session = self.make_session(config=config, goal=goal)
        short_pose = self.odometry_message(10, 0)
        short_pose.pose.covariance = [0.0] * 35
        session.on_odometry(short_pose)
        short_twist = self.odometry_message(10, 1)
        short_twist.twist.covariance = []
        session.on_odometry(short_twist)
        self.assertEqual(session.writer.odometry_samples, 0)
        self.assertEqual(session.auxiliary_errors, 2)


    def test_aux_json_over_budget_never_leaves_orphan_depth_file(self):
        config = self.make_config(
            camera_info_topic="/camera_info",
            depth_topic="/depth",
            save_sensor_depth=True,
            max_tf_bytes=512,
            source_timeout_sec=5.0,
            max_window_gap_sec=30.0,
            snapshot_limits=SnapshotSafetyLimits(
                max_frame_count=16,
                max_manifest_entries=64,
                max_metadata_bytes=1024 * 1024,
                max_file_bytes=1024,
                max_total_bytes=8 * 1024 * 1024,
                max_scene_id_bytes=128,
                max_relative_path_bytes=1024,
            ),
        )
        goal = validate_capture_goal(self.make_goal(max_frames=1), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokaux03"
            )
        )
        session = CaptureSession(config, goal, transaction, threading.Event())
        huge_ci = self.camera_info_message(10, 0)
        huge_ci.distortion_model = "x" * 4096
        session.on_camera_info(huge_ci)
        session.on_depth(self.depth_message(10, 0))
        session.on_image(self.image_message(10, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        self.assertTrue(session.writer.auxiliary_dropped > 0)
        committed = transaction.commit()
        validated = validate_snapshot(
            session.sealed.snapshot_uri,
            session.sealed.manifest_sha256,
            (config.snapshot_root,),
            config.snapshot_limits,
        )
        self.assertFalse(validated.has_camera_info)
        self.assertFalse(validated.has_depth)
        self.assertTrue(all(path is None for path in validated.depth_paths))
        self.assertFalse((committed / "depth").exists())

    def test_depth_write_failure_rewrites_aux_json_without_depth(self):
        import drone3d_reconstruction_core.capture as capture_module

        config = self.make_aux_config(max_frames=1)
        goal = validate_capture_goal(self.make_goal(max_frames=1), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokaux04"
            )
        )
        session = CaptureSession(config, goal, transaction, threading.Event())
        original = capture_module._write_file_atomically

        def failing_write(path, data, maximum, *, sync=True):
            if str(path).endswith("depth/frame_000000.bin"):
                raise OSError("injected depth write failure")
            return original(path, data, maximum, sync=sync)

        with mock.patch.object(capture_module, "_write_file_atomically", failing_write):
            session.on_camera_info(self.camera_info_message(10, 0))
            session.on_odometry(self.odometry_message(10, 0))
            session.on_depth(self.depth_message(10, 0))
            session.on_image(self.image_message(10, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        committed = transaction.commit()
        validated = validate_snapshot(
            session.sealed.snapshot_uri,
            session.sealed.manifest_sha256,
            (config.snapshot_root,),
            config.snapshot_limits,
        )
        self.assertTrue(validated.has_camera_info)
        self.assertTrue(validated.has_odometry)
        self.assertFalse(validated.has_depth)
        self.assertIsNone(validated.depth_paths[0])
        self.assertFalse((committed / "depth").exists())


    def test_zero_stamp_tf_static_saved_dynamic_rejected(self):
        config = self.make_aux_config(max_frames=1)
        goal = validate_capture_goal(self.make_goal(max_frames=1), config)
        transaction = self.track(
            allocate_snapshot_transaction(
                config, goal.scene_id, token_source=lambda: "tokaux05"
            )
        )
        session = CaptureSession(config, goal, transaction, threading.Event())
        session.on_tf(self.tf_message(0, parent="odom", child="body"), static=True)
        session.on_tf(self.tf_message(0, parent="map", child="body"), static=False)
        session.on_image(self.image_message(10, 0))
        self.assertTrue(session.done.is_set())
        self.assertIsNone(session.error)
        self.assertEqual(session.writer.tf_dropped, 0)
        self.assertEqual(session.auxiliary_errors, 1)
        committed = transaction.commit()
        validated = validate_snapshot(
            session.sealed.snapshot_uri,
            session.sealed.manifest_sha256,
            (config.snapshot_root,),
            config.snapshot_limits,
        )
        self.assertTrue(validated.has_tf)
        self.assertEqual(validated.tf_entries[0]["stamp"], {"sec": 0, "nsec": 0})
        self.assertTrue(validated.tf_entries[0]["static"])
        self.assertTrue((committed / "tf.json").is_file())


if __name__ == "__main__":
    unittest.main()


class ImageHandoffTest(unittest.TestCase):
    def test_decimates_to_the_goal_sample_period_and_keeps_returned_images(self):
        from drone3d_reconstruction_core.capture_handoff import ImageHandoff
        handoff = ImageHandoff(max_frames=100, max_bytes=10**9)
        handoff.min_interval_ns = 200_000_000  # 5 Hz goal
        for k in range(15):  # one second at 15 Hz
            handoff.push(CaptureTestBase.image_message(10, k * 66_666_667))
        kept = [stamp for stamp, _, _ in handoff.queue]
        self.assertEqual(len(kept), 5)
        self.assertTrue(all(b - a >= 200_000_000 for a, b in zip(kept, kept[1:])))
        self.assertEqual(handoff.decimated, 10)
        first = handoff.pop_after(0)
        handoff.push(first[1], front=True)  # a returned image is never decimated
        self.assertEqual(len(handoff.queue), 5)
