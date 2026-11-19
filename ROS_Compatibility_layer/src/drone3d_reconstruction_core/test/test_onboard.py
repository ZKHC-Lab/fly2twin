#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
import numpy as np
from recon_core.fusion import confidence_mask, partners, voxel_down_sample, colored_cloud
from recon_core.ply import write_ply


class OnboardTests(unittest.TestCase):
    def test_confidence_is_per_frame_strict_median(self):
        d = np.ones((6, 8), np.float32)
        c = np.arange(48).reshape(6, 8)
        np.testing.assert_array_equal(confidence_mask(d, c), c > np.median(c))

    def test_edge_reject(self):
        d = np.ones((8, 8)); d[:, 4:] = 2
        good = confidence_mask(d, np.arange(64).reshape(8, 8))
        self.assertFalse(good[:, 3:5].any())

    def test_partner_rotation_and_distance(self):
        p = np.tile(np.eye(4), (3, 1, 1)); p[1, 0, 3] = .3; p[2, 0, 3] = 2
        self.assertEqual(partners(p), [[1], [0], []])

    def test_voxel_centroids_and_color(self):
        xyz, rgb = voxel_down_sample([[0,0,0],[.001,0,0],[1,0,0]], [[1,0,0],[0,1,0],[0,0,1]], .02)
        np.testing.assert_allclose(xyz, [[.0005,0,0],[1,0,0]])
        np.testing.assert_allclose(rgb, [[.5,.5,0],[0,0,1]])

    def test_no_support_means_no_points(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rgb.png'; cv2.imwrite(str(path), np.zeros((8,8,3), np.uint8))
            xyz, _, _ = colored_cloud([0], np.eye(4)[None], np.eye(3)[None], np.ones((1,8,8)),
                np.ones((1,8,8),bool), {'0':path})
            self.assertEqual(len(xyz), 0)

    def test_ply_layout(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.ply'
            write_ply(path, np.array([[1.,2.,3.]]), np.array([[1.,.5,0.]]))
            header, body=path.read_bytes().split(b'end_header\n')
            self.assertIn(b'comment meters Z-up', header)
            self.assertEqual(len(body), 15)
            np.testing.assert_array_equal(np.frombuffer(body[:12],'<f4'), [1,2,3])
            self.assertEqual(body[12:], bytes([255,128,0]))

    def test_delivery_contract_and_hashes(self):
        import cv2, json
        from unittest.mock import patch
        from scipy.spatial.transform import Rotation
        from drone3d_onboard.export import export_session
        from drone3d_onboard.common import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); records=root/'sessions'; sid='s'+'a'*32
            backend=root/'g4'/sid; backend.mkdir(parents=True)
            (backend/'backend_profile.json').write_text(json.dumps({'frame_pose':{'pose_time_offset_ms':40.,'pose_time_offset_stage':'capture'}}))
            capture=root/'capture.yaml'; capture.write_text('pose_time_offset_ms: 40.0\n')
            image=root/'image.png'; cv2.imwrite(str(image),np.full((720,1280,3),127,np.uint8))
            pred=root/'pred.npz'; np.savez(pred,depth=np.ones((1,16,16,1)),depth_conf=np.ones((1,16,16)))
            K=np.array([[900.,0,640],[0,900,360],[0,0,1.]])
            row=dict(frame_id='source0',timestamp_ns=123,image_path=str(image),c2w=np.eye(4),K=K,camera_info={'K':K.ravel().tolist(),'D':[0.]*5,'width':1280,'height':720},revisit_log_scale=.02)
            ext=dict(translation=[.056558,-.006717,-.077751],rotation=Rotation.from_quat([-.494176,.506142,-.498990,.500618]).as_matrix().tolist())
            win=dict(window_id=0,source=dict(body_from_camera=ext,prediction_uri=str(pred),frame_ids=['source0'],timestamps_ns=[123]),alignment=dict(depth_scale=2.,intrinsics=K.tolist(),merge_frame_indices=[0]),selected=[(0,0)])
            result=(np.array([[1.,2.,3.]]),np.array([[1.,0.,0.]]),{},[row],[win])
            with patch('drone3d_reconstruction_core.continuous_web._formal_records_for',return_value=[]), patch('drone3d_onboard.export.fuse_session',return_value=result):
                manifest=export_session(records,sid,root/'delivery',capture)
                with self.assertRaises(FileExistsError):export_session(records,sid,root/'delivery',capture)
                export_session(records,sid,root/'plain',capture,revisit=False)
            self.assertEqual(manifest['pose_time_offset_stage'],'capture')
            self.assertEqual(cv2.imread(str(root/'delivery/keyframes/000000.jpg')).shape,(720,1280,3))
            for name,value in manifest['hashes'].items():self.assertEqual(digest(root/'delivery'/name),value)
            with np.load(root/'delivery/poses.npz') as data:
                np.testing.assert_array_equal(data['c2w'],np.eye(4)[None])
                np.testing.assert_array_equal(data['revisit_log_scale'],[.02])
            # The owned frame's exported scale carries the revisit correction; without the step nothing is added.
            with np.load(root/'delivery/windows/000000.npz') as data:np.testing.assert_allclose(data['frame_depth_scale'],[2.*np.exp(.02)],rtol=1e-15)
            with np.load(root/'plain/windows/000000.npz') as data:np.testing.assert_array_equal(data['frame_depth_scale'],[2.])
            with np.load(root/'plain/poses.npz') as data:self.assertNotIn('revisit_log_scale',data.files)

    def test_session_skips_frames_without_localization(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from drone3d_onboard.session import load_session
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pred = root/'pred.npz'; np.savez(pred, depth=np.ones((2, 4, 4, 1)), depth_conf=np.ones((2, 4, 4)))
            meta = dict(model_image_size=[4, 4], frame_ids=['f0', 'f1'], timestamps_ns=[1, 2], prediction_uri=str(pred))
            pose = dict(rotation=np.eye(3).tolist(), center=[1., 2., 3.])
            record = dict(window_index=0, snapshot_manifest_uri='file:///x', snapshot_manifest_sha256='0',
                          alignment=dict(placement='frame_pose', quality_passed=True, merge_frame_indices=[0, 1],
                                         frame_poses=[None, pose], intrinsics=np.eye(3).tolist(), depth_scale=2.))
            snapshot = SimpleNamespace(image_paths=['a.png', 'b.png'], camera_info_payloads=[{}, {}])
            with patch('drone3d_reconstruction_core.g4.input_adapter.WindowInputReader.load', return_value=SimpleNamespace(metadata=meta)), \
                    patch('drone3d_reconstruction_core.g4.input_adapter.local_path', return_value=root/'manifest.json'), \
                    patch('drone3d_reconstruction_core.snapshot_validator.validate_snapshot', return_value=snapshot):
                rows, windows, depth, _ = load_session([record], root/'cache')
                record['alignment']['frame_depth_scales'] = [None, 3.]
                _, _, per_frame, _ = load_session([record], root/'cache_frames')
            self.assertEqual([r['frame_id'] for r in rows], ['f1'])
            self.assertEqual(windows[0]['unposed'], ['f0'])
            np.testing.assert_array_equal(depth[0], 2.)
            np.testing.assert_array_equal(per_frame[0], 3.)  # a per-frame G4 scale wins over the window scale

    def test_incremental_fusion_matches_fresh_build_and_full_fusion(self):
        import cv2
        from types import SimpleNamespace
        from unittest.mock import patch
        from scipy.spatial import cKDTree
        from drone3d_onboard.session import SessionFusion, fuse_session
        rng = np.random.default_rng(3)
        H, W, per = 18, 24, 6
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metas, snapshots, records = {}, {}, []
            for w in range(4):
                ids = ['w%df%d' % (w, i) for i in range(per)]
                # Fronto-parallel wall 1.6 m ahead of cameras that only translate: consistent views.
                depth = 1.6 / (1. + .1 * w) * (1 + .002 * rng.random((per, H, W, 1)))
                pred = root / ('pred%d.npz' % w)
                np.savez(pred, depth=depth.astype(np.float32), depth_conf=rng.random((per, H, W)))
                paths = []
                for i in range(per):
                    paths.append(root / ('w%di%d.png' % (w, i)))
                    cv2.imwrite(str(paths[-1]), rng.integers(0, 255, (36, 48, 3), dtype=np.uint8))
                metas[w] = dict(model_image_size=[W, H], frame_ids=ids, timestamps_ns=list(range(per)),
                                prediction_uri=str(pred))
                snapshots[w] = SimpleNamespace(image_paths=paths, camera_info_payloads=[{}] * per)
                poses = [dict(rotation=np.eye(3).tolist(), center=[.5 * (w * per + i), 0., 0.]) for i in range(per)]
                poses[1] = None if w == 2 else poses[1]
                records.append(dict(window_index=w, snapshot_manifest_uri='file:///w%d' % w,
                                    snapshot_manifest_sha256=str(w), result_manifest_sha256='%064d' % w,
                                    alignment=dict(placement='frame_pose', quality_passed=True,
                                                   merge_frame_indices=list(range(1 if w else 0, per)),
                                                   frame_poses=poses, depth_scale=1. + .1 * w,
                                                   intrinsics=[[20., 0, 12], [0, 20., 9], [0, 0, 1]])))

            def rescale(record, scale):
                return dict(record, alignment=dict(record['alignment'], depth_scale=scale))

            steps = [records[:2], records[:3], [records[0], rescale(records[1], 1.05), records[2]], records,
                     records[:3] + [dict(records[3], alignment=dict(records[3]['alignment'], frame_poses=[
                         dict(p, center=[p['center'][0] + .01, 0., 0.]) if p else p
                         for p in records[3]['alignment']['frame_poses']]))]]
            fusion = SessionFusion(root / 'incremental', scale_tolerance=0)
            tolerant = SessionFusion(root / 'tolerant')
            counts = []
            with patch('drone3d_reconstruction_core.g4.input_adapter.WindowInputReader.load',
                       side_effect=lambda r: SimpleNamespace(metadata=metas[r['window_index']])), \
                    patch('drone3d_reconstruction_core.g4.input_adapter.local_path',
                          side_effect=lambda uri: root / uri[-2:]), \
                    patch('drone3d_reconstruction_core.snapshot_validator.validate_snapshot',
                          side_effect=lambda path, *args: snapshots[int(Path(path).name[1:])]):
                for step, selected in enumerate(steps):
                    before = fusion.recomputed
                    xyz, rgb, info = fusion.fuse(selected, max_points=0)
                    counts.append(fusion.recomputed - before)
                    # Running sums after updates equal a fresh build up to summation order.
                    fresh = SessionFusion(root / ('fresh%d' % step), scale_tolerance=0)
                    f_xyz, f_rgb, f_info = fresh.fuse(selected, max_points=0)
                    self.assertEqual(info, f_info)
                    np.testing.assert_allclose(xyz, f_xyz, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(rgb, f_rgb, rtol=0, atol=1e-12)
                    # Same points as the delivery fusion; only the 2 cm grid is anchored elsewhere.
                    ref_xyz, _, ref_info, _, _ = fuse_session(selected, root / ('full%d' % step), 0, False)
                    self.assertGreater(ref_info['points_2cm'], 50)
                    self.assertLess(abs(info['points_2cm'] - ref_info['points_2cm']), .2 * ref_info['points_2cm'])
                    self.assertLess(cKDTree(ref_xyz).query(xyz)[0].max(), .02 * 3 ** .5)
                    self.assertLess(cKDTree(xyz).query(ref_xyz)[0].max(), .02 * 3 ** .5)
                # Delivery fusion with the revisit step: every row carries its correction and the cached depth includes it.
                _, _, plain_info, plain_rows, _ = fuse_session(records, root / 'plain', 0, False)
                plain = np.array(np.load(root / 'plain/depth_m.npy'))
                _, _, revisit_info, revisit_rows, _ = fuse_session(records, root / 'revisit', 0, False, True)
                corrections = np.array([r['revisit_log_scale'] for r in revisit_rows])
                self.assertNotIn('revisit', plain_info)
                self.assertNotIn('revisit_log_scale', plain_rows[0])
                self.assertEqual(revisit_info['revisit']['frames'], len(revisit_rows))
                np.testing.assert_allclose(np.load(root / 'revisit/depth_m.npy'),
                                           plain * np.exp(corrections)[:, None, None].astype(np.float32), rtol=1e-6)
                # Default tolerance 0.5 %: a 0.2 % rescale reuses every frame, 1 % does not.
                tolerant.fuse(records[:3])
                for scale, expect_reuse in ((1.1 * 1.002, True), (1.1 * 1.01, False)):
                    before = tolerant.recomputed
                    tolerant.fuse([records[0], rescale(records[1], scale), records[2]])
                    self.assertEqual(tolerant.recomputed == before, expect_reuse, scale)
            # Rows per step: 11, 15, 15, 20, 20. Only new, rescaled or moved frames and their
            # neighbours (within 1.5 m, 3 frames either side here) are recomputed.
            self.assertEqual(counts[0], 11)
            self.assertTrue(4 <= counts[1] < 15, counts)
            self.assertTrue(5 <= counts[2] < 15, counts)
            self.assertTrue(5 <= counts[4] < 20, counts)

    def test_ba_no_ros_import(self):
        import sys
        from drone3d_onboard import ba
        self.assertTrue(callable(ba.run))
        self.assertNotIn('rospy', sys.modules)


if __name__ == '__main__':
    unittest.main()
