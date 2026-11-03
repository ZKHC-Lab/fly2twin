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
            row=dict(frame_id='source0',timestamp_ns=123,image_path=str(image),c2w=np.eye(4),K=K,camera_info={'K':K.ravel().tolist(),'D':[0.]*5,'width':1280,'height':720})
            ext=dict(translation=[.056558,-.006717,-.077751],rotation=Rotation.from_quat([-.494176,.506142,-.498990,.500618]).as_matrix().tolist())
            win=dict(window_id=0,source=dict(body_from_camera=ext,prediction_uri=str(pred),frame_ids=['source0'],timestamps_ns=[123]),alignment=dict(depth_scale=2.,intrinsics=K.tolist(),merge_frame_indices=[0]))
            result=(np.array([[1.,2.,3.]]),np.array([[1.,0.,0.]]),{},[row],[win])
            with patch('drone3d_reconstruction_core.continuous_web._formal_records_for',return_value=[]), patch('drone3d_onboard.export.fuse_session',return_value=result):
                manifest=export_session(records,sid,root/'delivery',capture)
                with self.assertRaises(FileExistsError):export_session(records,sid,root/'delivery',capture)
            self.assertEqual(manifest['pose_time_offset_stage'],'capture')
            self.assertEqual(cv2.imread(str(root/'delivery/keyframes/000000.jpg')).shape,(720,1280,3))
            for name,value in manifest['hashes'].items():self.assertEqual(digest(root/'delivery'/name),value)
            with np.load(root/'delivery/poses.npz') as data:np.testing.assert_array_equal(data['c2w'],np.eye(4)[None])

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
            self.assertEqual([r['frame_id'] for r in rows], ['f1'])
            self.assertEqual(windows[0]['unposed'], ['f0'])
            np.testing.assert_array_equal(depth[0], 2.)

    def test_ba_no_ros_import(self):
        import sys
        from drone3d_onboard import ba
        self.assertTrue(callable(ba.run))
        self.assertNotIn('rospy', sys.modules)


if __name__ == '__main__':
    unittest.main()
