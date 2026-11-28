import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from recon.io.export import export_scene
from recon.io.ply import read_ply
from recon.io.presentation import (
    field_of_view,
    viewer_intrinsics,
    within_capture_corridor,
    transform_scene_coordinates,
    require_full_sh_model,
)


@pytest.fixture
def scene_inputs(tmp_path):
    torch = pytest.importorskip('torch')
    root = tmp_path / 'run'
    out = root / 'export'
    for path in [root / 'train', root / 'fuse', out]:
        path.mkdir(parents=True)
    np.savez(
        root / 'fuse/initial.npz',
        points=np.array([[0.0, 0, 0], [1, 2, 3]], np.float32),
        colors=np.ones((2, 3), np.float32) * 0.5,
    )
    splats = dict(
        means=torch.tensor([[0.0, 0, 0], [1.0, 2, 3]]),
        scales=torch.zeros(2, 3),
        quats=torch.tensor([[1.0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]]),
        opacities=torch.zeros(2),
        sh0=torch.ones(2, 1, 3) * 0.2,
        shN=torch.arange(90, dtype=torch.float32).reshape(2, 15, 3) / 100,
    )
    torch.save(dict(splats=splats, config={'sh_degree': 3}), root / 'train/final.pt')
    K = np.array([[500.0, 0, 319.5], [0, 360 / (2 * np.tan(np.deg2rad(43.3 / 2))), 179.5], [0, 0, 1]])
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0, 1, 10]
    data = dict(
        meters_per_unit=2.0,
        center=[1.0, 2.0, 3.0],
        frame_ids=np.array([101, 102, 103]),
        c2w_m=poses,
        K_native=np.tile(K, (3, 1, 1)),
        timestamp_ns=np.arange(3),
        train_indices=np.array([0, 1]),
        manifest={'resolution': [640, 360]},
    )
    return root, out, data


def test_export_prefers_complete_sh_and_labels_dc_preview(scene_inputs):
    root, out, data = scene_inputs
    export_scene(root, out, data)
    scene = json.loads((out / 'scene.json').read_text())
    vertices = read_ply(out / 'gaussians.ply')
    assert scene['preferred_model']['path'] == 'gaussians.ply'
    assert scene['preferred_model']['full_sh'] and scene['preferred_model']['sh_degree'] == 3
    assert 'f_rest_44' in vertices.dtype.names and len(vertices.dtype.names) == 62
    np.testing.assert_allclose(vertices['f_rest_15'], [0.01, 0.46])
    assert scene['preview_model']['view_dependent_color_lost'] and scene['preview_model']['sh_degree'] == 0
    assert '细节明显发软' in scene['preview_model']['description']
    assert (out / 'preview.splat').stat().st_size == 64


def test_export_embeds_capture_intrinsics_and_actual_fov(scene_inputs):
    root, out, data = scene_inputs
    export_scene(root, out, data)
    scene = json.loads((out / 'scene.json').read_text())
    camera = scene['capture_cameras'][0]
    np.testing.assert_array_equal(camera['K'], data['K_native'][0])
    assert camera['fov']['vertical_deg'] == pytest.approx(43.3)
    assert camera['fov']['horizontal_deg'] == pytest.approx(np.degrees(2 * np.arctan(640 / 1000)))
    with np.load(out / 'cameras.npz') as c:
        np.testing.assert_array_equal(c['resolution'], [640, 360])


def test_viewer_rejects_58_degree_guess_and_resizes_calibrated_k(scene_inputs):
    root, out, data = scene_inputs
    export_scene(root, out, data)
    scene = json.loads((out / 'scene.json').read_text())
    with pytest.raises(ValueError, match='FOV differs'):
        viewer_intrinsics(scene, 101, 640, 360, {'vertical_deg': 58.0})
    resized = viewer_intrinsics(scene, 101, 320, 180)
    assert field_of_view(resized, 320, 180)['vertical_deg'] == pytest.approx(43.3)
    with pytest.raises(ValueError, match='aspect ratio'):
        viewer_intrinsics(scene, 101, 640, 480)


def test_offcenter_field_of_view_uses_sensor_edges():
    K = np.array([[100.0, 0, 25], [0, 200.0, 30], [0, 0, 1]])
    fov = field_of_view(K, 100, 80)
    expected = np.degrees(np.arctan(74.5 / 100) + np.arctan(25.5 / 100))
    assert fov['horizontal_deg'] == pytest.approx(expected)
    assert fov['horizontal_deg'] != pytest.approx(np.degrees(2 * np.arctan(0.5)))


def test_corridor_limits_use_same_training_camera(scene_inputs):
    root, out, data = scene_inputs
    export_scene(root, out, data)
    scene = json.loads((out / 'scene.json').read_text())
    c = scene['capture_corridor']
    assert c['distance_units'] == 'meters' and c['normalized_max_distance'] == 0.075
    assert within_capture_corridor(scene, [0.15, 0, 0], Rotation.from_euler('y', 15, degrees=True).as_matrix())
    assert not within_capture_corridor(scene, [0, 0, 0.151], np.eye(3))
    assert not within_capture_corridor(scene, [0, 0, 0], Rotation.from_euler('y', 15.1, degrees=True).as_matrix())
    assert not within_capture_corridor(scene, [10, 0, 0], np.eye(3))  # held-out camera is not training coverage
    # Position near one camera and direction near a different one must not pass.
    c['rotations_c2w'][0] = Rotation.from_euler('y', 45, degrees=True).as_matrix().tolist()
    assert not within_capture_corridor(scene, [0, 0, 0], np.eye(3))


def test_nonmetric_corridor_does_not_claim_meters(scene_inputs):
    root, out, data = scene_inputs
    data['arbitrary_scale'] = True
    export_scene(root, out, data)
    scene = json.loads((out / 'scene.json').read_text())
    assert scene['units'] == 'arbitrary' and scene['capture_corridor']['distance_units'] == 'scene_units'
    assert scene['coordinates']['world_up'] == [0, 0, 1] and scene['coordinates']['quaternion_order'] == 'wxyz'


def test_coordinate_roundtrip_preserves_positions_poses_and_covariances():
    points = np.array([[1.0, 2, 3], [-2, 0, 4]])
    orientations = Rotation.from_euler('xyz', [[20, 30, 40], [-10, 50, 5]], degrees=True)
    quats = orientations.as_quat()[:, [3, 0, 1, 2]]
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, 3] = [[4, 5, 6], [0, 2, 1]]
    poses[:, :3, :3] = orientations.as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('x', -90, degrees=True).as_matrix()
    T[:3, 3] = [2, 3, 4]
    p, q, c = transform_scene_coordinates(points, quats, poses, T)
    axes = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    covariance = orientations.as_matrix() @ np.diag([1.0, 4, 9]) @ orientations.as_matrix().transpose(0, 2, 1)
    actual = axes @ np.diag([1.0, 4, 9]) @ axes.transpose(0, 2, 1)
    np.testing.assert_allclose(actual, T[:3, :3] @ covariance @ T[:3, :3].T, atol=1e-12)
    # Camera-local coordinates must not change when world axes change together.
    np.testing.assert_allclose(
        (p - c[:, :3, 3])[:, None, :] @ c[:, :3, :3],
        (points - poses[:, :3, 3])[:, None, :] @ poses[:, :3, :3],
        atol=1e-12,
    )
    pp, qq, cc = transform_scene_coordinates(p, q, c, np.linalg.inv(T))
    np.testing.assert_allclose(pp, points, atol=1e-12)
    np.testing.assert_allclose(cc, poses, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(qq[:, [1, 2, 3, 0]]).as_matrix(), orientations.as_matrix(), atol=1e-12
    )


def test_coordinate_conversion_rejects_reflection():
    T = np.eye(4)
    T[0, 0] = -1
    with pytest.raises(ValueError, match='proper rigid'):
        transform_scene_coordinates(np.zeros((1, 3)), np.array([[1, 0, 0, 0]]), np.eye(4)[None], T)


def test_splat_only_export_warns_and_fails_completion(tmp_path):
    (tmp_path / 'preview.splat').write_bytes(b'')
    with pytest.warns(RuntimeWarning, match='Only preview.splat'):
        with pytest.raises(RuntimeError, match='full-SH'):
            require_full_sh_model(tmp_path)


def test_export_cannot_silently_skip_full_sh_ply(scene_inputs, monkeypatch):
    import recon.io.export as module

    root, out, data = scene_inputs
    original = module.write_ply
    monkeypatch.setattr(module, 'write_ply', lambda p, v: None if Path(p).name == 'gaussians.ply' else original(p, v))
    with pytest.warns(RuntimeWarning, match='Only preview.splat'):
        with pytest.raises(RuntimeError, match='full-SH'):
            export_scene(root, out, data)
    assert not (out / 'scene.json').exists()


def test_fixed_normalization_is_distinct_from_new_scene_recalculation():
    from recon.geometry.transforms import normalization

    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0, 2, 4]
    fixed = dict(
        normalization_center=[-7.171480196455258, 8.188519718807234, 0.9049076656900958],
        meters_per_unit=8.941428986458085,
        scene_scale=1.1180742025375368,
    )
    center, meters, scale = normalization(poses, [0, 1], fixed)
    np.testing.assert_allclose(center, fixed['normalization_center'], atol=1e-7, rtol=0)
    assert meters == fixed['meters_per_unit'] and scale == fixed['scene_scale']
    fresh_center, fresh_meters, _ = normalization(poses, [0, 1])
    np.testing.assert_array_equal(fresh_center, [1, 0, 0])
    assert fresh_meters == 1


def test_export_rejects_unconverted_non_z_up_scene(scene_inputs):
    root, out, data = scene_inputs
    data['manifest']['world_up'] = [0, 1, 0]
    with pytest.raises(ValueError, match='Z-up'):
        export_scene(root, out, data)
    assert not (out / 'scene.json').exists()


def test_drone_reference_export_quantization(scene_inputs):
    import torch

    root, out, data = scene_inputs
    checkpoint = torch.load(root / 'train/final.pt', weights_only=False)
    checkpoint['splats']['quats'] *= 3
    torch.save(checkpoint, root / 'train/final.pt')
    export_scene(root, out, data, reference_drone=True)
    vertices = read_ply(out / 'gaussians.ply')
    q = np.stack([vertices[f'rot_{i}'] for i in range(4)], 1)
    np.testing.assert_array_equal(np.linalg.norm(q, axis=1), np.ones(2))
    rows = np.fromfile(
        out / 'preview.splat', dtype=[('xyz', '<f4', 3), ('scale', '<f4', 3), ('rgba', 'u1', 4), ('quat', 'u1', 4)]
    )
    assert (rows['rgba'][:, 3] == 127).all()
    np.testing.assert_array_equal(rows['scale'], np.full((2, 3), 2.0))
