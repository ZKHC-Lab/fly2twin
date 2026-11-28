# SPDX-License-Identifier: Apache-2.0
"""CPU-side display contract; external viewers must consume/validate these fields."""

import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from recon.geometry.transforms import resize_K

PREVIEW_WARNING = '仅 DC 颜色的轻量预览，视角相关颜色丢失，细节明显发软'


def field_of_view(K, width, height):
    """Full sensor-edge angles; pixel centers use [-.5, size-.5] boundaries.

    Off-center principal points need asymmetric frusta; scalar FOV alone is not
    sufficient. Keep K authoritative rather than silently assuming a centered lens.
    """
    K = np.asarray(K, float)
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or width <= 0
        or height <= 0
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.allclose(K[2], [0, 0, 1])
        or not np.allclose([K[0, 1], K[1, 0]], 0)
    ):
        raise ValueError('Need finite zero-skew pinhole K and positive resolution')
    angles = []
    for size, focal, principal in ((width, K[0, 0], K[0, 2]), (height, K[1, 1], K[1, 2])):
        angles.append(
            float(np.degrees(np.arctan((size - 0.5 - principal) / focal) - np.arctan((-0.5 - principal) / focal)))
        )
    return dict(horizontal_deg=angles[0], vertical_deg=angles[1])


def camera_records(data):
    width, height = map(int, data['manifest']['resolution'])
    return [
        dict(
            frame_id=int(frame),
            resolution=[width, height],
            K=np.asarray(K).tolist(),
            fov=field_of_view(K, width, height),
        )
        for frame, K in zip(data['frame_ids'], data['K_native'])
    ]


def viewer_intrinsics(scene, frame_id, width, height, supplied_fov=None):
    """Return the calibrated viewport K, rejecting guessed FOV or stretched views.

    Use the returned off-axis projection; letterbox the viewport if necessary.
    """
    record = next(r for r in scene['capture_cameras'] if r['frame_id'] == frame_id)
    sw, sh = record['resolution']
    if width <= 0 or height <= 0 or not np.isclose(width / height, sw / sh, rtol=1e-6):
        raise ValueError('Preserve capture aspect ratio; letterbox instead of stretching/cropping')
    if supplied_fov is not None:
        for key in ('horizontal_deg', 'vertical_deg'):
            if key in supplied_fov and not np.isclose(supplied_fov[key], record['fov'][key], atol=1e-6, rtol=0):
                raise ValueError('Viewer FOV differs from capture calibration; do not use a guessed FOV')
    return resize_K(np.asarray(record['K']), width, height, sw, sh)


def capture_corridor(data):
    poses = np.asarray(data['c2w_m'])
    units = 'scene_units' if data.get('arbitrary_scale', False) else 'meters'
    return dict(
        frame_ids=np.asarray(data['frame_ids']).tolist(),
        centers=poses[:, :3, 3].tolist(),
        rotations_c2w=poses[:, :3, :3].tolist(),
        forward_axes=poses[:, :3, 2].tolist(),
        training_frame_ids=np.asarray(data['frame_ids'])[data['train_indices']].tolist(),
        recommended_max_distance=0.15,
        distance_units=units,
        normalized_max_distance=0.15 / float(data['meters_per_unit']),
        recommended_max_angle_deg=15.0,
        angle_metric='full SO(3) relative rotation',
        constraint='Both limits relative to the same training camera; no artificial altitude increase',
        guarantee='Conservative navigation recommendation, not a visual-quality guarantee',
    )


def within_capture_corridor(scene, position, rotation_c2w):
    """Check both bounds against the SAME training camera in exported coordinates."""
    corridor = scene['capture_corridor']
    train = np.isin(corridor['frame_ids'], corridor['training_frame_ids'])
    centers = np.asarray(corridor['centers'])[train]
    rotations = np.asarray(corridor['rotations_c2w'])[train]
    relative = np.asarray(rotation_c2w).T @ rotations
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    distances = np.linalg.norm(centers - np.asarray(position), axis=1)
    return bool(
        np.any(
            (distances <= corridor['recommended_max_distance'] + 1e-12)
            & (angles <= corridor['recommended_max_angle_deg'] + 1e-10)
        )
    )


def transform_scene_coordinates(points, quaternions_wxyz, c2w, target_from_source):
    """Rigid display-basis conversion: positions, ellipsoid axes AND cameras together.

    SH coefficients are expressed in the source-world basis. A scene-graph viewer
    must inverse-rotate its SH lookup directions, or rotate all SH bands before
    baking this transform into a standalone PLY. This helper does not rotate SH.
    """
    T = np.asarray(target_from_source, float)
    if T.shape != (4, 4):
        raise ValueError('Coordinate conversion requires a 4x4 proper rigid transform')
    R = T[:3, :3]
    if (
        not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(R.T @ R, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(R), 1)
    ):
        raise ValueError('Coordinate conversion must be a proper rigid transform, not a reflection/axis-only flip')
    q = np.asarray(quaternions_wxyz)
    orientations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    xyzw = Rotation.from_matrix(R @ orientations).as_quat()
    return np.asarray(points) @ R.T + T[:3, 3], xyzw[:, [3, 0, 1, 2]], T @ np.asarray(c2w)


def require_full_sh_model(out):
    """Never silently declare a DC-only preview to be a complete export."""
    if not (out / 'gaussians.ply').is_file():
        if (out / 'preview.splat').is_file():
            warnings.warn('Only preview.splat exists: ' + PREVIEW_WARNING, RuntimeWarning, stacklevel=2)
        raise RuntimeError('Complete export requires full-SH gaussians.ply')
