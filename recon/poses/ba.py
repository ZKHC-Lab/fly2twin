# SPDX-License-Identifier: Apache-2.0
"""Landing BA against odometry priors. pycolmap's prior BA constrains positions only, so a stretch of cameras
can turn with its points about the line through their centres (e.g. 12 deg about the optical axis on
office_gs_01) without changing any position residual, and which way it turns depends on match noise.

`rotation_sigma_deg` (off by default) is for scenes where that happens, i.e. few observations per frame:
rotations then stay fixed in the joint BA and move in a separate per-image step with a rotation prior, and
reprojection uses a Cauchy loss because the first round admits 16 px observations. On a well-observed scene
(office_gs_04) the prior pulls rotations towards the odometry and away from the images: held-out 3DGS PSNR
24.38 -> 23.22 dB, so it is not the default."""

import numpy as np
import json
import time
from recon.io.common import dump
from .options import known_model, pipeline_options, ba_options


def refine_rotations(rec, prior_cw, sigma_rad):
    """Per-image rotation about the fixed projection centre, points fixed: Cauchy reprojection (1 px) plus a
    rotation prior of `sigma_rad` towards `prior_cw[image_id]` (3x3 cam_from_world). Returns rotation changes (rad)."""
    import pycolmap as pc
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    ids = np.sort(np.fromiter(rec.point3D_ids(), np.int64))
    xyz = np.array([rec.point3D(int(i)).xyz for i in ids]).reshape(-1, 3)
    changes = {}
    for image_id in rec.reg_image_ids():
        im = rec.image(image_id)
        pose = im.cam_from_world()
        R0 = pose.rotation.matrix()
        centre = -R0.T @ pose.translation
        prior = prior_cw[image_id]
        observed = im.get_observation_points2D()
        if observed:
            uv = np.array([p.xy for p in observed])
            X = xyz[np.searchsorted(ids, [p.point3D_id for p in observed])] - centre
        else:
            uv, X = np.empty((0, 2)), np.empty((0, 3))
        fx, fy, cx, cy = im.camera.params[:4]

        def residuals(v):
            R = Rotation.from_rotvec(v).as_matrix() @ R0
            p = X @ R.T
            e = (np.c_[fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy] - uv).ravel()
            robust = np.sign(e) * np.sqrt(np.log1p(e * e))  # squared: Cauchy rho with 1 px scale, as in the BA
            return np.r_[robust, Rotation.from_matrix(R @ prior.T).as_rotvec() / sigma_rad]

        v = least_squares(residuals, np.zeros(3), method='lm' if len(uv) >= 2 else 'trf').x
        R = Rotation.from_rotvec(v).as_matrix() @ R0
        im.frame.set_cam_from_world(im.camera_id, pc.Rigid3d(pc.Rotation3d(R), -R @ centre))
        changes[image_id] = float(np.linalg.norm(v))
    return changes


def build(data, database, images, out, sigma=0.05, rotation_sigma_deg=None):
    import pycolmap as pc
    from scipy.spatial.transform import Rotation

    HERE, RGB, group = out, images, "prior"
    start = time.perf_counter()
    model = HERE / f'model_{group}'
    cameras = HERE / 'cameras.npz'
    if model.exists() or cameras.exists() or (HERE / f'{group}_round1').exists():
        raise FileExistsError(f'Existing {group} outputs; refusing to overwrite')
    rec = known_model(data, database)
    lookup = {name: i for i, name in enumerate(data['image_names'])}
    priors = [
        pc.PosePrior(
            pose_prior_id=im.image_id,
            corr_data_id=im.data_id,
            position=data['c2w_L'][lookup[im.name], :3, 3],
            position_covariance=np.eye(3) * sigma**2,
            coordinate_system=pc.PosePriorCoordinateSystem.CARTESIAN,
        )
        for im in rec.images.values()
    ]
    prior_cw = {im.image_id: data['c2w_L'][lookup[im.name], :3, :3].T for im in rec.images.values()}
    prior_options = pc.PosePriorBundleAdjustmentOptions()
    options_ba = ba_options()
    if rotation_sigma_deg is not None:
        options_ba.constant_rig_from_world_rotation = True
        options_ba.ceres.loss_function_type = pc.LossFunctionType.CAUCHY
        options_ba.ceres.loss_function_scale = 1.0
    dump(
        HERE / f'prior_options_{group}.json',
        dict(
            pycolmap=pc.__version__,
            sigma_m=sigma,
            rotation_sigma_deg=rotation_sigma_deg,
            prior_count=len(priors),
            prior_options=prior_options.summary(),
            ba_options=options_ba.summary(),
            final_sim3=False,
        ),
    )
    rounds = []
    for iteration, threshold in enumerate((16.0, 8.0, 4.0), 1):
        options = pipeline_options()
        tri = options.triangulation
        tri.merge_max_reproj_error = tri.complete_max_reproj_error = threshold
        angular = float(np.degrees(np.arctan(threshold / np.mean([data['K_L'][0, 0, 0], data['K_L'][0, 1, 1]]))))
        tri.create_max_angle_error = tri.continue_max_angle_error = tri.re_max_angle_error = angular
        options.mapper.filter_max_reproj_error = threshold
        out = HERE / f'{group}_round{iteration}'
        out.mkdir()
        rec = pc.triangulate_points(
            rec, database, RGB, out, clear_points=False, options=options, refine_intrinsics=False
        )
        turned = None
        if rotation_sigma_deg is not None:
            # Rotations get their own step with a prior; the joint BA keeps them fixed (see the module docstring).
            turned = np.degrees(list(refine_rotations(rec, prior_cw, np.radians(rotation_sigma_deg)).values()))
        config = pc.BundleAdjustmentConfig()
        for image_id in rec.reg_image_ids():
            config.add_image(image_id)
        for camera_id in rec.cameras:
            config.set_constant_cam_intrinsics(camera_id)
        adjuster = pc.create_pose_prior_bundle_adjuster(options_ba, prior_options, config, priors, rec)
        summary = adjuster.solve()
        del adjuster  # its Ceres problem is as large as the model (office_gs_04: 42 M residuals); not needed past here
        print(summary.brief_report(), flush=True)
        if not summary.is_solution_usable():
            raise RuntimeError('Global BA returned an unusable solution in round %d' % iteration)
        filtered = pc.ObservationManager(rec).filter_all_points3D(threshold, 1.5)
        rec.update_point_3d_errors()
        rec.write(out)
        centers = np.array(
            [
                np.linalg.norm(im.projection_center() - data['c2w_L'][lookup[im.name], :3, 3])
                for im in rec.images.values()
            ]
        )
        angles = np.array(
            [
                np.degrees(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            im.cam_from_world().rotation.matrix() @ prior_cw[im.image_id].T
                        ).as_rotvec()
                    )
                )
                for im in rec.images.values()
            ]
        )
        row = dict(
            round=iteration,
            threshold_px=threshold,
            angular_threshold_deg=angular,
            filter_threshold_px=threshold,
            filter_min_tri_angle_deg=1.5,
            points=rec.num_points3D(),
            filtered_observations=int(filtered),
            reprojection_mean_px=rec.compute_mean_reprojection_error(),
            center_m_median_p90=np.percentile(centers, [50, 90]).tolist(),
            rotation_deg_median_p90=np.percentile(angles, [50, 90]).tolist(),
            rotation_step_deg_median_p90_max=None if turned is None else [*np.percentile(turned, [50, 90]), turned.max()],
            prior_ba_summary=summary.brief_report(),
            prior_ba_termination=str(summary.termination_type),
            prior_ba_residuals=summary.num_residuals,
            prior_ba_solution_usable=summary.is_solution_usable(),
            elapsed_seconds=time.perf_counter() - start,
        )
        rounds.append(row)
        dump(HERE / f'{group}_rounds.json', rounds)
        print(json.dumps(row), flush=True)
    rec.extract_colors_for_all_images(RGB)
    model.mkdir()
    rec.write(model)
    poses = data['c2w_L'].copy()
    obs = np.zeros(len(poses), dtype=np.int64)
    registered = []
    for image_id in rec.reg_image_ids():
        im = rec.image(image_id)
        i = lookup[im.name]
        poses[i, :3] = im.cam_from_world().inverse().matrix()
        obs[i] = im.num_points3D
        registered.append(i)
    deviation = np.linalg.norm(poses[:, :3, 3] - data['c2w_L'][:, :3, 3], axis=1)
    far = deviation > 3 * sigma
    if len(registered) != len(poses):
        raise RuntimeError('BA lost registered frames; refusing silent initial-pose fallback')
    np.savez(
        cameras,
        frame_ids=data['frame_ids'],
        timestamp_ns=data['timestamp_ns'],
        blur=data['blur'],
        c2w=poses,
        K=data['K_L'],
        observations=obs,
        registered_indices=np.sort(registered),
        points=np.array([p.xyz for p in rec.points3D.values()], np.float32).reshape(-1, 3),
        colors=(np.array([p.color for p in rec.points3D.values()], float).reshape(-1, 3) / 255).astype(np.float32),
    )
    dump(
        HERE / f'reconstruction_{group}.json',
        dict(
            group=group,
            rounds=rounds,
            seconds=time.perf_counter() - start,
            points3D=rec.num_points3D(),
            registered_frames=rec.num_reg_images(),
            input_frames=len(poses),
            observations=int(obs.sum()),
            zero_observation_frame_ids=data['frame_ids'][obs == 0].tolist(),
            # Images and odometry disagree here by more than 3 sigma; seeds do not change these frames, so it is not
            # match noise, but which side is wrong is not decided here (office_gs_01: two stretches of 5 and 13 frames).
            prior_disagreement_threshold_m=3 * sigma,
            prior_disagreement_frame_ids=data['frame_ids'][far].tolist(),
            prior_disagreement_m=np.round(deviation[far], 4).tolist(),
            prior_disagreement_observations=obs[far].tolist(),
            camera_params={str(k): c.params.tolist() for k, c in rec.cameras.items()},
            final_sim3=False,
            mode='Cartesian position prior BA'
            if rotation_sigma_deg is None
            else 'Cartesian position prior BA with fixed rotations; per-image rotation prior step',
            rotation_sigma_deg=rotation_sigma_deg,
            sigma_m=sigma,
            zero_observation_pose='actual final model pose; not visually constrained',
        ),
    )
