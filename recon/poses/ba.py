# SPDX-License-Identifier: Apache-2.0
import numpy as np
import json
import time
from recon.io.common import dump
from .options import known_model, pipeline_options, ba_options

def build(data, database, images, out, sigma=.05):
    import pycolmap as pc
    HERE, RGB, group = out, images, "prior"
    start = time.perf_counter()
    model = HERE / f'model_{group}'
    cameras = HERE / 'cameras.npz'
    if model.exists() or cameras.exists() or (HERE / f'{group}_round1').exists():
        raise FileExistsError(f'Existing {group} outputs; refusing to overwrite')
    rec = known_model(data, database)
    lookup = {name: i for i,name in enumerate(data['image_names'])}
    priors = [pc.PosePrior(pose_prior_id=im.image_id, corr_data_id=im.data_id,
        position=data['c2w_L'][lookup[im.name], :3, 3],
        position_covariance=np.eye(3)*sigma**2,
        coordinate_system=pc.PosePriorCoordinateSystem.CARTESIAN)
        for im in rec.images.values()]
    prior_options = pc.PosePriorBundleAdjustmentOptions()
    options_ba = ba_options()
    dump(HERE / f'prior_options_{group}.json', dict(pycolmap=pc.__version__, sigma_m=sigma,
        prior_count=len(priors), prior_options=prior_options.summary(),
        ba_options=options_ba.summary(), final_sim3=False))
    rounds = []
    for iteration, threshold in enumerate((16., 8., 4.), 1):
        options = pipeline_options()
        tri = options.triangulation
        tri.merge_max_reproj_error = tri.complete_max_reproj_error = threshold
        angular = float(np.degrees(np.arctan(threshold / np.mean(
            [data['K_L'][0, 0, 0], data['K_L'][0, 1, 1]]))))
        tri.create_max_angle_error = tri.continue_max_angle_error = tri.re_max_angle_error = angular
        options.mapper.filter_max_reproj_error = threshold
        out = HERE / f'{group}_round{iteration}'
        out.mkdir()
        rec = pc.triangulate_points(rec, database, RGB, out, clear_points=False,
                                    options=options, refine_intrinsics=False)
        config = pc.BundleAdjustmentConfig()
        for image_id in rec.reg_image_ids():
            config.add_image(image_id)
        for camera_id in rec.cameras:
            config.set_constant_cam_intrinsics(camera_id)
        adjuster = pc.create_pose_prior_bundle_adjuster(options_ba, prior_options, config, priors, rec)
        summary = adjuster.solve()
        print(summary.brief_report(), flush=True)
        if not summary.is_solution_usable():
            raise RuntimeError('Global BA returned an unusable solution in round %d' % iteration)
        filtered = pc.ObservationManager(rec).filter_all_points3D(threshold, 1.5)
        rec.update_point_3d_errors()
        rec.write(out)
        centers = np.array([np.linalg.norm(im.projection_center()-data['c2w_L'][lookup[im.name], :3, 3])
                            for im in rec.images.values()])
        row = dict(round=iteration, threshold_px=threshold, angular_threshold_deg=angular,
            filter_threshold_px=threshold, filter_min_tri_angle_deg=1.5,
            points=rec.num_points3D(), filtered_observations=int(filtered),
            reprojection_mean_px=rec.compute_mean_reprojection_error(),
            center_m_median_p90=np.percentile(centers, [50, 90]).tolist(),
            prior_ba_summary=summary.brief_report(), prior_ba_termination=str(summary.termination_type),
            prior_ba_residuals=summary.num_residuals, prior_ba_solution_usable=summary.is_solution_usable(),
            elapsed_seconds=time.perf_counter()-start)
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
    if len(registered) != len(poses):
        raise RuntimeError('BA lost registered frames; refusing silent initial-pose fallback')
    np.savez(cameras, frame_ids=data['frame_ids'], timestamp_ns=data['timestamp_ns'], blur=data['blur'], c2w=poses, K=data['K_L'], observations=obs,
             registered_indices=np.sort(registered),
             points=np.array([p.xyz for p in rec.points3D.values()],np.float32).reshape(-1,3),
             colors=(np.array([p.color for p in rec.points3D.values()],float).reshape(-1,3)/255).astype(np.float32))
    dump(HERE / f'reconstruction_{group}.json', dict(group=group, rounds=rounds,
        seconds=time.perf_counter()-start, points3D=rec.num_points3D(),
        registered_frames=rec.num_reg_images(), input_frames=len(poses), observations=int(obs.sum()),
        zero_observation_frame_ids=data['frame_ids'][obs == 0].tolist(),
        camera_params={str(k): c.params.tolist() for k, c in rec.cameras.items()},
        final_sim3=False, mode='Cartesian position prior BA', sigma_m=sigma,
        zero_observation_pose='actual final model pose; not visually constrained'))
