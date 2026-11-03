# SPDX-License-Identifier: Apache-2.0
import numpy as np

def pipeline_options():
    import pycolmap as pc
    options = pc.IncrementalPipelineOptions()
    options.num_threads = options.mapper.num_threads = 4
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    options.mapper.filter_min_tri_angle = 1.5
    return options


def ba_options():
    import pycolmap as pc
    options = pc.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.ceres.solver_options.num_threads = 4
    options.ceres.solver_options.max_num_iterations = 100
    return options


def known_model(data, path):
    import pycolmap as pc
    if not path.is_file():
        raise FileNotFoundError(path)
    rec = pc.Reconstruction()
    lookup = {name: i for i, name in enumerate(data['image_names'])}
    with pc.Database.open(path) as db:
        for cam in db.read_all_cameras():
            rec.add_camera_with_trivial_rig(cam)
        for image in db.read_all_images():
            i = lookup[image.name]
            new = pc.Image(name=image.name, image_id=image.image_id, camera_id=image.camera_id,
                           keypoints=db.read_keypoints(image.image_id)[:, :2])
            rec.add_image_with_trivial_frame(new, pc.Rigid3d(np.linalg.inv(data['c2w_L'][i])[:3]))
    return rec
