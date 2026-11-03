"""Export normalized, non-metric phone models and the shared viewer contract."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .render_roam import export_ply, export_splat
from .pipeline import load_data
from recon.io.common import dump
from recon.io.presentation import camera_records, capture_corridor, PREVIEW_WARNING


def export(out):
    checkpoint=torch.load(out/'gs/final.pt',map_location='cpu',weights_only=False)
    data,images=load_data(out,0)
    # Viewer coordinates are normalized, as in render_roam; do not invent metric scale.
    with np.load(out/'cameras.npz',allow_pickle=False) as source:
        poses=source['c2w'].copy()
        poses[:,:3,3]=(poses[:,:3,3]-np.asarray(data['center']))/data['scale']
        native_K=source['K'].copy()
    data.update(c2w_m=poses,K_native=native_K,arbitrary_scale=True,
                manifest=dict(resolution=[images.width,images.height]),meters_per_unit=1.)
    corridor=capture_corridor(data)
    corridor.update(recommended_max_distance=.12,normalized_max_distance=.12,recommended_max_angle_deg=8.)
    ply=export_ply(checkpoint['splats'],out/'scene.ply',radius=5.)
    splat=export_splat(checkpoint['splats'],out/'scene.splat',.02,0,radius=5.)
    dump(out/'scene.json',dict(schema_version=2,units='arbitrary',
         coordinates=dict(world_up=[0,0,1],handedness='right',quaternion_order='wxyz',
                          gravity_alignment='not inferred',sh_basis='source world'),
         preferred_model=dict(path='scene.ply',format='INRIA Gaussian PLY',sh_degree=3,full_sh=True),
         preview_model=dict(path='scene.splat',sh_degree=0,full_sh=False,description=PREVIEW_WARNING),
         capture_cameras=camera_records(data),capture_corridor=corridor,
         gaussian_count=ply['gaussians'],normalization=checkpoint['config']['normalization_center'],
         export_report=dict(ply=ply,splat=splat)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path)
    export(parser.parse_args().out)


if __name__=='__main__':
    main()
