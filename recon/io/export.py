# SPDX-License-Identifier: Apache-2.0
import numpy as np
from .ply import write_ply,point_vertices
from .common import dump
from .presentation import (PREVIEW_WARNING, camera_records, capture_corridor, require_full_sh_model)


def scene_up(root):
    """Source-world up direction: the fitted floor normal for image-only SfM scenes scaled by an
    assumed camera height (pose/report.json scale_assumption), otherwise +Z."""
    from pathlib import Path
    import json
    report=Path(root)/'pose/report.json'
    fit=json.loads(report.read_text()).get('scale_assumption',{}) if report.exists() else {}
    n=np.asarray(fit.get('plane_normal',[0,0,1]),float)
    return (n/np.linalg.norm(n)).tolist()


def scene_bounds(points,lo=.5,hi=99.5):
    """Axis-aligned extent of the reconstruction in the source world (robust percentiles)."""
    return dict(min=np.percentile(points,lo,axis=0).tolist(),max=np.percentile(points,hi,axis=0).tolist(),
                percentiles=[lo,hi],source='cloud.ply')


def export_scene(root,out,data,reference_drone=False,name=None,subtitle=None):
    import torch
    # Do not silently reorient a full-SH model: its angular color basis also needs rotation.
    if not np.allclose(data['manifest'].get('world_up', [0, 0, 1]), [0, 0, 1]):
        raise ValueError('Export requires a declared Z-up source world; align the whole scene before training')
    cameras = camera_records(data)
    corridor = capture_corridor(data)
    with np.load(root/'fuse/initial.npz',allow_pickle=False) as z:
        write_ply(out/'cloud.ply',point_vertices(z['points'],z['colors']))
        bounds=scene_bounds(z['points'])
    up=scene_up(root)
    z_up=bool(np.allclose(up,[0,0,1]))
    checkpoint=torch.load(root/'train/final.pt',map_location='cpu',weights_only=False)
    splats={k:v.detach().numpy() for k,v in checkpoint['splats'].items()}
    meters=data['meters_per_unit'];center=np.asarray(data['center'])
    points=splats['means']*meters+center
    scales=splats['scales']+np.log(meters)
    # Standard Gaussian PLY: log axes, unactivated opacity, wxyz quaternion, channel-major SH.
    dc=splats['sh0'][:,0]
    coefficient_count = splats['sh0'].shape[1] + splats['shN'].shape[1]
    sh_degree = int(round(np.sqrt(coefficient_count))) - 1
    if (sh_degree+1)**2 != coefficient_count or checkpoint['config']['sh_degree'] != sh_degree:
        raise ValueError('Checkpoint SH coefficient count disagrees with declared degree')
    rest=splats['shN'].transpose(0,2,1).reshape(len(points),-1)
    columns={n:points[:,i] for i,n in enumerate(('x','y','z'))}
    columns.update({n:np.zeros(len(points)) for n in ('nx','ny','nz')})
    columns.update({f'f_dc_{i}':dc[:,i] for i in range(3)})
    columns.update({f'f_rest_{i}':rest[:,i] for i in range(rest.shape[1])})
    columns['opacity']=splats['opacities']
    columns.update({f'scale_{i}':scales[:,i] for i in range(3)})
    quats=splats['quats']
    if reference_drone: quats=quats/np.linalg.norm(quats,axis=1,keepdims=True)
    columns.update({f'rot_{i}':quats[:,i] for i in range(4)})
    vertices=np.empty(len(points),dtype=[(k,'<f4') for k in columns])
    for key,value in columns.items(): vertices[key]=value
    write_ply(out/'gaussians.ply',vertices)
    # 32-byte browser splat format; DC only, not SH3-equivalent rendering.
    preview=np.empty(len(points),dtype=[('position','<f4',(3,)),('scale','<f4',(3,)),('color','u1',(4,)),('rotation','u1',(4,))])
    preview['position']=points;preview['scale']=np.exp(scales)
    preview['color'][:,:3]=np.rint(np.clip(.5+.28209479177387814*dc,0,1)*255).astype('u1')
    preview['color'][:,3]=np.rint(255/(1+np.exp(-splats['opacities']))).astype('u1')
    q=splats['quats'];q=q/np.linalg.norm(q,axis=1,keepdims=True)
    preview['rotation']=np.clip(q*128+128,0,255).astype('u1')
    if reference_drone:
        preview['scale']=np.exp(splats['scales'])*meters
        preview['color'][:,:3]=np.clip((.5+.28209479177387814*dc)*255,0,255).astype('u1')
        preview['color'][:,3]=np.clip(255/(1+np.exp(-np.clip(splats['opacities'],-80,80))),0,255).astype('u1')
    preview.tofile(out/'preview.splat')
    require_full_sh_model(out)
    np.savez(out/'cameras.npz',frame_ids=data['frame_ids'],c2w=data['c2w_m'],K=data['K_native'],
             timestamp_ns=data['timestamp_ns'],resolution=data['manifest']['resolution'])
    dump(out/'scene.json',dict(schema_version=2,units='arbitrary' if data.get('arbitrary_scale',False) else 'meters',
         coordinate_system=('Z-up' if z_up else 'world_up (not Z)')+', right-handed world; optical camera right/down/forward; c2w maps camera into world',
         world_up=up,bounds=bounds,**({'name':name} if name else {}),**({'subtitle':subtitle} if subtitle else {}),
         coordinates=dict(world_up=up,handedness='right',quaternion_order='wxyz',
             gravity_alignment=('Not inferred from images; Z-up is an input/display convention' if z_up else
                 'Fitted floor-plane normal (pose/report.json scale_assumption); viewers rotate world_up onto their vertical'),
             basis_change='Transform positions, Gaussian orientations and cameras together with a proper rigid transform',
             sh_basis='source world; inverse-rotate lookup directions for a viewer basis change, or rotate SH bands when baking'),
         cloud='cloud.ply',gaussians='gaussians.ply',
         preferred_model=dict(path='gaussians.ply',format='INRIA Gaussian PLY',sh_degree=sh_degree,full_sh=True),
         cameras='cameras.npz',capture_cameras=cameras,capture_corridor=corridor,
         viewer_requirements=dict(projection='Use capture_cameras K and resolution, including off-center principal point',
             fov='Horizontal and vertical FOV must match the selected capture camera; never substitute a guessed FOV',
             navigation='Enforce capture_corridor limits; do not elevate the view for presentation'),
         preview='preview.splat',preview_sh_degree=0,training_sh_degree=sh_degree,
         preview_model=dict(path='preview.splat',sh_degree=0,full_sh=False,description=PREVIEW_WARNING,
             view_dependent_color_lost=True),
         checkpoint=str((root/'train/final.pt').resolve()),checkpoint_coordinates='normalized',
         world_from_normalized=dict(center=data['center'],meters_per_unit=meters),gaussian_count=len(points)))
