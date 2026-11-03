#!/usr/bin/env python3
"""Audit actual PLY/SPLAT arrays against reference export formulas and checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from recon.io.ply import read_ply


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['drone','phone'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root
    phone=args.kind=='phone'
    checkpoint=torch.load(root/('gs/final.pt' if phone else 'train/final.pt'),map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];a={k:v.numpy() for k,v in checkpoint['splats'].items()}
    destination=root if phone else root/'export'
    vertices=read_ply(destination/('scene.ply' if phone else 'gaussians.ply'))
    rows=np.fromfile(destination/('scene.splat' if phone else 'preview.splat'),dtype=[
        ('xyz','<f4',3),('scale','<f4',3),('rgba','u1',4),('quat','u1',4)])
    assert len(vertices.dtype.names)==62
    meters=1. if phone else cfg['meters_per_unit']
    center=np.zeros(3) if phone else np.array(cfg['normalization_center'])
    xyz=a['means'] if phone else a['means']*meters+center
    keep=np.linalg.norm(xyz,axis=1)<=5 if phone else np.ones(len(xyz),bool)
    q=a['quats'] if phone else a['quats']/np.linalg.norm(a['quats'],axis=1,keepdims=True)
    for j,name in enumerate(('x','y','z')):
        np.testing.assert_array_equal(vertices[name],xyz[keep,j].astype('f4'))
    rest=a['shN'].transpose(0,2,1).reshape(len(xyz),45)
    for j in range(45):np.testing.assert_array_equal(vertices[f'f_rest_{j}'],rest[keep,j])
    for j in range(3):
        np.testing.assert_array_equal(vertices[f'f_dc_{j}'],a['sh0'][keep,0,j])
        np.testing.assert_array_equal(vertices[f'scale_{j}'],(a['scales']+np.log(meters))[keep,j])
    for j in range(4):np.testing.assert_array_equal(vertices[f'rot_{j}'],q[keep,j])
    np.testing.assert_array_equal(vertices['opacity'],a['opacities'][keep])
    if phone:
        alpha=1/(1+np.exp(-np.clip(a['opacities'].astype(float),-80,80)))
        scale=np.exp(a['scales'].astype(float))
        quat=a['quats'].astype(float);norm=np.linalg.norm(quat,axis=1)
        eligible=keep&(alpha>=.02)&np.isfinite(norm)&(norm>1e-12)
        order=np.flatnonzero(eligible)
        order=order[np.argsort(-(scale[order].prod(1)*alpha[order]),kind='stable')]
    else:
        alpha=1/(1+np.exp(-np.clip(a['opacities'],-80,80)))
        scale=np.exp(a['scales'])*meters
        quat=a['quats'];norm=np.linalg.norm(quat,axis=1);order=np.arange(len(xyz))
    expected=np.empty(len(order),dtype=rows.dtype)
    expected['xyz']=xyz[order];expected['scale']=scale[order]
    expected['rgba'][:,:3]=np.clip((.5+.28209479177387814*a['sh0'][order,0])*255,0,255).astype('u1')
    expected['rgba'][:,3]=np.clip(alpha[order]*255,0,255).astype('u1')
    expected['quat']=np.clip(quat[order]/norm[order,None]*128+128,0,255).astype('u1')
    # For drone, preserve the source expression 255 / (1 + exp(-opacity)).
    if not phone: expected['rgba'][:,3]=np.clip(255/(1+np.exp(-np.clip(a['opacities'],-80,80))),0,255).astype('u1')
    assert rows.tobytes()==expected.tobytes()
    result=dict(passed=True,kind=args.kind,ply_fields=len(vertices.dtype.names),ply_gaussians=len(vertices),
                splat_gaussians=len(rows),splat_row_bytes=rows.dtype.itemsize,all_sh_columns_exact=True,
                all_positions_exact=True,all_quaternions_exact=True,splat_bytes_exact=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
