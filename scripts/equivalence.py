#!/usr/bin/env python3
"""Local acceptance runner. Inputs are explicit; outputs never enter the source tree.

Run inside the documented GPU flock and MemoryMax=16G scope. These assertions
implement ANALYSIS E1/E2/E3 without relaxing their numeric tolerances.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import yaml


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def call(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True)


def drone(args):
    import torch
    from recon.pipeline import run,gpu_data,export
    from recon.gaussian.render import render
    cfg=yaml.safe_load(args.config.read_text())
    root=run(cfg,through='fuse')
    fusion=load(root/'fuse/report.json')
    reference=load(args.reference/'drone_v1/inputs/fusion.json')
    assert fusion['frame_count']==670
    assert fusion['points']==848586
    assert fusion['voxel_m']==reference['voxel_m']
    np.testing.assert_array_equal(fusion['window_scales'],reference['window_scales'])
    with np.load(root/'fuse/initial.npz',allow_pickle=False) as new, np.load(args.reference/'drone_v1/inputs/init.npz',allow_pickle=False) as old:
        for key in ('points','colors'):
            np.testing.assert_array_equal(new[key],old[key])
    save(Path(cfg['output'])/'fusion-equivalence.json',dict(passed=True,new=fusion,reference=reference))
    root=run(cfg,through='train')
    data,image=gpu_data(cfg,root)
    checkpoint=torch.load(root/'train/final.pt',map_location='cpu',weights_only=False)
    splats={k:v.cuda() for k,v in checkpoint['splats'].items()}
    rows=[]
    with torch.inference_mode():
        for index in data['train_indices']:
            rgb=render(splats,data['c2w'][index:index+1],data['K'][index:index+1],data['width'],data['height'])[0][0,...,:3].clamp(0,1).cpu().numpy()
            target=image(int(data['frame_ids'][index]),data['width'],data['height'])/255.
            mse=float(np.mean((rgb-target)**2))
            rows.append(dict(frame=int(data['frame_ids'][index]),mse=mse,psnr=-10*np.log10(max(mse,1e-15))))
    psnr=float(np.mean([r['psnr'] for r in rows]));count=checkpoint['stats']['final_gaussians']
    result=dict(psnr=psnr,gaussians=count,gaussian_relative_error=count/1574807-1,
                target_psnr=26.43,psnr_tolerance=.3,target_gaussians=1574807,gaussian_tolerance=.03,
                passed=abs(psnr-26.43)<=.3 and abs(count/1574807-1)<=.03,per_frame=rows,root=str(root))
    save(Path(cfg['output'])/'acceptance.json',result)
    destination=root/'export';destination.mkdir(exist_ok=True)
    export(cfg,destination,root)
    assert result['passed'],result


def phone(args):
    from recon.video.normalization import normalize
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    ref=args.reference/'phone/runs/restroom_fix'
    metadata=load(ref/'extract.json')
    if not args.resume:
        call('-m','recon.video.extract_frames','--video',args.video,'--out',out,
             '--target-frames',metadata['target_frames'],'--sfm-multiple',metadata['sfm_multiple'],
             '--long-edge',metadata['long_edge'],'--scan-width',metadata['scan_size'][0],
             '--hdr',metadata['hdr_mode'],'--train-subset','all')
    ids=np.array([r['frame'] for r in load(out/'extract.json')['frames']])
    with np.load(ref/'cameras.npz',allow_pickle=False) as z:
        reference=dict(z)
    np.testing.assert_array_equal(ids,reference['frame_ids'])
    save(out/'frame-equivalence.json',dict(passed=True,frames=len(ids),exact_frame_ids=True))
    if args.stage=='e2':
        if not (args.resume and (out/'shared').exists()):
            shutil.copytree(ref/'shared',out/'shared')
        # Cleaned reference lacks frames.bin but retained rigs.bin. Loading only
        # the complete legacy triplet recreates trivial rigs without changing cameras.
        shared=out/'shared'
        if (shared/'rigs.bin').exists() and not (shared/'frames.bin').exists():
            source=(shared/'rigs.bin').resolve(); target=shared/'rigs.bin.incomplete'
            assert source.parent==shared.resolve() and shared.resolve().is_relative_to(out)
            if target.exists(): raise FileExistsError(target)
            source.rename(target)
        call('-m','recon.video.sfm','--out',out,'--export-only')
    else:
        call('-m','recon.video.sfm','--out',out,'--threads',8)
        registration=load(out/'sfm.json')['register_ratio']
        assert registration>=.99,registration
    with np.load(out/'cameras.npz',allow_pickle=False) as z:
        current=dict(z)
    norm=normalize(current); reference_norm=load(ref/'gs/metrics.json')['normalization']
    delta_center=float(np.max(np.abs(np.array(norm['center'])-reference_norm['center'])))
    delta_p95=abs(norm['p95_radius']-reference_norm['p95_radius'])
    preflight=dict(train=len(current['train_indices']),test=len(current['test_indices']),
                   center_max_difference=delta_center,p95_difference=delta_p95,
                   intrinsics_exact=bool(np.array_equal(current['K'],reference['K'])),
                   poses_exact=bool(np.array_equal(current['c2w'],reference['c2w'])),
                   sparse_points_exact=bool(np.array_equal(current['points'],reference['points'])),
                   sparse_colors_exact=bool(np.array_equal(current['colors'],reference['colors'])))
    if args.stage=='e2':
        assert preflight['intrinsics_exact']
        assert preflight['train']==1050 and preflight['test']==150
        assert delta_center<1e-9 and delta_p95<1e-9
    save(out/'preflight.json',preflight)
    # E2/E3 are historical RGB-only references, not the current B0 phone preset.
    historical=dict(train=dict(aniso_lambda=.1,depth_lambda=0,discard_initial_permutation=True,
                               exact_steps=True,image_cache_frames=1200,max_width=1600,
                               reset_at_zero=True,scale_densification=True,schedule_rounding='round',seed=42),
                    eval=dict(aligned_width=960,alignment_steps=200))
    historical_config=out/'rgb_reference.yaml'
    historical_config.write_text(yaml.safe_dump(historical))
    call('-m','recon.video.pipeline','--out',out,'--config',historical_config)
    metrics=load(out/'gs/metrics.json');psnr=metrics['rgb_metrics']['psnr']['mean'];count=metrics['final_gaussians']
    if args.stage=='e2':
        passed=abs(psnr-32.77)<=.3 and abs(count/1582100-1)<=.03
        result=dict(passed=passed,psnr=psnr,gaussians=count,target_psnr=32.77,
                    psnr_tolerance=.3,target_gaussians=1582100,gaussian_tolerance=.03)
    else:
        previous=load(args.e2/'acceptance.json')['psnr']
        result=dict(passed=abs(psnr-previous)<=.5,psnr=psnr,registration_ratio=registration,
                    e2_psnr=previous,psnr_tolerance=.5)
    save(out/'acceptance.json',result)
    call('-m','recon.video.export','--out',out)
    assert result['passed'],result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['e1','e2','e3'])
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--config',type=Path,help='E1 scene config; E2/E3 use embedded historical RGB-only settings')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--e2',type=Path)
    parser.add_argument('--resume',action='store_true',help='reuse already verified video extraction only')
    args=parser.parse_args()
    if args.stage=='e1' and args.config is None:
        parser.error('E1 requires --config')
    (drone if args.stage=='e1' else phone)(args)


if __name__=='__main__':
    main()
