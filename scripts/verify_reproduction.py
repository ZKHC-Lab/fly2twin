#!/usr/bin/env python3
"""Verify the frozen 507/93/71 acceptance protocol; no GPU or historical imports."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--normalization',type=Path,required=True)
    p.add_argument('--reference-fusion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    load=lambda path:json.loads(path.read_text())
    selection=load(a.selection);current=load(a.root/'prepare/selection.json')
    for key in ('train_indices','test_indices','train_frame_ids','holdout_frame_ids','interpolation_frame_ids'):
        np.testing.assert_array_equal(current[key],selection[key])
    assert len(selection['train_frame_ids'])==507
    assert len(selection['holdout_frame_ids'])==93
    assert len(selection['interpolation_frame_ids'])==71
    cfg=load(a.root/'train/config.json');ref=load(a.normalization)
    for key in ('normalization_center','meters_per_unit','scene_scale'):
        np.testing.assert_allclose(cfg[key],ref[key],atol=1e-7,rtol=0)
    assert cfg['seed']==42 and cfg['steps']==30000 and cfg['resolution']==[640,360]
    assert cfg['sh_degree']==3 and cfg['depth_lambda']==.1 and cfg['aniso_lambda']==.1
    evaluation=load(a.root/'eval/config.json')
    assert evaluation['alignment_steps']==200 and evaluation['frozen_gaussians']
    assert evaluation['target_rgb_used_for_alignment']
    from recon.evaluation.metrics import aggregate
    recomputed=aggregate(load(a.root/'eval/per_frame.json'),load(a.root/'prepare/subsets.json'))
    depth=np.load(a.root/'fuse/cache/depth_m.npy',mmap_mode='r',allow_pickle=False)
    good=np.load(a.root/'fuse/cache/good.npy',mmap_mode='r',allow_pickle=False)
    assert depth.shape==(673,288,512) and good.shape==depth.shape
    train=selection['train_indices'];outside=np.setdiff1d(np.arange(673),train)
    assert not good[outside].any() and not np.any(depth[outside])
    assert np.any(good,axis=(1,2)).sum()==507
    fusion=load(a.root/'fuse/report.json');old=load(a.reference_fusion)
    assert fusion['points']<=300000
    metrics=load(a.root/'eval/metrics.json')
    assert metrics==recomputed, 'Stored summary differs from per-frame arithmetic aggregation'
    stats=load(a.root/'train/train_stats.json')
    assert stats['completed_steps']==30000 and stats['train_frames']==507 and stats['test_frames']==164
    with np.load(a.root/'train/train_visits.npz',allow_pickle=False) as visits:
        np.testing.assert_array_equal(visits['frame_ids'],selection['train_frame_ids'])
        assert int(visits['visits'].sum())==30000 and int(visits['visits'].min())>=59
    targets={'holdout':{'fixed':16.651,'aligned':17.128},'interpolation':{'fixed':24.919,'aligned':25.433}}
    delta={subset:{mode:metrics[subset][mode]['psnr']-value for mode,value in modes.items()} for subset,modes in targets.items()}
    passed=all(abs(delta[s]['aligned'])<=.3 for s in targets)
    result=dict(passed=passed,protocol_checks_passed=True,target=targets,measured=metrics,delta_psnr_db=delta,
        final_gaussians=stats['final_gaussians'],reference_gaussians=1533519,
        gaussian_delta=stats['final_gaussians']-1533519,training=stats,
        fusion_comparison={k:{'new':fusion[k],'reference':old[k]} for k in ('points_at_2cm','points','voxel_m')},
        window_scale_max_abs_difference=float(np.max(np.abs(np.asarray(fusion['window_scales'])-old['window_scales']))),
        versions={k:cfg[k] for k in ('gsplat_version','torch_version')})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
