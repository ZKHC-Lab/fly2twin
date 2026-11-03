# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import time
import numpy as np
from recon.io.common import dump
from .metrics import aggregate


def evaluate(data, checkpoint, out, image, subsets, alignment_steps=200):
    import torch
    import torch.nn.functional as F
    import torchmetrics
    from torchvision.models import AlexNet_Weights
    from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from fused_ssim import fused_ssim
    from gsplat import rasterization
    from .camera import CameraDelta
    backbone = Path(torch.hub.get_dir())/'checkpoints'/Path(AlexNet_Weights.DEFAULT.url).name
    linear = Path(torchmetrics.__file__).parent/'functional/image/lpips_models/alex.pth'
    for path in (backbone,linear):
        if not path.is_file():
            raise FileNotFoundError(f'Install LPIPS weights explicitly; automatic download disabled: {path}')
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    lpips.requires_grad_(False)
    s = {k:v.cuda().requires_grad_(False) for k,v in checkpoint['splats'].items()}
    frozen = dict(means=s['means'],quats=s['quats'],scales=s['scales'].exp(),
                  opacities=s['opacities'].sigmoid(),colors=torch.cat([s['sh0'],s['shN']],1))
    w,h = data['width'],data['height']
    def render(pose,K):
        return rasterization(**frozen,viewmats=torch.linalg.inv(pose),Ks=K,width=w,height=h,
            sh_degree=3,near_plane=.01,far_plane=1e10,packed=False,sparse_grad=False,absgrad=False,
            rasterize_mode='classic',camera_model='pinhole',render_mode='RGB')[0]
    def loss(pred,target):
        return F.l1_loss(pred,target)+.2*(1-fused_ssim(pred.permute(0,3,1,2),target.permute(0,3,1,2),padding='valid'))
    @torch.no_grad()
    def metrics(pred,target):
        p,q = pred.clamp(0,1).permute(0,3,1,2),target.permute(0,3,1,2)
        values = dict(psnr=float(peak_signal_noise_ratio(p,q,data_range=1)),
                      ssim=float(structural_similarity_index_measure(p,q,data_range=1)),lpips=float(lpips(p,q)))
        lpips.reset()
        return values
    rows,before_poses,after_poses = [],[],[]
    dump(out/'config.json',dict(alignment_steps=alignment_steps,lr=1e-4,frozen_gaussians=True,
         target_rgb_used_for_alignment=True,aggregation='per-frame arithmetic mean',
         loss='L1 + .2*(1-SSIM valid)',parameterization='right-composed translation + continuous 6D rotation'))
    for index in data['test_indices']:
        start=time.perf_counter()
        frame=int(data['frame_ids'][index])
        base,K=data['c2w'][index:index+1].detach(),data['K'][index:index+1]
        target=torch.from_numpy(image(frame,w,h)).cuda().float()[None]/255
        pose=CameraDelta().cuda()
        optimizer=torch.optim.Adam(pose.parameters(),lr=1e-4)
        with torch.no_grad():
            before=metrics(render(base,K),target)
        for _ in range(alignment_steps):
            objective=loss(render(pose(base),K),target)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        with torch.no_grad():
            final_pose=pose(base)
            after=metrics(render(final_pose,K),target)
        row=dict(frame=frame,index=int(index),before=before,after=after,seconds=time.perf_counter()-start)
        rows.append(row)
        before_poses.append(base[0].cpu().numpy())
        after_poses.append(final_pose[0].cpu().numpy())
        dump(out/'per_frame.json',rows)
        print(row,flush=True)
    np.savez(out/'poses.npz',frame_ids=data['frame_ids'][data['test_indices']],before=before_poses,after=after_poses)
    result=aggregate(rows,subsets)
    dump(out/'metrics.json',result)
    return result
