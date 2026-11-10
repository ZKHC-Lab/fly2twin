# SPDX-License-Identifier: Apache-2.0
"""User-installed XFeat dense/refined matching, fixed pixel-space RANSAC."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from recon.io.common import dump


def _fundamental(xy):
    import cv2
    return cv2.findFundamentalMat(xy[:,:2],xy[:,2:],cv2.FM_RANSAC,1.,.999,10000)


def match(data,images,pairs,path,settings):
    import cv2
    import torch
    # XFeat must be installed by the user; no torch.hub download or main-branch fallback.
    from modules.xfeat import XFeat
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    model=XFeat(weights=settings['xfeat_weights'],top_k=4096)
    batch_size=int(settings.get('batch_size',8))
    # OpenCV's FM_RANSAC (4.x) seeds itself per call: the result of a pair does not depend on
    # the order or thread it runs in, so pairs verify in parallel while the next batch matches.
    threads=int(settings.get('ransac_threads',4))
    features=[]
    retained,chunks,matrices=[],[],[]
    raw=np.zeros(len(pairs),np.int64);inliers=np.zeros_like(raw)
    pending=deque()

    def collect(limit):
        while len(pending)>limit:
            i,xy,future=pending.popleft()
            F,mask=future.result()
            if F is None or mask is None or F.shape!=(3,3): continue
            keep=mask.ravel().astype(bool);inliers[i]=keep.sum()
            if inliers[i]>=30:
                retained.append(i);chunks.append(xy[keep]);matrices.append(F)

    with torch.inference_mode(), ThreadPoolExecutor(threads) as pool:
        for name in data['image_names']:
            image=cv2.imread(str(images/name))
            if image is None: raise ValueError(f'Cannot decode {name}')
            tensor=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).permute(2,0,1)
            result=model.detectAndComputeDense(tensor[None].float()/255,top_k=4096)
            features.append({k:v[0].cpu() for k,v in result.items()})
        for begin in range(0,len(pairs),batch_size):
            selected=pairs[begin:begin+batch_size]
            left,right=[{k:torch.stack([features[i][k] for i in indices]).to(model.dev) for k in features[0]}
                        for indices in selected.T]
            matches=model.batch_match(left['descriptors'],right['descriptors'])
            for b in range(len(selected)):
                i=begin+b
                xy=model.refine_matches(left,right,matches,b).cpu().numpy().astype(np.float32)
                raw[i]=len(xy)
                if len(xy)<8: continue
                pending.append((i,xy,pool.submit(_fundamental,xy)))
            collect(4*threads*batch_size)  # bounded queue: matches of at most a few batches in memory
        collect(0)
    retained=np.asarray(retained,np.int64)
    np.savez(path,frame_ids=data['frame_ids'],pair_indices=pairs[retained],offsets=np.r_[0,np.cumsum([len(x) for x in chunks])],
             coords=np.concatenate(chunks) if chunks else np.empty((0,4),np.float32),F=np.asarray(matrices).reshape(-1,3,3))
    dump(path.with_suffix('.json'),dict(top_k=4096,threshold_px=1.,confidence=.999,minimum_inliers=30,
        candidate_pairs=len(pairs),retained_pairs=len(retained),raw_matches=int(raw.sum()),retained_inliers=int(inliers[retained].sum())))
