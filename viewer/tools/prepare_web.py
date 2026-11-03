from pathlib import Path
import argparse, json, struct, gzip, hashlib
import numpy as np
from scipy.spatial import cKDTree
from recon.io.ply import read_ply

def encode(v,degree,path):
    n=len(v);parts=[struct.pack('<IIIBBBB',0x5053474e,3,n,degree,12,0,0)]
    xyz=np.stack([v[k] for k in ['x','y','z']],1)
    p=np.floor(xyz*4096+.5).astype(np.int64)
    assert np.max(np.abs(p))<8388607
    parts.append(np.stack([p&255,(p>>8)&255,(p>>16)&255],-1).astype('u1').tobytes())
    alpha=1/(1+np.exp(-np.clip(v['opacity'],-80,80)))
    parts.append(np.clip(np.floor(alpha*255+.5),0,255).astype('u1').tobytes())
    dc=np.stack([v[f'f_dc_{i}'] for i in range(3)],1)
    parts.append(np.clip(np.floor((dc*.15+.5)*255+.5),0,255).astype('u1').tobytes())
    scales=np.stack([v[f'scale_{i}'] for i in range(3)],1)
    parts.append(np.clip(np.floor((scales+10)*16+.5),0,255).astype('u1').tobytes())
    q=np.stack([v[f'rot_{i}'] for i in [1,2,3,0]],1).astype(np.float64);q/=np.linalg.norm(q,axis=1)[:,None]
    largest=np.abs(q).argmax(1);neg=q[np.arange(n),largest]<0;comp=largest.astype(np.uint32)
    for j in range(3):
        indices=np.array([[k for k in range(4) if k!=i] for i in range(4)])[largest,j]
        x=q[np.arange(n),indices];mag=np.floor(511*np.abs(x)/np.sqrt(.5)+.5).astype(np.uint32)
        comp=(comp<<10)|(((x<0)^neg).astype(np.uint32)<<9)|mag
    parts.append(comp.astype('<u4').tobytes())
    sh=np.stack([v[f'f_rest_{i}'] for i in range(45)],1).reshape(n,3,15).transpose(0,2,1)[:,:degree*(degree+2)].copy()
    values=np.floor(sh*128+.5)+128
    steps=np.full((1,sh.shape[1],1),16);steps[:,:3]=8
    parts.append(np.clip(np.floor((values+steps/2)/steps)*steps,0,255).astype('u1').tobytes())
    with path.open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=6,mtime=0) as g:
            for p in parts:g.write(p)
    return dict(gaussians=n,shDegree=degree,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',required=True,type=Path)
    args=parser.parse_args()
    scene=json.loads(args.scene.read_text()); folder=args.scene.resolve().parent
    v=read_ply(folder/scene['preferred_model']['path'])
    xyz=np.stack([v[k] for k in ['x','y','z']],1)
    corridor=scene['capture_corridor']
    train=np.isin(corridor['frame_ids'],corridor['training_frame_ids'])
    cameras=np.array(corridor['centers'])[train]
    dist,_=cKDTree(cameras).query(xyz,workers=2)
    scales=np.sort(np.exp(np.stack([v[f'scale_{i}'] for i in range(3)],1)),axis=1)
    alpha=1/(1+np.exp(-np.clip(v['opacity'],-80,80)))
    area=scales[:,1]*scales[:,2]/np.maximum(dist,.1)**2
    score=alpha*np.minimum(area,.1)**.35
    valid=np.isfinite(xyz).all(1)&np.isfinite(scales).all(1)&(alpha>=.025)&(scales[:,2]<np.quantile(scales[:,2],.9995))
    indices=np.flatnonzero(valid);indices=indices[np.argsort(-score[indices],kind='stable')]
    report={}; assets={}
    for tier,count,degree in [('low',350000,2),('medium',650000,2),('high',0,3)]:
        selected=np.sort(indices[:count]) if count else np.arange(len(v))
        path=folder/f'{tier}.spz';report[tier]=encode(v[selected],degree,path)
        assets[tier]=path.name
    scene['web_assets']=assets
    (folder/'scene-web.json').write_text(json.dumps(scene,indent=2)+'\n')
    (folder/'web-assets.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
