#!/usr/bin/env python3
"""CPU-only curated capture-following sofa route and floor-plane height audit."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from .render_roam import build_path,path_quality
from .normalization import normalize

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=570);p.add_argument('--end',type=int,default=1182);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 base=a.base.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 for name in ['cameras.npz','undistorted','gs']:
  if not (out/name).exists():(out/name).symlink_to((base/name).resolve())
 z=np.load(base/'cameras.npz');data={k:z[k] for k in z.files};norm=normalize(data);center=np.array(norm['center']);scale=norm['p95_radius'];allposes=data['c2w'].copy();allposes[:,:3,3]=(allposes[:,:3,3]-center)/scale
 ids=data['train_indices'];ids=ids[(data['frame_ids'][ids]>=a.start)&(data['frame_ids'][ids]<=a.end)];ids=ids[np.argsort(data['frame_ids'][ids])];poses=allposes[ids]
 if len(poses)<20:raise ValueError('Too few capture poses in curated range')
 # Source-frame contact sheet: native images, no labels or rescaling.
 contact=[];mapping=[]
 for seconds in [.5,5,10,15,20,25,30,35,39.4]:
  i=int(np.argmin(abs(data['frame_ids']-30*seconds)));frame=int(data['frame_ids'][i]);im=cv2.imread(str(base/'undistorted'/f'{frame:05d}.png'));contact.append(im);mapping.append(dict(seconds=seconds,frame=frame))
 canvas=cv2.vconcat([cv2.hconcat(contact[k:k+3]) for k in range(0,9,3)]);cv2.imwrite(str(out/'capture_route.png'),canvas);del contact,canvas
 # In the inspected first wide frame, the bottom-centre ROI is carpet.
 first=int(np.argmin(abs(data['frame_ids']-15)));pose=allposes[first];K=data['K'][first];image=cv2.imread(str(base/'undistorted'/f"{int(data['frame_ids'][first]):05d}.png"));h,w=image.shape[:2]
 xyz=(data['points']-center)/scale;pc=(xyz-pose[:3,3])@pose[:3,:3];uv=pc@K.T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
 mask=(pc[:,2]>0)&(uv[:,0]>.15*w)&(uv[:,0]<.95*w)&(uv[:,1]>.70*h)&(uv[:,1]<.95*h)&(np.linalg.norm(xyz,axis=1)<5);floor=xyz[mask]
 if len(floor)<100:raise ValueError('Insufficient carpet-plane points')
 rng=np.random.default_rng(42);sample=floor[rng.choice(len(floor),min(len(floor),10000),replace=False)];best=None;bestn=0
 for _ in range(256):
  q=sample[rng.choice(len(sample),3,replace=False)];normal=np.cross(q[1]-q[0],q[2]-q[0]);length=np.linalg.norm(normal)
  if length<1e-10:continue
  normal/=length;d=-normal@q[0];good=abs(sample@normal+d)<.01
  if good.sum()>bestn:bestn=int(good.sum());best=(normal,d)
 normal,d=best
 for _ in range(3):
  good=abs(floor@normal+d)<.01;points=floor[good];origin=np.mean(points,axis=0);_,vec=np.linalg.eigh((points-origin).T@(points-origin));normal=vec[:,0];d=-normal@origin
 up=np.median(-poses[:,:3,1],axis=0)
 if normal@up<0:normal=-normal;d=-d
 residual=abs(floor@normal+d);good=residual<.01
 if good.mean()<.5:raise ValueError('Carpet-plane support is too weak to audit height')
 candidates=[]
 for sigma in [1.,.7,.5,.25,0.]:
  positions,quats,info=build_path(poses,len(poses),sigma,600);quality=path_quality(positions,quats,poses,.12,8);span=float(np.ptp(positions@normal+d));candidates.append((sigma,positions,quats,info,quality,span))
 candidates.sort(key=lambda r:(min(r[4]['within_radius_fraction'],r[4]['within_angle_fraction']),r[0]),reverse=True)
 sigma,positions,quats,info,quality,span=candidates[0]
 report=dict(source='restroom_fix RGB-only',source_frame_range=[int(data['frame_ids'][ids[0]]),int(data['frame_ids'][ids[-1]])],source_cameras=len(ids),seconds=20,fps=30,frames=600,smooth_sigma=sigma,keyframes=len(poses),quality=quality,floor_plane=dict(normal=normal.tolist(),offset=float(d),roi='frame15 bottom-centre: x15%-95%, y70%-95%',candidate_points=len(floor),inlier_points=int(good.sum()),inlier_fraction=float(good.mean()),inlier_residual_p95=float(np.percentile(residual[good],95)),threshold=.01),camera_height_median=float(np.median(poses[:,:3,3]@normal+d)),camera_height_span=float(np.ptp(poses[:,:3,3]@normal+d)),output_height_span=span,height_limit=.15,units='normalized SfM units; floor-plane orientation, no metric/gravity sensor',contact_sheet_row_major=mapping,candidates=[dict(sigma=r[0],quality=r[4],height_span=r[5]) for r in candidates])
 (out/'route_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 if min(quality['within_radius_fraction'],quality['within_angle_fraction'])<.95 or span>.15:raise ValueError('Curated path does not meet coverage/height constraints; see route_audit.json')
 np.savez(out/'curated_path.npz',positions=positions,quats=quats,K=data['K'][ids[0]],source_frame_ids=data['frame_ids'][ids],width=w,height=h)
 print(json.dumps({k:v for k,v in report.items() if k not in ['candidates','contact_sheet_row_major']},indent=2))
if __name__=='__main__':main()
