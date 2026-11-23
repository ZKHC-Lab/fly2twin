# SPDX-License-Identifier: Apache-2.0
"""COLMAP orchestration. Partial registration is rejected; no silent pose fallback."""
from pathlib import Path
import numpy as np
from recon.io.common import dump


def estimate(data,manifest,out,settings,provider):
    import pycolmap as pc
    images=out/'images';images.mkdir()
    names=[]
    for i,row in enumerate(manifest['frames']):
        name=f'{i:08d}{Path(row["path"]).suffix}'
        (images/name).symlink_to(Path(row['path']).resolve())
        names.append(name)
    data=dict(data,image_names=names)
    if provider=='external':
        from .prior import reconstruct
        reconstruct(data,images,out,settings)
        return
    db=out/'database.db'
    pc.extract_features(db,images,image_names=names,camera_mode=pc.CameraMode.SINGLE,
        reader_options=pc.ImageReaderOptions(camera_model='OPENCV'),
        extraction_options=pc.FeatureExtractionOptions(num_threads=4,use_gpu=False),device=pc.Device.cpu)
    pc.match_sequential(db,matching_options=pc.FeatureMatchingOptions(num_threads=4,use_gpu=False),
        pairing_options=pc.SequentialPairingOptions(overlap=20,quadratic_overlap=True,loop_detection=False,num_threads=4),device=pc.Device.cpu)
    models_dir=out/'models';models_dir.mkdir()
    options=pc.IncrementalPipelineOptions(num_threads=4,random_seed=42,ba_refine_focal_length=True,
        ba_refine_principal_point=True,ba_refine_extra_params=True,ba_use_gpu=False)
    models=pc.incremental_mapping(db,images,models_dir,options=options)
    counts={str(k):int(v.num_reg_images()) for k,v in models.items()}
    largest=max(models,key=lambda k:models[k].num_reg_images()) if models else None
    n=len(names);registered=counts.get(str(largest),0)
    dump(out/'report.json',dict(provider='colmap',units='arbitrary',input_frames=n,registered=registered,
        registration_rate=registered/n,models=counts,disconnected_submodels=max(0,len(models)-1),largest_model=largest,
        recommendation=('For white-wall/weak-texture fragmentation, use pose.provider: glomap to solve globally' if registered<n else None)))
    if largest is None: raise RuntimeError('COLMAP registered no model; see report.json')
    if registered<n:
        raise RuntimeError('Partial reconstruction: registration and disconnected models recorded; export selected model separately or provide poses. No silent dropping of split frames.')
    rec=models[largest]
    import cv2
    camera=next(iter(rec.cameras.values()))
    fx,fy,cx,cy,k1,k2,p1,p2=camera.params
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    size=(camera.width,camera.height)
    newK,_=cv2.getOptimalNewCameraMatrix(K,np.array([k1,k2,p1,p2]),size,0,size)
    mx,my=cv2.initUndistortRectifyMap(K,np.array([k1,k2,p1,p2]),None,newK,size,cv2.CV_32FC1)
    undistorted=out/'undistorted';undistorted.mkdir()
    lookup={im.name:im for im in rec.images.values()}
    poses=np.tile(np.eye(4),(n,1,1))
    for i,name in enumerate(names):
        poses[i,:3]=lookup[name].cam_from_world().inverse().matrix()
        target=undistorted/(Path(name).stem+'.png')
        if not cv2.imwrite(str(target),cv2.remap(cv2.imread(str(images/name)),mx,my,cv2.INTER_LINEAR)):
            raise OSError(target)
        manifest['frames'][i]['path']=str(target.resolve())
    dump(out/'manifest.json',manifest)
    np.savez(out/'cameras.npz',frame_ids=data['frame_ids'],timestamp_ns=data['timestamp_ns'],c2w=poses,
             K=np.repeat(newK[None],n,axis=0),blur=data['blur'],
             points=np.array([p.xyz for p in rec.points3D.values()],np.float32).reshape(-1,3),
             colors=(np.array([p.color for p in rec.points3D.values()],float).reshape(-1,3)/255).astype(np.float32))
