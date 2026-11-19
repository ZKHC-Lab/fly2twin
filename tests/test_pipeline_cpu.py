"""CPU integration: standard manifest -> camera import -> depth cache -> fresh fusion."""
import json
import numpy as np
import pytest


def scene(tmp_path,cv2,windows=None):
    """Nine frames 0.2 m apart in front of a wall 2 m away; frame 7 is held out. Returns the config."""
    n,w,h=9,64,48
    ids=np.arange(101,101+n)
    poses=np.tile(np.eye(4),(n,1,1));poses[:,0,3]=np.arange(n)*.2
    K=np.array([[50.,0,32],[0,50,24],[0,0,1]])
    rows=[]
    image=np.full((h,w,3),127,np.uint8)
    for i,frame in enumerate(ids):
        path=tmp_path/f'{frame}.png';assert cv2.imwrite(str(path),image)
        rows.append(dict(frame_id=int(frame),timestamp_ns=int(i*200_000_000),path=str(path)))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(resolution=[w,h],K=K.tolist(),frames=rows)))
    cameras=tmp_path/'cameras.npz';np.savez(cameras,frame_ids=ids,c2w=poses,K=np.tile(K,(n,1,1)))
    cache=tmp_path/'depth';cache.mkdir()
    windows=windows or [('first',np.arange(n),np.arange(n))]
    (cache/'windows.json').write_text(json.dumps(dict(windows=[dict(id=name,idx=ids[idx].tolist(),new=ids[new].tolist()) for name,idx,new in windows])))
    extrinsics=np.linalg.inv(poses)[:,:3]
    for name,idx,_ in windows:
        (cache/name).mkdir()
        np.savez(cache/name/'prediction.npz',depth=np.full((len(idx),h,w,1),2.,np.float32),
                 depth_conf=np.tile(np.arange(h*w,dtype=np.float32).reshape(h,w),(len(idx),1,1)),extrinsic=extrinsics[idx])
    return dict(manifest=str(manifest),cameras=str(cameras),output=str(tmp_path/'output'),
                pose={'provider':'external','refine':'none','units':'meters'},depth={'cache':str(cache),'resolution':[w,h]})


def test_pipeline_fuse_and_cache(tmp_path):
    cv2=pytest.importorskip('cv2')
    from recon.pipeline import run
    cfg=scene(tmp_path,cv2)
    root=run(cfg,through='fuse')
    report=json.loads((root/'fuse/report.json').read_text())
    assert report['frame_count']==8 and report['populated_slots']==8 and report['points']>=4
    marker=root/'fuse/complete.json';original=marker.read_bytes()
    assert run(cfg,through='fuse')==root and marker.read_bytes()==original
    good=np.load(root/'fuse/cache/good.npy',allow_pickle=False)
    assert not good[7].any()
    # Artifact corruption is not silently accepted, even when inputs have not changed.
    (root/'fuse/report.json').write_text('{}')
    with pytest.raises(RuntimeError,match='artifact changed'):run(cfg,through='fuse')


def test_fuse_with_revisit_keeps_cache_scales_and_report_consistent(tmp_path):
    cv2=pytest.importorskip('cv2')
    from recon.pipeline import run
    two=[('K000',np.arange(6),np.arange(6)),('K001',np.arange(3,9),np.arange(6,9))]
    (tmp_path/'plain').mkdir();(tmp_path/'revisit').mkdir()
    plain=run(scene(tmp_path/'plain',cv2,two),through='fuse')
    cfg=scene(tmp_path/'revisit',cv2,two);cfg['fusion']={'revisit':True}
    root=run(cfg,through='fuse')
    report=json.loads((root/'fuse/report.json').read_text())
    assert report['revisit']['frames']==8 and report['revisit']['windows']==2 and 'revisit' not in json.loads((plain/'fuse/report.json').read_text())
    with np.load(root/'fuse/scales.npz') as z, np.load(plain/'fuse/scales.npz') as old:
        assert 'revisit_log_scale' not in old.files and z['revisit_log_scale'].shape==(8,)
        np.testing.assert_allclose(z['scales'],old['scales']*np.exp(z['revisit_log_scale']),rtol=1e-12)
        factor=np.ones(9);factor[np.searchsorted(np.arange(101,110),z['frame_ids'])]=np.exp(z['revisit_log_scale'])
    np.testing.assert_allclose(np.load(root/'fuse/cache/depth_m.npy'),np.load(plain/'fuse/cache/depth_m.npy')*factor[:,None,None].astype(np.float32),rtol=1e-6)


def test_blur_uses_candidates_before_keyframe_thinning(tmp_path,monkeypatch):
    cv2=pytest.importorskip('cv2')
    from types import SimpleNamespace
    from recon.pipeline import prepare
    import recon.keyframes.selection as selection
    n=30
    image=tmp_path/'frame.png';assert cv2.imwrite(str(image),np.zeros((48,64,3),np.uint8))
    poses=np.tile(np.eye(4),(n,1,1));poses[:,0,3]=np.arange(n)*.2
    rows=[dict(frame_id=i,timestamp_ns=i*200_000_000,path=str(image),c2w=poses[i].tolist()) for i in range(n)]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(resolution=[64,48],K=np.eye(3).tolist(),frames=rows)))
    sharp=np.full(n,100.);sharp[[10,20]]=40.
    values=iter(sharp)
    monkeypatch.setattr(cv2,'Laplacian',lambda *a,**k:SimpleNamespace(var=lambda:next(values)))
    monkeypatch.setattr(selection,'select',lambda *a,**k:np.array([10,20,29]))
    out=tmp_path/'prepared';out.mkdir()
    prepare(dict(manifest=str(manifest),keyframes={'enabled':True}),out)
    with np.load(out/'inputs.npz',allow_pickle=False) as z:
        assert z['blur'].tolist()==[True,True,False]
