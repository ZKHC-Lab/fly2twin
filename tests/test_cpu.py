import sqlite3
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from recon.geometry.transforms import transform,interpolate,deduplicate,resize_K,normalization
from recon.keyframes.selection import select,blur_mask
from recon.io.ply import read_ply,write_ply,point_vertices
from recon.poses.database import combine,points,matches,MAX_IMAGE_ID
from recon.poses.pairs import plan_pairs
from recon_core.scale import batch_window_scales
from recon.fusion.cloud import cache_depths
from recon.depth.backends import plan_windows
from recon.gaussian.schedule import schedule
from recon.evaluation.metrics import aggregate


def test_extrinsic_and_interpolation():
    t=np.array([0,1_000_000_000]);p=np.array([[0.,0,0],[2,0,0]])
    q=Rotation.from_euler('z',[0,90],degrees=True).as_quat()
    ext=transform([1,0,0],[0,0,0,1])
    poses,valid=interpolate(np.array([460_000_000]),t,p,q,ext,40)
    np.testing.assert_allclose(poses[0,:3,3],[1+2**-.5,2**-.5,0],atol=1e-7)
    assert valid.tolist()==[True]
    _,valid=interpolate(np.array([-1,1_000_000_001]),t,p,q)
    assert not valid.any()


def test_dedup_preserves_rotation():
    t=np.arange(3);p=np.zeros((3,3));q=Rotation.from_euler('z',[0,0,8],degrees=True).as_quat()
    tt,_,_=deduplicate(t,p,q)
    assert tt.tolist()==[0,2]


def test_resize_pixel_centers():
    K=np.array([[100,0,10],[0,100,20],[0,0,1.]])
    np.testing.assert_allclose(resize_K(K,50,50,100,100),[[50,0,4.75],[0,50,9.75],[0,0,1]])


def test_fixed_normalization():
    poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[0,1,2]
    fixed=dict(normalization_center=[3,4,5],meters_per_unit=8.941428986458085,scene_scale=1.1180742025375368)
    center,m,s=normalization(poses,[0,1],fixed)
    np.testing.assert_array_equal(center,[3,4,5]);assert m==fixed['meters_per_unit'];assert s==fixed['scene_scale']


def test_keyframes_motion_rotation_and_timeout():
    t=np.arange(21,dtype=np.int64)*200_000_000
    poses=np.tile(np.eye(4),(21,1,1))
    assert select(t,poses,trim_static=False).tolist()==[0,10,20]
    poses[1:,:3,:3]=Rotation.from_euler('z',9,degrees=True).as_matrix()
    assert select(t,poses,trim_static=True)[0]==1
    poses=np.tile(np.eye(4),(3,1,1));poses[1:,0,3]=.16
    assert select(t[:3],poses,trim_static=False).tolist()==[0,1]


def test_strict_gap_rejects_missing_candidates():
    with pytest.raises(ValueError): select(np.array([0,3_000_000_000]),np.tile(np.eye(4),(2,1,1)),trim_static=False)


def test_blur_local_median():
    sharp=np.full(30,100.);sharp[15]=49;sharp[16]=50
    blur,_=blur_mask(sharp)
    assert blur[15] and not blur[16]


@pytest.mark.parametrize('binary',[False,True])
def test_ply_roundtrip(tmp_path,binary):
    a=point_vertices(np.array([[1.5,2,3],[-2,0,1]]),np.array([[0,.5,1],[1,0,.2]]))
    path=tmp_path/'test.ply';write_ply(path,a,binary)
    np.testing.assert_array_equal(read_ply(path),a)


def test_ply_reject_truncated(tmp_path):
    path=tmp_path/'test.ply';write_ply(path,point_vertices(np.zeros((2,3)),np.zeros((2,3))))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):read_ply(path)


def test_database_union_reversal():
    old={1*MAX_IMAGE_ID+2:np.array([[0,1]],np.uint32)}
    new={7*MAX_IMAGE_ID+8:np.array([[2,3]],np.uint32)}
    result,count=combine(old,new,{7:2,8:1},{1:10,2:20})
    assert count==1
    np.testing.assert_array_equal(result[1*MAX_IMAGE_ID+2],[[0,1],[13,22]])


def test_database_sqlite_decode(tmp_path):
    with sqlite3.connect(tmp_path/'test.db') as con:
        con.execute('CREATE TABLE keypoints (image_id,rows,cols,data)')
        con.execute('CREATE TABLE matches (pair_id,rows,cols,data)')
        for i in [1,2]:con.execute('INSERT INTO keypoints VALUES (?,?,?,?)',(i,2,2,np.ones((2,2),np.float32).tobytes()))
        con.execute('INSERT INTO matches VALUES (?,?,?,?)',(MAX_IMAGE_ID+2,1,2,np.array([[0,1]],np.uint32).tobytes()))
        keys=points(con);result=matches(con,'matches',keys)
        np.testing.assert_array_equal(result[MAX_IMAGE_ID+2],[[0,1]])


def test_pairing_includes_offset_fifteen():
    poses=np.tile(np.eye(4),(20,1,1));poses[:,0,3]=np.arange(20)*2
    pairs=set(map(tuple,plan_pairs(poses)))
    assert (0,15) in pairs and (0,16) not in pairs


def window(ids):
    n=len(ids)
    return dict(id='w',idx=np.array(ids),new=np.array(ids),vc=np.stack([np.arange(n),np.zeros(n),np.zeros(n)],1),
                vR=np.tile(np.eye(3),(n,1,1)),depth=np.ones((n,24,24)),conf=np.tile(np.arange(576).reshape(24,24),(n,1,1)))


def test_window_scale_and_empty_test_slots(tmp_path):
    win=window([0,2,4]);poses=np.tile(np.eye(4),(5,1,1));poses[:,0,3]=np.arange(5)
    scales,_,_=batch_window_scales([win],np.arange(5),poses)
    np.testing.assert_allclose(scales,[2])
    depth,good,fs=cache_depths([win],scales,np.arange(5),tmp_path/'cache',24,24)
    assert depth.shape==(5,24,24)
    assert np.all(depth[[1,3]]==0) and not good[[1,3]].any()
    np.testing.assert_allclose(fs[[0,2,4]],2)


def test_degenerate_scale_rejected():
    with pytest.raises(ValueError):batch_window_scales([window([0,1,2])],np.arange(3),np.tile(np.eye(4),(3,1,1)))


def test_window_owner_unique():
    wins=plan_windows(range(73))
    assert [i for w in wins for i in w['new']]==list(range(73))
    assert len(wins[1]['idx'])==40


def test_schedule():
    assert schedule(507)['steps']==30000
    s=schedule(1588,94000)
    assert s['strategy']['refine_stop_iter']==47000 and s['sh_interval']==3133
    assert s['strategy']['reset_every']==9400


def test_losses_cpu_gradient():
    import torch
    from recon.gaussian.losses import inverse_depth_pearson,anisotropy
    x=torch.tensor([1.,2,3,4],requires_grad=True)
    loss=inverse_depth_pearson(x,x.detach()*2,torch.ones(4,dtype=torch.bool))
    assert abs(float(loss))<1e-6
    loss.backward();assert torch.isfinite(x.grad).all()
    a=anisotropy(torch.log(torch.tensor([[1.,1,20],[1.,1,1]])))
    assert abs(float(a)-5)<1e-5


def test_camera_delta_identity_and_gradient():
    import torch
    from recon.evaluation.camera import CameraDelta
    d=CameraDelta();base=torch.eye(4)[None]
    torch.testing.assert_close(d(base),base)
    d(base).sum().backward();assert torch.isfinite(d.embedding.weight.grad).all()


def test_metrics_per_frame_arithmetic():
    rows=[dict(frame=i,before=dict(psnr=p,ssim=.5,lpips=.2),after=dict(psnr=p+1,ssim=.6,lpips=.1)) for i,p in enumerate([10,30])]
    assert aggregate(rows,{'test':[0,1]})['test']['fixed']['psnr']==20
    with pytest.raises(ValueError):aggregate(rows+rows[:1],{'test':[0,1]})
