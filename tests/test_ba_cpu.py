"""Small synthetic calibrated prior-BA run against the actual pycolmap API."""
import json
import sqlite3
import numpy as np
import pytest


def test_prior_ba_cpu(tmp_path):
    pc=pytest.importorskip('pycolmap');cv2=pytest.importorskip('cv2')
    from recon.poses.ba import build
    n=4
    poses=np.tile(np.eye(4),(n,1,1));poses[:,0,3]=np.arange(n)*.3
    K=np.array([[120.,0,64],[0,120,48],[0,0,1]])
    xyz=np.array([[x,y,3+.1*x] for x in np.linspace(-.5,.5,6) for y in np.linspace(-.4,.4,5)])
    images=tmp_path/'images';images.mkdir()
    out=tmp_path/'out';out.mkdir()
    names=[f'{i}.png' for i in range(n)]
    dbpath=out/'mixed.db'
    with pc.Database.open(dbpath):pass
    with sqlite3.connect(dbpath) as db:
        db.execute('INSERT INTO cameras VALUES (?,?,?,?,?,?)',(1,1,128,96,np.array([120.,120.,64.,48.]).tobytes(),1))
        for i,name in enumerate(names):
            assert cv2.imwrite(str(images/name),np.full((96,128,3),127,np.uint8))
            camera=xyz-poses[i,:3,3]
            xy=(camera[:,:2]/camera[:,2:])*120+[64,48]
            db.execute('INSERT INTO images VALUES (?,?,?)',(i+1,name,1))
            db.execute('INSERT INTO keypoints VALUES (?,?,?,?)',(i+1,len(xy),2,xy.astype(np.float32).tobytes()))
        for i in range(n):
            for j in range(i+1,n):
                m=np.repeat(np.arange(len(xyz),dtype=np.uint32)[:,None],2,axis=1)
                pair=(i+1)*2147483647+j+1
                F=np.array([[0.,0,0],[0,0,-1],[0,1,0]])
                db.execute('INSERT INTO matches VALUES (?,?,?,?)',(pair,len(m),2,m.tobytes()))
                db.execute('INSERT INTO two_view_geometries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           (pair,len(m),2,m.tobytes(),3,F.tobytes(),None,None,None,None,None,None))
    data=dict(frame_ids=np.arange(n),timestamp_ns=np.arange(n)*200_000_000,blur=np.zeros(n,bool),
              c2w_L=poses,K_L=np.tile(K,(n,1,1)),image_names=names)
    build(data,dbpath,images,out)
    with np.load(out/'cameras.npz',allow_pickle=False) as z:
        assert len(z['registered_indices'])==n
        np.testing.assert_allclose(z['c2w'],poses,atol=1e-4)
    rounds=json.loads((out/'prior_rounds.json').read_text())
    assert [r['threshold_px'] for r in rounds]==[16.,8.,4.]
    assert all(r['prior_ba_solution_usable'] for r in rounds)
