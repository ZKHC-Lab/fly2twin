import json
import sqlite3
import numpy as np
import pytest
from recon.pipeline import fingerprint
from recon.poses.merge import merge
from recon.poses.database import MAX_IMAGE_ID


def test_cache_invalidates_bytes_and_config(tmp_path):
    image=tmp_path/'image.png';image.write_bytes(b'not decoded by hash function')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'frames':[{'path':'image.png'}]}))
    cfg={'manifest':str(manifest),'output':str(tmp_path/'out')}
    a=fingerprint(cfg)
    assert fingerprint(cfg)==a
    image.write_bytes(b'changed input bytes')
    assert fingerprint(cfg)!=a
    b=fingerprint(cfg)
    assert fingerprint(dict(cfg,train={'seed':43}))!=b


def test_cache_invalidates_shared_core_source(tmp_path, monkeypatch):
    import recon_core
    core=tmp_path/'recon_core';core.mkdir()
    init=core/'__init__.py';init.write_text('')
    source=core/'scale.py';source.write_text('VALUE = 1\n')
    tests=core/'tests';tests.mkdir()
    test=tests/'test_scale.py';test.write_text('# first test\n')
    monkeypatch.setattr(recon_core,'__file__',str(init))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'frames':[]}))
    cfg={'manifest':str(manifest),'output':str(tmp_path/'out')}
    original=fingerprint(cfg)
    test.write_text('# changed test\n')
    assert fingerprint(cfg)==original
    source.write_text('VALUE = 2\n')
    assert fingerprint(cfg)!=original


def test_real_database_merge(tmp_path):
    pc=pytest.importorskip('pycolmap')
    source=tmp_path/'sift.db'
    with pc.Database.open(source): pass
    K=np.array([100.,100.,20.,20.],np.float64)
    with sqlite3.connect(source) as db:
        db.execute('INSERT INTO cameras VALUES (?,?,?,?,?,?)',(1,1,40,40,K.tobytes(),1))
        # Intentionally reversed image IDs relative to manifest ordering.
        db.executemany('INSERT INTO images VALUES (?,?,?)',[(2,'a.png',1),(1,'b.png',1)])
        for i in (1,2):
            keys=np.array([[10,11],[12,13]],np.float32)
            db.execute('INSERT INTO keypoints VALUES (?,?,?,?)',(i,2,2,keys.tobytes()))
        m=np.array([[0,1]],np.uint32)
        db.execute('INSERT INTO matches VALUES (?,?,?,?)',(MAX_IMAGE_ID+2,1,2,m.tobytes()))
        db.execute('INSERT INTO two_view_geometries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   (MAX_IMAGE_ID+2,1,2,m.tobytes(),3,np.eye(3).tobytes(),None,None,None,None,None,None))
    x=tmp_path/'xfeat.npz'
    F=np.arange(9,dtype=float).reshape(3,3)
    np.savez(x,frame_ids=[101,105],pair_indices=[[0,1]],coords=np.array([[1,2,3,4],[1,2,5,6]],np.float32),
             offsets=[0,2],F=F[None])
    dst=tmp_path/'mixed.db'
    merge({'frame_ids':np.array([101,105]),'image_names':['a.png','b.png']},source,x,dst)
    with sqlite3.connect(dst) as db:
        n={i:rows for i,rows in db.execute('SELECT image_id,rows FROM keypoints')}
        assert n=={1:4,2:3}
        rows,blob,f=db.execute('SELECT rows,data,F FROM two_view_geometries').fetchone()
        assert rows==3
        np.testing.assert_array_equal(np.frombuffer(blob,np.uint32).reshape(-1,2),[[0,1],[2,2],[3,2]])
        np.testing.assert_array_equal(np.frombuffer(f,np.float64).reshape(3,3),F.T)
        assert db.execute('SELECT COUNT(*) FROM descriptors').fetchone()[0]==0
