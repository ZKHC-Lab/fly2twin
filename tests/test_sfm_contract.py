import json
from types import SimpleNamespace
import numpy as np
import pytest


def test_uncalibrated_manifest_can_prepare(tmp_path):
    cv2=pytest.importorskip('cv2')
    from recon.pipeline import prepare
    path=tmp_path/'frame.png';assert cv2.imwrite(str(path),np.zeros((48,64,3),np.uint8))
    rows=[dict(frame_id=i,timestamp_ns=i*200_000_000,path=str(path)) for i in range(4)]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(resolution=[64,48],frames=rows)))
    out=tmp_path/'prepared';out.mkdir()
    prepare(dict(manifest=str(manifest)),out)
    with np.load(out/'inputs.npz',allow_pickle=False) as z:
        assert 'c2w' not in z.files and len(z['frame_ids'])==4


def test_partial_sfm_reports_registration_and_submodels(tmp_path,monkeypatch):
    pc=pytest.importorskip('pycolmap')
    from recon.poses.pipeline import estimate
    path=tmp_path/'frame.png';path.write_bytes(b'features are mocked in this contract test')
    rows=[dict(frame_id=i,timestamp_ns=i*200_000_000,path=str(path)) for i in range(4)]
    manifest=dict(resolution=[64,48],frames=rows)
    monkeypatch.setattr(pc,'extract_features',lambda *a,**k:None)
    monkeypatch.setattr(pc,'match_sequential',lambda *a,**k:None)
    monkeypatch.setattr(pc,'incremental_mapping',lambda *a,**k:{0:SimpleNamespace(num_reg_images=lambda:3),1:SimpleNamespace(num_reg_images=lambda:2)})
    out=tmp_path/'pose';out.mkdir()
    with pytest.raises(RuntimeError,match='Partial reconstruction'):
        estimate({'frame_ids':np.arange(4)},manifest,out,{},'colmap')
    report=json.loads((out/'report.json').read_text())
    assert report['registration_rate']==.75 and report['disconnected_submodels']==1
    assert report['models']=={'0':3,'1':2} and report['units']=='arbitrary'
    assert 'pose.provider: glomap' in report['recommendation']
