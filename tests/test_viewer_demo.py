"""CPU-only demo construction, PLY quantiles and HTTP cache regression tests."""
import functools
import http.server
import json
from pathlib import Path
import struct
import subprocess
import shutil
import threading
from urllib.request import Request, urlopen

import pytest

from viewer.tools.demo import build_site, cloud_bounds, ensure_server, main
from viewer.tools.serve import NoCache


def ply(path, fmt='ascii'):
    header = ('ply\nformat ' + fmt + ' 1.0\nelement vertex 3\n'
              'property uchar red\nproperty float z\nproperty double x\nproperty float y\nend_header\n')
    rows = [(255, 20, 0, 10), (128, 30, 10, 20), (0, 40, 20, 30)]
    data = '\n'.join(' '.join(map(str, row)) for row in rows).encode() if fmt == 'ascii' else b''.join(
        struct.pack(('<' if fmt == 'binary_little_endian' else '>') + 'Bfdf', *row) for row in rows)
    path.write_bytes(header.encode() + data)


@pytest.mark.parametrize('fmt', ['ascii', 'binary_little_endian', 'binary_big_endian'])
def test_quantiles(tmp_path, fmt):
    path = tmp_path / 'cloud.ply'
    ply(path, fmt)
    bounds = cloud_bounds(path)
    assert bounds['min'] == pytest.approx([.1, 10.1, 20.1])
    assert bounds['max'] == pytest.approx([19.9, 29.9, 39.9])


def export(tmp_path, name, scene):
    root = tmp_path / name
    root.mkdir()
    (root / 'scene.json').write_text(json.dumps(scene))
    (root / 'gaussians.ply').write_bytes(b'model placeholder')
    ply(root / 'cloud.ply')
    return root


def test_build_overlay_preserves_sources(tmp_path):
    old = {'preferred_model': {'path': 'gaussians.ply', 'full_sh': True}}
    first = export(tmp_path, 'old export', old)
    second_scene = {**old, 'bounds': {'min': [0, 1, 2], 'max': [3, 4, 5]},
                    'name': 'Original', 'world_up': [0, 1, 0], 'subtitle': 'Preserve me'}
    second = export(tmp_path, 'new', second_scene)
    cache = tmp_path / 'cache'
    site, manifests, ids = build_site([first, second], ['Custom', 'Ignored'], cache)
    assert ids == ['scene', 'scene-1']
    assert len(manifests) == 2
    assert (site / 'viewer').is_symlink()
    assert (site / 'export-0').resolve() == first
    assert (site / 'scene-0' / 'gaussians.ply').is_symlink()
    assert (site / 'scene-0' / 'gaussians.ply').resolve() == first / 'gaussians.ply'
    overlay = json.loads((site / 'scene-0' / 'scene.json').read_text())
    assert overlay['name'] == 'Custom'
    assert overlay['bounds']['min'] == pytest.approx([.1, 10.1, 20.1])
    new = json.loads((site / 'scene-1' / 'scene.json').read_text())
    assert all(new[key] == second_scene[key] for key in second_scene)
    assert json.loads((first / 'scene.json').read_text()) == old
    assert json.loads((second / 'scene.json').read_text()) == second_scene
    repeated, _, _ = build_site([first, second], [], cache)
    assert repeated == site
    assert json.loads((site / 'scene-0' / 'scene.json').read_text())['name'] == first.name


def test_invalid_inputs(tmp_path):
    with pytest.raises(SystemExit):
        main(['missing', '--margin', 'nan'])
    with pytest.raises(SystemExit):
        main(['missing', '--start', '0'])
    source = export(tmp_path, 'old', {})
    with pytest.raises(ValueError, match='More --name'):
        build_site([source], ['a', 'b'], tmp_path / 'cache')


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for scene-contract CPU test')
def test_margin_and_legacy_contract():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const root = process.argv[2];
const three = pathToFileURL(root + '/viewer/vendor/three/build/three.module.js').href;
const source = fs.readFileSync(root + '/viewer/src/scene-contract.js', 'utf8').replace("'three'", JSON.stringify(three));
const {loadSceneContract} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const scene = {preferred_model:{full_sh:true,path:'model.ply'},
 capture_corridor:{training_frame_ids:[0],frame_ids:[0],centers:[[10,20,30]],
 rotations_c2w:[[[1,0,0],[0,1,0],[0,0,1]]]}, capture_cameras:[{frame_id:0}],
 bounds:{min:[0,0,0],max:[1,2,3]}};
globalThis.fetch = async () => ({ok:true,json:async()=>scene});
async function bounds(search) {
 globalThis.location = {href:'http://localhost/viewer/',search};
 return (await loadSceneContract('/scene.json')).navigationData.bounds;
}
assert.deepEqual(await bounds(''), {min:[-1.5,-1.5,-3.5],max:[2.5,4.5,1.5]});
assert.deepEqual(await bounds('?margin=.25'), {min:[-.25,-.25,-2.25],max:[1.25,3.25,.25]});
assert.deepEqual(await bounds('?margin=-1'), await bounds(''));
assert.equal((await loadSceneContract('/scene.json')).pointUrl, undefined);
scene.cloud = 'cloud.ply';
assert.equal((await loadSceneContract('/scene.json')).pointUrl, 'http://localhost/cloud.ply');
scene.world_up = [0,1,0];
assert.deepEqual(await bounds('?margin=0'), {min:[0,0,0],max:[1,2,3]});
delete scene.bounds; delete scene.world_up;
assert.deepEqual(await bounds('?margin=99'), {min:[9,29,-21],max:[11,31,-19]});
'''
    subprocess.run(['node', '--input-type=module', '-', str(root)], input=script, text=True, check=True)


def test_no_cache_and_occupied_port(tmp_path):
    (tmp_path / 'hello.js').write_text('current')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(NoCache, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        port = server.server_port
        ensure_server(tmp_path, port)  # Reuse the matching tool server.
        with urlopen(Request(f'http://127.0.0.1:{port}/hello.js', headers={
                'If-Modified-Since': 'Wed, 21 Oct 2099 07:28:00 GMT'})) as response:
            assert response.status == 200
            assert 'no-store' in response.headers['Cache-Control']
            assert response.read() == b'current'
        with pytest.raises(RuntimeError, match='occupied'):
            ensure_server(tmp_path / 'other-cache', port)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
