# SPDX-License-Identifier: Apache-2.0
import json
import struct
import sys

import numpy as np
import pytest

from recon.evaluation import lidar


def test_distances_quantiles_thresholds_layers(tmp_path):
    ref = np.array([[0, 0, z] for z in [-1, 1, 2, 3, 4]], dtype=float)
    p = ref.copy()
    p[:, 0] = [.01, .05, .1, .2, .4]
    d = p[:, 0].astype(np.float32)
    result = lidar.distances(p, ref)
    assert result['n'] == 5
    assert result['p50'] == float(np.median(d))
    assert result['p90'] == float(np.percentile(d, 90))
    assert result['within_distance_fraction'] == {'0.05': .2, '0.1': .4, '0.2': .6}
    assert [a['n'] for a in result['height_layers']] == [1]*5
    assert [a['p50'] for a in result['height_layers']] == d.tolist()
    path = tmp_path/'ref.npz'
    np.savez_compressed(path, points=ref)
    assert lidar.distances(p, path) == result
    assert lidar.distances(p[:1], ref)['height_layers'][1]['p50'] is None


@pytest.mark.parametrize('bad', [np.empty((0, 3)), [[np.nan, 0, 0]], [[1, 2]]])
def test_reject_invalid_points(bad):
    with pytest.raises(ValueError, match='finite Nx3'):
        lidar.distances(bad, [[0, 0, 0]])
    with pytest.raises(ValueError, match='finite Nx3'):
        lidar.distances([[0, 0, 0]], bad)


def custom_scan():
    # ROS1 Header, uint64 timebase, uint32 point_num, uint8 lidar_id,
    # uint8[3] reserved, uint32 array length, packed 19-byte CustomPoints.
    header = struct.pack('<IIII', 7, 12, 345, 5) + b'livox'
    points = (struct.pack('<IfffBBB', 10, 1., 2., 3., 9, 16, 1)
              + struct.pack('<IfffBBB', 20, -1., 0., .5, 10, 32, 2))
    return header + struct.pack('<QIB3sI', 12000000345, 2, 0, b'\0'*3, 2) + points


def test_custommsg_parser():
    raw = custom_scan()
    stamp, tb, records, points = lidar._scan(raw)
    assert stamp == tb == 12000000345
    assert records['t'].tolist() == [10, 20]
    assert records['tag'].tolist() == [16, 32]
    np.testing.assert_array_equal(points, [[1, 2, 3], [-1, 0, .5]])
    with pytest.raises(ValueError, match='layout'):
        lidar._scan(raw[:-1])
    with pytest.raises(ValueError, match='layout'):
        lidar._scan(raw[:37] + struct.pack('<I', 3) + raw[41:])


def test_exact_pair_support_and_voxels():
    ks = np.array([1, 1, 1, 2, 2, 3, 3])
    sid = np.array([0, 1, 2, 0, 3, 0, 4], np.uint16)
    ns = np.array([0, 11, 5, 12, 10], np.int64)*10**9
    pos = np.array([[0, 0, 0], [.1, 0, 0], [2, 0, 0], [1.1, 0, 0], [1, 0, 0]])
    _, good, _, _ = lidar._static_support(ks, sid, ns, pos)
    assert good.tolist() == [False, True, True]
    shuffled = np.array([3, 0, 5, 1, 4, 2, 6])
    order = np.argsort(ks[shuffled], kind='stable')
    indexed = lidar._static_support(ks[shuffled], sid[shuffled], ns, pos, order)
    for actual, expected in zip(indexed, lidar._static_support(ks, sid, ns, pos)):
        np.testing.assert_array_equal(actual, expected)
    assert len(np.unique(lidar._keys(np.array([[-.01, 0, 0], [0, 0, 0], [.049, 0, 0], [.05, 0, 0]]), .05))) == 3


def test_trajectory_dedup_conflict_and_no_extrapolation():
    ts = np.arange(5, dtype=np.int64)*10000000
    pq = np.tile([0., 0, 0, 0, 0, 0, 1], (5, 1)); pq[3:, 0] = 1
    traj = lidar._Trajectory(ts, pq)
    assert traj.ts.tolist() == [0, 30000000]
    np.testing.assert_allclose(traj.at([15000000])[0], [[.5, 0, 0]])
    with pytest.raises(ValueError, match='extrapolation'):
        traj.at([-1])
    with pytest.raises(ValueError, match='conflicting'):
        lidar._Trajectory(np.zeros(5, np.int64), pq)


def test_cli_ply(tmp_path, monkeypatch):
    from recon.cli import main
    from recon.io.ply import write_ply, point_vertices
    p = np.array([[0, 0, 0], [.1, 0, 1]], np.float32)
    cloud, ref, out = [tmp_path/n for n in ('cloud.ply', 'ref.npz', 'out.json')]
    write_ply(cloud, point_vertices(p, np.zeros_like(p)))
    np.savez_compressed(ref, points=p)
    monkeypatch.setattr(sys, 'argv', ['recon', 'lidar-distance', '--cloud', str(cloud), '--reference', str(ref), '--out', str(out)])
    main()
    assert json.loads(out.read_text())['p90'] == 0


def test_build_reference_synthetic_messages(tmp_path, monkeypatch):
    from types import SimpleNamespace
    lidar_conn = SimpleNamespace(topic='/sensor')
    odom_conn = SimpleNamespace(topic='/poses')
    messages = []
    for sec in range(0, 22, 2):
        header = struct.pack('<IIII', 0, sec, 0, 0)
        pose = struct.pack('<I7d', 0, sec, 0, 0, 0, 0, 0, 1)
        messages.append((odom_conn, sec*10**9, header+pose))
        point = struct.pack('<IfffBBB', 0, 20-sec, 0, 0, 1, 0, 0)
        scan = header + struct.pack('<QIB3sI', sec*10**9, 1, 0, b'\0'*3, 1) + point
        messages.append((lidar_conn, sec*10**9, scan))

    class Reader:
        def __init__(self, path):
            self.connections = [lidar_conn, odom_conn]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def messages(self, connections):
            return (m for m in messages if m[0] in connections)

    monkeypatch.setitem(sys.modules, 'rosbags.rosbag1', SimpleNamespace(Reader=Reader))
    monkeypatch.setitem(sys.modules, 'numba', SimpleNamespace(njit=lambda f: f))
    bag, out = tmp_path/'synthetic.bag', tmp_path/'ref.npz'
    bag.write_bytes(b'parser inputs supplied by synthetic Reader')
    report = lidar.build_reference(bag, out, lidar_topic='/sensor',
                                  odometry_topic='/poses', body_from_lidar=np.eye(4))
    assert report['scans'] == 11
    assert report['clean_points'] == 1
    assert report['filtering']['near'] == 1
    assert report['fixed_voxels'] == 0
    assert report['reference_sha256'] == lidar._digest(out)
    assert json.loads(out.with_suffix('.json').read_text()) == report
    with np.load(out) as z:
        np.testing.assert_array_equal(z['points'], [[20, 0, 0]])
        assert z['witness_scan_a'].tolist() == [0]
        assert z['witness_scan_b'].tolist() == [5]
        assert z['witness_scan_b'].dtype == np.int32


def test_missing_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'rosbags.rosbag1', None)
    with pytest.raises(ImportError, match=r'\.\[lidar\]'):
        lidar.build_reference(tmp_path/'input.bag', tmp_path/'ref.npz')
