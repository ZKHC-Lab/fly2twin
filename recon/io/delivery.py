"""Onboard delivery package (drone3d_onboard.export) as a recon input.

Frames are the package keyframes in package order; K is the shared CameraInfo K.
Poses are the landing-BA cameras when the package records a completed onboard BA,
otherwise the onboard LIO frame_pose poses. Both are metric camera-to-world.
"""

from pathlib import Path
import json
import numpy as np


def files(package):
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    return [package / 'manifest.json'] + [package / name for name in sorted(manifest['hashes'])]


def load(package):
    """Return (recon manifest dict with absolute paths and per-frame c2w, pose source name)."""
    package = Path(package).resolve()
    meta = json.loads((package / 'manifest.json').read_text())
    if meta.get('schema_version') != 1 or meta.get('units') != 'meters':
        raise ValueError('Expected a metric onboard delivery package, schema 1')
    with np.load(package / 'poses.npz', allow_pickle=False) as z:
        ids, stamps, c2w, K = z['frame_ids'], z['timestamp_ns'], z['c2w'], z['K']
    source = 'onboard LIO frame_pose'
    ba = meta.get('onboard_ba')
    if ba and ba.get('completed'):
        with np.load(package / ba['poses'], allow_pickle=False) as z:
            if not np.array_equal(z['frame_ids'], ids):
                raise ValueError('Landing BA cameras do not match package frames')
            c2w = z['c2w']
        source = 'landing global BA'
    if not np.allclose(K, K[0]):
        raise ValueError('Delivery frames do not share one camera')
    width, height = meta['resolution']
    frames = [
        dict(frame_id=int(i), timestamp_ns=int(t), path=str(package / row['path']), c2w=pose.tolist())
        for i, t, row, pose in zip(ids, stamps, meta['frames'], c2w)
    ]
    return dict(K=K[0].tolist(), resolution=[width, height], frames=frames), source
