"""Landing-only global SIFT+XFeat position-prior BA (no ROS dependency)."""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def stage(name, start):
    row = dict(stage=name, seconds=time.perf_counter()-start,
               peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    print(json.dumps(row), flush=True)
    return row


def run(package, xfeat_source, xfeat_weights):
    # Set before importing NumPy/Torch/COLMAP; CPU phases never hold the GPU lock.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT'):
        os.environ[key] = '4'
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(Path(xfeat_source).resolve()))
    for name in ('numpy', 'cv2', 'torch', 'pycolmap', 'modules.xfeat'):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError('Landing BA dependency unavailable: %s. Install it for this architecture; BA was NOT skipped.' % name) from exc
    if not Path(xfeat_weights).is_file():
        raise FileNotFoundError('XFeat weights unavailable: ' + str(xfeat_weights))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from .common import digest, dump
    from recon.poses.prior import reconstruct
    package = Path(package).resolve()
    manifest = json.loads((package/'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('units') != 'meters':
        raise ValueError('Expected metric onboard delivery schema 1')
    for relative, expected in manifest['hashes'].items():
        path = (package/relative).resolve()
        if package not in path.parents or digest(path) != expected:
            raise ValueError('Package file changed or escaped: ' + relative)
    out = package/'onboard_ba'
    if out.exists():
        raise FileExistsError('Refusing to overwrite onboard_ba')
    with np.load(package/'poses.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in ('frame_ids', 'timestamp_ns', 'c2w', 'K', 'blur')}
    # The reference method is PINHOLE. Never silently feed distorted images.
    if any(np.any(np.abs(ci.get('D', [])) > 1e-10) for ci in manifest['camera_info']):
        raise ValueError('Reference BA requires rectified images / zero distortion; rectify package first')
    out.mkdir(); images = out/'images'; images.mkdir()
    names = []
    for i, row in enumerate(manifest['frames']):
        name = '%08d.jpg' % i
        source = (package/row['path']).resolve()
        if package not in source.parents:
            raise ValueError('Keyframe path escapes delivery package')
        (images/name).symlink_to(os.path.relpath(source, images))
        names.append(name)
    data['image_names'] = names
    start = time.perf_counter()
    try:
        reconstruct(data, images, out, dict(position_sigma_m=.05, xfeat_weights=str(Path(xfeat_weights).resolve()), batch_size=8))
    except Exception as exc:
        dump(out/'failure.json', dict(error=str(exc), **stage('failed', start)))
        raise
    with np.load(out/'cameras.npz', allow_pickle=False) as f:
        offsets = np.linalg.norm(f['c2w'][:, :3, 3]-data['c2w'][:, :3, 3], axis=1)
        report = dict(**stage('total', start), threads=4, input_frames=len(offsets),
            observed_fraction=float(np.mean(f['observations'] > 0)), center_offset_m_p50_p90=np.percentile(offsets,[50,90]).tolist(),
            method='SIFT+XFeat; Cartesian LIO prior sigma=.05m; global 16/8/4px; no fixed cameras')
    from .quality import adjacent_sampson
    with np.load(out/'cameras.npz', allow_pickle=False) as f:
        epipolar = adjacent_sampson(out/'mixed.db', names, {'LIO':data['c2w'], 'BA':f['c2w']}, data['K'])
    dump(out/'epipolar.json', epipolar)
    report['sampson_p50_p90_px'] = epipolar['rows']['BA']['p50_p90_px']
    report['checks'] = dict(observed_at_least_97pct=report['observed_fraction'] >= .97,
        sampson_p50_at_most_07px=report['sampson_p50_p90_px'][0] is not None and report['sampson_p50_p90_px'][0] <= .7)
    dump(out/'timing_quality.json', report)
    manifest['onboard_ba'] = dict(poses='onboard_ba/cameras.npz', sparse_model='onboard_ba/model_prior',
        quality_report='onboard_ba/timing_quality.json', completed=True, quality_checks=report['checks'])
    for path in sorted(out.rglob('*')):
        if path.is_file():
            manifest['hashes'][str(path.relative_to(package))] = digest(path)
    temporary = package/'manifest.ba.tmp'
    dump(temporary, manifest)
    os.replace(temporary, package/'manifest.json')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package', type=Path)
    p.add_argument('--xfeat-source', required=True, type=Path)
    p.add_argument('--xfeat-weights', required=True, type=Path)
    args = p.parse_args()
    try:
        run(args.package, args.xfeat_source, args.xfeat_weights)
    except Exception as exc:
        p.exit(1, 'Landing BA failed: %s\n' % exc)


if __name__ == '__main__':
    main()
