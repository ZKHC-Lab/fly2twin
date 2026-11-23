import sys
import types
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
torch = pytest.importorskip('torch')


class FakeXFeat:
    """Stands in for the user-installed modules.xfeat: fixed synthetic correspondences per pair."""
    dev = 'cpu'

    def __init__(self, weights, top_k):
        pass

    def detectAndComputeDense(self, tensor, top_k):
        seed = int(tensor.sum().item()) % 1000
        return {'descriptors': torch.full((1, 4), float(seed))}

    def batch_match(self, left, right):
        return [(int(a[0]), int(b[0])) for a, b in zip(left, right)]

    def refine_matches(self, left, right, matches, b):
        rng = np.random.default_rng(matches[b][0] * 1000 + matches[b][1])
        n = 40 + 20 * (matches[b][0] % 5)  # some pairs fall below 8 or 30 inliers
        x = rng.random((n, 2)) * 400
        y = x + rng.normal(0, .3, (n, 2)) + [3, 1]
        y[: n // 3] = rng.random((n // 3, 2)) * 400  # outliers
        return torch.from_numpy(np.hstack([x, y]))


def test_parallel_ransac_matches_one_thread(tmp_path, monkeypatch):
    package = types.ModuleType('modules')
    package.xfeat = types.ModuleType('modules.xfeat')
    package.xfeat.XFeat = FakeXFeat
    monkeypatch.setitem(sys.modules, 'modules', package)
    monkeypatch.setitem(sys.modules, 'modules.xfeat', package.xfeat)
    from recon.poses.xfeat import match
    names = []
    for i in range(12):
        names.append('%02d.png' % i)
        cv2.imwrite(str(tmp_path / names[-1]), np.full((8, 8, 3), 10 * i, np.uint8))
    data = dict(frame_ids=np.arange(12), image_names=names)
    pairs = np.array([(i, j) for i in range(12) for j in range(i + 1, min(i + 4, 12))])
    outputs = []
    for threads in (1, 4):
        path = tmp_path / ('x%d.npz' % threads)
        match(data, tmp_path, pairs, path, dict(xfeat_weights='unused', batch_size=3, ransac_threads=threads))
        with np.load(path) as f:
            outputs.append({k: f[k] for k in f.files})
        outputs[-1]['json'] = path.with_suffix('.json').read_text()
    assert 0 < len(outputs[0]['pair_indices']) < len(pairs)
    for key, value in outputs[0].items():
        assert np.array_equal(value, outputs[1][key]), key


def test_xfeat_source_is_imported(tmp_path, monkeypatch):
    checkout = tmp_path / 'accelerated_features'
    (checkout / 'modules').mkdir(parents=True)
    (checkout / 'modules' / '__init__.py').touch()
    (checkout / 'modules' / 'xfeat.py').write_text(
        'class XFeat:\n    def __init__(self, weights, top_k):\n        raise RuntimeError("loaded from checkout")\n')
    monkeypatch.delitem(sys.modules, 'modules', raising=False)
    monkeypatch.delitem(sys.modules, 'modules.xfeat', raising=False)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    from recon.poses.xfeat import match
    with pytest.raises(RuntimeError, match='loaded from checkout'):
        match(dict(image_names=[]), tmp_path, np.empty((0, 2), int), tmp_path / 'x.npz',
              dict(xfeat_weights='unused', xfeat_source=str(checkout)))


def test_xfeat_source_conflicting_with_an_imported_checkout_fails(tmp_path, monkeypatch):
    def checkout(name):
        root = tmp_path / name
        (root / 'modules').mkdir(parents=True)
        (root / 'modules' / '__init__.py').touch()
        (root / 'modules' / 'xfeat.py').write_text(
            'class XFeat:\n    def __init__(self, weights, top_k):\n        raise RuntimeError("%s")\n' % name)
        return root
    first, second = checkout('first'), checkout('second')
    monkeypatch.delitem(sys.modules, 'modules', raising=False)
    monkeypatch.delitem(sys.modules, 'modules.xfeat', raising=False)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    from recon.poses.xfeat import match
    arguments = dict(image_names=[]), tmp_path, np.empty((0, 2), int), tmp_path / 'x.npz'
    with pytest.raises(RuntimeError, match='first'):
        match(*arguments, dict(xfeat_weights='unused', xfeat_source=str(first)))
    # Same process, another checkout configured: the cached import must not be used silently.
    with pytest.raises(ImportError, match='already imported from ' + str(first)):
        match(*arguments, dict(xfeat_weights='unused', xfeat_source=str(second)))


def test_xfeat_source_without_module_fails(tmp_path):
    from recon.poses.xfeat import match
    with pytest.raises(FileNotFoundError, match='pose.xfeat_source has no modules/xfeat.py: ' + str(tmp_path)):
        match(dict(image_names=[]), tmp_path, np.empty((0, 2), int), tmp_path / 'x.npz',
              dict(xfeat_weights='unused', xfeat_source=str(tmp_path)))
