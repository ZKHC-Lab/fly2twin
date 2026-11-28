"""Build a symlink-only multi-scene demo and launch a discrete-GPU browser."""

import argparse
from array import array
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen

TOOLS = Path(__file__).resolve().parent
GPU_ENV = {
    '__NV_PRIME_RENDER_OFFLOAD': '1',
    '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
    '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
}
PLY_TYPES = {
    'char': 'b',
    'int8': 'b',
    'uchar': 'B',
    'uint8': 'B',
    'short': 'h',
    'int16': 'h',
    'ushort': 'H',
    'uint16': 'H',
    'int': 'i',
    'int32': 'i',
    'uint': 'I',
    'uint32': 'I',
    'float': 'f',
    'float32': 'f',
    'double': 'd',
    'float64': 'd',
}


def cloud_bounds(path):
    """Exact per-axis linear 0.5/99.5 percentiles; finite XYZ vertices only.

    Supports ASCII and little/big-endian binary PLY, scalar vertex properties.
    Vertex must be the first nonempty element; later faces are not read.
    """
    axes = [array('d') for _ in range(3)]
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Not a PLY: {path}')
        fmt, count, element, props = None, None, None, []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            words = line.decode('ascii').split()
            if not words:
                continue
            if words[0] == 'end_header':
                break
            if words[0] == 'format':
                fmt = words[1]
            elif words[0] == 'element':
                element = words[1]
                if element == 'vertex':
                    count = int(words[2])
                elif count is None and int(words[2]):
                    raise ValueError('PLY vertex must be first nonempty element')
            elif words[0] == 'property' and element == 'vertex':
                if words[1] not in PLY_TYPES:
                    raise ValueError('Only scalar PLY vertex properties are supported')
                props.append((words[2], PLY_TYPES[words[1]]))
        if count is None or count <= 0:
            raise ValueError('PLY has no vertices')
        indices = [[p[0] for p in props].index(axis) for axis in ('x', 'y', 'z')]
        if fmt not in ('ascii', 'binary_little_endian', 'binary_big_endian'):
            raise ValueError(f'Unsupported PLY format: {fmt}')
        record = struct.Struct(('<' if fmt == 'binary_little_endian' else '>') + ''.join(p[1] for p in props))
        for _ in range(count):
            values = f.readline().split() if fmt == 'ascii' else record.unpack(f.read(record.size))
            xyz = [float(values[i]) for i in indices]
            if all(map(math.isfinite, xyz)):
                for axis, value in zip(axes, xyz):
                    axis.append(value)
    if not axes[0]:
        raise ValueError('PLY has no finite XYZ vertices')

    def percentile(values, q):
        at = (len(values) - 1) * q
        lo = int(at)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)

    bounds = {'min': [], 'max': []}
    for axis in axes:
        ordered = sorted(axis)
        bounds['min'].append(percentile(ordered, 0.005))
        bounds['max'].append(percentile(ordered, 0.995))
    return bounds


def link(target, destination):
    if destination.is_symlink() and destination.resolve() == target.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'Unexpected cache entry: {destination}')
    destination.symlink_to(target, target_is_directory=target.is_dir())


def build_site(exports, names=(), cache=None):
    exports = [Path(p).resolve(strict=True) for p in exports]
    if len(names) > len(exports):
        raise ValueError('More --name values than exports')
    cache = Path(cache or Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'drone3d-demo')
    key = hashlib.sha256(json.dumps([str(p) for p in exports]).encode()).hexdigest()[:20]
    site = cache / key
    site.mkdir(parents=True, exist_ok=True)
    with (site / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        link(TOOLS.parent, site / 'viewer')
        ids, manifests = [], []
        for i, export in enumerate(exports):
            scene = json.loads((export / 'scene.json').read_text())
            link(export, site / f'export-{i}')
            overlay = site / f'scene-{i}'
            overlay.mkdir(exist_ok=True)
            for child in export.iterdir():
                if child.name != 'scene.json':
                    link(child, overlay / child.name)
            if 'bounds' not in scene:
                scene['bounds'] = cloud_bounds(export / 'cloud.ply')
            if 'name' not in scene:
                scene['name'] = names[i] if i < len(names) else export.name
            scene_id = scene.get('id', 'scene')
            if scene_id in ids:
                scene_id = f'{scene_id}-{i}'
            ids.append(scene_id)
            # Use unique IDs exactly as main.js does, including absent IDs.
            scene['id'] = scene_id
            temporary = overlay / 'scene.json.tmp'
            temporary.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(overlay / 'scene.json')
            manifests.append(f'/{key}/scene-{i}/scene.json')
    return site, manifests, ids


def ensure_server(cache, port):
    """Reuse only this tool's server serving the same cache; never kill listeners."""
    endpoint = f'http://127.0.0.1:{port}/__drone3d_demo__'

    def ours():
        try:
            with urlopen(endpoint, timeout=1) as r:
                info = json.load(r)
            return info == {'tool': 'drone3d-demo-v1', 'root': str(cache.resolve())}
        except (OSError, ValueError):
            return False

    if ours():
        return
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is occupied by another service; choose --port') from exc
    with (cache / 'server.log').open('ab') as log:
        process = subprocess.Popen(
            [sys.executable, str(TOOLS / 'serve.py'), str(cache), '--port', str(port)],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    for _ in range(50):
        if ours():
            return
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise RuntimeError(f'Server failed; see {cache / "server.log"}')


def find_browser(explicit=None):
    if explicit:
        found = shutil.which(explicit)
        if not found:
            raise ValueError(f'Browser not executable: {explicit}')
        return found
    return shutil.which('microsoft-edge') or shutil.which('microsoft-edge-stable')


def launch_browser(browser, url, site):
    # Profiles are outside the served cache. Use a fresh instance per launch so an
    # already-open browser cannot forward into an old process or race cache deletion.
    base = site.parent.parent / 'drone3d-demo-profiles' / site.name
    profile = base / str(time.time_ns())
    profile.mkdir(parents=True)
    for name in ('Cache', 'Code Cache'):
        target = profile / 'Default' / name
        assert target.absolute().is_relative_to(profile.absolute())
        if target.exists():
            shutil.rmtree(target)
    with (profile / 'browser.log').open('ab') as log:
        subprocess.Popen(
            [
                browser,
                f'--user-data-dir={profile}',
                '--use-angle=gl-egl',
                '--no-first-run',
                '--no-default-browser-check',
                url,
            ],
            env={**os.environ, **GPU_ENV},
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exports', nargs='+', type=Path)
    parser.add_argument('--name', action='append', default=[])
    parser.add_argument('--start', type=int, default=1, help='1-based scene number')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--margin', type=float, default=1.5)
    parser.add_argument('--guided', action='store_true')
    parser.add_argument('--no-launch', action='store_true')
    parser.add_argument('--check', type=Path)
    parser.add_argument('--browser')
    args = parser.parse_args(argv)
    if not 1 <= args.start <= len(args.exports):
        parser.error('--start must be between 1 and the number of exports')
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error('--margin must be finite and nonnegative')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    try:
        site, manifests, ids = build_site(args.exports, args.name)
        ensure_server(site.parent, args.port)
        query = urlencode(
            {
                'manifest': ','.join(manifests),
                'scene': ids[args.start - 1],
                'margin': args.margin,
                'guided': int(args.guided),
                'quality': 'high',
            }
        )
        url = f'http://127.0.0.1:{args.port}/{site.name}/viewer/index.html?{query}'
        print(url, flush=True)
        print(
            'WASD 移动 · Q/E 转向 · R/F 升降 · Shift 加速 · 鼠标拖动转向 · Tab 飞行 · [ ] 切换场景 · P 巡航 · G 引导 · H 隐藏界面',
            flush=True,
        )
        browser = find_browser(args.browser)
        if args.check:
            args.check.mkdir(parents=True, exist_ok=True)
            if not browser:
                raise RuntimeError('--check requires Edge or --browser PATH')
            result = subprocess.run(
                ['node', str(TOOLS / 'check_demo.cjs'), url, str(args.check.resolve()), browser, json.dumps(ids)],
                env={**os.environ, **GPU_ENV},
            )
            if result.returncode:
                return result.returncode
        if not args.no_launch:
            if browser:
                launch_browser(browser, url, site)
            else:
                print('Edge not found; open the URL manually or use --browser PATH.')
        return 0
    except (OSError, ValueError, RuntimeError, struct.error) as exc:
        print(f'demo: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
