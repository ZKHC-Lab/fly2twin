#!/usr/bin/env python3
"""Phone video -> uniformly spaced, sharpness-picked JPEG frames.

Pass 1 decodes every frame at low resolution as luma only (no tone mapping) and
records the Laplacian variance per frame.  Frames are then binned into
`--target-frames` equal spans of the source timeline and the sharpest frame of
each span is kept, which gives uniform temporal coverage without motion blur.
Pass 2 decodes the video once more with an HDR->SDR tone map (only when the
source really is HLG/PQ) and writes just the selected frames.

Output: <out>/frames/<source_frame>.jpg plus <out>/extract.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HDR_TRANSFERS = {'arib-std-b67', 'smpte2084'}  # HLG (iPhone/Android default), PQ (HDR10)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def probe(video):
    result = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                  '-show_entries',
                  'stream=codec_name,width,height,color_transfer,color_primaries,pix_fmt,'
                  'r_frame_rate,avg_frame_rate:stream_tags=rotate:stream_side_data=rotation',
                  '-show_entries', 'format=duration,size', '-of', 'json', str(video)],
                 capture_output=True, text=True)
    info = json.loads(result.stdout)
    if not info.get('streams'):
        raise RuntimeError(f'No video stream in {video}')
    stream, fmt = info['streams'][0], info.get('format', {})
    rotation = 0
    for item in stream.get('side_data_list', []):
        if 'rotation' in item:
            rotation = int(float(item['rotation']))
    if not rotation and 'rotate' in (stream.get('tags') or {}):
        rotation = int(float(stream['tags']['rotate']))
    width, height = int(stream['width']), int(stream['height'])
    # ffmpeg autorotates before filters, so pass 1/2 see the displayed orientation.
    if abs(rotation) % 180 == 90:
        width, height = height, width
    fps = stream.get('avg_frame_rate') or stream.get('r_frame_rate') or '30/1'
    try:
        num, den = (int(x) for x in fps.split('/'))
        fps = num / den if den else 30.0
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    fmt_size = int(fmt.get('size') or 0)
    return dict(path=str(video), width=width, height=height, coded_width=int(stream['width']),
                coded_height=int(stream['height']), rotation=rotation, fps=fps,
                duration=float(fmt.get('duration') or 0),
                pix_fmt=stream.get('pix_fmt'), color_transfer=stream.get('color_transfer'),
                color_primaries=stream.get('color_primaries'),
                codec=stream.get('codec_name'), bytes=fmt_size)


def is_hdr(meta, mode):
    if mode == 'on':
        return True
    if mode == 'off':
        return False
    transfer = (meta['color_transfer'] or '').lower()
    if transfer in HDR_TRANSFERS:
        return True
    return bool(meta['pix_fmt'] and ('10le' in meta['pix_fmt'] or '12le' in meta['pix_fmt']))


def display_chain(meta, hdr, long_edge, scale_prefix=True):
    """HDR->SDR conversion, then downscale to `long_edge` on the long side."""
    width, height = meta['width'], meta['height']
    if width >= height:
        scale = f'scale={long_edge}:-2:flags=lanczos'
    else:
        scale = f'scale=-2:{long_edge}:flags=lanczos'
    if not hdr:
        return f'{scale},setsar=1,format=yuv420p'
    # Standard HLG/PQ -> BT.709 SDR chain; npl=100 maps diffuse white.
    tonemap = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,'
               'tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
    return f'{scale},{tonemap},setsar=1'


def sharpness_scan(meta, video, width):
    """Decode every frame as low-res luma and return the Laplacian variance per frame."""
    height = max(2, int(round(width * meta['height'] / meta['width'] / 2)) * 2)
    chain = f'scale={width}:{height}:flags=area,format=gray'
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video), '-map', '0:v:0',
           '-vf', chain, '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    size = width * height
    values = []
    started = time.perf_counter()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size * 8) as proc:
        read = proc.stdout.read
        while True:
            buffer = read(size)
            if not buffer or len(buffer) < size:
                break
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width)
            values.append(float(cv2.Laplacian(frame, cv2.CV_64F).var()))
            if len(values) % 500 == 0:
                print(json.dumps(dict(event='scan', frames=len(values),
                                      seconds=round(time.perf_counter() - started, 1))), flush=True)
        stderr = proc.stderr.read().decode('utf-8', 'replace')
        if proc.wait() != 0:
            raise RuntimeError(f'Sharpness scan failed: {stderr[-2000:]}')
    return np.asarray(values, dtype=np.float64), height


def select(sharpness, target, minimum):
    """Sharpest frame inside each of `target` equal timeline bins."""
    total = len(sharpness)
    count = int(min(target, total))
    if count <= 0:
        raise RuntimeError('No frames decoded; check the video and ffmpeg output')
    if count >= total:
        return list(range(total))
    edges = np.linspace(0, total, count + 1).astype(np.int64)
    picked = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        picked.append(int(start + int(np.argmax(sharpness[start:stop]))))
    if len(picked) < minimum:
        raise RuntimeError(f'Only {len(picked)} bins for a requested minimum of {minimum} frames')
    return picked


def extract(video, out, dense, keep, meta, hdr, long_edge, quality):
    """Single pass with an explicit frame-number select expression.

    `frames_sfm/` receives every dense frame; `frames/` is a symlink view of the
    subset that 3DGS will train on.  SfM benefits from a shorter baseline than
    training needs, and a texture-poor room needs the denser chain.
    """
    folder = out / 'frames_sfm'
    for stale in (folder, out / 'frames'):
        if stale.exists() or stale.is_symlink():
            raise FileExistsError(f'Use a fresh extraction directory: {stale.resolve()}')
    folder.mkdir(parents=True)
    expression = '+'.join(f'eq(n\\,{index})' for index in dense)
    chain = f"select='{expression}',{display_chain(meta, hdr, long_edge)}"
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video), '-map', '0:v:0', '-an',
           '-vf', chain, '-fps_mode', 'passthrough', '-q:v', str(quality), '-start_number', '1',
           '-f', 'image2', str(folder / '%05d.jpg')]
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Frame extraction failed: {result.stderr[-4000:]}')
    written = sorted(folder.glob('*.jpg'), key=lambda p: int(p.stem))
    if len(written) != len(dense):
        raise RuntimeError(f'ffmpeg wrote {len(written)} frames, expected {len(dense)}')
    # Two-phase rename: the sequential ffmpeg names and the source frame numbers
    # share one namespace, so a direct rename would overwrite a pending frame.
    staged = [path.rename(folder / f'stage{position:05d}.jpg') for position, path in enumerate(written)]
    for path, frame in zip(staged, dense):
        path.rename(folder / f'{frame:05d}.jpg')
    final = sorted(folder.glob('*.jpg'), key=lambda p: int(p.stem))
    if [int(p.stem) for p in final] != sorted(dense):
        raise RuntimeError(f'Dense frame rename mismatch: {len(final)} files for {len(dense)} frames')
    subset = out / 'frames'
    if keep is dense:
        # Train on every SfM frame: one symlink instead of 1200 links.
        subset.symlink_to('frames_sfm', target_is_directory=True)
    else:
        subset.mkdir()
        for frame in keep:
            (subset / f'{frame:05d}.jpg').symlink_to(f'../frames_sfm/{frame:05d}.jpg')
    if not all((subset / f'{frame:05d}.jpg').is_file() for frame in keep):
        raise RuntimeError('Kept frame symlinks do not resolve')
    return dict(dense_frames=len(final), kept_frames=len(keep),
                seconds=time.perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--target-frames', type=int, default=300, help='preferred number of frames (200-400)')
    parser.add_argument('--min-frames', type=int, default=200)
    parser.add_argument('--long-edge', type=int, default=1600, help='output frames long side, px')
    parser.add_argument('--scan-width', type=int, default=640)
    parser.add_argument('--quality', type=int, default=2, help='JPEG -q:v (2 is visually lossless)')
    parser.add_argument('--hdr', choices=['auto', 'on', 'off'], default='auto')
    parser.add_argument('--train-subset', choices=['all', 'subset'], default='all',
                        help='all: train on every SfM frame; subset: train on every Nth bin only')
    parser.add_argument('--sfm-multiple', type=int, default=3,
                        help='extract this many times more frames for SfM; every Nth is kept for '
                             'training (1 disables the dense SfM set)')
    args = parser.parse_args(argv)

    cv2.setNumThreads(1)
    if not args.video.is_file():
        sys.exit(f'Video not found: {args.video}')
    meta = probe(args.video)
    hdr = is_hdr(meta, args.hdr)
    print(json.dumps(dict(event='probe', **meta, hdr=hdr)), flush=True)

    started = time.perf_counter()
    sharpness, scan_height = sharpness_scan(meta, args.video, args.scan_width)
    multiple = max(1, args.sfm_multiple)
    dense = select(sharpness, args.target_frames * multiple,
                   args.min_frames * multiple) if multiple > 1 else \
        select(sharpness, args.target_frames, args.min_frames)
    frames = dense if args.train_subset == 'all' else (dense[::multiple] if multiple > 1 else dense)
    picked = np.array([sharpness[i] for i in frames])
    median = float(np.median(picked))
    blurry = [int(frames[i]) for i in np.flatnonzero(picked < 0.3 * median)]
    extraction = extract(args.video, args.out, dense, frames, meta, hdr, args.long_edge, args.quality)

    rows = [dict(index=int(i), frame=int(f), sharpness=float(sharpness[f]),
                 time_s=round(float(f) / meta['fps'], 3)) for i, f in enumerate(frames)]
    dump(args.out / 'extract.json', dict(
        video=meta, hdr=hdr, hdr_mode=args.hdr,
        tonemap=('zscale linear npl=100 -> bt709 primaries -> hable -> bt709 transfer/matrix' if hdr else 'none'),
        filter_chain=display_chain(meta, hdr, args.long_edge), long_edge=args.long_edge,
        decoded_frames=len(sharpness), scan_size=[args.scan_width, scan_height],
        target_frames=args.target_frames, selected_frames=len(frames),
        sfm_frames=len(dense), sfm_multiple=multiple,
        selection='equal timeline bins; sharpest frame per bin (Laplacian variance on low-res luma); '
                  'SfM uses a 1/sfm_multiple denser chain, training keeps every sfm_multiple-th bin',
        sharpness_all_median=float(np.median(sharpness)), sharpness_all_p10=float(np.percentile(sharpness, 10)),
        sharpness_selected_median=median, sharpness_selected_p10=float(np.percentile(picked, 10)),
        blurry_below_0p3_median=blurry, extraction=extraction,
        seconds=time.perf_counter() - started, frames=rows))
    print(json.dumps(dict(event='complete', selected=len(frames), decoded=len(sharpness), hdr=hdr,
                          sharpness_median=median, blurry=len(blurry),
                          seconds=round(time.perf_counter() - started, 1))), flush=True)


if __name__ == '__main__':
    main()
