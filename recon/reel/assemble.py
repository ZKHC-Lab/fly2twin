# SPDX-License-Identifier: Apache-2.0
"""Assemble the edit from an EDL (json): frame-accurate compositing in numpy, piped to ffmpeg, music muxed.

EDL:
{"fps": 30, "size": [1920, 1080], "out": "x.mp4", "crf": 16,   # x264 CRF; lower = larger, sharper
 "music": {"file": "...mp3", "start_s": 0.0, "fade_in_s": 0.5, "fade_out_s": 2.0},
 "clips": [
   {"kind": "frames", "src": dir, "first": 0, "dur_s": 3.2, "speed": 1.0, "step": 1,
    "xfade_s": 0.0,                          # crossfade from the previous clip (overlaps it)
    "overlays": [{"png": p, "t0": 0.2, "t1": 2.8, "fade": 0.3}],
    "kenburns": [1.0, 1.05]},                # zoom from/to (centre), any kind
   {"kind": "image", "src": png/jpg, "dur_s": 2.1, ...},
   {"kind": "video", "src": mp4, "start_s": 3.0, "dur_s": 4.0, ...},
   {"kind": "wipe", "a": image, "b": frames dir, "b_first": 0, "dur_s": 4.3, "hold_s": 1.0, "wipe_s": 1.2,
    "line": [255,255,255]},                  # a (still) is revealed into b by a vertical line moving left->right
   {"kind": "color", "rgb": [14,17,22], "dur_s": 1.0}
 ]}
Frames dirs hold %05d.jpg rendered at the shot fps; missing indices (preview stride) reuse the previous frame.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np


def load_rgb(path, size):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.shape[2] == 4:
        im = im[..., :3]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    w, h = size
    s = max(w / im.shape[1], h / im.shape[0])
    im = cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (im.shape[0] - h) // 2, (im.shape[1] - w) // 2
    return im[y0 : y0 + h, x0 : x0 + w]


class Frames:
    def __init__(self, folder, size):
        self.folder, self.size, self.cache = Path(folder), size, {}
        self.files = sorted(self.folder.glob('*.jpg')) or sorted(self.folder.glob('*.png'))
        self.index = {int(f.stem): f for f in self.files}

    def get(self, i):
        i = int(i)
        while i not in self.index and i > 0:
            i -= 1
        if i not in self.cache:
            self.cache = {i: load_rgb(self.index[i], self.size)}
        return self.cache[i]


class Video:
    def __init__(self, path, size, fps):
        self.path, self.size, self.fps, self.cap, self.pos, self.frame = path, size, fps, None, -1, None

    def get(self, t):
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.path))
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
        target = int(round(t * src_fps))
        if target < self.pos or self.frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.pos = target - 1
        while self.pos < target:
            ok, bgr = self.cap.read()
            if not ok:
                break
            self.pos += 1
            self.frame = bgr
        return load_rgb_array(self.frame, self.size)


def load_rgb_array(bgr, size):
    im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    w, h = size
    sc = max(w / im.shape[1], h / im.shape[0])
    im = cv2.resize(im, (int(round(im.shape[1] * sc)), int(round(im.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    y0, x0 = (im.shape[0] - h) // 2, (im.shape[1] - w) // 2
    return im[y0 : y0 + h, x0 : x0 + w]


def zoom(im, s):
    if abs(s - 1) < 1e-4:
        return im
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Overlay:
    cache = {}

    @classmethod
    def get(cls, path):
        if path not in cls.cache:
            im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
            a = (
                (im[..., 3:4].astype(np.float32) / 255.0)
                if im.shape[2] == 4
                else np.ones(im.shape[:2] + (1,), np.float32)
            )
            cls.cache[path] = (rgb, a)
        return cls.cache[path]


def clip_frame(c, t, size, fps, sources):
    kind = c['kind']
    if kind == 'color':
        im = np.full((size[1], size[0], 3), c['rgb'], np.uint8)
    elif kind == 'image':
        if 'img' not in sources and c.get('fit') == 'contain_blur':
            # Small source: fit whole picture by height (Lanczos), fill the sides with a blurred, darkened copy.
            src = cv2.cvtColor(cv2.imread(str(c['src']), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            w, h = size
            s_in = min(w / src.shape[1], h / src.shape[0])
            fg = cv2.resize(
                src, (round(src.shape[1] * s_in), round(src.shape[0] * s_in)), interpolation=cv2.INTER_LANCZOS4
            )
            fg = cv2.addWeighted(fg, 1.35, cv2.GaussianBlur(fg, (0, 0), 1.2), -0.35, 0)  # mild unsharp after upscaling
            bg = cv2.GaussianBlur(load_rgb(c['src'], size), (0, 0), 40) * 0.55
            out = bg.astype(np.uint8)
            y0, x0 = (h - fg.shape[0]) // 2, (w - fg.shape[1]) // 2
            out[y0 : y0 + fg.shape[0], x0 : x0 + fg.shape[1]] = fg
            sources['img'] = out
        if 'img' not in sources:
            sources['img'] = load_rgb(c['src'], size)
            if c.get('grade'):
                from .grade import grade

                sources['img'] = grade(sources['img'], c['grade'])
        im = sources['img']
    elif kind == 'frames':
        if 'f' not in sources:
            sources['f'] = Frames(c['src'], size)
        src = sources['f']
        im = src.get(c.get('first', 0) + t * fps * c.get('speed', 1.0))
    elif kind == 'video':
        src = sources.setdefault('v', Video(c['src'], size, fps))
        im = src.get(c.get('start_s', 0) + t * c.get('speed', 1.0))
        if c.get('grade'):
            from .grade import grade

            im = grade(im, c['grade'])
    elif kind == 'wipe':
        if 'a' not in sources:
            sources['a'] = load_rgb(c['a'], size)
            if c.get('grade'):
                from .grade import grade

                sources['a'] = grade(sources['a'], c['grade'])
        if 'b' not in sources:
            sources['b'] = Frames(c['b'], size)
        b = sources['b'].get(c.get('b_first', 0) + max(0, t - c['hold_s'] - c['wipe_s']) * fps)
        x = np.clip((t - c['hold_s']) / c['wipe_s'], 0, 1)
        x = x * x * (3 - 2 * x)
        cut = int(round(x * size[0]))
        im = sources['a'].copy()
        im[:, :cut] = b[:, :cut]
        if 0 < cut < size[0]:
            im[:, max(0, cut - 2) : cut + 2] = c.get('line', [255, 255, 255])
        if t >= c['hold_s'] + c['wipe_s']:
            im = b
    else:
        raise ValueError(kind)
    if c.get('gamma'):
        lut = sources.setdefault('lut', (255 * (np.arange(256) / 255.0) ** c['gamma']).round().astype(np.uint8))
        im = lut[im]
    if c.get('kenburns'):
        z0, z1 = c['kenburns']
        im = zoom(im, z0 + (z1 - z0) * t / c['dur_s'])
    if c.get('overlays'):
        out = im.astype(np.float32)
        for o in c['overlays']:
            if not (o['t0'] <= t <= o['t1']):
                continue
            f = o.get('fade', 0.3)
            k = min(1.0, (t - o['t0']) / f if f else 1, (o['t1'] - t) / f if f else 1)
            rgb, a = Overlay.get(o['png'])
            if rgb.shape[:2] != (size[1], size[0]):
                rgb = cv2.resize(rgb, size)
                a = cv2.resize(a, size)[..., None]
            out = out * (1 - a * k) + rgb * a * k
        im = np.clip(out, 0, 255).astype(np.uint8)
    return im


def timeline(storyboard):
    """Compile contiguous nominal slots; crossfades extend incoming clips backwards."""
    import copy

    clips, slots = [], copy.deepcopy(storyboard['slots'])
    end = 0.0
    names = set()
    for slot in slots:
        name, t0, t1 = slot['name'], slot['start_s'], slot['end_s']
        if name in names or abs(t0 - end) > 1e-6 or t1 <= t0:
            raise ValueError('Slots must have unique names and contiguous increasing boundaries starting at zero')
        names.add(name)
        trans = slot.get('transition', 'cut')
        if trans not in ('none', 'cut', 'xfade'):
            raise ValueError(f'Unknown transition: {trans}')
        xf = slot.get('xfade_s', 0.25) if trans == 'xfade' else 0.0
        if xf < 0 or xf > t0 or (clips and xf > clips[-1]['dur_s']):
            raise ValueError('Crossfade must fit inside the preceding clip')
        source = copy.deepcopy(storyboard['sources'][name])
        multi = isinstance(source, list)
        parts = source if multi else [dict(source, dur_s=t1 - t0)]
        if not parts or abs(sum(p['dur_s'] for p in parts) - (t1 - t0)) > 1e-6:
            raise ValueError(f'Part durations do not fill slot {name}')
        custom = multi and any('overlays' in p for p in parts)
        offset = t0
        for j, part in enumerate(parts):
            if part['dur_s'] <= 0:
                raise ValueError('Part durations must be positive')
            extension = xf if j == 0 else 0.0
            own = part.pop('overlays', [])
            dur = round(part['dur_s'] + extension, 4)
            defaults = storyboard.get('overlays', {}).get(name, []) if j == 0 and not custom else []
            overlays = [
                dict(o, t0=o['t0'] + extension, t1=min(o['t1'] + extension, dur - 0.05)) for o in defaults + own
            ]
            c = dict(part, slot=name, start_s=offset - extension, dur_s=dur, xfade_s=extension)
            # Source-video start_s is media time, not timeline time.
            if part['kind'] == 'video':
                c['start_s'] = part.get('start_s', 0.0)
            c['timeline_start_s'] = offset - extension
            if overlays:
                c['overlays'] = overlays
            clips.append(c)
            offset += part['dur_s']
        end = t1
    fps, size = storyboard.get('fps', 30), storyboard.get('size', [1920, 1080])
    if fps <= 0 or len(size) != 2 or any(n <= 0 or n % 2 for n in size):
        raise ValueError('fps must be positive; size must be two positive even dimensions')
    # Retain the compositor's arithmetic order: int(t * fps) selects source frames,
    # so even 1e-14 s differences at integer frame times can change a chosen frame.
    cursor = 0.0
    for clip in clips:
        cursor -= clip['xfade_s']
        clip['timeline_start_s'] = cursor
        cursor += clip['dur_s']
    return dict(
        fps=fps,
        size=size,
        slots=slots,
        clips=clips,
        seconds=end,
        crf=storyboard.get('crf', 16),
        frames=round(end * fps),
        music=copy.deepcopy(storyboard.get('music')),
    )


def encode(edl, out):
    """Stream frames, check encoder status, mux music and write actual timestamps."""
    import tempfile
    from .common import require_ffmpeg, dump

    ffmpeg = require_ffmpeg()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps, size, clips = edl['fps'], tuple(edl['size']), edl['clips']
    starts = [c['timeline_start_s'] for c in clips]
    total, n = edl['seconds'], edl['frames']
    Overlay.cache.clear()
    with tempfile.TemporaryDirectory(prefix='reel-', dir=out.parent) as temp:
        silent = Path(temp) / 'video.mp4'
        ff = subprocess.Popen(
            [
                ffmpeg,
                '-v',
                'error',
                '-y',
                '-f',
                'rawvideo',
                '-pix_fmt',
                'rgb24',
                '-s',
                f'{size[0]}x{size[1]}',
                '-r',
                str(fps),
                '-i',
                '-',
                '-c:v',
                'libx264',
                '-preset',
                'slow',
                '-crf',
                str(edl.get('crf', 16)),
                '-pix_fmt',
                'yuv420p',
                '-threads',
                '4',
                str(silent),
            ],
            stdin=subprocess.PIPE,
        )
        sources = [dict() for _ in clips]
        try:
            for i in range(n):
                tt, frame = i / fps, None
                for k, c in enumerate(clips):
                    if starts[k] <= tt < starts[k] + c['dur_s']:
                        im = clip_frame(c, tt - starts[k], size, fps, sources[k]).astype(np.float32)
                        a = np.clip((tt - starts[k]) / max(c.get('xfade_s', 0.0), 1e-6), 0, 1)
                        frame = im if frame is None else frame * (1 - a) + im * a
                    elif starts[k] + c['dur_s'] < tt:
                        video = sources[k].get('v')
                        if video and video.cap:
                            video.cap.release()
                        sources[k] = {}
                if frame is None:
                    frame = np.zeros((size[1], size[0], 3), np.float32)
                ff.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
        finally:
            ff.stdin.close()
            rc = ff.wait()
            for source in sources:
                video = source.get('v')
                if video and video.cap:
                    video.cap.release()
            Overlay.cache.clear()
        if rc:
            raise RuntimeError(f'ffmpeg encoder exited with status {rc}')
        music = edl.get('music')
        if music:
            fade = music.get('fade', {})
            fi = music.get('fade_in_s', fade.get('in_s', 0.3))
            fo = music.get('fade_out_s', fade.get('out_s', 2.0))
            af = f'afade=t=in:st=0:d={fi},afade=t=out:st={max(0, total - fo):.3f}:d={fo}'
            subprocess.run(
                [
                    ffmpeg,
                    '-v',
                    'error',
                    '-y',
                    '-i',
                    str(silent),
                    '-ss',
                    str(music.get('start_s', 0)),
                    '-i',
                    music['file'],
                    '-t',
                    f'{total:.3f}',
                    '-filter:a',
                    af,
                    '-c:v',
                    'copy',
                    '-c:a',
                    'aac',
                    '-b:a',
                    '192k',
                    '-shortest',
                    str(out),
                ],
                check=True,
            )
        else:
            import shutil

            shutil.copyfile(silent, out)
    result = dict(edl, out=str(out))
    dump(out.with_suffix('.json'), result)
    return result
