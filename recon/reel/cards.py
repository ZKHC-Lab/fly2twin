# SPDX-License-Identifier: Apache-2.0
"""Full-frame RGBA text overlays (1920x1080) in one typeface, Noto Sans CJK SC.

Styles:
  title   : chapter title, lower-left, large, with a short accent bar
  label   : small caption (e.g. 实拍 / 重建, 小窗名), placed at (x, y) anchor
  number  : one line of result numbers, lower-left under where a title would be
  endcard : project name, tagline, GitHub, footer (default 开源) — on the shared dark background
Usage from code: cards.title('边飞边建', path) etc.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
def font(weight, size, style=None):
    """Find the SC face via fontconfig, or use an explicit font file/index."""
    import shutil
    import subprocess
    style = style or {}
    path, index = style.get('font'), style.get('font_index', 0)
    if not path:
        if not shutil.which('fc-match'):
            raise RuntimeError('Noto Sans CJK SC not found: install fontconfig and Noto CJK, or supply font/font_index')
        result = subprocess.check_output(['fc-match', '-f', '%{family}\n%{file}\n%{index}',
                                          'Noto Sans CJK SC:style=' + weight], text=True).splitlines()
        if len(result) < 3 or 'Noto Sans CJK SC' not in result[0]:
            raise RuntimeError('Noto Sans CJK SC not found: install Noto CJK or supply font/font_index')
        path, index = result[1], int(result[2])
    try:
        return ImageFont.truetype(str(path), size, index=index)
    except OSError as exc:
        raise RuntimeError(f'Cannot load card font: {path} (face {index})') from exc


def _canvas():
    return Image.new('RGBA', (W, H), (0, 0, 0, 0))


def _shadow_text(d, xy, text, f, fill=(255, 255, 255, 255)):
    x, y = xy
    for dx, dy, a in ((0, 2, 110), (0, 0, 60)):
        d.text((x + dx, y + dy), text, font=f, fill=(0, 0, 0, a))
    d.text((x, y), text, font=f, fill=fill)


def title(text, out, sub=None, top=False, style=None):
    im = _canvas()
    d = ImageDraw.Draw(im)
    f = font('Medium', 64, style)
    x, y = (120, 110) if top else (120, H - 220)
    d.rectangle([x, y - 26, x + 56, y - 20], fill=tuple((style or {}).get('accent', [64, 200, 190])) + (255,))
    _shadow_text(d, (x, y), text, f)
    if sub:
        _shadow_text(d, (x, y + 92), sub, font('Regular', 30, style), (230, 235, 240, 255))
    im.save(out)


def headline(text, out, style=None):
    """Centred main title over picture (opening), on a soft dark band for bright scenes."""
    from PIL import ImageFilter
    band = Image.new('L', (W, H), 0)
    ImageDraw.Draw(band).rectangle([0, H / 2 - 110, W, H / 2 + 70], fill=110)
    band = band.filter(ImageFilter.GaussianBlur(60))
    im = Image.new('RGBA', (W, H), tuple((style or {}).get('background', [8, 10, 14])) + (0,))
    im.putalpha(band)
    d = ImageDraw.Draw(im)
    f = font('Bold', 84, style)
    w = d.textlength(text, font=f)
    _shadow_text(d, ((W - w) / 2, H / 2 - 60), text, f)
    im.save(out)


def label(items, out, style=None):
    """items: [(text, x, y, anchor)] with anchor in la/ra/ma (left/right/middle, top)."""
    im = _canvas()
    d = ImageDraw.Draw(im)
    f = font('Regular', 34, style)
    for text, x, y, anchor in items:
        w = d.textlength(text, font=f)
        x0 = x if anchor == 'la' else x - w if anchor == 'ra' else x - w / 2
        d.rounded_rectangle([x0 - 18, y - 8, x0 + w + 18, y + 50], 10,
                            fill=tuple((style or {}).get('background', [0, 0, 0])) + (120,))
        d.text((x0, y), text, font=f, fill=(255, 255, 255, 255))
    im.save(out)


def number(text, out, y=H - 120, style=None):
    im = _canvas()
    d = ImageDraw.Draw(im)
    _shadow_text(d, (120, y), text, font('Regular', 34, style), (235, 240, 245, 255))
    im.save(out)


def endcard(name, tagline, url, out, logo=None, style=None, footer='开源'):
    im = Image.new('RGBA', (W, H), tuple((style or {}).get('background', [14, 17, 22])) + (255,))
    d = ImageDraw.Draw(im)
    y = 330
    if logo and Path(logo).exists():
        lg = Image.open(logo).convert('RGBA')
        lg.thumbnail((360, 140))
        im.alpha_composite(lg, ((W - lg.width) // 2, 180))
        y = 360
    for text, f, fill, dy in ((name, font('Bold', 96, style), (255, 255, 255, 255), 150),
                              (tagline, font('Regular', 40, style), (215, 222, 230, 255), 110),
                              (url, font('Regular', 32, style), tuple((style or {}).get('accent', [64, 200, 190])) + (255,), 70),
                              (footer, font('Medium', 34, style), (255, 255, 255, 255), 0)):
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=fill)
        y += dy
    im.save(out)
