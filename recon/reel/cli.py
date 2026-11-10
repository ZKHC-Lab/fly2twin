# SPDX-License-Identifier: Apache-2.0
"""Register only the six supported reel commands."""
import json
from pathlib import Path


def register(commands):
    p = commands.add_parser('reel', help='Design, audit, render and edit exported scenes')
    sub = p.add_subparsers(dest='reel_command', required=True)
    s = sub.add_parser('shot')
    s.add_argument('spec', type=Path)
    s.add_argument('out', type=Path)
    s.add_argument('--preview', action='store_true')
    s.add_argument('--only', nargs='+')
    s = sub.add_parser('search')
    s.add_argument('scene', type=Path)
    s.add_argument('templates', type=Path)
    s.add_argument('out', type=Path)
    s.add_argument('--step', type=int, default=6)
    s.add_argument('--min-score', type=float, default=.9)
    s = sub.add_parser('growth')
    s.add_argument('scene', type=Path)
    s.add_argument('out', type=Path)
    for flag, default in [('start-frac', 0.), ('grow-s', 5.3), ('dissolve-s', 2.8), ('height', 4.5),
                          ('back', 6.5), ('orbit-deg', 40.), ('trail-smooth', 0.), ('point-size', .012), ('trail-gain', 1.)]:
        s.add_argument('--'+flag, type=float, default=default)
    s.add_argument('--stable', action='store_true')
    s.add_argument('--grade', type=json.loads, help='JSON exposure/curve/cool/saturation object')
    s.add_argument('--background', type=float, nargs=3, default=[.055, .067, .086])
    s = sub.add_parser('card')
    s.add_argument('kind', choices=['headline', 'title', 'label', 'number', 'endcard'])
    s.add_argument('out', type=Path)
    s.add_argument('--text', default='')
    s.add_argument('--sub')
    s.add_argument('--top', action='store_true')
    s.add_argument('--tagline', default='')
    s.add_argument('--url', default='')
    s.add_argument('--footer', default='开源', help='Endcard last line')
    s.add_argument('--logo', type=Path)
    s.add_argument('--items', type=json.loads, help='Label [[text,x,y,anchor],...]')
    s.add_argument('--font', type=Path)
    s.add_argument('--font-index', type=int, default=0)
    s.add_argument('--background', type=int, nargs=3, default=[14,17,22])
    s.add_argument('--accent', type=int, nargs=3, default=[64,200,190])
    s = sub.add_parser('cut')
    s.add_argument('storyboard', type=Path)
    s.add_argument('out', type=Path)
    s = sub.add_parser('beats')
    s.add_argument('music', type=Path)


def dispatch(args):
    from . import api
    data = vars(args).copy()
    name = data.pop('reel_command')
    data.pop('command')
    if name == 'card':
        kind, out = data.pop('kind'), data.pop('out')
        style = {k: data[k] for k in ('font', 'font_index', 'background', 'accent')}
        options = dict(style=style)
        if kind == 'endcard':
            options.update(name=data['text'], tagline=data['tagline'], url=data['url'], logo=data['logo'], footer=data['footer'])
        elif kind == 'label':
            options['items'] = data['items'] or [[data['text'], 120, 100, 'la']]
        else:
            options['text'] = data['text']
            if kind == 'title':
                options.update(sub=data['sub'], top=data['top'])
        result = api.card(kind, out, **options)
    else:
        result = getattr(api, name)(**data)
    print(json.dumps(result, ensure_ascii=False, default=str))
