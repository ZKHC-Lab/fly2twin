# SPDX-License-Identifier: Apache-2.0
"""recon sim prepare | pairs | eval."""
import json
from pathlib import Path


def register(commands):
    p = commands.add_parser('sim', help='Evaluate a vision flight policy inside an exported scene')
    sub = p.add_subparsers(dest='sim_command', required=True)
    s = sub.add_parser('prepare', help='Level the floor and close the ceiling of an export')
    s.add_argument('export', type=Path)
    s.add_argument('out', type=Path, help='scene .npz')
    s.add_argument('--ceiling-z', type=float, help='Synthetic ceiling height above the floor (m); default: top of the cloud')
    s.add_argument('--seed', type=int, default=20261020)
    s = sub.add_parser('pairs', help='Sample start/goal pairs at flight height')
    s.add_argument('scene', type=Path)
    s.add_argument('out', type=Path, help='pairs .npz')
    s.add_argument('--count', type=int, default=50)
    s.add_argument('--seed', type=int, default=20261030)
    s.add_argument('--height', type=float, default=1.2)
    s.add_argument('--min-distance', type=float, default=6.)
    s.add_argument('--max-distance', type=float, default=20.)
    s.add_argument('--blocked-min', type=int, help='Pairs whose straight line is blocked; default: half')
    s = sub.add_parser('eval', help='Fly every pair once with a DiffDrone-PyTorch3D policy')
    s.add_argument('scene', type=Path)
    s.add_argument('pairs', type=Path)
    s.add_argument('out', type=Path, help='output directory')
    s.add_argument('--diffdrone', type=Path, required=True, help='DiffDrone-PyTorch3D checkout')
    s.add_argument('--checkpoint', type=Path, required=True)
    s.add_argument('--steps', type=int, default=200)
    s.add_argument('--seed', type=int, default=20261020)
    s.add_argument('--limit', type=int)
    s.add_argument('--save-depth', action='store_true', help='Also store every 48x64 policy depth frame')


def _load(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def dispatch(args):
    import numpy as np
    from . import scene as S
    if getattr(args, 'out', None) and any(Path(p).resolve() == args.out.resolve() for p in
                                          (getattr(args, k, None) for k in ('export', 'scene', 'pairs')) if p):
        raise ValueError('Output must not overwrite an input')
    if args.sim_command == 'prepare':
        result = S.prepare(args.export, args.ceiling_z, args.seed)
        report = json.loads(str(result['info']))
    elif args.sim_command == 'pairs':
        result = S.sample_pairs(_load(args.scene), args.count, args.seed, args.height,
                                distance_range=(args.min_distance, args.max_distance), blocked_min=args.blocked_min)
        report = dict(pairs=len(result['starts']), straight_blocked=int(result['straight_blocked'].sum()),
                      proposals=int(result['proposals']), distance_m=[float(result['distances'].min()),
                                                                      float(result['distances'].max())])
    else:
        from .flight import evaluate, load_diffdrone
        agent = load_diffdrone(args.diffdrone, args.checkpoint)
        summary, arrays = evaluate(_load(args.scene), _load(args.pairs), agent, args.steps, args.seed,
                                   args.limit, args.save_depth)
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out / 'episodes.npz', **arrays)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print(json.dumps(report, indent=2))
