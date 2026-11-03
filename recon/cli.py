# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
import yaml
from .pipeline import run,STAGES


def main():
    p=argparse.ArgumentParser(description='Offline reconstruction and Gaussian splatting')
    commands=p.add_subparsers(dest='command',required=True)
    r=commands.add_parser('run')
    r.add_argument('--config',required=True,type=Path)
    r.add_argument('--through',choices=STAGES,default='export')
    r.add_argument('--only',choices=STAGES)
    ref=commands.add_parser('lidar-reference', help='Build clean global 5 cm Livox reference')
    ref.add_argument('--bag', required=True, type=Path)
    ref.add_argument('--out', required=True, type=Path)
    ref.add_argument('--lidar-topic', default='/livox/lidar')
    ref.add_argument('--odometry-topic', default='/Odometry')
    ref.add_argument('--body-from-lidar', type=Path, help='JSON rigid 4x4 T_BL (LiDAR to body)')
    dist=commands.add_parser('lidar-distance', help='Pooled nearest-reference distances, no alignment')
    dist.add_argument('--cloud', required=True, type=Path)
    dist.add_argument('--reference', required=True, type=Path)
    dist.add_argument('--out', required=True, type=Path)
    from .reel.cli import register, dispatch
    register(commands)
    args=p.parse_args()
    if args.command == 'reel':
        try:
            dispatch(args)
        except (ImportError, ValueError, RuntimeError, FileNotFoundError) as exc:
            p.error(str(exc))
        return
    if args.command.startswith('lidar-'):
        import json
        from .evaluation.lidar import build_reference, distances
        try:
            if args.command == 'lidar-reference':
                transform = json.loads(args.body_from_lidar.read_text()) if args.body_from_lidar else None
                result = build_reference(args.bag, args.out, lidar_topic=args.lidar_topic,
                                         odometry_topic=args.odometry_topic, body_from_lidar=transform)
            else:
                if args.out.resolve() in (args.cloud.resolve(), args.reference.resolve()):
                    raise ValueError('Output must not overwrite input')
                result = distances(args.cloud, args.reference)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        except (ImportError, ValueError) as exc:
            p.error(str(exc))
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    cfg=yaml.safe_load(args.config.read_text())
    base=args.config.resolve().parent
    for key in ('manifest','video','delivery','cameras','selection','normalization_file','body_poses','output'):
        if cfg.get(key): cfg[key]=str((base/Path(cfg[key]).expanduser()).resolve())
    for section,keys in [('depth',('cache','weights','omega_source')),('pose',('xfeat_source','xfeat_weights')),('eval',('omega_dir',))]:
        for key in keys:
            if cfg.get(section,{}).get(key): cfg[section][key]=str((base/Path(cfg[section][key]).expanduser()).resolve())
    print(run(cfg,args.through,args.only))


if __name__=='__main__': main()
