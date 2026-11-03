"""Configuration boundary for the staged reconstruction pipeline."""


def validate(cfg):
    pose = cfg.get('pose', {})
    errors = []
    if 'mode' in pose:
        errors.append('pose.mode was removed; set pose.provider and pose.refine')
    if 'units' in cfg:
        errors.append('top-level units was removed; set pose.units for external poses')
    if sum(bool(cfg.get(k)) for k in ('manifest', 'video', 'delivery')) != 1:
        errors.append('provide exactly one of manifest, video or delivery')
    if not cfg.get('output'):
        errors.append('output is required')
    if cfg.get('protocol', 'drone') not in ('drone', 'phone'):
        errors.append('protocol must be drone or phone')
    provider = pose.get('provider', 'external' if cfg.get('cameras') or cfg.get('body_poses') else 'colmap')
    if provider not in ('external', 'glomap', 'colmap'):
        errors.append('pose.provider must be external, glomap or colmap')
    if pose.get('refine', 'none') not in ('none', 'ba'):
        errors.append('pose.refine must be none or ba')
    if 'units' in pose and (pose.get('provider', 'external') != 'external' or pose['units'] not in ('meters', 'arbitrary')):
        errors.append('pose.units is meters or arbitrary and only applies to external poses')
    height = pose.get('metric', {}).get('camera_height_m')
    if height is not None and (not isinstance(height, (float, int)) or height <= 0):
        errors.append('pose.metric.camera_height_m must be positive')
    if errors:
        raise ValueError('Invalid reconstruction configuration: ' + '; '.join(errors))
