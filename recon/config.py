"""Configuration boundary for the staged reconstruction pipeline."""


def _keys(*names, **sections):
    return dict(dict.fromkeys(names), **sections)


# Every key the pipeline reads; INPUTS.md documents each one. A misspelt key would otherwise
# switch its feature off silently and still give the run a new fingerprint directory.
# None marks a value, a mapping marks a section with its own allowed keys.
KEYS = _keys(
    'manifest', 'video', 'delivery', 'output', 'exclude_frames', 'protocol', 'cameras',
    'body_poses', 'body_from_camera', 'time_offset_ms', 'selection', 'normalization_file',
    'normalization_cast_first', 'threads', 'dependency_revision',
    normalization=_keys('normalization_center', 'meters_per_unit', 'scene_scale'),
    extract=_keys('target_frames', 'min_frames', 'long_edge', 'scan_width', 'quality', 'hdr',
                  'train_subset', 'sfm_multiple'),
    sfm=_keys('camera_model', 'focal_factor', 'overlap', 'loop_keyframes', 'loop_max_pairs',
              'loop_min_gap', 'max_features', 'peak_threshold', 'min_model_size', 'threads'),
    pose=_keys('provider', 'refine', 'units', 'position_sigma_m', 'xfeat_source', 'xfeat_weights',
               'batch_size', 'ransac_threads',
               metric=_keys('camera_height_m')),
    keyframes=_keys('enabled', 'translation_m', 'rotation_deg', 'max_gap_s', 'trim_static'),
    split=_keys('mode', 'exclude_blur'),
    depth=_keys('weights', 'cache', 'omega_source', 'resolution', 'first_window', 'new_frames',
                'overlap'),
    fusion=_keys('max_points', 'revisit'),
    train=_keys('seed', 'resolution', 'steps', 'exact_steps', 'scale_densification',
                'schedule_rounding', 'sh_interval', 'reset_at_zero', 'discard_initial_permutation',
                'depth_lambda', 'aniso_lambda', 'pose_opt', 'pose_lr', 'pose_reg', 'max_width',
                'image_cache_frames'),
    eval=_keys('alignment_steps', 'aligned_width', 'omega_dir'),
    export=_keys('reference_drone', 'name', 'subtitle'),
)


def unknown_keys(cfg, allowed=KEYS, prefix=''):
    errors = []
    where = prefix[:-1] or 'the top level'
    for key, value in cfg.items():
        if key not in allowed:
            errors.append(f'unknown key {prefix}{key} ({where} allows {", ".join(allowed)})')
        elif allowed[key] is not None:
            if isinstance(value, dict):
                errors += unknown_keys(value, allowed[key], f'{prefix}{key}.')
            else:
                errors.append(f'{prefix}{key} must be a mapping')
    return errors


def validate(cfg):
    errors = unknown_keys(cfg)
    if errors:
        raise ValueError('Invalid reconstruction configuration: ' + '; '.join(errors))
    pose = cfg.get('pose', {})
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
