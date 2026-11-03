# SPDX-License-Identifier: Apache-2.0

def schedule(frame_count, steps=None, *, exact_steps=False, scale_densification=True,
             schedule_rounding='truncate', sh_interval=None):
    if frame_count < 1:
        raise ValueError('Training split is empty')
    if steps is None:
        steps = frame_count * 60 if exact_steps else max(30000, round(frame_count*60/1000)*1000)
    steps = int(steps)
    if steps < 1:
        raise ValueError('Steps must be positive')
    if schedule_rounding not in ('truncate', 'round'):
        raise ValueError('Unknown schedule_rounding')
    convert = round if schedule_rounding == 'round' else int
    factor = steps/30000 if scale_densification else 1
    times = {k: max(1, int(convert(v*factor))) for k,v in dict(refine_start_iter=500,
        refine_stop_iter=15000, reset_every=3000, refine_every=100).items()}
    interval = max(1, int(steps/30)) if sh_interval is None else int(sh_interval)
    if interval < 1:
        raise ValueError('SH interval must be positive')
    return dict(steps=steps, sh_interval=interval, strategy=times)
