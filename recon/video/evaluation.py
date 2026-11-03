import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from recon.gaussian.render import render
from recon.io.common import dump

class OmegaPrior:
    """Native 288x512 Omega depth, resized per frame, with a confidence mask.

    Pearson correlation is invariant to affine transforms of INVERSE depth.
    Consequently multiplicative window depth scales need no fitting; an additive
    offset in depth itself is not invariant after taking reciprocals.
    """

    def __init__(self, folder, frame_ids, width, height, conf_percentile=50.0):
        folder = Path(folder)
        self.depth = np.load(folder / 'depth.npy', mmap_mode='r')
        self.conf = np.load(folder / 'conf.npy', mmap_mode='r')
        rows = np.load(folder / 'frame_ids.npy')
        self.row = {int(frame): position for position, frame in enumerate(rows)}
        self.frame_ids = frame_ids
        self.size = (width, height)
        self.conf_percentile = conf_percentile
        self.cache = {}
        self.empty = 0

    def read(self, frame):
        row = self.row.get(frame)
        if row is None:
            raise RuntimeError(f'No Omega depth for frame {frame}')
        return np.asarray(self.depth[row], dtype=np.float32), np.asarray(self.conf[row], dtype=np.float32)

    def __call__(self, index):
        if index in self.cache:
            return self.cache[index]
        frame = int(self.frame_ids[index])
        depth, conf = self.read(frame)
        valid = np.isfinite(depth) & np.isfinite(conf) & (depth > 0)
        if valid.any():
            valid &= conf >= np.percentile(conf[valid], self.conf_percentile)
        import cv2
        depth = cv2.resize(depth, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(valid.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST).astype(bool)
        mask &= np.isfinite(depth) & (depth > 0)
        if mask.sum() < 2:
            self.empty += 1
            mask[:] = False
            depth[:] = 1.0
        result = (depth, mask)
        # Bound the full-resolution CPU cache; 1096 float32 maps would exceed 6 GiB.
        self.cache.clear()
        self.cache[index] = result
        return result


class WindowPrior(OmegaPrior):
    """Same phone evaluation mask, reading raw new-owner window predictions."""

    def __init__(self, folder, frame_ids, width, height, conf_percentile=50.0):
        folder = Path(folder)
        windows = json.loads((folder / 'windows.json').read_text())['windows']
        self.row = {int(frame): (folder / win['id'] / 'prediction.npz', win['idx'].index(frame))
                    for win in windows for frame in win['new']}
        self.frame_ids, self.size = frame_ids, (width, height)
        self.conf_percentile, self.cache, self.empty = conf_percentile, {}, 0
        self.loaded_path, self.loaded = None, None

    def read(self, frame):
        path, row = self.row[frame]
        if path != self.loaded_path:
            with np.load(path, allow_pickle=False) as z:
                self.loaded = (z['depth'][..., 0], z['depth_conf'])
            self.loaded_path = path
        return tuple(np.asarray(array[row], dtype=np.float32) for array in self.loaded)


def se3_exp(twist):
    """6-vector (rotation, translation) -> 4x4 SE3 matrix, differentiable at zero.

    Built with concatenation only: writing into a tensor created by torch.eye would
    drop the autograd graph and the pose fit would have nothing to differentiate.
    """
    rotation, translation = twist[:3], twist[3:]
    theta = torch.linalg.norm(rotation)
    axis = rotation / theta.clamp_min(1e-12)
    zero = torch.zeros((), dtype=twist.dtype, device=twist.device)
    skew = torch.stack([torch.stack([zero, -axis[2], axis[1]]),
                        torch.stack([axis[2], zero, -axis[0]]),
                        torch.stack([-axis[1], axis[0], zero])])
    # skew holds the unit axis, so Rodrigues needs sin(theta) and 1-cos(theta) directly.
    a = torch.sin(theta)
    b = 1.0 - torch.cos(theta)
    matrix = torch.eye(3, dtype=twist.dtype, device=twist.device) + a * skew + b * (skew @ skew)
    top = torch.cat([matrix, translation[:, None]], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=twist.dtype, device=twist.device)
    return torch.cat([top, bottom], dim=0)


def median_depth(c2w, K, points, width, height):
    """Median optical depth of the SfM points that project inside the frame."""
    c2w = np.asarray(c2w)
    points_c = (c2w[:3, :3].T @ (points - c2w[:3, 3]).T).T
    depth = points_c[:, 2]
    inside = depth > 1e-6
    if not inside.any():
        return float('nan')
    uv = (K @ points_c[inside].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return float(np.median(depth[inside][keep])) if keep.any() else float('nan')


def aligned_evaluation(splats, data, images, out, steps, width, fit_colour=False):
    """Score each test view with the SfM pose and with a pose re-fitted to the target RGB."""
    from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    scale = width / images.width
    size = (int(round(images.width * scale)), int(round(images.height * scale)))
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda().eval()
    rows = []
    original_requires = {key: value.requires_grad for key, value in splats.items()}
    for value in splats.values():
        value.requires_grad_(False)
    try:
        for index in data['test_indices']:
            index = int(index)
            full, frame = images.get(index)
            resized = cv2.resize(full, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else full
            target = torch.from_numpy(resized).cuda().float()[None] / 255.0
            y = target.permute(0, 3, 1, 2)
            c2w = data['c2w'][index:index + 1]
            K = data['K'][index:index + 1].clone()
            K[:, 0, :] *= scale
            K[:, 1, :] *= scale

            def score(prediction):
                x = prediction[..., :3].clamp(0, 1).permute(0, 3, 1, 2)
                values = dict(psnr=float(peak_signal_noise_ratio(x, y, data_range=1.0)),
                              ssim=float(structural_similarity_index_measure(x, y, data_range=1.0)),
                              lpips=float(metric(x, y)))
                metric.reset()
                return values

            with torch.no_grad():
                fixed, _, _ = render(splats, c2w, K, size[0], size[1])
            fixed_values = score(fixed)
            # Start just off zero so the axis-angle derivative is well defined.
            twist = torch.full((6,), 1e-6, device='cuda', requires_grad=True)
            # Colour is fitted together with the pose: an exposure/white-balance
            # mismatch and a pose error look similar in PSNR but not in the fit.
            colour = torch.cat([torch.ones(3, device='cuda'), torch.zeros(3, device='cuda')])
            colour = colour.clone().requires_grad_(fit_colour)
            optimizer = torch.optim.Adam([twist, colour] if fit_colour else [twist], lr=1e-4)
            with torch.enable_grad():
                for _ in range(steps):
                    pose = se3_exp(twist)[None] @ c2w
                    prediction, _, _ = render(splats, pose, K, size[0], size[1])
                    rgb = prediction[..., :3] * colour[:3][None, None, :] + colour[3:][None, None, :]
                    loss = F.l1_loss(rgb.clamp(0, 1), target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                pose = se3_exp(twist)[None] @ c2w
                prediction, _, _ = render(splats, pose, K, size[0], size[1])
                rgb = prediction[..., :3] * colour[:3][None, None, :] + colour[3:][None, None, :]
                fitted = dict(rgb=rgb)
                aligned_values = score(rgb)
                pose_only = score(prediction)
            rows.append(dict(index=index, frame=frame, fixed=fixed_values, aligned=aligned_values,
                             pose_only=pose_only,
                             twist=twist.detach().cpu().numpy().tolist(),
                             colour=colour.detach().cpu().numpy().tolist()))
            print(json.dumps(dict(event='aligned', frame=frame, **{k: round(v, 3) for k, v in
                                  fixed_values.items()},
                                  aligned_psnr=round(aligned_values['psnr'], 3))), flush=True)
    finally:
        for key, flag in original_requires.items():
            splats[key].requires_grad_(flag)

    def summarize(mode, key):
        values = [row[mode][key] for row in rows]
        return dict(mean=float(np.mean(values)), median=float(np.median(values))) if values else None

    dump(out / 'aligned_per_frame.json', rows)
    return dict(resolution=list(size), fit_steps=steps, fit_colour=fit_colour,
                protocol=f'per test frame: Adam 1e-4 for {steps} steps on a 6-DoF SE3 delta; '
                         f'colour affine fitted={fit_colour}; L1 clamped RGB; splats frozen; '
                         'all modes scored at the same resolution; pose_only uses jointly fitted pose when colour enabled',
                fixed={key: summarize('fixed', key) for key in ('psnr', 'ssim', 'lpips')},
                aligned={key: summarize('aligned', key) for key in ('psnr', 'ssim', 'lpips')},
                pose_only={key: summarize('pose_only', key) for key in ('psnr', 'ssim', 'lpips')},
                frames=len(rows))


def floater_fraction(rendered_depth, prior, mask, ratio):
    """Share of valid pixels whose rendered depth sits in front of the Omega surface.

    Ordinary least squares fits rendered_z = scale * Omega_z + shift per frame.
    Only finite, positive depths and positive aligned depths enter the denominator.
    This self-aligned statistic is a proxy, not independent geometry ground truth.
    """
    valid = mask & np.isfinite(rendered_depth) & np.isfinite(prior) & (rendered_depth > 1e-6) & (prior > 0)
    if valid.sum() < 16:
        return None
    render = rendered_depth[valid].astype(np.float64)
    target = prior[valid].astype(np.float64)
    matrix = np.column_stack([target, np.ones_like(target)])
    (scale, shift), _, rank, _ = np.linalg.lstsq(matrix, render, rcond=None)
    if rank < 2 or not np.isfinite(scale + shift) or scale <= 0:
        return None
    aligned = scale * target + shift
    positive = aligned > 1e-6
    if positive.sum() < 16:
        return None
    front = render[positive] < ratio * aligned[positive]
    return dict(fraction=float(front.mean()), pixels=int(positive.sum()), scale=float(scale), shift=float(shift))


def evaluate(splats, data, images, stats, out, args):
    import torchmetrics
    from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    height, width = images.height, images.width
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda().eval()
    prior = None
    omega = Path(args.omega_dir) if args.omega_dir else (out.parent / 'omega_v2')
    if getattr(args, 'depth_windows', None):
        prior = WindowPrior(args.depth_windows, data['frame_ids'], width, height, args.depth_conf_percentile)
        print(json.dumps(dict(event='floater_prior', frames=len(prior.row))), flush=True)
    elif args.omega_dir is not None and (omega / 'depth.npy').is_file():
        prior = OmegaPrior(omega, data['frame_ids'], width, height, args.depth_conf_percentile)
        print(json.dumps(dict(event='floater_prior', frames=len(prior.row))), flush=True)
    rows, preview_done = [], False
    for index in data['test_indices']:
        index = int(index)
        original, frame = images.get(index)
        target = torch.from_numpy(original).cuda().float()[None] / 255.0
        prediction, _, _ = render(splats, data['c2w'][index:index + 1], data['K'][index:index + 1],
                                  width, height, depth=prior is not None)
        rgb = prediction[..., :3].clamp(0, 1)
        x, y = rgb.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
        row = dict(index=index, frame=frame,
                   psnr=float(peak_signal_noise_ratio(x, y, data_range=1.0)),
                   ssim=float(structural_similarity_index_measure(x, y, data_range=1.0)),
                   lpips=float(metric(x, y)))
        metric.reset()
        if prior is not None:
            depth_map, mask = prior(index)
            measured = floater_fraction(prediction[0, ..., 3].cpu().numpy(), depth_map, mask,
                                        args.floater_ratio)
            if measured is not None:
                row.update(floater_fraction=measured['fraction'],
                           floater_scale=measured['scale'], floater_shift=measured['shift'], floater_pixels=measured['pixels'])
        rows.append(row)
        if not preview_done:
            rendered = np.rint(rgb[0].cpu().numpy() * 255).astype(np.uint8)
            canvas = np.concatenate([original, rendered], axis=0)
            cv2.imwrite(str(out / f'preview_frame_{frame:05d}.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            preview_done = True
        print(json.dumps(dict(event='eval', **row)), flush=True)

    sampled = data['train_indices'][::20]
    train_rows = []
    for index in sampled:
        index = int(index)
        original, frame = images.get(index)
        target = torch.from_numpy(original).cuda().float()[None] / 255.0
        prediction, _, _ = render(splats, data['c2w'][index:index + 1], data['K'][index:index + 1],
                                  width, height)
        mse = F.mse_loss(prediction[..., :3].clamp(0, 1), target)
        train_rows.append(dict(frame=frame, psnr=float(-10 * torch.log10(mse))))

    def summary(key):
        values = [row[key] for row in rows]
        return dict(mean=float(np.mean(values)), median=float(np.median(values))) if values else None

    floater = None
    fractions = [row['floater_fraction'] for row in rows if 'floater_fraction' in row]
    if fractions:
        floater = dict(definition=f'pixels with rendered depth < {args.floater_ratio} x aligned Omega depth',
                       frames=len(fractions), mean=float(np.mean(fractions)),
                       median=float(np.median(fractions)), p90=float(np.percentile(fractions, 90)),
                       frames_without_prior=len(rows) - len(fractions))
    depths = []
    for index in data['test_indices']:
        index = int(index)
        depths.append(dict(frame=int(data['frame_ids'][index]),
                           median_depth=median_depth(data['c2w'][index].cpu(), data['K'][index].cpu(),
                                                     data['points'], images.width, images.height)))
    values = np.array([row['median_depth'] for row in depths], dtype=np.float64)
    values = values[np.isfinite(values)]
    global_median = float(np.median(values)) if len(values) else float('nan')
    depth_summary = dict(
        units='normalised reconstruction units (a phone video has no metric scale)',
        per_frame=depths, global_median_depth=global_median,
        closeup_definition='per-frame median optical depth below half the global median depth',
        closeup_frames=int((values < 0.5 * global_median).sum()) if len(values) else 0,
        closeup_fraction=float((values < 0.5 * global_median).mean()) if len(values) else None,
        metric_note='an absolute 0.5 m threshold needs a metric reference (LiDAR or a known-size '
                    'object). With a typical capture distance T, 0.5 m equals 0.5/T of the global '
                    'median depth, so the ratio above converts directly once T is known.')
    aligned = None
    if args.aligned_steps > 0 and len(data['test_indices']):
        aligned = aligned_evaluation(splats, data, images, out, args.aligned_steps, args.aligned_width,
                                     args.aligned_colour)

    train_psnr = [row['psnr'] for row in train_rows]
    result = dict(**stats, group=data['group'],
                  evaluated_test_frames=len(rows), input_frames=len(data['frame_ids']),
                  registered_frames=int(len(data['registered_indices'])),
                  normalization=data['normalization'],
                  rgb_metrics={key: summary(key) for key in ('psnr', 'ssim', 'lpips')},
                  depth=depth_summary, aligned=aligned, floater=floater,
                  training_psnr=dict(frames=len(train_rows),
                                     mean=float(np.mean(train_psnr)) if train_psnr else None,
                                     median=float(np.median(train_psnr)) if train_psnr else None,
                                     selection='train_indices[::20]'),
                  metric_implementation=f'torchmetrics {torchmetrics.__version__}; PSNR/SSIM functional '
                                        f'data_range=1; LPIPS alex normalize=True; full frame, no test-time fitting')
    dump(out / 'per_frame.json', rows)
    dump(out / 'train_psnr_per_frame.json', train_rows)
    dump(out / 'metrics.json', result)
    print(json.dumps(dict(event='complete', **{k: result[k] for k in
          ('evaluated_test_frames', 'rgb_metrics', 'final_gaussians')})), flush=True)
    return result
