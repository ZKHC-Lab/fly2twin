"""Phone preset's assumed-camera-height scale, not a measured metric reference."""
import numpy as np


def camera_height_scale(c2w, points, camera_height_m):
    """Fit the floor plane and return SfM-to-assumed-metres scale and diagnostics."""
    centers = c2w[:, :3, 3]
    center = np.median(centers, axis=0)
    radius = np.percentile(np.linalg.norm(centers-center, axis=1), 95)
    up = -np.median(c2w[:, :3, 1], axis=0)
    up /= np.linalg.norm(up)
    height = (points-center) @ up
    candidate = points[height < -.1*radius]
    rng = np.random.default_rng(42)
    sample = candidate[rng.choice(len(candidate), min(30000, len(candidate)), replace=False)]
    threshold = .01*radius
    best = None
    for _ in range(3000):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p[1]-p[0], p[2]-p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ up < 0:
            n = -n
        if n @ up < np.cos(np.deg2rad(25)):
            continue
        d = -n @ p[0]
        h = centers @ n+d
        if np.median(h) < .2*radius or np.mean(h > 0) < .95:
            continue
        residual = np.abs(sample @ n+d)
        count = int(np.sum(residual < threshold))
        if best is None or count > best[0]:
            best = (count, n.copy(), float(d))
    if best is None:
        raise ValueError('No floor plane found below the cameras; camera-height scale needs a visible floor')
    _, n, d = best
    for _ in range(3):
        inliers = candidate[np.abs(candidate @ n+d) < threshold]
        origin = inliers.mean(0)
        _, v = np.linalg.eigh((inliers-origin).T @ (inliers-origin))
        n = v[:, 0]
        n = n if n @ up > 0 else -n
        d = float(-n @ origin)
    heights = centers @ n+d
    scale = camera_height_m/float(np.median(heights))
    residual = np.abs(candidate @ n+d)
    support = residual < threshold
    plane = dict(
        assumed_median_camera_height_m=camera_height_m, measured=False,
        sfm_units_to_assumed_meters=scale, plane_normal=n.tolist(), plane_offset_sfm=d,
        median_camera_height_sfm=float(np.median(heights)),
        camera_height_assumed_m_p10_p50_p90=np.percentile(heights*scale, [10, 50, 90]).tolist(),
        reference_radius_sfm=float(radius), camera_up_prior=up.tolist(),
        candidate_points=len(candidate), inliers=int(support.sum()),
        candidate_inlier_fraction=float(support.mean()), threshold_sfm=float(threshold),
        threshold_assumed_m=float(threshold*scale),
        inlier_residual_p95_assumed_m=float(np.percentile(residual[support]*scale, 95)),
        method='RANSAC3000, seed42; candidates >0.1 trajectory P95 radius below median camera along median -camera_y; normal within25deg of that up; >=95% cameras above; plane height>0.2radius; max support within0.01radius; three PCA refits')
    return scale, plane
