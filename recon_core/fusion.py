"""Per-frame depth mask and multiview-consistent coloured point cloud.

Rules: keep pixels above the frame's confidence median and off depth edges;
reproject each pixel into up to six neighbours (within 1.5 m, < 30 deg); a
neighbour supports it when |log depth ratio| < 0.03; keep pixels supported by
min(2, views that see it), with 0.3 m < depth < 4 m; stride 2; 2 cm voxels,
enlarged until the cloud fits ``max_points``.
"""
import numpy as np

EDGE_REL = 0.03
NEIGHBOR_RADIUS_M = 1.5
NEIGHBOR_ANGLE_DEG = 30.0
MAX_NEIGHBORS = 6
SUPPORT_LOG_RATIO = 0.03
DEPTH_RANGE_M = (0.3, 4.0)
VOXEL_M = 0.02


def confidence_mask(depth, confidence):
    """Confidence above the frame median, positive depth, not on a 3x3 depth edge."""
    from scipy.ndimage import maximum_filter, minimum_filter
    edge = (maximum_filter(depth, 3) - minimum_filter(depth, 3)) / np.maximum(depth, 1e-6) > EDGE_REL
    return (confidence > np.median(confidence)) & (depth > 0) & ~edge


def ranked_partners(a, centers, rotations, tree):
    """Up to six views for frame ``a`` within 1.5 m and 30 deg, by angle/30 + |distance - 0.3 m|.

    Depends only on the frames within 1.5 m of ``a``; ``tree`` is a cKDTree of ``centers``.
    """
    center, scored = centers[a], []
    for b in tree.query_ball_point(center, NEIGHBOR_RADIUS_M):
        if a == b:
            continue
        angle = np.degrees(np.arccos(np.clip((np.trace(rotations[a].T @ rotations[b])-1)/2, -1, 1)))
        if angle < NEIGHBOR_ANGLE_DEG:
            scored.append((angle / NEIGHBOR_ANGLE_DEG + abs(np.linalg.norm(centers[b]-center)-0.3), b))
    return [b for _, b in sorted(scored)[:MAX_NEIGHBORS]]


def symmetric_partners(ranked):
    """Neighbour lists from per-frame rankings: a pair counts if either frame picked the other."""
    pairs = {(min(a, b), max(a, b)) for a, picks in enumerate(ranked) for b in picks}
    neighbors = [[] for _ in ranked]
    for a, b in sorted(pairs):
        neighbors[a].append(b)
        neighbors[b].append(a)
    return neighbors


def partners(poses):
    """Neighbour lists: up to six views per frame, ranked by angle/30 + |distance - 0.3 m|."""
    from scipy.spatial import cKDTree
    centers, rotations = poses[:, :3, 3], poses[:, :3, :3]
    tree = cKDTree(centers)
    return symmetric_partners([ranked_partners(a, centers, rotations, tree) for a in range(len(poses))])


_PACK_BITS = 21  # voxel index per axis below 2**21: 42 km at 2 cm


def voxel_down_sample(points, colors, voxel):
    """Mean point and colour per voxel; grid origin min(points) - voxel/2 as in Open3D.

    Output is in lexicographic voxel order (Open3D uses hash-map order). The three
    indices are packed into one int64 key and summed with bincount in input order,
    which gives the same voxels, order and sums as a row-wise unique with add.at.
    """
    points, colors = np.asarray(points, np.float64), np.asarray(colors, np.float64)
    if not len(points):
        return points.reshape(-1, 3), colors.reshape(-1, 3)
    indices = np.floor((points - (points.min(axis=0) - voxel * .5)) / voxel).astype(np.int64)
    if indices.max() < 1 << _PACK_BITS:
        keys = (indices[:, 0] << 2 * _PACK_BITS) | (indices[:, 1] << _PACK_BITS) | indices[:, 2]
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    else:
        _, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(counts)
    xyz = np.stack([np.bincount(inverse, points[:, k], n) for k in range(3)], 1)
    rgb = np.stack([np.bincount(inverse, colors[:, k], n) for k in range(3)], 1)
    return xyz / counts[:, None], rgb / counts[:, None]


def frame_points(i, poses, K, depth, good, neighbors):
    """Points of frame ``i`` that pass the multiview check, stride 2.

    ``depth`` and ``good`` are indexed by frame (``depth[j]`` is H x W metric depth).
    Returns (xyz float64, keep mask on the stride-2 grid).
    """
    depth_i = depth[i]
    H, W = depth_i.shape
    v, u = np.mgrid[0:H, 0:W]
    Ki = K[i]
    ray = np.stack([(u-Ki[0, 2])/Ki[0, 0], (v-Ki[1, 2])/Ki[1, 1], np.ones_like(u)], -1)
    world = (ray * depth_i[..., None]).reshape(-1, 3) @ poses[i, :3, :3].T + poses[i, :3, 3]
    support, seen = np.zeros(H*W, np.int16), np.zeros(H*W, np.int16)
    for j in neighbors[i]:
        depth_j = depth[j]
        xyz = (world-poses[j, :3, 3]) @ poses[j, :3, :3]
        z = xyz[:, 2]
        front = z > 0.1
        uu, vv = np.full(H*W, -1, int), np.full(H*W, -1, int)
        uu[front] = np.rint(K[j, 0, 0]*xyz[front, 0]/z[front] + K[j, 0, 2]).astype(int)
        vv[front] = np.rint(K[j, 1, 1]*xyz[front, 1]/z[front] + K[j, 1, 2]).astype(int)
        inside = front & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        observed = np.full(H*W, np.nan)
        observed[inside] = depth_j[vv[inside], uu[inside]]
        seen += inside
        support += inside & (np.abs(np.log(np.maximum(observed, 1e-6)/np.maximum(z, 1e-6))) < SUPPORT_LOG_RATIO)
    mask = good[i] & (support.reshape(H, W) >= np.minimum(2, np.maximum(seen.reshape(H, W), 1)))
    mask &= (depth_i > DEPTH_RANGE_M[0]) & (depth_i < DEPTH_RANGE_M[1])
    keep = mask[::2, ::2]
    return world.reshape(H, W, 3)[::2, ::2][keep], keep


def supported_points(i, poses, K, depth, good, neighbors, rows, cols):
    """The multiview check of ``frame_points`` for the pixels (rows, cols) of frame ``i`` only.

    Returns (xyz float64 of the pixels that pass, keep mask over the given pixels): the
    decisions ``frame_points`` takes for those pixels, at a cost proportional to their number.
    """
    depth_i = np.asarray(depth[i])
    H, W = depth_i.shape
    d, n = depth_i[rows, cols], len(rows)
    Ki = K[i]
    ray = np.stack([(cols-Ki[0, 2])/Ki[0, 0], (rows-Ki[1, 2])/Ki[1, 1], np.ones(n)], -1)
    world = (ray * d[:, None]) @ poses[i, :3, :3].T + poses[i, :3, 3]
    support, seen = np.zeros(n, np.int16), np.zeros(n, np.int16)
    for j in neighbors[i]:
        depth_j = depth[j]
        xyz = (world-poses[j, :3, 3]) @ poses[j, :3, :3]
        z = xyz[:, 2]
        front = z > 0.1
        uu, vv = np.full(n, -1, int), np.full(n, -1, int)
        uu[front] = np.rint(K[j, 0, 0]*xyz[front, 0]/z[front] + K[j, 0, 2]).astype(int)
        vv[front] = np.rint(K[j, 1, 1]*xyz[front, 1]/z[front] + K[j, 1, 2]).astype(int)
        inside = front & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        observed = np.full(n, np.nan)
        observed[inside] = depth_j[vv[inside], uu[inside]]
        seen += inside
        support += inside & (np.abs(np.log(np.maximum(observed, 1e-6)/np.maximum(z, 1e-6))) < SUPPORT_LOG_RATIO)
    keep = np.asarray(good[i])[rows, cols] & (support >= np.minimum(2, np.maximum(seen, 1)))
    keep &= (d > DEPTH_RANGE_M[0]) & (d < DEPTH_RANGE_M[1])
    return world[keep], keep


def frame_image(path, H, W):
    """BGR image resized to the depth grid, stride 2 (uint8), for point colours."""
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Cannot decode RGB: " + str(path))
    return cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)[::2, ::2]


def frame_colors(image, keep):
    """RGB in [0, 1] for the kept points of a stride-2 BGR image."""
    return image[keep][:, ::-1] / 255.0


def adaptive_cloud(base_xyz, base_rgb, max_points):
    """Enlarge the voxel from 2 cm until the cloud fits ``max_points`` (0 keeps the 2 cm cloud)."""
    if max_points < 0:
        raise ValueError("max_points must be nonnegative (zero disables adaptation)")
    xyz, rgb, voxel = base_xyz, base_rgb, VOXEL_M
    while max_points and len(xyz) > max_points:
        voxel *= max(1.1, (len(xyz) / max_points) ** (1/3))
        xyz, rgb = voxel_down_sample(base_xyz, base_rgb, voxel)
    return xyz, rgb, dict(voxel_m=voxel, points_2cm=len(base_xyz), points=len(xyz))


def cloud_from_frames(points, colors, max_points):
    """2 cm voxel cloud of per-frame points, enlarged until it fits ``max_points``."""
    if max_points < 0:
        raise ValueError("max_points must be nonnegative (zero disables adaptation)")
    if not points:
        return np.empty((0, 3)), np.empty((0, 3)), dict(voxel_m=VOXEL_M, points_2cm=0, points=0)
    join = lambda arrays: arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
    return adaptive_cloud(*voxel_down_sample(join(points), join(colors), VOXEL_M), max_points)


def colored_cloud(ids, poses, K, depth, good, image_paths, max_points=1000000):
    """Fused metric cloud of frames ``ids``.

    ``poses`` camera-to-world (N, 4, 4); ``K`` intrinsics at the depth
    resolution (N, 3, 3); ``depth`` metric (N, H, W), may be a read-only mmap;
    ``good`` boolean masks (N, H, W); ``image_paths`` maps str(frame id) to RGB,
    read one frame at a time. ``max_points`` 0 keeps the 2 cm cloud.
    Returns (xyz, rgb in [0, 1], info dict with voxel_m, points_2cm, points).
    """
    if max_points < 0:
        raise ValueError("max_points must be nonnegative (zero disables adaptation)")
    H, W = depth.shape[1:]
    neighbors = partners(poses)
    points, colors = [], []
    for i, frame in enumerate(ids):
        xyz, keep = frame_points(i, poses, K, depth, good, neighbors)
        points.append(xyz)
        colors.append(frame_colors(frame_image(image_paths[str(frame)], H, W), keep))
    return cloud_from_frames(points, colors, max_points)
