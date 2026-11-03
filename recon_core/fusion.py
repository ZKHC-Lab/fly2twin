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


def partners(poses):
    """Neighbour lists: up to six views per frame, ranked by angle/30 + |distance - 0.3 m|."""
    from scipy.spatial import cKDTree
    centers, rotations = poses[:, :3, 3], poses[:, :3, :3]
    tree, pairs = cKDTree(centers), set()
    for a, center in enumerate(centers):
        scored = []
        for b in tree.query_ball_point(center, NEIGHBOR_RADIUS_M):
            if a == b:
                continue
            angle = np.degrees(np.arccos(np.clip((np.trace(rotations[a].T @ rotations[b])-1)/2, -1, 1)))
            if angle < NEIGHBOR_ANGLE_DEG:
                scored.append((angle / NEIGHBOR_ANGLE_DEG + abs(np.linalg.norm(centers[b]-center)-0.3), b))
        for _, b in sorted(scored)[:MAX_NEIGHBORS]:
            pairs.add((min(a, b), max(a, b)))
    neighbors = [[] for _ in poses]
    for a, b in sorted(pairs):
        neighbors[a].append(b)
        neighbors[b].append(a)
    return neighbors


def voxel_down_sample(points, colors, voxel):
    """Mean point and colour per voxel; grid origin min(points) - voxel/2 as in Open3D.

    Output is in lexicographic voxel order (Open3D uses hash-map order).
    """
    points, colors = np.asarray(points, np.float64), np.asarray(colors, np.float64)
    if not len(points):
        return points.reshape(-1, 3), colors.reshape(-1, 3)
    indices = np.floor((points - (points.min(axis=0) - voxel * .5)) / voxel).astype(np.int64)
    _, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    xyz, rgb = np.zeros((len(counts), 3)), np.zeros((len(counts), 3))
    np.add.at(xyz, inverse, points)
    np.add.at(rgb, inverse, colors)
    return xyz / counts[:, None], rgb / counts[:, None]


def colored_cloud(ids, poses, K, depth, good, image_paths, max_points=1000000):
    """Fused metric cloud of frames ``ids``.

    ``poses`` camera-to-world (N, 4, 4); ``K`` intrinsics at the depth
    resolution (N, 3, 3); ``depth`` metric (N, H, W), may be a read-only mmap;
    ``good`` boolean masks (N, H, W); ``image_paths`` maps str(frame id) to RGB,
    read one frame at a time. ``max_points`` 0 keeps the 2 cm cloud.
    Returns (xyz, rgb in [0, 1], info dict with voxel_m, points_2cm, points).
    """
    import cv2
    if max_points < 0:
        raise ValueError("max_points must be nonnegative (zero disables adaptation)")
    H, W = depth.shape[1:]
    neighbors = partners(poses)
    v, u = np.mgrid[0:H, 0:W]
    points, colors = [], []
    for i, frame in enumerate(ids):
        Ki = K[i]
        ray = np.stack([(u-Ki[0, 2])/Ki[0, 0], (v-Ki[1, 2])/Ki[1, 1], np.ones_like(u)], -1)
        world = (ray * depth[i, ..., None]).reshape(-1, 3) @ poses[i, :3, :3].T + poses[i, :3, 3]
        support, seen = np.zeros(H*W, np.int16), np.zeros(H*W, np.int16)
        for j in neighbors[i]:
            xyz = (world-poses[j, :3, 3]) @ poses[j, :3, :3]
            z = xyz[:, 2]
            front = z > 0.1
            uu, vv = np.full(H*W, -1, int), np.full(H*W, -1, int)
            uu[front] = np.rint(K[j, 0, 0]*xyz[front, 0]/z[front] + K[j, 0, 2]).astype(int)
            vv[front] = np.rint(K[j, 1, 1]*xyz[front, 1]/z[front] + K[j, 1, 2]).astype(int)
            inside = front & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            observed = np.full(H*W, np.nan)
            observed[inside] = depth[j, vv[inside], uu[inside]]
            seen += inside
            support += inside & (np.abs(np.log(np.maximum(observed, 1e-6)/np.maximum(z, 1e-6))) < SUPPORT_LOG_RATIO)
        mask = good[i] & (support.reshape(H, W) >= np.minimum(2, np.maximum(seen.reshape(H, W), 1)))
        mask &= (depth[i] > DEPTH_RANGE_M[0]) & (depth[i] < DEPTH_RANGE_M[1])
        keep = mask[::2, ::2]
        points.append(world.reshape(H, W, 3)[::2, ::2][keep])
        path = image_paths[str(frame)]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Cannot decode RGB: " + str(path))
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        colors.append(image[::2, ::2][keep][:, ::-1] / 255.0)
    if not points:
        return np.empty((0, 3)), np.empty((0, 3)), dict(voxel_m=VOXEL_M, points_2cm=0, points=0)
    base_xyz, base_rgb = voxel_down_sample(np.concatenate(points), np.concatenate(colors), VOXEL_M)
    xyz, rgb, voxel = base_xyz, base_rgb, VOXEL_M
    while max_points and len(xyz) > max_points:
        voxel *= max(1.1, (len(xyz) / max_points) ** (1/3))
        xyz, rgb = voxel_down_sample(base_xyz, base_rgb, voxel)
    return xyz, rgb, dict(voxel_m=voxel, points_2cm=len(base_xyz), points=len(xyz))
