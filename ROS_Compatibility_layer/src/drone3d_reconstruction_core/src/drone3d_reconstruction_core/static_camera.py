"""Resolve a rigid body-to-camera transform from a frozen static TF graph."""
import numpy as np
from .odom_local import _extract_transform


def rotation_quaternion(matrix):
    """Unit quaternion x,y,z,w from a proper rotation, including 180° turns."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-8) or np.linalg.det(r) < 0:
        raise ValueError("expected a proper rotation")
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                  [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
                  [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def resolve_static_camera(entries, body, camera):
    graph = {}
    for entry in entries:
        if not entry.get("static"):
            continue
        rotation, translation = _extract_transform(entry)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        parent, child = entry["parent_frame_id"], entry["child_frame_id"]
        graph.setdefault(parent, []).append((child, transform))
        graph.setdefault(child, []).append((parent, np.linalg.inv(transform)))
    pending, seen = [(body, np.eye(4))], {}
    while pending:
        node, pose = pending.pop(0)
        if node in seen:
            if not np.allclose(seen[node], pose, atol=1e-8, rtol=0):
                raise ValueError("inconsistent static TF graph")
            continue
        seen[node] = pose
        for child, transform in graph.get(node, []):
            pending.append((child, pose @ transform))
    return seen.get(camera)
