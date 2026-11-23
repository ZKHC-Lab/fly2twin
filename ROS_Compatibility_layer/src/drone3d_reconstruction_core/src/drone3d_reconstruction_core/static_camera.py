"""Resolve a rigid body-to-camera transform from a frozen static TF graph."""
import numpy as np
from .odom_local import _extract_transform


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
