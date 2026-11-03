"""Overlay a complete, verified G4 revision on immutable window records."""
import hashlib
import json
from pathlib import Path
from .backend_process import BACKEND_G4, output_root


def apply_map_snapshot(records, records_root, session_id):
    selected = [r for r in records if r.get("alignment", {}).get("method") == BACKEND_G4]
    if not selected:
        return records
    root = output_root(records_root, session_id)
    try:
        pointer = json.loads((root / "latest.json").read_text())
    except FileNotFoundError:
        return records
    revision_path = Path(pointer["path"])
    if revision_path.parent.resolve() != (root / "revisions").resolve():
        raise ValueError("G4 revision path is outside this session")
    raw = (revision_path / "revision.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
        raise ValueError("G4 revision digest changed")
    revision = json.loads(raw)
    if revision["session_id"] != session_id or revision["replacement"] != "complete_map":
        raise ValueError("G4 revision session or replacement contract differs")
    display = pointer.get("display", pointer)
    stale = display["sha256"] != pointer["sha256"]
    if stale:
        revision_path = Path(display["path"])
        if revision_path.parent.resolve() != (root / "revisions").resolve():
            raise ValueError("G4 display path is outside this session")
        raw = (revision_path / "revision.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != display["sha256"]:
            raise ValueError("G4 display digest changed")
        revision = json.loads(raw)
        if revision["session_id"] != session_id or revision["replacement"] != "complete_map":
            raise ValueError("G4 display session or replacement contract differs")
    windows = {w["window_id"]: w for w in revision["windows"]}
    if len(windows) != len(revision["windows"]):
        raise ValueError("duplicate G4 revision window")
    # Formal input can advance while both pointers still name the previous solve.
    stale = stale or any(record["window_index"] not in windows for record in selected)
    by_id = {r["window_index"]: r for r in records}
    for index, w in windows.items():
        r = by_id.get(index)
        if r is None or r.get("alignment", {}).get("method") != BACKEND_G4:
            raise ValueError("G4 source is absent from verified formal records")
        for field in ("result_manifest_sha256", "snapshot_manifest_sha256"):
            if w["source"][field] != r[field]:
                raise ValueError("G4 revision source binding differs: " + field)
    result = []
    for record in records:
        index = record["window_index"]
        if index not in windows:
            result.append(record)
            continue
        w = windows[index]
        usable = bool(w["included_in_global_preview"])
        source = w["source"]
        alignment = {**record["alignment"], **w["prediction_to_map"],
            "method": BACKEND_G4, "coordinate_state": "ODOM_LOCAL_APPROX" if usable else "MODEL_LOCAL_UNALIGNED",
            "map_id": session_id + "-g4", "map_revision": revision["revision"],
            "map_revision_sha256": display["sha256"], "map_stale": stale,
            "quality_passed": usable,
            "pending": not usable, "unreliable": not usable,
            "geometry_accepted": False, "reason": w["global_frame_status"],
            "reference_frame_id": revision["reference_frame_id"],
            "merge_frame_indices": list(range(source["shared_prefix"], len(source["frame_ids"]))),
            "gauge": source["gauge"], "confidence_percentile": 50., "edge_relative_threshold": .03,
            "model_frame_count": len(source["frame_ids"]),
            "new_frame_count": len(source["frame_ids"]) - source["shared_prefix"]}
        if w.get("placement") == "frame_pose" and usable:
            # Per-frame localization poses replace the window Sim(3) for points;
            # prediction_to_map above stays the window-level fallback.
            alignment.update(placement="frame_pose", frame_poses=w["frame_poses"],
                             depth_scale=w["depth_scale"], intrinsics=w["intrinsics"])
        result.append({**record, "alignment": alignment})
    return result
