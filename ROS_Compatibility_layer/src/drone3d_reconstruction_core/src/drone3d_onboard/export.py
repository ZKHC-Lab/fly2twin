"""Export a retained frame_pose session without ROS or a running master."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import dump, digest
from recon_core.ply import write_ply
from .session import frame_depth_scale, fuse_session


def export_session(records_root, session_id, destination, capture_profile, max_points=1000000, revisit=True):
    import cv2
    import yaml
    from drone3d_reconstruction_core.continuous_web import _formal_records_for
    from drone3d_reconstruction_core.g4.backend_process import output_root
    if not 4 <= max_points <= 1000000:
        raise ValueError('Delivery fusion max_points must be 4..1000000')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite delivery: ' + str(destination))
    root = output_root(records_root, session_id)
    profile = json.loads((root/'backend_profile.json').read_text())
    offset_ms = profile.get('frame_pose', {}).get('pose_time_offset_ms')
    if offset_ms is None or profile.get('frame_pose', {}).get('pose_time_offset_stage') != 'capture':
        raise ValueError('Delivery requires the pose time offset to be applied at capture')
    capture = yaml.safe_load(Path(capture_profile).read_text())
    records = _formal_records_for(session_id, Path(records_root))
    destination.mkdir(parents=True)
    # manifest is the completion marker; failures never publish a valid package.
    xyz, rgb, fusion, rows, windows = fuse_session(records, destination/'cache', max_points, revisit=revisit)
    (destination/'keyframes').mkdir()
    (destination/'windows').mkdir()
    # One fixed camera: a frame whose CameraInfo was not associated at capture uses the shared one.
    known = [r['camera_info'] for r in rows if r['camera_info'] is not None]
    if not known:
        raise ValueError('Session has no CameraInfo')
    shared = known[0]
    if any(any(c.get(k) != shared.get(k) for k in ('K', 'D', 'width', 'height')) for c in known):
        raise ValueError('CameraInfo changes within the session; a delivery assumes one fixed camera')
    frames, poses, Ks, stamps, blur = [], [], [], [], []
    camera_infos, filled = [], []
    for i, row in enumerate(rows):
        image = cv2.imread(row['image_path'])
        if image is None:
            raise ValueError('Cannot decode full-resolution snapshot RGB')
        if image.shape[:2] != (shared['height'], shared['width']):
            raise ValueError('Snapshot RGB is not at CameraInfo resolution (model thumbnail?)')
        name = 'keyframes/%06d.jpg' % i
        if not cv2.imwrite(str(destination/name), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(name)
        ci = row['camera_info']
        if ci is None:
            ci = shared
            filled.append(row['frame_id'])
        camera_infos.append(ci)
        K = np.array(ci['K'], dtype=float).reshape(3, 3)
        poses.append(row['c2w']); Ks.append(K); stamps.append(row['timestamp_ns'])
        gray = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2GRAY)
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        frames.append(dict(frame_id=i, source_frame_id=row['frame_id'], timestamp_ns=row['timestamp_ns'], path=name))
    ext = windows[0]['source']['body_from_camera']
    for win in windows:
        e = win['source']['body_from_camera']
        if e is None or not np.allclose(e['translation'], ext['translation'], atol=2e-6, rtol=0) or not np.allclose(e['rotation'], ext['rotation'], atol=2e-6, rtol=0):
            raise ValueError('Body-from-camera extrinsic changes within the session')
    np.savez(destination/'poses.npz', frame_ids=np.arange(len(rows)), timestamp_ns=np.array(stamps, np.int64),
        c2w=np.array(poses), K=np.array(Ks), blur=np.array(blur),
        **({'revisit_log_scale': np.array([r['revisit_log_scale'] for r in rows])} if revisit else {}))
    window_list = []
    for win in windows:
        name = 'windows/%06d.npz' % win['window_id']
        frame_scale = [frame_depth_scale(win['alignment'], k) for k in range(len(win['source']['frame_ids']))]
        if revisit:   # owned frames carry the correction, like cache/depth_m.npy; shared frames keep the solved scale
            for row, k in win['selected']:
                frame_scale[k] *= float(np.exp(rows[row]['revisit_log_scale']))
        with np.load(win['source']['prediction_uri'], allow_pickle=False) as f:
            np.savez_compressed(destination/name, depth=f['depth'], confidence=f['depth_conf'],
                depth_scale=win['alignment']['depth_scale'], frame_depth_scale=np.array(frame_scale, float),
                K=win['alignment']['intrinsics'],
                frame_ids=np.array(win['source']['frame_ids']), timestamp_ns=win['source']['timestamps_ns'],
                owned_indices=win['alignment']['merge_frame_indices'])
        window_list.append(dict(window_id=win['window_id'], path=name, depth_scale=win['alignment']['depth_scale']))
    write_ply(destination/'cloud.ply', xyz, rgb)
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    manifest = dict(schema_version=1, session_id=session_id, units='meters', world_up='+Z',
        pose_convention='camera_to_world', pose_source='LIO frame_pose revision', pose_time_offset_ms=offset_ms,
        body_from_camera=ext, resolution=[shared['width'], shared['height']], jpeg_quality=95, camera_info=camera_infos,
        camera_info_shared_for_frames=filled,
        capture_profile=capture, backend_profile=profile, frames=frames, windows=window_list, fusion=fusion,
        pose_time_offset_stage='capture', backend_additional_offset_ms=0,
        pose_offset_policy='Deduplicated dense LIO interpolation at RGB + pose_time_offset_ms; frames without a bracketing pose are not exported',
        hashes={str(p.relative_to(destination)):digest(p) for p in files})
    dump(destination/'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records-root', required=True, type=Path)
    p.add_argument('--session', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--capture-profile', required=True, type=Path)
    p.add_argument('--max-points', type=int, default=1000000)
    p.add_argument('--no-revisit', action='store_true', help='Keep the frame scales of the last revision (skip recon_core.revisit)')
    p.add_argument('--wait-stop-service', help='Optional ROS service: export automatically after this session reaches STOPPED')
    p.add_argument('--wait-timeout-sec', type=float, default=86400.)
    args = p.parse_args()
    if args.wait_stop_service:
        import time
        import rospy
        from drone3d_reconstruction_msgs.srv import ContinuousSessionCommand
        rospy.init_node('onboard_delivery_export', anonymous=True)
        rospy.wait_for_service(args.wait_stop_service, timeout=30)
        service = rospy.ServiceProxy(args.wait_stop_service, ContinuousSessionCommand)
        deadline = time.monotonic() + args.wait_timeout_sec
        while True:
            response = service(command=5)
            if response.error_code or response.session_id != args.session:
                raise RuntimeError('Stopped-session watcher lost its requested session')
            if response.state == 'STOPPED':
                break
            if rospy.is_shutdown() or time.monotonic() >= deadline:
                raise TimeoutError('Session did not reach STOPPED before export deadline')
            time.sleep(1)
    export_session(args.records_root, args.session, args.output, args.capture_profile, args.max_points, not args.no_revisit)


if __name__ == '__main__':
    main()
