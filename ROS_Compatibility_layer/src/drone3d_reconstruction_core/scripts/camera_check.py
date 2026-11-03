#!/usr/bin/env python3
"""Bench check of the RealSense color stream before a recording flight.

    python3 camera_check.py [--seconds 5]

Prints size, measured frame rate, mean brightness (0-255), sharpness and,
when /camera/color/metadata is published, the actual exposure and gain.
Sharpness is the variance of the 4-neighbour Laplacian on the grey image
downsampled to <= 640 px wide (matches cv2.Laplacian on testflight3 frames,
correlation 1.000). testflight3 keyframes: brightness median 123 (P10-P90
97-148), sharpness median 110; blurred ones 31, well-matched ones 174.
Sensor noise also raises the Laplacian variance (bench, similar brightness:
gain 64 -> 164, gain 128 -> 406), so compare sharpness only at one gain: hold
the camera still, then turn it at flight speed.
"""
import argparse
import json
import time

import numpy as np
import rospy
from sensor_msgs.msg import Image

CHANNELS = {"rgb8": 3, "bgr8": 3, "mono8": 1}


def grey(msg):
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    a = a[:, :msg.width * CHANNELS[msg.encoding]].reshape(msg.height, msg.width, -1)
    g = a.astype(np.float32).mean(axis=2)
    while g.shape[1] > 640:
        h, w = g.shape[0] // 2 * 2, g.shape[1] // 2 * 2
        g = g[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return g


def sharpness(g):
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    rospy.init_node("camera_check", anonymous=True)
    frames, meta = [], []
    rospy.Subscriber("/camera/color/image_raw", Image,
                     lambda m: frames.append((time.monotonic(), m)), queue_size=5)
    try:
        from realsense2_camera.msg import Metadata
        rospy.Subscriber("/camera/color/metadata", Metadata,
                         lambda m: meta.append(json.loads(m.json_data)), queue_size=5)
    except ImportError:
        pass
    time.sleep(args.seconds)
    if len(frames) < 2:
        raise SystemExit("no frames on /camera/color/image_raw")
    msg = frames[-1][1]
    fps = (len(frames) - 1) / (frames[-1][0] - frames[0][0])
    stats = [(float(g.mean()), sharpness(g)) for g in (grey(m) for _, m in frames[::max(1, len(frames) // 30)])]
    bright, sharp = np.array(stats).T
    print(f"size {msg.width}x{msg.height} {msg.encoding}, {fps:.1f} fps over {len(frames)} frames")
    print(f"brightness median {np.median(bright):.0f} (testflight3: 123)")
    print(f"sharpness median {np.median(sharp):.0f}, min {sharp.min():.0f} (testflight3: 110)")
    if meta:
        exp = [m.get("actual_exposure") for m in meta if "actual_exposure" in m]
        gain = [m.get("gain_level") for m in meta if "gain_level" in m]
        if exp:
            print(f"actual_exposure (raw metadata value) median {np.median(exp):.0f}")
        if gain:
            print(f"gain median {np.median(gain):.0f}")
    else:
        print("no /camera/color/metadata (exposure not reported)")


if __name__ == "__main__":
    main()
