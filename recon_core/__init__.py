"""Shared numerical core of the onboard (ROS, Python 3.8) and offline (Python 3.10) pipelines.

NumPy/SciPy only; OpenCV is imported lazily where RGB is read. Conventions:
poses are camera-to-world 4x4 (OpenCV optical axes: right, down, forward),
world coordinates follow the pose provider; metric depth is z along the optical axis.
"""
