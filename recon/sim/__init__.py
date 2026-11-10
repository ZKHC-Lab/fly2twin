# SPDX-License-Identifier: Apache-2.0
"""Fly a vision policy inside an exported scene: scene preparation, start/goal sampling, closed-loop evaluation.

`scene` turns a recon export into a closed flight volume (levelled floor, synthetic ceiling where the
capture did not see one) and samples start/goal pairs; `flight` renders policy depth from the point cloud
and runs the DiffDrone-PyTorch3D policy and dynamics, which are loaded from a separate checkout at run time.
See recon/docs/SIM.md.
"""
