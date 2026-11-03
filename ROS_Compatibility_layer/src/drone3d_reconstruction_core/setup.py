#!/usr/bin/env python3

from setuptools import setup

setup(
    name="drone3d_reconstruction_core",
    version="0.1.0",
    packages=["drone3d_reconstruction_core", "drone3d_reconstruction_core.g4", "g4_backend", "drone3d_onboard", "recon_core"],
    # src/recon_core is a symlink to the repository-root package shared with the offline pipeline.
    package_dir={"": "src"},
)
