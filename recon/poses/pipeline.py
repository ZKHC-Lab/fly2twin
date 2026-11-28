# SPDX-License-Identifier: Apache-2.0
"""Prior BA on the provider poses: images linked under one name scheme, then XFeat/SIFT and BA."""

from pathlib import Path


def estimate(data, manifest, out, settings):
    images = out / 'images'
    images.mkdir()
    names = []
    for i, row in enumerate(manifest['frames']):
        name = f'{i:08d}{Path(row["path"]).suffix}'
        (images / name).symlink_to(Path(row['path']).resolve())
        names.append(name)
    from .prior import reconstruct

    reconstruct(dict(data, image_names=names), images, out, settings)
