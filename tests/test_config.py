from pathlib import Path
import pytest
import yaml
from recon.config import validate

BASE = dict(manifest='frames.json', output='out')


@pytest.mark.parametrize(
    'extra,message',
    [
        (dict(colour=1), r'unknown key colour \(the top level allows manifest, video'),
        (dict(train=dict(stepz=10)), r'unknown key train\.stepz \(train allows seed, resolution'),
        (dict(fusion=dict(revist=True)), r'unknown key fusion\.revist \(fusion allows max_points, init_max_points, revisit\)'),
        (
            dict(pose=dict(metric=dict(height_m=1.4))),
            r'unknown key pose\.metric\.height_m \(pose\.metric allows camera_height_m, speed_mps\)',
        ),
        (dict(fusion=True), r'fusion must be a mapping'),
    ],
)
def test_unknown_keys_are_rejected(extra, message):
    with pytest.raises(ValueError, match=message):
        validate(dict(BASE, **extra))


def test_known_keys_pass():
    validate(
        dict(
            BASE,
            dependency_revision='gsplat-1.5.3',
            fusion=dict(revisit=True),
            pose=dict(provider='colmap', metric=dict(camera_height_m=1.4)),
        )
    )


@pytest.mark.parametrize('path', sorted((Path(__file__).parents[1] / 'configs').glob('*.yaml')), ids=lambda p: p.name)
def test_presets_pass(path):
    cfg = yaml.safe_load(path.read_text())
    # drone_with_poses.yaml is merged with a scene's paths; it names no input or output itself.
    validate(dict(BASE, **cfg) if not cfg.get('video') else cfg)
