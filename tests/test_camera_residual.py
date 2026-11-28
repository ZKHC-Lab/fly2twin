import torch
from recon.gaussian.camera import CameraResidual


def test_camera_residual_identity_and_gradients():
    module = CameraResidual(3).double()
    base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    base[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    result = module(base, [0, 2])
    torch.testing.assert_close(result, base, rtol=0, atol=0)
    (result * torch.arange(32, dtype=torch.float64).reshape(2, 4, 4)).sum().backward()
    assert torch.isfinite(module.values.grad).all()
    assert module.values.grad[[0, 2], :3].abs().sum() > 0
    assert module.values.grad[[0, 2], 3:].abs().sum() > 0
    assert module.values.grad[1].count_nonzero() == 0
    with torch.no_grad():
        module.values.add_(torch.randn_like(module.values) * 0.01)
    rotation = module(base, [0, 2])[:, :3, :3]
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, torch.eye(3, dtype=torch.float64).expand(2, 3, 3))
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2, dtype=torch.float64))
