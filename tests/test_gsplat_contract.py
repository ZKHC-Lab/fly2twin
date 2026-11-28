"""Guard reset-at-zero and never-reset regressions in public gsplat releases."""

import pytest


def test_no_step_zero_opacity_reset(monkeypatch):
    pytest.importorskip('gsplat')
    import recon.gaussian.strategy as module

    strategy = module.BaselineStrategy()
    calls = []
    monkeypatch.setattr(strategy, '_update_state', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'reset_opa', lambda **kw: calls.append(kw))
    strategy.step_post_backward({}, {}, {}, 0, {})
    assert not calls, 'Baseline must not reset opacity at step zero, regardless of dependency version'


def test_positive_reset_intervals_and_callback_order(monkeypatch):
    torch = pytest.importorskip('torch')
    import recon.gaussian.strategy as module

    strategy = module.BaselineStrategy()
    events = []
    monkeypatch.setattr(strategy, '_update_state', lambda *a, **kw: events.append('update'))
    monkeypatch.setattr(strategy, '_grow_gs', lambda *a, **kw: events.append('grow') or (0, 0))
    monkeypatch.setattr(strategy, '_prune_gs', lambda *a, **kw: events.append('prune') or 0)
    monkeypatch.setattr(module, 'reset_opa', lambda **kw: events.append(('reset', kw['value'])))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    state = {'grad2d': torch.ones(1), 'count': torch.ones(1)}
    strategy.step_post_backward({}, {}, state, 3000, {})
    assert events == ['update', 'grow', 'prune', ('reset', 0.01)]
    assert not state['count'].any() and not state['grad2d'].any()
    events.clear()
    strategy.step_post_backward({}, {}, state, 6000, {})
    assert events == ['update', 'grow', 'prune', ('reset', 0.01)]
    events.clear()
    strategy.step_post_backward({}, {}, state, 15000, {})
    assert events == []


def test_reset_replaces_actual_opacity_and_optimizer_parameter(monkeypatch):
    torch = pytest.importorskip('torch')
    from recon.gaussian.strategy import BaselineStrategy

    strategy = BaselineStrategy()
    monkeypatch.setattr(strategy, '_update_state', lambda *a, **kw: None)
    monkeypatch.setattr(strategy, '_grow_gs', lambda *a, **kw: (0, 0))
    monkeypatch.setattr(strategy, '_prune_gs', lambda *a, **kw: 0)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    params = {'opacities': torch.nn.Parameter(torch.logit(torch.tensor([0.5, 0.02, 0.005])))}
    optimizers = {'opacities': torch.optim.Adam([params['opacities']], lr=0.05)}
    state = {'grad2d': torch.zeros(3), 'count': torch.zeros(3)}
    strategy.step_post_backward(params, optimizers, state, 3000, {})
    torch.testing.assert_close(params['opacities'].sigmoid(), torch.tensor([0.01, 0.01, 0.005]))
    assert optimizers['opacities'].param_groups[0]['params'][0] is params['opacities']
