"""Frozen G4 graph: right Sim(3) increments and position-only anchors.

The legacy order is intentional: time-block normalization precedes the Huber
kernel. The later R1 correction is a different method and is not applied here.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from . import sim3
from .types import GraphConfig, GraphResult, METHOD_ID, Sim3


def build_initial_states(edges, window_ids):
    """Return a deterministic spanning-forest initialization and component IDs."""
    ids = list(window_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("window IDs must be nonempty and unique")
    neighbors = {i: [] for i in ids}
    for edge in edges:
        if edge.i not in neighbors or edge.j not in neighbors:
            raise ValueError("edge references an unknown window")
        neighbors[edge.i].append((edge.j, edge.measurement))
        neighbors[edge.j].append((edge.i, sim3.inverse(edge.measurement)))
    states, components, component = {}, {}, 0
    for root in ids:
        if root in states:
            continue
        states[root], components[root], todo = sim3.identity(), component, [root]
        while todo:
            i = todo.pop()
            for j, transform in neighbors[i]:
                if j not in states:
                    states[j], components[j] = sim3.compose(states[i], transform), component
                    todo.append(j)
        component += 1
    return {i: states[i] for i in ids}, {i: components[i] for i in ids}


def robust_block(vector, delta=2.5):
    norm = np.linalg.norm(vector)
    return vector if norm <= delta else vector*np.sqrt(2*delta*norm-delta**2)/norm


def graph_residuals(states, edges, anchors, config=None, details=False):
    """Evaluate the exact legacy objective at supplied local→world states.

    Edge translation is normalized in the source j local units before covariance
    whitening. Position block weight is applied before Huber, as in frozen G4.
    """
    config = config or GraphConfig()
    rows, diagnostic = [], []
    for edge in edges:
        raw = sim3.edge_error(edge.measurement, states[edge.i], states[edge.j])
        raw[:3] /= edge.length_source
        z = np.linalg.solve(np.linalg.cholesky(edge.covariance), raw)
        norm = float(np.linalg.norm(z))
        rows.extend(robust_block(z, config.huber))
        diagnostic.append({"type": "edge", "i": edge.i, "j": edge.j, "norm": norm,
                           "huber_influence_weight": min(1., config.huber/max(norm, 1e-15))})
    for anchor in anchors:
        z = (sim3.apply(states[anchor.window_id], anchor.local_center)-anchor.reference_center)/config.position_sigma_m*anchor.weight
        norm = float(np.linalg.norm(z))
        rows.extend(robust_block(z, config.huber))
        diagnostic.append({"type": "position", "i": anchor.window_id, "j": None, "norm": norm,
                           "huber_influence_weight": min(1., config.huber/max(norm, 1e-15))})
    return diagnostic if details else np.asarray(rows)


def solve_g4(initial, edges, anchors, config=None, method_id=METHOD_ID):
    """Solve the frozen objective. ``method_id`` travels with the result.

    The numbers are identical for every identity; the identity records which
    grouping and input policy produced them, so two runs are never conflated
    merely because they share a class name.
    """
    config = config or GraphConfig()
    ids, edges, anchors = list(initial), list(edges), list(anchors)
    if not ids or any(not isinstance(initial[i], Sim3) for i in ids):
        raise ValueError("initial must map nonempty window IDs to Sim3 states")
    _, comp = build_initial_states(edges, ids)
    if any(a.window_id not in initial for a in anchors):
        raise ValueError("anchor references an unknown window")
    base, fixed, pending, frame_status = dict(initial), [], [], {}
    component_nodes = [[i for i in ids if comp[i] == k] for k in sorted(set(comp.values()))]
    for k, nodes in enumerate(component_nodes):
        aa = [a for a in anchors if a.window_id in nodes]
        if aa:
            x = np.array([sim3.apply(base[a.window_id], a.local_center) for a in aa])
            y = np.array([a.reference_center for a in aa])
            if len(x) >= 3 and sim3.support(x, config.support_ratio)[0] and sim3.support(y, config.support_ratio)[0] and np.sqrt(np.mean(np.sum((y-y.mean(0))**2, axis=1))) >= config.initialization_min_reference_rms_m:
                transform = sim3.fit(x, y)
                frame_status[k] = "localization_anchored"
            else:
                transform = Sim3(1., np.eye(3), np.median(y-x, axis=0))
                pending.append({"component": k, "reason": "initial_location_baseline_weak"})
                frame_status[k] = "pending"
            for j in nodes:
                base[j] = sim3.compose(transform, base[j])
        else:
            fixed.append(nodes[0])
            frame_status[k] = "local_only"
            if anchors:
                pending.append({"component": k, "reason": "no_localization_anchor"})
    variables = [i for i in ids if i not in fixed]
    variable_map = {i: k for k, i in enumerate(variables)}
    whitening = [np.linalg.inv(np.linalg.cholesky(edge.covariance)) for edge in edges]
    begin, best = time.monotonic(), [np.inf, np.zeros(len(variables)*7)]

    def decode(vector):
        states = dict(base)
        for i, k in variable_map.items():
            states[i] = sim3.compose(base[i], sim3.exp(vector[k*7:k*7+7]))
        return states

    def residual(vector, check_time=True):
        if check_time and config.time_limit_seconds is not None and time.monotonic()-begin > config.time_limit_seconds:
            raise TimeoutError("graph_time_limit")
        states, rows = decode(vector), []
        for edge, W in zip(edges, whitening):
            z = sim3.edge_error(edge.measurement, states[edge.i], states[edge.j])
            z[:3] /= edge.length_source
            rows.extend(robust_block(W@z, config.huber))
        for a in anchors:
            z = (sim3.apply(states[a.window_id], a.local_center)-a.reference_center)/config.position_sigma_m*a.weight
            rows.extend(robust_block(z, config.huber))
        rr = np.asarray(rows)
        cost = float(rr@rr)
        if cost < best[0]:
            best[:] = [cost, vector.copy()]
        return rr

    zero = np.zeros(len(variables)*7)
    r0 = residual(zero, check_time=False)
    info = {"method_id": method_id, "initial_cost": float(r0@r0), "fixed_nodes": fixed,
            "components": comp, "pending_components": pending, "global_frame_status": frame_status,
            "component_nodes": component_nodes, "variables": len(zero), "geometry_accepted": False,
            "orientation_factors": 0, "edge_count": len(edges), "position_factors": len(anchors),
            "robust_order": "legacy_block_before_huber"}
    J = None
    try:
        if len(zero) and np.linalg.norm(r0) > 1e-10:
            bound = np.tile([np.inf]*6+[config.logscale_increment_bound], len(variables))
            result = least_squares(residual, zero, method="trf", tr_solver="exact", x_scale="jac",
                                   ftol=config.tolerance, xtol=config.tolerance, gtol=config.tolerance,
                                   max_nfev=config.max_nfev, bounds=(-bound, bound))
            final, J = result.x, result.jac
            info.update(status=int(result.status), message=str(result.message), nfev=int(result.nfev), optimality=float(result.optimality))
        else:
            final = zero
            info.update(status=1, message="zero residual tree or no variables", nfev=0)
            J = np.empty((len(r0), len(zero)))
            for k in range(len(zero)):
                step = np.zeros(len(zero))
                step[k] = 1e-6
                J[:, k] = (residual(zero+step)-residual(zero-step))/2e-6
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(sv > sv[0]*config.jacobian_relative_sv)) if len(sv) and sv[0] > 0 else 0
        info.update(jacobian_rank=rank, jacobian_singular_values=sv.tolist(), weak_directions=len(zero)-rank)
        info["publishable_numerical"] = info["status"] > 0 and info["weak_directions"] == 0 and not pending
    except TimeoutError as error:
        final = best[1]
        info.update(status=-2, message=str(error), publishable_numerical=False)
    component_diagnostics = []
    for k, nodes in enumerate(component_nodes):
        columns = [column for node in nodes if node in variable_map
                   for column in range(variable_map[node]*7, variable_map[node]*7+7)]
        detail = {"component": k, "nodes": nodes, "variable_count": len(columns),
                  "global_frame_status": frame_status[k], "publishable_numerical": False}
        if J is not None:
            singular_values = np.linalg.svd(J[:, columns], compute_uv=False)
            rank = int(np.sum(singular_values > singular_values[0]*config.jacobian_relative_sv)) if len(singular_values) and singular_values[0] > 0 else 0
            detail.update(jacobian_rank=rank, weak_directions=len(columns)-rank,
                          jacobian_singular_values=singular_values.tolist())
            detail["publishable_numerical"] = bool(info["status"] > 0 and rank == len(columns)
                                                    and frame_status[k] == "localization_anchored")
        else:
            detail.update(jacobian_rank=None, weak_directions=None, reason="jacobian_unavailable")
        component_diagnostics.append(detail)
    info["component_diagnostics"] = component_diagnostics
    states = decode(final)
    rr = residual(final, check_time=False)
    info.update(final_cost=float(rr@rr), final_factors=graph_residuals(states, edges, anchors, config, True), seconds=time.monotonic()-begin)
    # Numerical success never grants independent surface or application acceptance.
    return GraphResult(states, info, method_id=method_id)
