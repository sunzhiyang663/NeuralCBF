"""QP-based safe controller with fallback cascade (CBF + CLF)."""

import time
import numpy as np
import torch
import cvxpy as cp


class QPController:
    """Solves a QP at each timestep with three fallback strategies:
      1. Full QP (CLF stability + CBF safety)
      2. Safety-only QP (drop CLF)
      3. Gradient escape (move along -grad_h)
    """

    def __init__(self, cbf_net, alpha_cbf=1.0, gamma_clf=0.1,
                 clf_penalty=1000.0, use_slack=True, escape_gain=2.0):
        self.cbf_net = cbf_net
        self.alpha_cbf = alpha_cbf
        self.gamma_clf = gamma_clf
        self.clf_penalty = clf_penalty
        self.use_slack = use_slack
        self.escape_gain = escape_gain
        self.device = next(cbf_net.parameters()).device

    def get_control(self, x):
        """Compute safe+stable control with fallback cascade."""
        x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)
        x_tensor.requires_grad_(True)
        h = self.cbf_net(x_tensor)
        grad_h = torch.autograd.grad(h, x_tensor,
            grad_outputs=torch.ones_like(h), create_graph=False, retain_graph=False)[0]
        h_val = h.detach().cpu().item()
        grad_h_np = grad_h.detach().cpu().numpy().flatten()

        # Strategy 1: Full QP (CLF + CBF)
        u = self._solve_qp(x, h_val, grad_h_np, use_clf=True)
        if u is not None:
            return u

        # Strategy 2: Safety-only QP (drop CLF)
        u = self._solve_qp(x, h_val, grad_h_np, use_clf=False)
        if u is not None:
            return u

        # Strategy 3: Gradient escape
        grad_norm = np.linalg.norm(grad_h_np)
        if grad_norm > 1e-8:
            return -self.escape_gain * grad_h_np / grad_norm
        return np.zeros(2)

    def _solve_qp(self, x, h_val, grad_h_np, use_clf=True):
        V_val = 0.5 * np.sum(x ** 2)
        grad_V_val = x

        u = cp.Variable(2)
        if self.use_slack:
            delta = cp.Variable(1)
            objective = cp.Minimize(cp.sum_squares(u) + self.clf_penalty * cp.sum_squares(delta))
            constraints = [
                -grad_h_np @ u <= self.alpha_cbf * h_val,
                delta >= 0,
            ]
            if use_clf:
                constraints.insert(0, grad_V_val @ u <= -self.gamma_clf * V_val + delta)
        else:
            objective = cp.Minimize(cp.sum_squares(u))
            constraints = [-grad_h_np @ u <= self.alpha_cbf * h_val]
            if use_clf:
                constraints.insert(0, grad_V_val @ u <= -self.gamma_clf * V_val)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value
        except Exception:
            pass
        return None
