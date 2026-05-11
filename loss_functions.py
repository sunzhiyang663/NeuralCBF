"""Improved CBF loss functions with discrete-time invariance and zero-levelset alignment."""

import copy
import numpy as np
import torch
import torch.nn as nn

from models import NeuralCBF, ResidualCBF
from dynamics import system_dynamics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Stratified Data Sampling (with replay buffer + exact boundary support)
# ============================================================================

def sample_data(obstacle_set, n_total=4096, xlim=(-4, 5), ylim=(-4, 5),
                boundary_thickness=0.3, replay_buffer=None, replay_replace_ratio=0.3):
    """Stratified sampling with exact boundary points and collision replay.

    Returns:
        x_safe, x_unsafe, x_general, x_boundary_exact
    """
    n_boundary = int(n_total * 0.25)
    n_exact    = int(n_total * 0.05)
    n_safe_int = int(n_total * 0.25)
    n_unsafe   = int(n_total * 0.25)

    # Replace a fraction of unsafe samples with collision replay points
    if replay_buffer and len(replay_buffer) > 0:
        n_replay = min(int(n_unsafe * replay_replace_ratio), len(replay_buffer))
    else:
        n_replay = 0
    n_unsafe = max(0, n_unsafe - n_replay)
    n_uniform = max(0, n_total - n_boundary - n_exact - n_safe_int - n_unsafe - n_replay)

    # --- exact boundary points ---
    x_boundary_exact = obstacle_set.exact_boundary_points(
        n_per_obs=max(60, n_exact // len(obstacle_set.obstacles)))
    if len(x_boundary_exact) > n_exact:
        idx = np.random.choice(len(x_boundary_exact), n_exact, replace=False)
        x_boundary_exact = x_boundary_exact[idx]

    # --- boundary ring ---
    x_boundary = []
    for obs in obstacle_set.obstacles:
        n = n_boundary // len(obstacle_set.obstacles)
        r = np.random.uniform(obs['radius'], obs['radius'] + boundary_thickness, n)
        th = np.random.uniform(0, 2 * np.pi, n)
        x_boundary.append(np.stack([r * np.cos(th), r * np.sin(th)], 1) + obs['center'])
    x_boundary = np.vstack(x_boundary)[:n_boundary]

    # --- safe interior ---
    x_safe_int = []
    attempts = 0
    while len(x_safe_int) < n_safe_int and attempts < n_safe_int * 20:
        attempts += 1
        n_cand = min(n_safe_int * 2, n_safe_int * 20 - attempts)
        cand = np.random.uniform([xlim[0], ylim[0]], [xlim[1], ylim[1]], (n_cand, 2))
        for pt in cand:
            if len(x_safe_int) >= n_safe_int:
                break
            md = min(np.linalg.norm(pt - o['center']) - o['radius']
                     for o in obstacle_set.obstacles)
            if md > boundary_thickness:
                x_safe_int.append(pt)
    x_safe_int = np.array(x_safe_int[:n_safe_int])

    # --- unsafe interior ---
    x_unsafe = []
    for obs in obstacle_set.obstacles:
        n = n_unsafe // len(obstacle_set.obstacles)
        r = np.sqrt(np.random.uniform(0, 1, n)) * obs['radius']
        th = np.random.uniform(0, 2 * np.pi, n)
        x_unsafe.append(np.stack([r * np.cos(th), r * np.sin(th)], 1) + obs['center'])
    x_unsafe = np.vstack(x_unsafe)[:n_unsafe]

    # --- collision replay ---
    x_replay = np.zeros((0, 2))
    if n_replay > 0 and replay_buffer and len(replay_buffer) >= n_replay:
        x_replay = replay_buffer.sample(n_replay)

    # --- uniform ---
    x_uniform = np.random.uniform([xlim[0], ylim[0]], [xlim[1], ylim[1]],
                                  (max(n_uniform, 0), 2))

    # --- combine ---
    x_safe = np.vstack([x_boundary, x_boundary_exact, x_safe_int])
    x_unsafe_all = np.vstack([x_unsafe, x_replay]) if len(x_replay) > 0 else x_unsafe
    parts = [x for x in [x_boundary, x_boundary_exact, x_safe_int,
                          x_unsafe, x_uniform, x_replay] if len(x) > 0]
    x_general = np.vstack(parts)

    return (
        torch.tensor(x_safe, dtype=torch.float32, device=DEVICE),
        torch.tensor(x_unsafe_all, dtype=torch.float32, device=DEVICE),
        torch.tensor(x_general, dtype=torch.float32, device=DEVICE),
        torch.tensor(x_boundary_exact, dtype=torch.float32, device=DEVICE),
    )


# ============================================================================
# Improved CBF Loss
# ============================================================================

class ImprovedCBFLoss:
    """Loss function with 5 components:
      1. boundary_loss:       h > eps on safe, h < -eps on unsafe
      2. zero_levelset_loss:  h = 0 exactly on obstacle boundaries
      3. invariance_loss:     discrete-time CBF condition
      4. converse_loss:       h matches signed distance (rollout or TD Bellman)
      5. smoothness_loss:     gradient norm regularization
    """

    def __init__(self, cbf_net, controller_net, obstacle_set,
                 lambda1=1.0, lambda2=1.0, lambda3=0.5, lambda4=0.01,
                 lambda_zero=0.5,
                 alpha=1.0, epsilon=0.1,
                 rollout_steps=10, dt=0.05,
                 converse_mode="td_bellman", gamma=0.99, tau=0.005):
        self.cbf_net = cbf_net
        self.controller_net = controller_net
        self.obstacle_set = obstacle_set
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_zero = lambda_zero
        self.alpha = alpha
        self.epsilon = epsilon
        self.rollout_steps = rollout_steps
        self.dt = dt
        self.converse_mode = converse_mode
        self.gamma = gamma
        self.tau = tau

        self.h_target = None
        if converse_mode == "td_bellman":
            self._init_target()

    def _init_target(self):
        if isinstance(self.cbf_net, ResidualCBF):
            self.h_target = ResidualCBF(
                self.cbf_net.pretrained,
                hidden_dim=self.cbf_net.residual_hidden_dim,
                num_layers=self.cbf_net.residual_num_layers,
            ).to(DEVICE)
            self.h_target.load_state_dict(copy.deepcopy(self.cbf_net.state_dict()))
        else:
            c = self.cbf_net
            hd = c.network[0].out_features
            nl = sum(1 for m in c.network if isinstance(m, nn.Linear)) - 1
            self.h_target = NeuralCBF(input_dim=2, hidden_dim=hd, num_layers=nl).to(DEVICE)
            self.h_target.load_state_dict(copy.deepcopy(c.state_dict()))
        for p in self.h_target.parameters():
            p.requires_grad = False

    def update_target(self):
        if self.h_target is None:
            return
        with torch.no_grad():
            for tp, op in zip(self.h_target.parameters(), self.cbf_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(op.data, alpha=self.tau)

    # ---- Loss components ----

    def boundary_loss(self, x_safe, x_unsafe):
        h_s = self.cbf_net(x_safe)
        h_u = self.cbf_net(x_unsafe)
        L_in = torch.mean(torch.relu(self.epsilon - h_s))
        L_out = torch.mean(torch.relu(h_u + self.epsilon))
        return L_in + L_out

    def zero_levelset_loss(self, x_boundary_exact):
        """Enforce h(x) = 0 exactly on obstacle boundaries."""
        h = self.cbf_net(x_boundary_exact)
        return torch.mean(h ** 2)

    def invariance_loss_discrete(self, x):
        """Discrete-time CBF condition.

        h(x_{k+1}) - h(x_k) >= -alpha * h(x_k) * dt
        """
        u = self.controller_net(x)
        x_next = x + system_dynamics(x, u) * self.dt
        h_current = self.cbf_net(x)
        h_next = self.cbf_net(x_next)
        dh_discrete = (h_next - h_current) / self.dt
        return torch.mean(torch.relu(-dh_discrete - self.alpha * h_current))

    def converse_loss_rollout(self, x):
        """N-step rollout + softmin over future distances."""
        traj = [x]
        cur = x.clone()
        for _ in range(self.rollout_steps):
            u = self.controller_net(cur)
            cur = cur + system_dynamics(cur, u) * self.dt
            traj.append(cur)
        dists = torch.stack([self.obstacle_set.distance_to_unsafe(t) for t in traj], dim=1)
        w = torch.softmax(-dists / 1.0, dim=1)
        smooth_min = torch.sum(w * dists, dim=1)
        h = self.cbf_net(x)
        return torch.mean((h - smooth_min) ** 2)

    def converse_loss_td(self, x):
        """1-step TD/Bellman backup: h(x) = min(dist(x), gamma * h_target(x_next))."""
        u = self.controller_net(x)
        x_next = x + system_dynamics(x, u) * self.dt
        with torch.no_grad():
            h_next = self.h_target(x_next)
            dist_x = self.obstacle_set.distance_to_unsafe(x)
            y_target = torch.minimum(dist_x, self.gamma * h_next)
        h = self.cbf_net(x)
        return torch.mean((h - y_target) ** 2)

    def converse_loss(self, x):
        if self.converse_mode == "td_bellman":
            return self.converse_loss_td(x)
        return self.converse_loss_rollout(x)

    def smoothness_loss(self, x):
        x_g = x.clone().requires_grad_(True)
        h = self.cbf_net(x_g)
        gh = torch.autograd.grad(h, x_g, grad_outputs=torch.ones_like(h),
                                 create_graph=True, retain_graph=True)[0]
        L_h = torch.mean(torch.sum(gh ** 2, dim=1))
        x_u = x.clone().requires_grad_(True)
        u = self.controller_net(x_u)
        gu = 0
        for i in range(u.shape[1]):
            gi = torch.autograd.grad(u[:, i].sum(), x_u,
                                     create_graph=True, retain_graph=True)[0]
            gu += torch.mean(torch.sum(gi ** 2, dim=1))
        return L_h + gu

    def total_loss(self, x_safe, x_unsafe, x_inv, x_conv, x_smooth, x_boundary_exact):
        L_boundary = self.boundary_loss(x_safe, x_unsafe)
        L_invariance = self.invariance_loss_discrete(x_inv)
        L_converse = self.converse_loss(x_conv)
        L_smooth = self.smoothness_loss(x_smooth)
        L_zero = self.zero_levelset_loss(x_boundary_exact)

        L_total = (self.lambda1 * L_boundary +
                   self.lambda2 * L_invariance +
                   self.lambda3 * L_converse +
                   self.lambda4 * L_smooth +
                   self.lambda_zero * L_zero)

        return L_total, {
            'boundary': L_boundary.item(),
            'invariance': L_invariance.item(),
            'converse': L_converse.item(),
            'smoothness': L_smooth.item(),
            'zero_levelset': L_zero.item(),
            'total': L_total.item(),
        }
