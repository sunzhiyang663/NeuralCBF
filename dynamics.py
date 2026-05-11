"""System dynamics, analytical CLF, and obstacle definitions."""

import numpy as np
import torch


def system_dynamics(x, u):
    """Single integrator: dx/dt = u."""
    return u


def analytical_clf(x):
    """V(x) = 0.5 * (x^2 + y^2)."""
    if isinstance(x, torch.Tensor):
        if x.dim() == 1:
            return 0.5 * torch.sum(x ** 2)
        return 0.5 * torch.sum(x ** 2, dim=1, keepdim=True)
    if x.ndim == 1:
        return 0.5 * np.sum(x ** 2)
    return 0.5 * np.sum(x ** 2, axis=1, keepdims=True)


def grad_analytical_clf(x):
    """Gradient of CLF: dV/dx = x."""
    return x


class ObstacleSet:
    """Collection of circular obstacles defining unsafe regions."""

    def __init__(self, obstacles=None):
        self.obstacles = obstacles if obstacles else self.three_circles()

    @staticmethod
    def three_circles():
        return [
            {'center': np.array([0.0, 2.5]), 'radius': 1.0},
            {'center': np.array([-2.0, -2.0]), 'radius': 1.0},
            {'center': np.array([2.0, -2.0]), 'radius': 1.0},
        ]

    @staticmethod
    def narrow_gap():
        """Two large obstacles forming a narrow passage (~0.4 unit wide)."""
        return [
            {'center': np.array([-1.0, 2.0]), 'radius': 1.8},
            {'center': np.array([-1.0, -2.0]), 'radius': 1.8},
        ]

    def distance_to_unsafe(self, x):
        """Minimum signed distance to any obstacle boundary (positive = safe)."""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        min_dist = np.full(x_np.shape[0], np.inf)
        for obs in self.obstacles:
            dist_to_center = np.linalg.norm(x_np - obs['center'], axis=1)
            dist_to_boundary = dist_to_center - obs['radius']
            min_dist = np.minimum(min_dist, dist_to_boundary)

        if isinstance(x, torch.Tensor):
            return torch.tensor(min_dist, dtype=torch.float32, device=x.device).reshape(-1, 1)
        return min_dist

    def is_safe(self, x):
        dist = self.distance_to_unsafe(x)
        if isinstance(dist, torch.Tensor):
            return (dist > 0).squeeze()
        return dist > 0

    def exact_boundary_points(self, n_per_obs=200):
        """Sample points exactly on obstacle boundaries (for zero-levelset loss)."""
        pts = []
        for obs in self.obstacles:
            theta = np.linspace(0, 2 * np.pi, n_per_obs, endpoint=False)
            x = obs['radius'] * np.cos(theta) + obs['center'][0]
            y = obs['radius'] * np.sin(theta) + obs['center'][1]
            pts.append(np.stack([x, y], axis=1))
        return np.vstack(pts)
