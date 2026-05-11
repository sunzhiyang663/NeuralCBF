#!/usr/bin/env python
"""
Improved Neural CBF with four safety enhancements:
  1. Discrete-time CBF condition
  2. Exact boundary level-set alignment
  3. Collision replay fine-tuning
  4. QP fallback strategies

Plus residual training (Option B): freeze pretrained CBF + learn correction.
"""

import os, sys, time, copy, argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ---- Core library imports ----
from models import NeuralCBF, NeuralController, ResidualCBF
from dynamics import system_dynamics, ObstacleSet
from loss_functions import ImprovedCBFLoss, sample_data
from qp_controller import QPController

# ============================================================================
# GPU setup
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
if USE_AMP:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

def gpu_report():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} | {gpu.total_memory/1e9:.1f} GB | CUDA {torch.version.cuda}")
        print(f"AMP: ON | TF32: ON | batch={BATCH_SIZE}")
    else:
        print("GPU: N/A, running on CPU")

# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE          = 4096
EPOCHS              = 3000
LR                  = 3e-4
HIDDEN_DIM          = 256
CBF_LAYERS          = 5
CTRL_LAYERS         = 4

# Loss weights
LAMBDA_BOUNDARY     = 1.0
LAMBDA_INVARIANCE   = 1.0
LAMBDA_CONVERSE     = 0.5
LAMBDA_SMOOTHNESS   = 0.01
LAMBDA_ZERO_LS      = 0.5

# QP controller
ALPHA_CBF           = 1.0
GAMMA_CLF           = 0.1
CLF_PENALTY         = 1000.0
USE_SLACK           = True
ESCAPE_GAIN         = 2.0

# CBF condition
EPSILON             = 0.1
ROLLOUT_STEPS       = 10
DT                  = 0.05
CONVERSE_MODE       = "td_bellman"
GAMMA_TD            = 0.99
TAU_TARGET          = 0.005

# Scene
SCENE               = "three_circles"
XLIM, YLIM          = (-4.0, 5.0), (-4.0, 5.0)
BOUNDARY_THICKNESS  = 0.3

# Simulation
SIM_T               = 100.0
SIM_DT              = 0.05
CONVERGENCE_THRESH  = 0.01
STABILITY_WINDOW    = 50

# Collision replay
EVAL_INTERVAL       = 500
REPLAY_BUFFER_SIZE  = 2000
REPLAY_REPLACE_RATIO = 0.3

# Residual training
RESIDUAL_HIDDEN_DIM = 64
RESIDUAL_LAYERS     = 3
RESIDUAL_LR         = 1e-4
RESIDUAL_EPOCHS     = 1000
PRETRAINED_PATH     = ""

# Output
RESULTS_DIR         = "results_improved"
LOG_INTERVAL        = 250
SEEDS               = [42, 123, 456]

# ============================================================================
# Collision Replay Buffer
# ============================================================================
class CollisionReplayBuffer:
    """Stores collision points for targeted retraining (Priority 4)."""
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.points = deque(maxlen=max_size)

    def add(self, pts):
        for p in pts:
            self.points.append(p.copy())

    def sample(self, n):
        if len(self.points) < n:
            return None
        idx = np.random.choice(len(self.points), n, replace=False)
        return np.array([self.points[i] for i in idx])

    def __len__(self):
        return len(self.points)

# ============================================================================
# Experiment Runner
# ============================================================================
class CBFExperiment:
    def __init__(self, seed=42, replay_buffer=None,
                 residual_mode=False, pretrained_path=""):
        self.seed = seed
        self.replay_buffer = replay_buffer
        self.residual_mode = residual_mode
        self._set_seed(seed)

        if SCENE == "narrow_gap":
            self.obstacle_set = ObstacleSet(ObstacleSet.narrow_gap())
        else:
            self.obstacle_set = ObstacleSet(ObstacleSet.three_circles())

        if residual_mode and pretrained_path:
            self._init_residual(pretrained_path)
        else:
            self.cbf_net = NeuralCBF(hidden_dim=HIDDEN_DIM, num_layers=CBF_LAYERS).to(DEVICE)
            self.controller_net = NeuralController(hidden_dim=HIDDEN_DIM,
                                                   num_layers=CTRL_LAYERS).to(DEVICE)

        self.loss_fn = ImprovedCBFLoss(
            self.cbf_net, self.controller_net, self.obstacle_set,
            lambda1=LAMBDA_BOUNDARY, lambda2=LAMBDA_INVARIANCE,
            lambda3=LAMBDA_CONVERSE, lambda4=LAMBDA_SMOOTHNESS,
            lambda_zero=LAMBDA_ZERO_LS,
            alpha=ALPHA_CBF, epsilon=EPSILON,
            rollout_steps=ROLLOUT_STEPS, dt=DT,
            converse_mode=CONVERSE_MODE, gamma=GAMMA_TD, tau=TAU_TARGET,
        )

        trainable_params = (
            list(self.cbf_net.trainable_parameters()) if residual_mode
            else list(self.cbf_net.parameters())
        )
        self.optimizer = optim.AdamW(
            trainable_params + list(self.controller_net.parameters()),
            lr=RESIDUAL_LR if residual_mode else LR, weight_decay=1e-6
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=500, T_mult=2
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
        self.loss_history = []

    def _init_residual(self, pretrained_path):
        """Load pretrained CBF + controller, freeze CBF, add residual."""
        ckpt = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        pretrained_cbf = NeuralCBF(hidden_dim=HIDDEN_DIM, num_layers=CBF_LAYERS).to(DEVICE)
        pretrained_cbf.load_state_dict(ckpt['cbf_net'])
        pretrained_ctrl = NeuralController(hidden_dim=HIDDEN_DIM,
                                           num_layers=CTRL_LAYERS).to(DEVICE)
        pretrained_ctrl.load_state_dict(ckpt['controller_net'])

        self.cbf_net = ResidualCBF(
            pretrained_cbf,
            hidden_dim=RESIDUAL_HIDDEN_DIM,
            num_layers=RESIDUAL_LAYERS,
        ).to(DEVICE)
        self.controller_net = pretrained_ctrl

        n_pretrained = sum(p.numel() for p in pretrained_cbf.parameters())
        n_residual = sum(p.numel() for p in self.cbf_net.residual.parameters())
        print(f"[Residual] Loaded pretrained: {pretrained_path}")
        print(f"  Pretrained CBF: {n_pretrained:,} params (frozen)")
        print(f"  Residual net:   {n_residual:,} params (trainable)")
        print(f"  Controller:     {sum(p.numel() for p in pretrained_ctrl.parameters()):,} params (trainable)")

    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        epochs = RESIDUAL_EPOCHS if self.residual_mode else EPOCHS
        n_params = sum(p.numel() for p in self.cbf_net.parameters())
        n_trainable = sum(p.numel() for p in self.cbf_net.parameters() if p.requires_grad)
        mode_str = "RESIDUAL" if self.residual_mode else "FULL"
        print(f"[Seed {self.seed}] {mode_str} training {epochs} epochs, "
              f"{n_params:,} total params ({n_trainable:,} trainable), "
              f"mode={CONVERSE_MODE}, scene={SCENE}")

        # Timing accumulators (milliseconds)
        t_sample = t_forward = t_backward = t_target = t_eval = 0.0

        for epoch in range(epochs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            x_safe, x_unsafe, x_general, x_boundary_exact = sample_data(
                self.obstacle_set, BATCH_SIZE, XLIM, YLIM,
                BOUNDARY_THICKNESS, self.replay_buffer, REPLAY_REPLACE_RATIO,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                loss, loss_dict = self.loss_fn.total_loss(
                    x_safe, x_unsafe,
                    x_general[:BATCH_SIZE // 2],  # invariance
                    x_general,                      # converse
                    x_general,                      # smoothness
                    x_boundary_exact,               # exact boundary
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.cbf_net.parameters()) + list(self.controller_net.parameters()), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            self.scheduler.step()

            if CONVERSE_MODE == "td_bellman":
                self.loss_fn.update_target()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t4 = time.perf_counter()

            self.loss_history.append(loss_dict)

            # Periodic evaluation
            eval_during = 0.0
            if (epoch + 1) % EVAL_INTERVAL == 0 and self.replay_buffer is not None:
                te0 = time.perf_counter()
                self._collect_collisions()
                eval_during = time.perf_counter() - te0

            # Accumulate timing (ms)
            t_sample   += (t1 - t0) * 1000
            t_forward  += (t2 - t1) * 1000
            t_backward += (t3 - t2) * 1000
            t_target   += (t4 - t3) * 1000
            t_eval     += eval_during * 1000

            if (epoch + 1) % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                parts = [f"E {epoch+1}/{epochs} total={loss_dict['total']:.4f}"]
                for k in ['boundary', 'invariance', 'converse', 'zero_levelset']:
                    parts.append(f"{k}={loss_dict[k]:.4f}")
                print("  " + " | ".join(parts))
                total_per_step = (t_sample + t_forward + t_backward + t_target + t_eval) / n
                print(f"  Timing (ms/step): "
                      f"sample={t_sample/n:.2f} | "
                      f"forward={t_forward/n:.2f} | "
                      f"backward={t_backward/n:.2f} | "
                      f"target+other={t_target/n:.2f} | "
                      f"eval={t_eval/n:.2f} | "
                      f"total={total_per_step:.2f}")
                t_sample = t_forward = t_backward = t_target = t_eval = 0.0

        print(f"[Seed {self.seed}] Training done.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _collect_collisions(self):
        """Run evaluation and add collision points to replay buffer."""
        sim_results = self.simulate(use_fallback=True)
        collision_pts = []
        for r in sim_results:
            for c in r['collisions']:
                collision_pts.append(c['position'])
        if collision_pts:
            self.replay_buffer.add(collision_pts)
            print(f"  [Replay] Collected {len(collision_pts)} collision points, "
                  f"buffer size={len(self.replay_buffer)}")

    def simulate(self, initial_positions=None, use_fallback=True):
        qp = QPController(
            self.cbf_net,
            alpha_cbf=ALPHA_CBF, gamma_clf=GAMMA_CLF,
            clf_penalty=CLF_PENALTY, use_slack=USE_SLACK, escape_gain=ESCAPE_GAIN,
        )
        self.cbf_net.eval()

        if initial_positions is None:
            initial_positions = self._default_starts()

        results = []
        for x0 in initial_positions:
            results.append(self._sim_one(x0, qp))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def _default_starts(self):
        if SCENE == "narrow_gap":
            return [
                np.array([-4, 0]), np.array([-3, 0.5]), np.array([-3, -0.5]),
                np.array([-4, 2]), np.array([-4, -2]),
                np.array([-3, 3]), np.array([-3, -3]),
                np.array([0, 4]), np.array([0, -4]),
                np.array([3, 3]), np.array([3, -3]),
                np.array([3, 0]),
            ]
        return [
            np.array([0, 4]), np.array([4, 0]), np.array([0, -4]), np.array([-4, 0]),
            np.array([3, 3]), np.array([-3, 3]), np.array([3, -3]), np.array([-3, -3]),
            np.array([1.5, 3.5]), np.array([-1.5, -3.5]),
            np.array([3.5, 1.5]), np.array([-3.5, -1.5]),
        ]

    def _sim_one(self, x0, qp):
        steps = int(SIM_T / SIM_DT)
        traj = np.zeros((steps, 2))
        ctrl = np.zeros((steps, 2))
        h_vals = np.zeros(steps)
        V_vals = np.zeros(steps)
        dists = np.zeros(steps)
        cbf_sat = np.zeros(steps, dtype=bool)
        qp_times = np.zeros(steps)

        traj[0] = x0
        converged, conv_step = False, steps

        for i in range(steps - 1):
            t0 = time.perf_counter()
            u = qp.get_control(traj[i])
            qp_times[i] = time.perf_counter() - t0

            ctrl[i] = u if u is not None else np.zeros(2)
            traj[i+1] = traj[i] + ctrl[i] * SIM_DT

            xt = torch.tensor(traj[i+1].reshape(1, -1), dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                h_vals[i+1] = self.cbf_net(xt).cpu().item()
            V_vals[i+1] = 0.5 * np.sum(traj[i+1]**2)
            dists[i+1] = self.obstacle_set.distance_to_unsafe(traj[i+1].reshape(1, -1))[0]

            if i > 0:
                dh = (h_vals[i+1] - h_vals[i]) / SIM_DT
                cbf_sat[i+1] = dh + ALPHA_CBF * h_vals[i] >= 0

            pos = np.linalg.norm(traj[i+1])
            if i >= STABILITY_WINDOW:
                recent = np.linalg.norm(traj[i+1-STABILITY_WINDOW:i+2], axis=1)
                if np.all(recent < CONVERGENCE_THRESH):
                    converged, conv_step = True, i + 1
                    traj = traj[:i+2]; ctrl = ctrl[:i+2]
                    h_vals = h_vals[:i+2]; V_vals = V_vals[:i+2]
                    dists = dists[:i+2]; cbf_sat = cbf_sat[:i+2]
                    qp_times = qp_times[:i+2]
                    break

        collisions = []
        for i, pt in enumerate(traj):
            for oi, obs in enumerate(self.obstacle_set.obstacles):
                if np.linalg.norm(pt - obs['center']) <= obs['radius'] + 1e-6:
                    collisions.append({
                        'step': i, 'obstacle_id': oi+1,
                        'position': pt.copy(),
                        'distance': np.linalg.norm(pt - obs['center']),
                    })

        return {
            'trajectory': traj, 'controls': ctrl,
            'h_values': h_vals, 'V_values': V_vals,
            'distances': dists, 'cbf_satisfaction': cbf_sat,
            'qp_times': qp_times,
            'converged': converged, 'convergence_step': conv_step,
            'collisions': collisions,
            'min_obstacle_distance': float(np.min(dists)) if len(dists) > 0 else np.inf,
            'h_min': float(np.min(h_vals)) if len(h_vals) > 0 else np.inf,
            'final_position_error': float(np.linalg.norm(traj[-1])),
            'initial_position': x0,
        }

    def save(self, path):
        ckpt = {
            'cbf_net': self.cbf_net.state_dict(),
            'controller_net': self.controller_net.state_dict(),
            'loss_history': self.loss_history,
            'seed': self.seed,
            'residual_mode': self.residual_mode,
        }
        if self.residual_mode:
            ckpt['residual_hidden_dim'] = RESIDUAL_HIDDEN_DIM
            ckpt['residual_num_layers'] = RESIDUAL_LAYERS
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.residual_mode = ckpt.get('residual_mode', False)
        if self.residual_mode:
            pretrained_cbf = NeuralCBF(hidden_dim=HIDDEN_DIM, num_layers=CBF_LAYERS).to(DEVICE)
            self.cbf_net = ResidualCBF(
                pretrained_cbf,
                hidden_dim=ckpt.get('residual_hidden_dim', RESIDUAL_HIDDEN_DIM),
                num_layers=ckpt.get('residual_num_layers', RESIDUAL_LAYERS),
            ).to(DEVICE)
        else:
            self.cbf_net = NeuralCBF(hidden_dim=HIDDEN_DIM, num_layers=CBF_LAYERS).to(DEVICE)
        self.cbf_net.load_state_dict(ckpt['cbf_net'])
        self.controller_net.load_state_dict(ckpt['controller_net'])
        self.loss_history = ckpt['loss_history']
        self.seed = ckpt.get('seed', 0)
        print(f"Loaded: {path}")

# ============================================================================
# Evaluation metrics
# ============================================================================
def compute_metrics(sim_results):
    n = len(sim_results)
    n_col = sum(1 for r in sim_results if len(r['collisions']) > 0)
    min_d = np.min([r['min_obstacle_distance'] for r in sim_results])
    h_mins = np.array([r['h_min'] for r in sim_results])
    final_errs = np.array([r['final_position_error'] for r in sim_results])
    conv = sum(1 for r in sim_results if r['converged'])

    decay_rates = []
    for r in sim_results:
        V = r['V_values']; V = V[V > 1e-10]
        if len(V) > 10:
            logV = np.log(V)
            slope = np.polyfit(np.arange(len(V)), logV, 1)[0]
            decay_rates.append(-slope)
    lyap = np.mean(decay_rates) if decay_rates else 0.0

    return {
        'collision_rate': n_col / n,
        'min_obstacle_distance': min_d,
        'h_min': float(np.min(h_mins)),
        'h_min_std': float(np.std(h_mins)),
        'convergence_rate': conv / n,
        'mean_final_error': float(np.mean(final_errs)),
        'lyapunov_decay_rate': lyap,
        'n_trajectories': n, 'n_collisions': n_col,
    }

# ============================================================================
# Visualization
# ============================================================================
def plot_heatmap(cbf_net, obstacle_set, sim_results, save_path, resolution=120):
    """CBF heatmap + zero level set + obstacle boundaries + all trajectories."""
    fig, ax = plt.subplots(figsize=(10, 9))

    xr = np.linspace(XLIM[0], XLIM[1], resolution)
    yr = np.linspace(YLIM[0], YLIM[1], resolution)
    X, Y = np.meshgrid(xr, yr)
    pts = np.stack([X.flatten(), Y.flatten()], 1)
    pts_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        H = cbf_net(pts_t).cpu().numpy().reshape(X.shape)

    clev = ax.contourf(X, Y, H, levels=30, cmap='RdYlGn', alpha=0.5)
    ax.contour(X, Y, H, levels=[0], colors='black', linewidths=2.5, linestyles='-')
    plt.colorbar(clev, ax=ax, label='h(x)', shrink=0.8)

    for obs in obstacle_set.obstacles:
        ax.add_patch(Circle(obs['center'], obs['radius'], fill=True,
                     color='red', alpha=0.25, ec='darkred', lw=2))
        ax.add_patch(Circle(obs['center'], obs['radius'], fill=False,
                     color='white', lw=1.5, ls='--'))

    for i, r in enumerate(sim_results):
        t = r['trajectory']
        col = 'red' if len(r['collisions']) > 0 else 'blue'
        alpha = 0.5 if len(sim_results) > 6 else 0.7
        ax.plot(t[:, 0], t[:, 1], '-', color=col, lw=1.2, alpha=alpha)
        ax.plot(t[0, 0], t[0, 1], 'go', ms=6, alpha=0.8)
        ax.plot(t[-1, 0], t[-1, 1], '*', color=col, ms=8)

    ax.plot(0, 0, 'ks', ms=12, label='Goal', zorder=20)
    ax.legend(handles=[
        Line2D([0], [0], color='blue', lw=2, label='Safe traj'),
        Line2D([0], [0], color='red', lw=2, label='Collision traj'),
        Line2D([0], [0], color='black', lw=2.5, label='h(x)=0 (learned)'),
        Line2D([0], [0], color='white', lw=1.5, ls='--', label='True boundary'),
    ], loc='upper right', fontsize=8)

    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Neural CBF — Heatmap & Trajectories\n{len(sim_results)} trajectories', fontsize=13)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_safety_profiles(sim_results, save_path):
    """Multi-traj safety profiles: h(t) and distance(t)."""
    n = min(4, len(sim_results))
    fig, axes = plt.subplots(n, 2, figsize=(14, 3*n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        r = sim_results[i]
        steps = np.arange(len(r['h_values']))
        col = 'red' if len(r['collisions']) > 0 else 'blue'

        ax = axes[i, 0]
        ax.plot(steps, r['h_values'], color=col, lw=1.5, label='h(x)')
        ax.axhline(y=0, color='k', ls='--', lw=0.8)
        ax.set_ylabel('h(x)'); ax.set_title(f'Traj {i} — CBF value'); ax.grid(alpha=0.3)

        ax = axes[i, 1]
        ax.plot(steps, r['distances'], color=col, lw=1.5, label='dist to unsafe')
        ax.axhline(y=0, color='r', ls='--', lw=0.8, label='Collision boundary')
        ax.set_ylabel('Distance'); ax.set_title(f'Traj {i} — Obstacle distance'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_loss(loss_history, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(loss_history)+1)
    ax.plot(epochs, [l['total'] for l in loss_history], 'k-', lw=2, label='Total')
    for key, color in [('boundary','tab:blue'), ('invariance','tab:orange'),
                        ('converse','tab:green'), ('zero_levelset','tab:purple')]:
        ax.plot(epochs, [l[key] for l in loss_history], color=color, alpha=0.6, lw=1, label=key)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Improved Neural CBF — Safety-Enhanced")
    parser.add_argument('--quick', action='store_true', help='Quick test: 500 epochs, single seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--no-replay', action='store_true', help='Disable collision replay')
    parser.add_argument('--mode', choices=['full','single'], default='full',
                       help='full=multi-seed, single=single best seed')
    parser.add_argument('--residual', action='store_true',
                       help='Residual training: freeze pretrained CBF + learn correction')
    parser.add_argument('--pretrained', type=str, default='',
                       help='Path to pretrained model (required for --residual)')
    parser.add_argument('--scene', choices=['three_circles','narrow_gap'], default='three_circles',
                       help='Obstacle scene: three_circles or narrow_gap')
    args = parser.parse_args()

    global EPOCHS, BATCH_SIZE, EVAL_INTERVAL, RESIDUAL_EPOCHS, SCENE, XLIM, YLIM
    SCENE = args.scene
    if SCENE == "narrow_gap":
        XLIM, YLIM = (-5.0, 5.0), (-5.0, 5.0)
    else:
        XLIM, YLIM = (-4.0, 5.0), (-4.0, 5.0)

    if args.quick:
        EPOCHS = 500
        BATCH_SIZE = 2048
        EVAL_INTERVAL = 200
        RESIDUAL_EPOCHS = 300
        print("[QUICK MODE] 500 epochs, batch=2048")
        seeds_to_run = [42]
    else:
        seeds_to_run = args.seeds if args.seeds else SEEDS

    if args.residual and not args.pretrained:
        parser.error("--pretrained PATH is required when using --residual")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    gpu_report()

    all_metrics_list = []
    best_seed, best_collision = None, 1.0

    for seed in seeds_to_run:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        if args.residual:
            print(f"Mode: RESIDUAL (pretrained={args.pretrained})")
        print(f"{'='*60}")

        replay = None if args.no_replay else CollisionReplayBuffer()
        exp = CBFExperiment(seed=seed, replay_buffer=replay,
                           residual_mode=args.residual,
                           pretrained_path=args.pretrained)

        t0 = time.time()
        exp.train()
        train_time = time.time() - t0

        sim_results = exp.simulate()
        metrics = compute_metrics(sim_results)
        metrics['training_time_min'] = train_time / 60
        all_metrics_list.append(metrics)

        model_path = os.path.join(RESULTS_DIR, f"seed{seed}.pth")
        exp.save(model_path)

        tag = f"seed{seed}"
        plot_heatmap(exp.cbf_net, exp.obstacle_set, sim_results,
                     os.path.join(RESULTS_DIR, f"heatmap_{tag}.png"))
        plot_loss(exp.loss_history, os.path.join(RESULTS_DIR, f"loss_{tag}.png"))
        if len(sim_results) > 0:
            plot_safety_profiles(sim_results, os.path.join(RESULTS_DIR, f"safety_{tag}.png"))

        print(f"\n[Seed {seed}] Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if metrics['collision_rate'] < best_collision:
            best_collision = metrics['collision_rate']
            best_seed = seed

    if len(all_metrics_list) > 1:
        print(f"\n{'='*60}")
        print(f"Multi-Seed Summary ({len(all_metrics_list)} seeds)")
        print(f"{'='*60}")
        keys = ['collision_rate','min_obstacle_distance','h_min','convergence_rate',
                'mean_final_error','lyapunov_decay_rate']
        for k in keys:
            vals = [m[k] for m in all_metrics_list]
            print(f"  {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        print(f"  Best seed: {best_seed} (collision_rate={best_collision:.4f})")

        with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
            json.dump(all_metrics_list, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
