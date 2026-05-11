"""Visualization for neural CBF experiments — memory-leak fixed, lower-DPI for speed."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Dict, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Core plot helpers
# ============================================================================

def _plot_obstacles(ax, obstacle_set, fill=True, linestyle='--'):
    for obs in obstacle_set.obstacles:
        if fill:
            circle = Circle(obs['center'], obs['radius'], fill=True,
                          color='red', alpha=0.3, edgecolor='red', linewidth=2)
        else:
            circle = Circle(obs['center'], obs['radius'], fill=False,
                          edgecolor='red', linewidth=2, linestyle=linestyle)
        ax.add_patch(circle)


def _plot_goal(ax):
    ax.plot(0, 0, 'ks', markersize=10, label='Goal (Origin)', zorder=10)


# ============================================================================
# Trajectory plots
# ============================================================================

def plot_trajectory(obstacle_set, sim_result, ax=None, title="Safe Trajectory",
                    show_goal=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    _plot_obstacles(ax, obstacle_set)
    traj = sim_result['trajectory']
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
    ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=12, label='End')
    if show_goal:
        _plot_goal(ax)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=9)
    return ax


def plot_trajectory_overlay(obstacle_set, sim_results: List[Dict],
                            ax=None, title="Trajectories Overlay",
                            color_by_safety=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    _plot_obstacles(ax, obstacle_set)
    _plot_goal(ax)

    for i, r in enumerate(sim_results):
        traj = r['trajectory']
        collided = len(r['collisions']) > 0
        color = 'red' if (color_by_safety and collided) else 'blue'
        alpha = 0.4 if len(sim_results) > 5 else 0.7
        label = None
        if i == 0:
            label = 'Safe trajectory' if not collided else 'Unsafe trajectory'
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5,
                alpha=alpha, label=label)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=10)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=9)
    return ax


# ============================================================================
# CBF heatmap
# ============================================================================

def plot_cbf_heatmap(cbf_net, obstacle_set, trajectory=None,
                     xlim=(-5, 5), ylim=(-5, 5), resolution=80,
                     ax=None, title="Neural CBF h(x)"):
    """Plot learned CBF as a heatmap. Resolution lowered for GPU speed."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    x_range = np.linspace(xlim[0], xlim[1], resolution)
    y_range = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)

    cbf_net.eval()
    with torch.no_grad():
        points_t = torch.tensor(points, dtype=torch.float32, device=device)
        H = cbf_net(points_t).cpu().numpy().reshape(X.shape)

    contour = ax.contourf(X, Y, H, levels=20, cmap='RdYlGn', alpha=0.6)
    ax.contour(X, Y, H, levels=[0], colors='black', linewidths=2)
    plt.colorbar(contour, ax=ax, label='h(x)')

    for obs in obstacle_set.obstacles:
        circle = Circle(obs['center'], obs['radius'], fill=False,
                       edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(circle)

    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


# ============================================================================
# Safety & Stability profiles
# ============================================================================

def plot_safety_profile(sim_result, ax=None, title="Safety Profile"):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    steps = np.arange(len(sim_result['h_values']))
    ax.plot(steps, sim_result['h_values'], 'b-', label='h(x)', linewidth=1.5)
    ax.plot(steps, sim_result['distances'], 'g-', label='dist(x, unsafe)', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)

    cbf_sat = sim_result['cbf_satisfaction'].astype(float)
    violated = np.where(cbf_sat < 0.5)[0]
    if len(violated) > 0:
        for v in violated:
            ax.axvline(x=v, color='red', alpha=0.05)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_stability_profile(sim_result, ax=None, title="Stability Profile"):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    steps = np.arange(len(sim_result['V_values']))
    ax.plot(steps, sim_result['V_values'], 'r-', label='V(x)', linewidth=1.5)
    ax.plot(steps, np.linalg.norm(sim_result['trajectory'], axis=1),
            'b-', label='||x||', linewidth=1.5)
    ax.plot(steps[:-1], np.linalg.norm(sim_result['controls'][:-1], axis=1),
            'g-', label='||u||', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ============================================================================
# Training curves
# ============================================================================

def plot_loss_history(loss_history, ax=None, title="Training Loss"):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, [l['total'] for l in loss_history], label='Total', linewidth=2)
    ax.plot(epochs, [l['boundary'] for l in loss_history], label='Boundary', alpha=0.7)
    ax.plot(epochs, [l['invariance'] for l in loss_history], label='Invariance', alpha=0.7)
    ax.plot(epochs, [l['converse'] for l in loss_history], label='Converse', alpha=0.7)
    ax.plot(epochs, [l['regularization'] for l in loss_history], label='Reg/Smooth', alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    return ax


def plot_loss_comparison(loss_histories: Dict[str, List], ax=None,
                         title="Loss Comparison"):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for label, history in loss_histories.items():
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, [l['total'] for l in history], label=label,
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    return ax


# ============================================================================
# Ablation comparison charts
# ============================================================================

def plot_ablation_comparison(metrics_list, metric_key_path, labels=None,
                             ax=None, title="Ablation Comparison", kind='bar'):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    values = []
    for m in metrics_list:
        v = m
        for key in metric_key_path:
            v = getattr(v, key)
        values.append(v)

    if labels is None:
        labels = [m.config_name for m in metrics_list]

    x = np.arange(len(values))
    if kind == 'bar':
        ax.bar(x, values, color='steelblue', edgecolor='white')
    elif kind == 'barh':
        ax.barh(x, values, color='steelblue', edgecolor='white')
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        return ax

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    return ax


def plot_radar_comparison(metrics_list, ax=None, title="Metrics Radar"):
    metric_fns = [
        ('Safety\n(CBF Rate)', lambda m: m.safety.cbf_satisfaction_rate),
        ('Safety\n(1-Collision)', lambda m: 1.0 - m.safety.collision_rate),
        ('h_min\n(normalized)', lambda m: max(0, m.safety.h_min / 2.0)),
        ('Convergence\nRate', lambda m: m.stability.convergence_rate),
        ('Low Final\nError', lambda m: max(0, 1.0 - min(m.stability.mean_final_error, 2.0) / 2.0)),
        ('Lyapunov\nDecay', lambda m: min(m.stability.lyapunov_decay_rate / 0.1, 1.0)),
    ]

    n_metrics = len(metric_fns)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    for i, m in enumerate(metrics_list):
        values = [fn(m) for _, fn in metric_fns]
        values += values[:1]
        ax.plot(angles, values, 'o-', color=colors[i], linewidth=2, label=m.config_name)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name for name, _ in metric_fns], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    return ax


# ============================================================================
# Multi-panel summary figure
# ============================================================================

def plot_experiment_summary(cbf_net, obstacle_set, sim_results, loss_history=None,
                            save_path=None, config_label=""):
    """4-panel summary — closes figure to prevent memory leak."""
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    plot_trajectory_overlay(obstacle_set, sim_results, ax=ax1,
                            title=f"Trajectories - {config_label}")

    ax2 = fig.add_subplot(2, 2, 2)
    plot_cbf_heatmap(cbf_net, obstacle_set,
                     trajectory=sim_results[0]['trajectory'] if sim_results else None,
                     ax=ax2, title=f"CBF Heatmap - {config_label}")

    ax3 = fig.add_subplot(2, 2, 3)
    if sim_results:
        plot_safety_profile(sim_results[0], ax=ax3,
                           title=f"Safety Profile (Trajectory 0)")

    ax4 = fig.add_subplot(2, 2, 4)
    if loss_history:
        plot_loss_history(loss_history, ax=ax4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
