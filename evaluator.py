"""Quantitative evaluation metrics for neural CBF experiments."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SafetyMetrics:
    collision_rate: float = 0.0
    min_obstacle_distance: float = np.inf
    mean_obstacle_distance: float = 0.0
    h_min: float = np.inf
    h_mean: float = 0.0
    cbf_satisfaction_rate: float = 0.0
    n_trajectories: int = 0
    n_collisions: int = 0


@dataclass
class StabilityMetrics:
    convergence_rate: float = 0.0
    mean_convergence_time: float = np.inf
    mean_final_error: float = np.inf
    lyapunov_decay_rate: float = 0.0
    clf_slack_frequency: float = 0.0
    mean_control_effort: float = 0.0


@dataclass
class EfficiencyMetrics:
    mean_qp_time_ms: float = 0.0
    training_time_seconds: float = 0.0
    epochs_to_converge: int = 0


@dataclass
class ExperimentMetrics:
    config_name: str = ""
    safety: SafetyMetrics = field(default_factory=SafetyMetrics)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    raw_results: List[Dict] = field(default_factory=list)


# ============================================================================
# Metric Computation
# ============================================================================

def compute_safety_metrics(sim_results: List[Dict]) -> SafetyMetrics:
    """Compute safety metrics across all simulated trajectories."""
    n = len(sim_results)
    n_collisions = sum(1 for r in sim_results if len(r['collisions']) > 0)
    min_dists = [r['min_obstacle_distance'] for r in sim_results]
    mean_dists = [np.mean(r['distances']) for r in sim_results]
    h_mins = [r['h_min'] for r in sim_results]
    h_means = [np.mean(r['h_values']) for r in sim_results]
    cbf_rates = [np.mean(r['cbf_satisfaction'].astype(float)) for r in sim_results]

    return SafetyMetrics(
        collision_rate=n_collisions / n if n > 0 else 0.0,
        min_obstacle_distance=np.min(min_dists) if min_dists else np.inf,
        mean_obstacle_distance=np.mean(mean_dists) if mean_dists else 0.0,
        h_min=np.min(h_mins) if h_mins else np.inf,
        h_mean=np.mean(h_means) if h_means else 0.0,
        cbf_satisfaction_rate=np.mean(cbf_rates) if cbf_rates else 0.0,
        n_trajectories=n,
        n_collisions=n_collisions,
    )


def compute_stability_metrics(sim_results: List[Dict]) -> StabilityMetrics:
    """Compute stability metrics across all simulated trajectories."""
    n = len(sim_results)
    converged = [r for r in sim_results if r['converged']]
    conv_times = [r['convergence_step'] for r in converged]
    final_errors = [r['final_position_error'] for r in sim_results]
    control_efforts = [np.mean(np.linalg.norm(r['controls'], axis=1)) for r in sim_results]

    # Lyapunov decay rate: linear fit to log V(t)
    decay_rates = []
    for r in sim_results:
        V = r['V_values']
        V = V[V > 1e-10]
        if len(V) > 10:
            t = np.arange(len(V))
            logV = np.log(V)
            slope = np.polyfit(t, logV, 1)[0]
            decay_rates.append(-slope)  # positive = decreasing
    mean_decay = np.mean(decay_rates) if decay_rates else 0.0

    # Slack frequency
    slack_freqs = []
    for r in sim_results:
        slacks = r.get('slack_values', np.zeros(1))
        if len(slacks) > 0:
            slack_freqs.append(np.mean(slacks > 0.01))
    mean_slack_freq = np.mean(slack_freqs) if slack_freqs else 0.0

    return StabilityMetrics(
        convergence_rate=len(converged) / n if n > 0 else 0.0,
        mean_convergence_time=np.mean(conv_times) if conv_times else np.inf,
        mean_final_error=np.mean(final_errors) if final_errors else np.inf,
        lyapunov_decay_rate=mean_decay,
        clf_slack_frequency=mean_slack_freq,
        mean_control_effort=np.mean(control_efforts) if control_efforts else 0.0,
    )


def compute_all_metrics(sim_results: List[Dict], config_name: str = "",
                        training_time: float = 0.0) -> ExperimentMetrics:
    """Compute all metrics for a set of simulation results."""
    return ExperimentMetrics(
        config_name=config_name,
        safety=compute_safety_metrics(sim_results),
        stability=compute_stability_metrics(sim_results),
        efficiency=EfficiencyMetrics(
            mean_qp_time_ms=np.mean([r.get('qp_times', [0]) for r in sim_results]) * 1000,
            training_time_seconds=training_time,
        ),
        raw_results=sim_results,
    )


# ============================================================================
# Comparison
# ============================================================================

def compare_experiments(metrics_list: List[ExperimentMetrics]) -> str:
    """Generate a comparison table in markdown format."""
    lines = [
        "| Metric | " + " | ".join(m.config_name for m in metrics_list) + " |",
        "|" + "|".join(" --- " for _ in range(len(metrics_list) + 1)) + "|",
    ]

    rows = [
        ("Collision Rate", lambda m: f"{m.safety.collision_rate:.3f}"),
        ("Min Obstacle Dist", lambda m: f"{m.safety.min_obstacle_distance:.3f}"),
        ("Mean Obstacle Dist", lambda m: f"{m.safety.mean_obstacle_distance:.3f}"),
        ("h_min", lambda m: f"{m.safety.h_min:.3f}"),
        ("h_mean", lambda m: f"{m.safety.h_mean:.3f}"),
        ("CBF Satisfaction Rate", lambda m: f"{m.safety.cbf_satisfaction_rate:.3f}"),
        ("Convergence Rate", lambda m: f"{m.stability.convergence_rate:.3f}"),
        ("Mean Conv. Time (steps)", lambda m: f"{m.stability.mean_convergence_time:.1f}"),
        ("Mean Final Error", lambda m: f"{m.stability.mean_final_error:.4f}"),
        ("Lyapunov Decay Rate", lambda m: f"{m.stability.lyapunov_decay_rate:.4f}"),
        ("CLF Slack Freq", lambda m: f"{m.stability.clf_slack_frequency:.3f}"),
        ("Mean Control Effort", lambda m: f"{m.stability.mean_control_effort:.3f}"),
        ("Mean QP Time (ms)", lambda m: f"{m.efficiency.mean_qp_time_ms:.3f}"),
    ]

    for label, fn in rows:
        vals = " | ".join(fn(m) for m in metrics_list)
        lines.append(f"| {label} | {vals} |")

    return "\n".join(lines)


def aggregate_across_seeds(metrics_list: List[ExperimentMetrics]) -> Dict:
    """Aggregate metrics across multiple seeds (mean +/- std)."""
    def _agg(values):
        arr = np.array(values)
        return {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}

    conv_times = [m.stability.mean_convergence_time for m in metrics_list
                  if m.stability.mean_convergence_time != np.inf]
    conv_time_result = _agg(conv_times) if conv_times else {'mean': float('nan'), 'std': float('nan')}

    return {
        'collision_rate': _agg([m.safety.collision_rate for m in metrics_list]),
        'min_obstacle_distance': _agg([m.safety.min_obstacle_distance for m in metrics_list]),
        'mean_obstacle_distance': _agg([m.safety.mean_obstacle_distance for m in metrics_list]),
        'h_min': _agg([m.safety.h_min for m in metrics_list]),
        'cbf_satisfaction_rate': _agg([m.safety.cbf_satisfaction_rate for m in metrics_list]),
        'convergence_rate': _agg([m.stability.convergence_rate for m in metrics_list]),
        'mean_convergence_time': conv_time_result,
        'mean_final_error': _agg([m.stability.mean_final_error for m in metrics_list]),
        'lyapunov_decay_rate': _agg([m.stability.lyapunov_decay_rate for m in metrics_list]),
    }
