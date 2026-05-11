#!/usr/bin/env python
"""
Experiment: Converse Loss (Foresight) & Residual Network Effects on Narrow-Gap Safety
======================================================================================

Design: 3 converse modes x 2 training modes = 6 groups
  Converse:  no_converse  /  td_bellman  /  rollout
  Training:  full         /  residual

Scene: narrow_gap (two radius-1.8 obstacles, 0.4-unit gap)
Seeds: 3 per group for statistical significance

Output: results/experiment_gap/
  - {group}/seed{seed}.pth          model checkpoints
  - {group}/heatmap_seed{seed}.png  CBF level set visualization
  - {group}/metrics.json            per-group metrics summary
  - comparison.json                 aggregated comparison
  - comparison_table.txt            human-readable comparison table
"""

import os, sys, time, json, copy, argparse
import numpy as np
import torch

# ---- Import improved_cbf as a module ----
import improved_cbf as cbf

RESULTS_ROOT = "results/experiment_gap"

# ============================================================================
# Experiment Groups
# ============================================================================
GROUPS = [
    # (name,         converse_mode,   residual_mode, lambda_converse)
    ("A_full_noConv",    "none",         False,         0.0),
    ("B_full_TD",        "td_bellman",   False,         0.5),
    ("C_full_Rollout",   "rollout",      False,         0.5),
    ("D_res_noConv",     "none",         True,          0.0),
    ("E_res_TD",         "td_bellman",   True,          0.5),
    ("F_res_Rollout",    "rollout",      True,          0.5),
]


def configure_globals(group, seed):
    """Set improved_cbf globals for a given experiment group."""
    name, conv_mode, residual_mode, lam_conv = group

    cbf.SCENE             = "narrow_gap"
    cbf.CONVERSE_MODE     = "rollout" if conv_mode == "none" else conv_mode
    cbf.LAMBDA_CONVERSE   = lam_conv
    cbf.LAMBDA_BOUNDARY   = 1.0
    cbf.LAMBDA_INVARIANCE = 1.0
    cbf.LAMBDA_SMOOTHNESS = 0.01
    cbf.LAMBDA_ZERO_LS    = 0.5

    # Use fewer epochs for faster iteration
    cbf.EPOCHS            = 1500
    cbf.RESIDUAL_EPOCHS   = 500
    cbf.BATCH_SIZE        = 4096
    cbf.EVAL_INTERVAL     = 500
    cbf.LOG_INTERVAL      = 250

    return name, conv_mode, residual_mode, lam_conv


def compute_metrics(sim_results):
    """Compute safety + stability metrics from simulation results."""
    n = len(sim_results)
    n_col = sum(1 for r in sim_results if len(r['collisions']) > 0)
    min_d = np.min([r['min_obstacle_distance'] for r in sim_results])
    h_mins = np.array([r['h_min'] for r in sim_results])
    final_errs = np.array([r['final_position_error'] for r in sim_results])
    conv = sum(1 for r in sim_results if r['converged'])

    cbf_rates = []
    for r in sim_results:
        sat = r['cbf_satisfaction']
        if len(sat) > 1:
            cbf_rates.append(np.mean(sat[1:].astype(float)))
    mean_cbf = np.mean(cbf_rates) if cbf_rates else 0.0

    # Lyapunov decay rate
    decay_rates = []
    for r in sim_results:
        V = r['V_values']; V = V[V > 1e-10]
        if len(V) > 10:
            slope = np.polyfit(np.arange(len(V)), np.log(V), 1)[0]
            decay_rates.append(-slope)
    lyap = np.mean(decay_rates) if decay_rates else 0.0

    return {
        'collision_rate': n_col / n if n > 0 else 0.0,
        'min_obstacle_distance': float(min_d),
        'h_min': float(np.min(h_mins)),
        'h_min_std': float(np.std(h_mins)),
        'cbf_satisfaction_rate': float(mean_cbf),
        'convergence_rate': conv / n if n > 0 else 0.0,
        'mean_final_error': float(np.mean(final_errs)),
        'lyapunov_decay_rate': float(lyap),
        'n_trajectories': n,
        'n_collisions': n_col,
    }


def run_single(group, seed, pretrained_path=""):
    """Train and evaluate a single (group, seed) configuration."""
    name, conv_mode, residual_mode, lam_conv = group
    configure_globals(group, seed)

    print(f"\n{'='*60}")
    print(f"[{name}] Seed={seed}  converse={conv_mode}  residual={residual_mode}")
    print(f"{'='*60}")

    t0 = time.time()

    if residual_mode:
        if not pretrained_path or not os.path.exists(pretrained_path):
            print(f"  SKIP: pretrained model not found: {pretrained_path}")
            return None
        exp = cbf.CBFExperiment(seed=seed, replay_buffer=None,
                                residual_mode=True, pretrained_path=pretrained_path)
    else:
        exp = cbf.CBFExperiment(seed=seed, replay_buffer=None, residual_mode=False)

    exp.train()
    train_time = time.time() - t0

    sim_results = exp.simulate()
    metrics = compute_metrics(sim_results)
    metrics['training_time_min'] = train_time / 60

    # Save
    group_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(group_dir, exist_ok=True)
    model_path = os.path.join(group_dir, f"seed{seed}.pth")
    exp.save(model_path)

    # Heatmap
    try:
        cbf.plot_heatmap(exp.cbf_net, exp.obstacle_set, sim_results,
                         os.path.join(group_dir, f"heatmap_seed{seed}.png"))
    except Exception as e:
        print(f"  [WARN] heatmap failed: {e}")

    print(f"\n[{name} Seed={seed}] Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return {
        'metrics': metrics,
        'model_path': model_path,
        'group': name,
        'seed': seed,
        'converse_mode': conv_mode,
        'residual_mode': residual_mode,
    }


def run_group(group, seeds):
    """Run all seeds for a group. For residual groups, use pretrained from full group."""
    name, conv_mode, residual_mode, lam_conv = group
    results = []

    # Determine pretrained mapping: D_res_noConv -> A_full_noConv
    pretrained_base = None
    if residual_mode:
        base_name = name.replace("_res_", "_full_")
        pretrained_base = os.path.join(RESULTS_ROOT, base_name)

    for seed in seeds:
        pretrained_path = ""
        if residual_mode and pretrained_base:
            pretrained_path = os.path.join(pretrained_base, f"seed{seed}.pth")

        result = run_single(group, seed, pretrained_path)
        if result:
            results.append(result)

    if not results:
        return None

    # Aggregate across seeds
    metrics_list = [r['metrics'] for r in results]
    keys = ['collision_rate', 'min_obstacle_distance', 'h_min',
            'cbf_satisfaction_rate', 'convergence_rate', 'mean_final_error',
            'lyapunov_decay_rate', 'training_time_min']

    agg = {'group': name, 'converse_mode': conv_mode, 'residual_mode': residual_mode}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        agg[f'{k}_mean'] = float(np.mean(vals))
        agg[f'{k}_std']  = float(np.std(vals))

    # Save group metrics
    group_dir = os.path.join(RESULTS_ROOT, name)
    with open(os.path.join(group_dir, 'metrics.json'), 'w') as f:
        json.dump({'aggregate': agg, 'per_seed': metrics_list}, f, indent=2)

    return agg


def generate_report(all_aggregates):
    """Generate comparison report."""
    if not all_aggregates:
        print("No results to report.")
        return

    # Save comparison JSON
    with open(os.path.join(RESULTS_ROOT, 'comparison.json'), 'w') as f:
        json.dump(all_aggregates, f, indent=2)

    # Generate text table
    lines = []
    lines.append("=" * 110)
    lines.append("Narrow-Gap Experiment: Converse Loss & Residual Network Effects")
    lines.append("=" * 110)
    lines.append("")

    header = (f"{'Group':<20} {'ConvMode':<14} {'Train':<10} "
              f"{'Coll%':>7} {'minDist':>8} {'h_min':>8} "
              f"{'CBFSat%':>8} {'Conv%':>7} {'FinalErr':>9} {'LyapDecay':>10}")
    sep = "-" * len(header)

    lines.append(header)
    lines.append(sep)

    for a in all_aggregates:
        line = (f"{a['group']:<20} {a['converse_mode']:<14} "
                f"{'residual' if a['residual_mode'] else 'full':<10} "
                f"{a['collision_rate_mean']*100:>6.1f}% "
                f"{a['min_obstacle_distance_mean']:>8.3f} "
                f"{a['h_min_mean']:>8.3f} "
                f"{a['cbf_satisfaction_rate_mean']*100:>7.1f}% "
                f"{a['convergence_rate_mean']*100:>6.1f}% "
                f"{a['mean_final_error_mean']:>9.4f} "
                f"{a['lyapunov_decay_rate_mean']:>10.4f}")
        lines.append(line)

    lines.append(sep)
    lines.append("")

    # Factor analysis
    lines.append("=" * 110)
    lines.append("Factor Analysis")
    lines.append("=" * 110)

    # Converse mode comparison (average over training modes)
    for mode in ['none', 'td_bellman', 'rollout']:
        mode_results = [a for a in all_aggregates if a['converse_mode'] == mode]
        if mode_results:
            avg_col = np.mean([a['collision_rate_mean'] for a in mode_results])
            avg_cbf = np.mean([a['cbf_satisfaction_rate_mean'] for a in mode_results])
            avg_err = np.mean([a['mean_final_error_mean'] for a in mode_results])
            lines.append(f"  converse={mode:<14} avg_collision={avg_col*100:.1f}%  "
                         f"avg_cbf_sat={avg_cbf*100:.1f}%  avg_final_err={avg_err:.4f}")

    lines.append("")

    # Training mode comparison
    for train_mode, label in [(False, 'full'), (True, 'residual')]:
        mode_results = [a for a in all_aggregates if a['residual_mode'] == train_mode]
        if mode_results:
            avg_col = np.mean([a['collision_rate_mean'] for a in mode_results])
            avg_cbf = np.mean([a['cbf_satisfaction_rate_mean'] for a in mode_results])
            avg_err = np.mean([a['mean_final_error_mean'] for a in mode_results])
            lines.append(f"  train={label:<10} avg_collision={avg_col*100:.1f}%  "
                         f"avg_cbf_sat={avg_cbf*100:.1f}%  avg_final_err={avg_err:.4f}")

    lines.append("")
    lines.append("=" * 110)

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(RESULTS_ROOT, 'comparison_table.txt'), 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {os.path.join(RESULTS_ROOT, 'comparison_table.txt')}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment: Converse Loss & Residual Network on Narrow Gap")
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 300 epochs full, 100 epochs residual, 1 seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Seeds to run (default: 42 123 456)')
    parser.add_argument('--groups', type=str, nargs='+', default=None,
                       help='Specific groups to run (e.g. A_full_noConv C_full_Rollout)')
    parser.add_argument('--skip-full', action='store_true',
                       help='Skip full training (only run residual, needs existing full models)')
    args = parser.parse_args()

    if args.quick:
        cbf.EPOCHS = 300
        cbf.RESIDUAL_EPOCHS = 100
        cbf.EVAL_INTERVAL = 150
        cbf.LOG_INTERVAL = 100
        seeds = [42]
        print("[QUICK MODE]")
    else:
        seeds = args.seeds

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    cbf.gpu_report()

    # Filter groups if specified
    groups_to_run = GROUPS
    if args.groups:
        groups_to_run = [g for g in GROUPS if g[0] in args.groups]

    print(f"\nExperiment: {len(groups_to_run)} groups x {len(seeds)} seeds = "
          f"{len(groups_to_run) * len(seeds)} runs")
    print(f"Scene: narrow_gap")
    print(f"Groups: {[g[0] for g in groups_to_run]}")
    print(f"Seeds: {seeds}")
    print(f"Results: {os.path.abspath(RESULTS_ROOT)}")
    print()

    all_aggregates = []

    # Run full training groups first
    full_groups = [g for g in groups_to_run if not g[2]]
    residual_groups = [g for g in groups_to_run if g[2]]

    if not args.skip_full:
        for group in full_groups:
            print(f"\n{'#'*60}")
            print(f"# GROUP: {group[0]} (full training)")
            print(f"{'#'*60}")
            agg = run_group(group, seeds)
            if agg:
                all_aggregates.append(agg)

    # Run residual groups
    for group in residual_groups:
        print(f"\n{'#'*60}")
        print(f"# GROUP: {group[0]} (residual training)")
        print(f"{'#'*60}")
        agg = run_group(group, seeds)
        if agg:
            all_aggregates.append(agg)

    # Generate report
    generate_report(all_aggregates)

    print("\nDone!")


if __name__ == "__main__":
    main()
