"""
Plot FIM results from rooney_biegler_fim_2exp_verification.json
Creates 2D contour plots for D-optimality, A-optimality, and Pseudo-A-optimality
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment as RooneyBieglerExperimentParmest,
)
from pyomo.contrib.doe import DesignOfExperiments

# Add pyomo root and local prototype directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from pyomo.contrib.doe.utils import compute_FIM_metrics
from rooney_biegler_multiexperiment import RooneyBieglerExperiment as RBExperiment

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the results file from the same directory as this script
with open(script_dir / 'rooney_biegler_fim_2exp_verification_w_o_prior.json', 'r') as f:
    data = json.load(f)

# Convert results to DataFrame
results_df = pd.DataFrame(data['results'])

print(f"Loaded {len(results_df)} results")
print(f"Computing FIM metrics for all results...\n")

# Compute prior FIM from existing data

# Data Setup
data = pd.DataFrame(
    data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
    columns=['hour', 'y'],
)
theta = {'asymptote': 15, 'rate_constant': 0.5}
measurement_error = 0.1

print("Computing prior FIM from existing data...")
FIM_0 = np.zeros((2, 2))
for i in range(len(data)):
    exp_data = data.loc[i, :]
    exp = RooneyBieglerExperimentParmest(
        data=exp_data, theta=theta, measure_error=measurement_error
    )
    doe_obj = DesignOfExperiments(
        experiment_list=exp, objective_option='determinant', prior_FIM=None, tee=False
    )
    FIM_0 += doe_obj.compute_FIM()


def run_doe_optimization(objective_option, prior_FIM, theta, measurement_error):
    """
    Run optimize_experiments for 2 experiments and return the optimal (h1, h2).
    Initial point for both experiments is hour=1 (low) and hour=7 (high) to
    help find the global basin regardless of prior.
    """
    # Use two different starting points
    best_val = None
    best_h = None
    metric_key = {
        'determinant': 'log10 D-opt',
        'trace': 'log10 A-opt',
        'pseudo_trace': 'log10 pseudo A-opt',
    }[objective_option]
    maximise = objective_option in ('determinant', 'pseudo_trace')

    for h1_init, h2_init in [(1.0, 2.0), (5.0, 5.0), (9.0, 9.0)]:
        exp_list = [
            RBExperiment(
                data=pd.Series({'hour': h_init, 'y': 10.0}),
                theta=theta,
                measure_error=measurement_error,
            )
            for h_init in (h1_init, h2_init)
        ]
        try:
            doe = DesignOfExperiments(
                experiment_list=exp_list,
                objective_option=objective_option,
                prior_FIM=prior_FIM,
                tee=False,
                _Cholesky_option=True,
                _only_compute_fim_lower=True,
            )
            doe.optimize_experiments()
            scen = doe.results['Scenarios'][0]
            val = scen.get(metric_key)
            if val is None:
                continue
            is_better = (
                best_val is None
                or (maximise and val > best_val)
                or (not maximise and val < best_val)
            )
            if is_better:
                best_val = val
                best_h = tuple(
                    exp['Experiment Design'][0] for exp in scen['Experiments']
                )
        except Exception as e:
            print(f"  Optimization failed for {objective_option} init=({h1_init},{h2_init}): {e}")
    return best_h, best_val


print("\nRunning optimize_experiments with prior FIM (FIM_0)...")
opt_with_prior = {}
for obj in ('determinant', 'trace', 'pseudo_trace'):
    h, v = run_doe_optimization(obj, FIM_0.copy(), theta, measurement_error)
    opt_with_prior[obj] = h
    print(f"  {obj}: h=({h[0]:.4f}, {h[1]:.4f}), metric={v:.4f}")

# Compute metrics for each result
d_optimality = []
a_optimality = []
pseudo_a_optimality = []

for idx, row in results_df.iterrows():
    # Use stored FIM values (JSON note: stored WITHOUT prior) rather than relying on
    # a `log10_det` flag which may be null. Compute metrics when a numeric FIM is present.
    if row.get('FIM') is not None:
        try:
            FIM = np.asarray(row['FIM'], dtype=float) + FIM_0
        except Exception as e:
            print(f"Failed to construct FIM array at idx={idx}: {e}")
            d_optimality.append(np.nan)
            a_optimality.append(np.nan)
            pseudo_a_optimality.append(np.nan)
            continue

        try:
            (
                det_FIM,
                trace_cov,
                trace_FIM,
                E_vals,
                E_vecs,
                D_opt,
                A_opt,
                pseudo_A_opt,
                E_opt,
                ME_opt,
            ) = compute_FIM_metrics(FIM)

            d_optimality.append(D_opt)
            a_optimality.append(A_opt)
            pseudo_a_optimality.append(pseudo_A_opt)
        except Exception as e:
            print(f"compute_FIM_metrics failed at idx={idx}: {e}")
            try:
                print(f"  FIM (shape): {np.asarray(row['FIM']).shape}")
                print(
                    f"  FIM (sample): {np.asarray(row['FIM'], dtype=float).ravel()[:6]}"
                )
            except Exception:
                print(f"  Could not display FIM for idx={idx}")
            d_optimality.append(np.nan)
            a_optimality.append(np.nan)
            pseudo_a_optimality.append(np.nan)
    else:
        d_optimality.append(np.nan)
        a_optimality.append(np.nan)
        pseudo_a_optimality.append(np.nan)

# Add metrics to dataframe
results_df['D_optimality'] = d_optimality
results_df['A_optimality'] = a_optimality
results_df['pseudo_A_optimality'] = pseudo_a_optimality

# Find optimal values for each metric (guard against empty valid sets)
# D-optimality: maximize
valid_d = results_df[results_df['D_optimality'].notna()].copy()
if not valid_d.empty:
    best_d_idx = valid_d['D_optimality'].idxmax()
    best_d = valid_d.loc[best_d_idx]
else:
    best_d = None

# A-optimality: minimize (trace of covariance)
valid_a = results_df[results_df['A_optimality'].notna()].copy()
if not valid_a.empty:
    best_a_idx = valid_a['A_optimality'].idxmin()
    best_a = valid_a.loc[best_a_idx]
else:
    best_a = None

# Pseudo-A-optimality: maximize (trace of FIM)
valid_pa = results_df[results_df['pseudo_A_optimality'].notna()].copy()
if not valid_pa.empty:
    best_pa_idx = valid_pa['pseudo_A_optimality'].idxmax()
    best_pa = valid_pa.loc[best_pa_idx]
else:
    best_pa = None

print(f"Valid results: {len(valid_d)}")
if best_d is not None:
    print(f"\nBest D-optimality design:")
    print(f"  Hour1: {best_d['hour1']:.4f}, Hour2: {best_d['hour2']:.4f}")
    print(f"  log10(det): {best_d['D_optimality']:.4f}")
else:
    print("\nBest D-optimality design: None (no valid D-optimality results)")

if best_a is not None:
    print(f"\nBest A-optimality design (minimize trace of covariance):")
    print(f"  Hour1: {best_a['hour1']:.4f}, Hour2: {best_a['hour2']:.4f}")
    print(f"  log10(trace(cov)): {best_a['A_optimality']:.4f}")
else:
    print("\nBest A-optimality design: None (no valid A-optimality results)")

if best_pa is not None:
    print(f"\nBest Pseudo-A-optimality design (maximize trace of FIM):")
    print(f"  Hour1: {best_pa['hour1']:.4f}, Hour2: {best_pa['hour2']:.4f}")
    print(f"  log10(trace(FIM)): {best_pa['pseudo_A_optimality']:.4f}")
else:
    print("\nBest Pseudo-A-optimality design: None (no valid pseudo-A results)")


# Create the plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Get unique hour values and create grid
hour1_unique = np.unique(results_df['hour1'])
hour2_unique = np.unique(results_df['hour2'])

print(f"\nGrid size: {len(hour1_unique)} x {len(hour2_unique)}")

# Create meshgrid
H1, H2 = np.meshgrid(hour1_unique, hour2_unique)

# Axis limits: pad the grid range so boundary markers are fully visible
_h_min = min(hour1_unique.min(), hour2_unique.min())
_h_max = max(hour1_unique.max(), hour2_unique.max())
_pad = (_h_max - _h_min) * 0.07
_xlim = (_h_min - _pad, _h_max + _pad)
_ylim = (_h_min - _pad, _h_max + _pad)


# Helper function to create grid for metric
# The stored data only covers combinations_with_replacement (hour1 <= hour2),
# but FIM is additive so FIM(h1,h2) = FIM(h2,h1). Mirror the symmetric half.
def create_metric_grid(df, metric_name):
    Z = np.full(H1.shape, np.nan)
    for idx, row in df.iterrows():
        val = row[metric_name] if not np.isnan(row[metric_name]) else np.nan
        i = np.where(hour2_unique == row['hour2'])[0][0]
        j = np.where(hour1_unique == row['hour1'])[0][0]
        Z[i, j] = val
        # Mirror: swap hour1/hour2 roles (symmetric)
        Z[j, i] = val
    return Z


# 1. D-optimality plot
ax1 = axes[0]
Z_d = create_metric_grid(results_df, 'D_optimality')
contour1 = ax1.contourf(H1, H2, Z_d, levels=20, cmap='viridis')
ax1.contour(H1, H2, Z_d, levels=20, colors='black', alpha=0.3, linewidths=0.5)
ax1.scatter(
    [best_d['hour1']],
    [best_d['hour2']],
    color='red',
    s=600,
    marker='*',
    edgecolors='black',
    linewidths=1.5,
    alpha=1.0,
    label=f'Brute-force best: ({best_d["hour1"]:.2f}, {best_d["hour2"]:.2f})',
    zorder=10,
    clip_on=False,
)
# Optimum from optimize_experiments() with prior FIM
ax1.scatter(
    opt_with_prior['determinant'][0],
    opt_with_prior['determinant'][1],
    color='orange',
    s=400,
    marker='P',
    edgecolors='black',
    linewidths=1.5,
    alpha=1.0,
    label=f'Opt (with prior): ({opt_with_prior["determinant"][0]:.2f}, {opt_with_prior["determinant"][1]:.2f})',
    zorder=9,
    clip_on=False,
)
ax1.set_xlabel('Hour 1', fontsize=12)
ax1.set_ylabel('Hour 2', fontsize=12)
ax1.set_title(
    'D-Optimality (Maximize)\nlog10(det(FIM))', fontsize=13, fontweight='bold'
)
ax1.legend(loc='best', fontsize=9)
ax1.set_xlim(_xlim)
ax1.set_ylim(_ylim)
ax1.grid(True, alpha=0.3)
fig.colorbar(contour1, ax=ax1, label='log10(det(FIM))')

# 2. A-optimality plot
ax2 = axes[1]
Z_a = create_metric_grid(results_df, 'A_optimality')
contour2 = ax2.contourf(H1, H2, Z_a, levels=20, cmap='viridis')
ax2.contour(H1, H2, Z_a, levels=20, colors='black', alpha=0.3, linewidths=0.5)
ax2.scatter(
    [best_a['hour1']],
    [best_a['hour2']],
    color='red',
    s=600,
    marker='*',
    edgecolors='black',
    linewidths=1.5,
    alpha=1.0,
    label=f'Brute-force best: ({best_a["hour1"]:.2f}, {best_a["hour2"]:.2f})',
    zorder=10,
    clip_on=False,
)
ax2.scatter(
    opt_with_prior['trace'][0],
    opt_with_prior['trace'][1],
    color='orange',
    s=400,
    marker='P',
    edgecolors='black',
    linewidths=1.5,
    alpha=1.0,
    label=f'Opt (with prior): ({opt_with_prior["trace"][0]:.2f}, {opt_with_prior["trace"][1]:.2f})',
    zorder=9,
    clip_on=False,
)
ax2.set_xlabel('Hour 1', fontsize=12)
ax2.set_ylabel('Hour 2', fontsize=12)
ax2.set_title(
    'A-Optimality (Minimize)\nlog10(trace(FIM⁻¹))', fontsize=13, fontweight='bold'
)
ax2.legend(loc='best', fontsize=9)
ax2.set_xlim(_xlim)
ax2.set_ylim(_ylim)
ax2.grid(True, alpha=0.3)
fig.colorbar(contour2, ax=ax2, label='log10(trace(cov))')

# 3. Pseudo-A-optimality plot
ax3 = axes[2]
Z_pa = create_metric_grid(results_df, 'pseudo_A_optimality')
contour3 = ax3.contourf(H1, H2, Z_pa, levels=20, cmap='viridis')
ax3.contour(H1, H2, Z_pa, levels=20, colors='black', alpha=0.3, linewidths=0.5)
ax3.scatter(
    [best_pa['hour1']],
    [best_pa['hour2']],
    color='red',
    s=600,
    marker='*',
    edgecolors='black',
    linewidths=1.5,
    alpha=1.0,
    label=f'Brute-force best: ({best_pa["hour1"]:.2f}, {best_pa["hour2"]:.2f})',
    zorder=10,
    clip_on=False,
)
ax3.scatter(
    opt_with_prior['pseudo_trace'][0],
    opt_with_prior['pseudo_trace'][1],
    color='orange',
    s=400,
    marker='P',
    edgecolors='black',
    linewidths=1.5,
    alpha=1.0,
    label=f'Opt (with prior): ({opt_with_prior["pseudo_trace"][0]:.2f}, {opt_with_prior["pseudo_trace"][1]:.2f})',
    zorder=9,
    clip_on=False,
)


ax3.set_xlabel('Hour 1', fontsize=12)
ax3.set_ylabel('Hour 2', fontsize=12)
ax3.set_title(
    'Pseudo-A-Optimality (Maximize)\nlog10(trace(FIM))', fontsize=13, fontweight='bold'
)
ax3.legend(loc='best', fontsize=9)
ax3.set_xlim(_xlim)
ax3.set_ylim(_ylim)
ax3.grid(True, alpha=0.3)
fig.colorbar(contour3, ax=ax3, label='log10(trace(FIM))')

plt.tight_layout()
output_path = script_dir / 'rooney_biegler_fim_metrics_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{output_path}'")
plt.show()
