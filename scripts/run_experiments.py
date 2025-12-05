"""
Experiment script to generate results to CSV + plots.

Ties together the deterministic (Task 2) and Monte-Carlo (Task 3, Task 5) methods
and produces the data for Task 4 plotting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import set_seed, f_from_points, volume_ball
from src.integration import spherical_grid_integral_equal_m
from src.montecarlo import mc_estimate_with_repeats

D = 3           # change to 2,3,4 for experiments
SAMPLING = 'direct'  # 'direct' or 'rejection'

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_reference_integral(m_ref=80):
    """Compute a deterministic integral used as true value.
    """
    print(f"Computing reference integral with m={m_ref} (this may take a while)...")
    integral_ref, n_evals = spherical_grid_integral_equal_m(f_from_points, m_ref)
    print(f"Reference integral computed. n_evals={n_evals}")
    return integral_ref

def run_all_experiments():
    set_seed(12345)

    # 1) reference
    integral_ref = run_reference_integral(m_ref=80)

    # 2) deterministic: pick budgets mapped to m^3 evaluations
    N_det_list = [1000, 8000, 27000]  # example budgets (will be mapped to m)
    det_results = []
    for N in N_det_list:
        m = max(2, int(round(N ** (1/3))))
        integral_est, n_evals = spherical_grid_integral_equal_m(f_from_points, m)
        det_results.append({
            'N_budget': n_evals,
            'estimate': integral_est,
            'abs_error': abs(integral_est - integral_ref)
        })

    det_df = pd.DataFrame(det_results)
    det_df.to_csv(os.path.join(OUTPUT_DIR, 'deterministic_results.csv'), index=False)

    # 3) Monte-Carlo experiments (standard and symmetrized)
    N_mc = [500, 1000, 2000, 5000]
    mc_std = mc_estimate_with_repeats(f_from_points, N_mc, repeats=10, symmetrize=False, d=D, sampling=SAMPLING)
    mc_sym = mc_estimate_with_repeats(f_from_points, N_mc, repeats=10, symmetrize=True, d=D, sampling=SAMPLING)

    rows = []
    for N in N_mc:
        ests = mc_std[N]['estimates']
        rows.append({
            'method': 'mc_standard',
            'N': N,
            'mean_est': np.mean(ests),
            'std_est': np.std(ests, ddof=1),
            'abs_error_mean': abs(np.mean(ests) - integral_ref)
        })

        ests_s = mc_sym[N]['estimates']
        rows.append({
            'method': 'mc_symmetrized',
            'N': N,
            'mean_est': np.mean(ests_s),
            'std_est': np.std(ests_s, ddof=1),
            'abs_error_mean': abs(np.mean(ests_s) - integral_ref)
        })

    mc_df = pd.DataFrame(rows)
    mc_df.to_csv(os.path.join(OUTPUT_DIR, 'mc_results.csv'), index=False)

    # quick plots
    plt.figure()
    for method in mc_df['method'].unique():
        dfm = mc_df[mc_df['method'] == method]
        plt.errorbar(dfm['N'], dfm['abs_error_mean'], yerr=dfm['std_est'], label=method, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of function evaluations (N)')
    plt.ylabel('Absolute error (vs reference)')
    plt.legend()
    plt.title('Error vs N (MC variants)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mc_error_plot.png'))
    plt.close()

    print('Experiments completed. Outputs in', OUTPUT_DIR)


if __name__ == '__main__':
    run_all_experiments()
