#scripts/run_experiments.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import set_seed, f_from_points, volume_ball, f_radial_norm2, analytic_integral_radial_norm2
from src.integration import spherical_grid_integral_equal_m
from src.montecarlo import mc_estimate_with_repeats

try:
    from src.integration import spherical_grid_integral_equal_m
    HAVE_3D_DET = True
except Exception:
    HAVE_3D_DET = False
    
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_mc_for_config(f_points, d, sampling, N_mc, repeats=10, symmetrize=True):
    """
    Run MC experiments for the given integrand f_points (callable),
    dimension d, sampling mode, Ns in N_mc, and repeats.
    Returns a DataFrame of results.
    """
    out_rows = []
    # Standard MC
    mc_std = mc_estimate_with_repeats(f_points, N_mc, repeats=repeats, symmetrize=False, d=d, sampling=sampling)
    # Symmetrized MC
    mc_sym = mc_estimate_with_repeats(f_points, N_mc, repeats=repeats, symmetrize=True, d=d, sampling=sampling)

    for N in N_mc:
        ests = mc_std[N]['estimates']
        n_evals_std = mc_std[N]['n_evals']
        rows_std = {
            'method': 'mc_standard',
            'd': d,
            'sampling': sampling,
            'N': N,
            'n_evals': n_evals_std,
            'mean_est': np.mean(ests),
            'std_est': np.std(ests, ddof=1),
            'abs_error_mean': None  # fill later once reference known
        }
        out_rows.append(rows_std)

        ests_s = mc_sym[N]['estimates']
        n_evals_sym = mc_sym[N]['n_evals']
        rows_sym = {
            'method': 'mc_symmetrized',
            'd': d,
            'sampling': sampling,
            'N': N,
            'n_evals': n_evals_sym,
            'mean_est': np.mean(ests_s),
            'std_est': np.std(ests_s, ddof=1),
            'abs_error_mean': None
        }
        out_rows.append(rows_sym)

    df = pd.DataFrame(out_rows)
    return df

def run_reference_integral(m_ref=80):
    """Compute a deterministic integral used as true value.
    """
    print(f"Computing reference integral with m={m_ref} (this may take a while)...")
    integral_ref, n_evals = spherical_grid_integral_equal_m(f_from_points, m_ref)
    print(f"Reference integral computed. n_evals={n_evals}")
    return integral_ref

def run_all_experiments(d_list=[2,3,4], sampling_modes=['direct','rejection']):
    set_seed(12345)

    N_mc = [500, 1000, 2000, 5000]
    repeats = 10

    summary_rows = []

    for d in d_list:
        # default: use radial integrand and analytic reference
        f_points = f_radial_norm2
        integral_ref = analytic_integral_radial_norm2(d)
        print(f"\n=== Dimension d={d} (analytic ref for ||x||^2) ===")
        print(f"Analytic reference integral (radial norm^2): {integral_ref:.8g}")

        # compute 3D deterministic reference if available (for the original 3D integrand)
        det_ref = None
        if d == 3 and HAVE_3D_DET:
            try:
                # spherical_grid_integral_equal_m returns (integral, n_evals)
                integral_det, n_evals_det = spherical_grid_integral_equal_m(f_from_points, m=80)
                det_ref = integral_det
                print(f"Deterministic 3D reference (grid m=80) for original integrand: {det_ref:.8g} (n_evals={n_evals_det})")
            except Exception as e:
                print("Warning: deterministic 3D reference could not be computed:", e)
                det_ref = None

        # if d==3, switch the MC experiment to use the original 3D integrand and, if available, use the deterministic grid result as the MC reference
        if d == 3:
            # use original 3D integrand for MC runs
            f_points = f_from_points
            # if we successfully computed a deterministic reference on f_from_points, use it
            if det_ref is not None:
                integral_ref = det_ref
                print(f"Using deterministic grid reference as analytic reference for d=3 (original integrand): {integral_ref:.8g}")
            else:
                # fallback: no deterministic ref available; keep analytic radial ref or warn
                print("Note: deterministic 3D reference not available; MC d=3 will be compared to analytic radial reference (not the original integrand).")

            for sampling in sampling_modes:
                # skip rejection sampling for higher d if acceptance will be tiny
                if sampling == 'rejection':
                    # compute expected acceptance rate
                    vol = volume_ball(1.0, d=d)
                    acc_expected = vol / (2.0 ** d)
                    print(f"[d={d}, sampling=rejection] expected acceptance = {acc_expected:.6g}")
                print(f"-- running MC experiments for d={d}, sampling={sampling} --")

                df = run_mc_for_config(f_points, d=d, sampling=sampling, N_mc=N_mc, repeats=repeats, symmetrize=True)

                # fill absolute error column using analytic reference
                df['abs_error_mean'] = df['mean_est'].apply(lambda x: abs(x - integral_ref))

                csv_name = os.path.join(OUTPUT_DIR, f"mc_results_d{d}_{sampling}.csv")
                df.to_csv(csv_name, index=False)
                print(f"Saved results to: {csv_name}")

                # plot (log-log) error vs N with std errorbars
                plt.figure()
                for method in df['method'].unique():
                    dfm = df[df['method'] == method].sort_values('N')
                    plt.errorbar(dfm['N'], dfm['abs_error_mean'], yerr=dfm['std_est'], label=method, marker='o')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Number of samples N')
                plt.ylabel('Absolute error (vs analytic ref)')
                plt.legend()
                plt.title(f'Error vs N (d={d}, sampling={sampling})')
                png_name = os.path.join(OUTPUT_DIR, f"mc_error_d{d}_{sampling}.png")
                plt.savefig(png_name)
                plt.close()
                print(f"Saved plot to: {png_name}")

                for method in df['method'].unique():
                    dfm = df[df['method'] == method]
                    summary_rows.append({
                        'd': d,
                        'sampling': sampling,
                        'method': method,
                        'N_values': ','.join(map(str, sorted(dfm['N'].unique()))),
                        'mean_abs_error_over_N': float(dfm['abs_error_mean'].mean())
                    })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUTPUT_DIR, 'mc_experiments_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print("\nAll experiments completed. Summary saved to:", summary_csv)

if __name__ == '__main__':
    run_all_experiments()