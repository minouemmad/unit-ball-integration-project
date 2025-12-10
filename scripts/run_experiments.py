# scripts/run_experiments.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import (
    set_seed,
    f_from_points,
    volume_ball,
    f_radial_norm2,
    analytic_integral_radial_norm2
)

from src.integration import spherical_grid_integral_equal_m
from src.montecarlo import mc_estimate_with_repeats

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_mc_for_config(f_points, d, sampling, N_mc, repeats=10):
    """
    Run both standard MC and symmetrized MC for a given integrand.

    Returns:
        DataFrame with columns:
        method, d, sampling, N, n_evals, mean_est, std_est
    """
    out_rows = []

    mc_std = mc_estimate_with_repeats(
        f_points, N_mc, repeats=repeats,
        symmetrize=False, d=d, sampling=sampling
    )

    mc_sym = mc_estimate_with_repeats(
        f_points, N_mc, repeats=repeats,
        symmetrize=True, d=d, sampling=sampling
    )

    for N in N_mc:

        # Standard Monte Carlo
        ests = mc_std[N]["estimates"]
        out_rows.append({
            "method": "mc_standard",
            "d": d,
            "sampling": sampling,
            "N": N,
            "n_evals": mc_std[N]["n_evals"],
            "mean_est": float(np.mean(ests)),
            "std_est": float(np.std(ests, ddof=1)),
        })

        # Symmetrized MC
        ests_s = mc_sym[N]["estimates"]
        out_rows.append({
            "method": "mc_symmetrized",
            "d": d,
            "sampling": sampling,
            "N": N,
            "n_evals": mc_sym[N]["n_evals"],
            "mean_est": float(np.mean(ests_s)),
            "std_est": float(np.std(ests_s, ddof=1)),
        })

    return pd.DataFrame(out_rows)


def compute_reference_value(d):
    """
    Returns:
        reference_value, reference_type, integrand_function
    """

    # d = 2 and d = 4 use radial analytic integrand
    if d in [2, 4]:
        ref = analytic_integral_radial_norm2(d)
        return ref, f"analytic radial reference for d={d}", f_radial_norm2

    # d = 3 uses the actual project integrand f(x,y,z)
    if d == 3:
        print("Computing deterministic grid reference for d=3 integrand...")
        ref, n_evals = spherical_grid_integral_equal_m(f_from_points, m=80)
        print(f"Reference value = {ref:.8f} (n_evals={n_evals})")
        return ref, "deterministic spherical grid (m=80) for d=3", f_from_points

    raise ValueError("Unsupported dimension")


def run_all_experiments(d_list=[2,3,4], sampling_modes=["direct","rejection"]):

    set_seed(12345)

    N_mc = [500, 1000, 2000, 5000]
    repeats = 10

    summary_rows = []

    for d in d_list:

        # Get the correct integrand + reference value
        reference_value, reference_type, f_points = compute_reference_value(d)

        print("\n====================================================")
        print(f"Running experiments for dimension d={d}")
        print(f"Reference value = {reference_value:.8f}")
        print(f"Reference type = {reference_type}")
        print("====================================================\n")

        for sampling in sampling_modes:

            # Rejection sampling acceptance info
            if sampling == "rejection":
                vol = volume_ball(1.0, d=d)
                acc = vol / (2**d)
                print(f"[d={d}, rejection] expected acceptance probability = {acc:.6f}")

            print(f"-- Running MC for d={d}, sampling='{sampling}' --")

            df = run_mc_for_config(
                f_points=f_points,
                d=d,
                sampling=sampling,
                N_mc=N_mc,
                repeats=repeats
            )

            # Compute absolute and relative errors
            df["reference_value"] = reference_value
            df["reference_type"] = reference_type
            df["abs_error"] = df["mean_est"].apply(lambda x: abs(x - reference_value))
            df["relative_error"] = df["abs_error"] / abs(reference_value)

            # Save CSV
            csv_path = os.path.join(OUTPUT_DIR, f"mc_results_d{d}_{sampling}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved results: {csv_path}")

            # Plot error vs N
            plt.figure()
            for method in df["method"].unique():
                df_m = df[df["method"] == method].sort_values("N")
                plt.errorbar(
                    df_m["N"],
                    df_m["abs_error"],
                    yerr=df_m["std_est"],
                    marker="o",
                    label=method
                )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("N (samples)")
            plt.ylabel("Absolute error")
            plt.title(f"Error vs N (d={d}, sampling={sampling})")
            plt.legend()

            png_path = os.path.join(OUTPUT_DIR, f"mc_error_d{d}_{sampling}.png")
            plt.savefig(png_path)
            plt.close()
            print(f"Saved plot: {png_path}")

            # Add to summary
            for method in df["method"].unique():
                mean_err = df[df["method"] == method]["abs_error"].mean()
                summary_rows.append({
                    "d": d,
                    "sampling": sampling,
                    "method": method,
                    "mean_abs_error_over_N": float(mean_err)
                })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "mc_experiments_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\nAll experiments complete.")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    run_all_experiments()
