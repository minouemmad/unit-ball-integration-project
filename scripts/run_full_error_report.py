import os
import numpy as np
import pandas as pd
from src.utils import (
    set_seed,
    f_from_points,
    f_radial_norm2,
    analytic_integral_radial_norm2,
    volume_ball
)
from src.integration import spherical_grid_integral_equal_m
from src.montecarlo import mc_estimate_with_repeats

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_reference_integral(d):
    """
    Returns (reference_value, description_string)
    """
    # d = 2 or 4 → analytic reference for ||x||^2 integrand
    if d in [2, 4]:
        ref = analytic_integral_radial_norm2(d)
        return ref, f"analytic radial reference for d={d}"

    # d = 3 → try deterministic grid using the real integrand
    if d == 3:
        try:
            print("Computing deterministic reference for d=3 using spherical grid...")
            ref, n_eval = spherical_grid_integral_equal_m(f_from_points, m=80)
            return ref, f"deterministic spherical grid (m=80) for d=3"
        except Exception as e:
            print("Warning: deterministic 3D reference failed:", e)

        # fallback: analytic reference of ||x||^2
        ref = analytic_integral_radial_norm2(3)
        return ref, "fallback analytic radial reference (d=3)"

    raise ValueError("Unsupported dimension.")

def get_integrand_for_dimension(d):
    """
    Returns the appropriate f(x) to evaluate.
    """
    if d == 3:
        return f_from_points  # project-specified integrand
    return f_radial_norm2    # radial integrand for d != 3

def run_error_report(d_list=[2, 3, 4], sampling_modes=["direct", "rejection"]):
    set_seed(12345)

    N_list = [500, 1000, 2000, 5000]
    repeats = 10

    rows = []

    for d in d_list:
        print(f"\n=== Running dimension d={d} ===")

        f_points = get_integrand_for_dimension(d)
        ref_val, ref_desc = compute_reference_integral(d)

        print(f"Reference value for d={d}: {ref_val} ({ref_desc})")

        for sampling in sampling_modes:

            # skip rejection sampling in high dimensions (low acceptance)
            if sampling == "rejection" and d > 3:
                print(f"Skipping rejection sampling for d={d} (acceptance too small).")
                continue

            print(f"-- Sampling mode: {sampling} --")

            # Run both standard and symmetrized MC
            mc_variants = {
                "mc_standard": False,
                "mc_symmetrized": True
            }

            for method, sym_flag in mc_variants.items():

                out = mc_estimate_with_repeats(
                    f_points,
                    N_list,
                    repeats=repeats,
                    symmetrize=sym_flag,
                    d=d,
                    sampling=sampling
                )

                for N in N_list:
                    ests = out[N]["estimates"]
                    n_evals = out[N]["n_evals"]

                    mean_est = float(np.mean(ests))
                    std_est = float(np.std(ests, ddof=1))
                    abs_err = abs(mean_est - ref_val)
                    rel_err = abs_err / abs(ref_val)

                    rows.append({
                        "d": d,
                        "sampling": sampling,
                        "method": method,
                        "N": N,
                        "n_evals": n_evals,
                        "mean_est": mean_est,
                        "std_est": std_est,
                        "abs_error": abs_err,
                        "relative_error": rel_err,
                        "reference_value": ref_val,
                        "reference_type": ref_desc,
                    })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "full_error_report.csv")
    df.to_csv(out_path, index=False)
    print("\nSaved full error report to:", out_path)


if __name__ == "__main__":
    run_error_report()
