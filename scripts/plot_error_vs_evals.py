import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "outputs"

def plot_error_vs_evals(csv_path, title_suffix):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 4))

    for method in df['method'].unique():
        dfm = df[df['method'] == method].sort_values('n_evals')

        plt.loglog(
            dfm['n_evals'],
            dfm['abs_error_mean'],
            marker='o',
            label=method
        )

    plt.xlabel("Number of function evaluations (n_evals)")
    plt.ylabel("Absolute error vs reference")
    plt.title(f"Error vs n_evals {title_suffix}")
    plt.legend()
    
    out_name = os.path.join(OUTPUT_DIR, f"error_vs_evals_{title_suffix}.png")
    plt.savefig(out_name, dpi=150)
    plt.close()

    print(f"Saved: {out_name}")


def main():
    # You can add d=2,4 if desired
    paths = [
        ("outputs/mc_results_d3_direct.csv", "d3_direct"),
        ("outputs/mc_results_d3_rejection.csv", "d3_rejection"),
    ]

    for csv_path, tag in paths:
        plot_error_vs_evals(csv_path, tag)


if __name__ == "__main__":
    main()
