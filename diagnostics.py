import pandas as pd
import numpy as np
from math import sqrt
from scipy import stats  

DETERMINISTIC_REF = 6.6288953  
def analyze_csv(path):
    df = pd.read_csv(path)
    print("\nFile:", path)
    for method in df['method'].unique():
        dfm = df[df['method'] == method].sort_values('N')
        print("\nMethod:", method)
        for _, row in dfm.iterrows():
            N = int(row['N'])
            mean_est = float(row['mean_est'])
            std_est = float(row['std_est'])
            # repeats used in  experiments
            repeats = 10
            # 95% t-critical
            tcrit = stats.t.ppf(0.975, df=repeats-1)
            sem = std_est / np.sqrt(repeats)
            ci_low = mean_est - tcrit * sem
            ci_high = mean_est + tcrit * sem
            abs_err = abs(mean_est - DETERMINISTIC_REF)
            rel_err = abs_err / abs(DETERMINISTIC_REF)
            print(f"N={N:5d}  mean={mean_est:.6g}  std={std_est:.4g}  abs_err={abs_err:.6g}  rel_err={rel_err:.4%}  95%CI=[{ci_low:.6g},{ci_high:.6g}]")

if __name__ == "__main__":
    analyze_csv("outputs/mc_results_d3_direct.csv")
    analyze_csv("outputs/mc_results_d3_rejection.csv")
