# unit-ball-integration-project
Numerical Methods Project. Due Dec. 10.

# Interpretation of 3D Monte Carlo Results
## Reference Integral

All reported errors use the same deterministic spherical-grid reference integral:

**I_ref ≈ 6.1344**

The exact value is not important—only that the same reference is used consistently.

---

## Behavior of the Estimators

### 1. Standard Monte Carlo (Direct Sampling)

Standard MC produces estimates in the range **6.38–6.44**, which is expected
for sample sizes N = 500–5000.

Approximate absolute errors:

| N | Abs Error |
|---|-----------|
| 500  | 0.2017 |
| 1000 | 0.1211 |
| 2000 | 0.2157 |
| 5000 | 0.2353 |

Notes:
- Errors fluctuate due to randomness.
- The overall scale of the error matches the expected **O(N^{-1/2})** behavior.
- The small error at N = 1000 is lucky draw.

---

### 2. Standard Monte Carlo (Rejection Sampling)

Rejection sampling shows slightly larger errors on average:

Mean absolute error over all N:
- **Direct MC:** ~0.193  
- **Rejection MC:** ~0.219

- Rejection sampling has higher variance (fewer effective samples).
- The acceptance rate in 3D is ~0.524, so half of proposed points are thrown away.

---

### 3. Symmetrized Monte Carlo (Variance Reduction)

Symmetrized MC evaluates f(x) over all 8 sign-flipped versions of x and averages the results.

Key observation:
- **Variance is greatly reduced**, as shown by much smaller `std_est` values.
- **Absolute error does not necessarily decrease** for this particular integrand.

Example at N = 500 (direct sampling):

| Method | std_est |
|--------|---------|
| Standard MC     | 0.1335 |
| Symmetrized MC  | 0.0510 |

Variance drops by ~62%.

However, the mean absolute errors are similar:
- Standard: 0.2017  
- Symmetrized: 0.1960  

This is expected because the integrand  
**f(x, y, z) = (1 + x² + y²)e^z − x/(1 + z²)**  
contains a *mostly even* component and only a *small odd* term.  
Symmetrization only cancels odd components, so its effect is limited.


