#tests/test_general_d_sampling.py
import numpy as np
from src.utils import volume_ball
from src.montecarlo import sample_uniform_ball_direct, sample_ball_rejection

def test_direct_sampling_moments():
    for d in [2, 3, 4]:
        N = 20000
        X = sample_uniform_ball_direct(N, d)
        assert X.shape == (N, d)
        # estimated fraction inside unit ball should be 1 (by construction)
        radii = np.linalg.norm(X, axis=1)
        assert np.all(radii <= 1.0 + 1e-12)

def test_rejection_acceptance_formula():
    for d in [2, 3, 4]:
        vol = volume_ball(1.0, d=d)
        acc = vol / (2.0 ** d)
        # acc is the expected acceptance rate
        print(f"d={d}, vol={vol:.6g}, expected acceptance={acc:.6g}")
if __name__ == "__main__":
    # quick smoke-run when executing the file directly
    print("Running quick sampling smoke tests...")

    try:
        test_direct_sampling_moments()
        test_rejection_acceptance_formula()
    except AssertionError as e:
        print("Test FAILED:", e)
        raise SystemExit(1)

    print("All quick sampling smoke tests passed.")