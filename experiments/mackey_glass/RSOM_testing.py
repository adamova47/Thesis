from joblib import Parallel, delayed
import pandas as pd
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.RSOM import RSOM
from src.models.utils import *


def run_config(params):
    m, n, init, metric, kernel, alpha, beta, gamma1, gamma2, x, y, epochs = params
    rsom = RSOM(
        m=m,
        n=n,
        dim=x.shape[1],
        weight_init_method=init,
        grid_metric=metric,
        neighborhood_kernel=kernel,
        alpha=alpha,
        beta=beta,
        gamma1=gamma1,
        gamma2=gamma2,
        seed=42
    )
    rsom.train(x, num_epochs=epochs)
    q = rsom.q_error[-1]
    return {
        "m": m,
        "n": n,
        "init": init,
        "metric": metric,
        "kernel": kernel,
        "alpha": alpha,
        "beta": beta,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "q": q,
        "rsom": rsom
    }


def main():
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "mackey_glass.xlsx"))

    print(df.head())
    print(df.shape)

    x = df["t-taw"].values.reshape(-1, 1)
    y = df["t+1"].values.reshape(-1, 1)

    dims = [(m, n) for m in range(8, 16) for n in range(m, 16) if 80 <= m*n <= 150]
    inits = ["uniform", "sample", "pca"]
    metrics = ["euclid", "manhattan", "toroidal"]
    kernels = ["gaussian", "bubble"]

    alphas = [0.1, 0.2, 0.3]
    betas = [0.2, 0.4, 0.6]
    gammas1 = [0.05, 0.1, 0.2]
    gammas2 = [0.05, 0.1, 0.2]

    epochs = 50

    configs = [
        (m, n, init, metric, kernel, a, b, g1, g2, x, y, epochs)
        for m, n in dims
        for init in inits
        for metric in metrics
        for kernel in kernels
        for a in alphas
        for b in betas
        for g1 in gammas1
        for g2 in gammas2
    ]

    results = Parallel(n_jobs=-1)(delayed(run_config)(cfg) for cfg in configs)

    best = min(results, key=lambda r: r["q"])
    best_rsom = best["rsom"]

    print(f"Best config: m={best['m']}, n={best['n']}, init={best['init']}, "
        f"metric={best['metric']}, kernel={best['kernel']}, "
        f"alpha={best['alpha']}, beta={best['beta']}, "
        f"gamma1={best['gamma1']}, gamma2={best['gamma2']}, QE={best['q']}")

    """plot_quantization_error(best_rsom)
    plot_trajectory_map(best_rsom)
    plot_recursive_state_evolution(best_rsom, 100)
    plot_temporal_similarity(best_rsom)
    plot_context_norms(best_rsom)"""



if __name__ == "__main__":
    main()
