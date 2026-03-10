import pickle
import cupy as cp
from joblib import Parallel, delayed
import pandas as pd
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.RSOM_cp_vectorized import RSOM
from src.models.utils import *


def run_config(params):
    m, n, init, metric, kernel, alpha, beta, x, y, epochs = params
    rsom = RSOM(
        m=m,
        n=n,
        dim=2,
        weight_init_method=init,
        grid_metric=metric,
        neighborhood_kernel=kernel,
        alpha=alpha,
        beta=beta,
        seed=42
    )
    rsom.train(x, num_epochs=epochs)
    qe = rsom.q_error_history[-1]

    hits = cp.zeros((m, n), dtype=cp.int32)

    for xi in cp.asarray(x):
        _, _, bmu = rsom._compute_energy_and_activity(xi)
        hits[bmu] += 1

    hits = hits.get()

    dead_neurons = (hits == 0).sum()
    utilization = hits / hits.sum()

    p = utilization[utilization > 0]
    entropy = -np.sum(p * np.log(p))

    return {
        "m": m,
        "n": n,
        "init": init,
        "metric": metric,
        "kernel": kernel,
        "alpha": alpha,
        "beta": beta,
        "qe": qe,
        "entropy": entropy,
        "dead_neurons": dead_neurons,
        "rsom": rsom
    }


def main():
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "mackey_glass.xlsx"))
    x = df[["t", "t-taw"]].values
    x = cp.asarray(smart_normalize(x))
    y = df["t+1"].values.reshape(-1, 1)

    # dims = [(m, n) for m in range(8, 16) for n in range(m, 16) if 80 <= m*n <= 150]
    dims = [(10, 10)]
    inits = ["sample"]
    metrics = ["euclid"]
    kernels = ["gaussian"]

    alphas = [1.0, 0.5, 0.2]
    betas = [0.0, 0.1, 0.3, 0.5]

    epochs = 100

    configs = [
        (m, n, init, metric, kernel, a, b, x, y, epochs)
        for m, n in dims
        for init in inits
        for metric in metrics
        for kernel in kernels
        for a in alphas
        for b in betas
    ]

    results = Parallel(n_jobs=3)(
        delayed(run_config)(cfg) for cfg in configs
    )


    best = min(results, key=lambda r: r["qe"])
    best_rsom = best["rsom"]

    print(f"Best config: m={best['m']}, n={best['n']}, init={best['init']}, "
        f"metric={best['metric']}, kernel={best['kernel']}, "
        f"alpha={best['alpha']}, beta={best['beta']}, "
        f"QE={best['qe']}, Entropy={best['entropy']}, Dead neurons={best['dead_neurons']}")

    plot_quantization_error(best_rsom)
    plot_trajectory_map(best_rsom)
    # plot_recursive_state_evolution(best_rsom, 100)
    # plot_temporal_similarity(best_rsom)
    # plot_context_norms(best_rsom)



if __name__ == "__main__":
    main()
