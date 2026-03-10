import pickle
import cupy as cp
from joblib import Parallel, delayed
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.SOM_vectorized import SOM_vectorized as SOM
from src.models.utils import *


def run_config(params):
    m, n, init_method, grid_metric, kernel, x, y, epochs = params

    som = SOM(
        m, n,
        dim=x.shape[1],
        weight_init_method=init_method,
        grid_metric=grid_metric,
        neighborhood_kernel=kernel,
        seed=42,
    )

    som.train(x, num_epochs=epochs)
    qe = float(som.q_error_history[-1].get())

    hits = cp.zeros((m, n), dtype=cp.int32)

    for xi in x:
        bi, bj = som.find_bmu(xi)
        hits[bi, bj] += 1

    dead_neurons = int(cp.sum(hits == 0).get())

    p = hits / cp.sum(hits)
    p = p[p > 0]
    entropy = float((-cp.sum(p * cp.log(p))).get())

    return {
        "m": m,
        "n": n,
        "init": init_method,
        "metric": grid_metric,
        "kernel": kernel,
        "epochs": epochs,
        "qe": qe,
        "entropy": entropy,
        "dead_neurons": dead_neurons,
        "som": som,
    }


def main():
    # load data
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "seeds.txt")
    data = np.loadtxt(data_path)
    x = cp.asarray(data[:, :-1])
    y = data[:, -1].astype(int)

    # generate map dimensions - number of neurons between 80 and 150
    dims = [(12,12)]

    # other hyperparameters
    inits = ["data_range"]
    metrics = ["euclid"]
    kernels = ["gaussian"]
    epochs = 150

    configs = [
        (m, n, init_method, grid_metric, kernel, x, y, epochs)
        for m, n in dims
        for init_method in inits
        for grid_metric in metrics
        for kernel in kernels
    ]

    results = Parallel(n_jobs=2)(delayed(run_config)(cfg) for cfg in configs)

    best = min(results, key=lambda r: r["qe"])
    best_som = best["som"]
    print(
        f"Best config: m={best['m']}, n={best['n']}, init={best['init']},"
        f" metric={best['metric']}, kernel={best['kernel']}, QE={best['qe']},"
        f" Entropy={best['entropy']}, Dead neurons={best['dead_neurons']}"
    )

    plot_quantization_error(best_som)
    plot_avg_adjustment(best_som)
    plot_winner_map(best_som, x, y)
    plot_feature_heatmaps(best_som)
    plot_u_matrix(best_som)
    plt.show()


if __name__ == "__main__":
    main()
