import pickle
import numpy as np
import cupy as cp
from joblib import Parallel, delayed
import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.SOM_vectorized import SOM_vectorized as SOM
from src.models.utils import *


def make_result_key(result):
    return (
        result["m"],
        result["n"],
        result["init"],
        result["metric"],
        result["kernel"],
        result["train_epochs"],
        42
    )


def pickle_dump(obj, filepath):
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(tmp_path, filepath)


def load_results_dict(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        return {}

    with open(filepath, "rb") as f:
        return pickle.load(f)


def run_config(params):
    m, n, init_method, grid_metric, kernel, x, epochs = params

    som = SOM(
        m, n,
        dim=x.shape[1],
        weight_init_method=init_method,
        grid_metric=grid_metric,
        neighborhood_kernel=kernel,
        seed=42,
    )

    som.train(x, num_epochs=epochs)
    qe = float(som.q_error_history[-1])

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
        "train_epochs": epochs,
        "best_epoch": som.best_epoch + 1,
        "init": init_method,
        "metric": grid_metric,
        "kernel": kernel,
        "weights": cp.asnumpy(som.weights),
        "qe": qe,
        "entropy": entropy,
        "dead_neurons": dead_neurons,
        "som": som,
    }


def main():
    # load data
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "seeds_dataset.txt")
    data = np.loadtxt(data_path)
    x = cp.asarray(smart_normalize(data[:, :-1]))
    y = data[:, -1].astype(int)

    # generate map dimensions - number of neurons between 80 and 150
    dims = [(m, n) for m in range(8, 16) for n in range(m, 16) if 80 <= m*n <= 150]

    # other hyperparameters
    inits = ["uniform", "data_range", "sample", "pca", "kmeans"]
    metrics = ["euclid", "manhattan", "chebyshev", "toroidal"]
    kernels = ["gaussian", "bubble", "epanechnikov", "triangular", "inverse"]
    epochs = 200

    results_file = os.path.join(os.path.dirname(__file__), "som_results_exponential.pkl")

    configs = [
        (m, n, init_method, grid_metric, kernel, x, epochs)
        for m, n in dims
        for init_method in inits
        for grid_metric in metrics
        for kernel in kernels
    ]

    results = Parallel(n_jobs=3)(delayed(run_config)(cfg) for cfg in configs)

    best = min(results, key=lambda r: r["qe"])
    best_som = best["som"]
    print(
        f"Best config: m={best['m']}, n={best['n']}, init={best['init']},"
        f" metric={best['metric']}, kernel={best['kernel']}, QE={best['qe']},"
        f" Entropy={best['entropy']}, Dead neurons={best['dead_neurons']}"
    )

    # load the old big dictionary if it exists
    all_results = load_results_dict(results_file)

    # update or overwrite only the configs from this run
    for result in results:
        key = make_result_key(result)
        all_results[key] = {
            "weights": result["weights"],
            "qe": result["qe"],
            "entropy": result["entropy"],
            "dead_neurons": result["dead_neurons"],
            "best_epoch": result["best_epoch"],
        }

    # save at the end
    pickle_dump(all_results, results_file)

    plot_quantization_error(best_som)
    plot_avg_adjustment(best_som)
    plot_winner_map(best_som, x, y)
    plot_feature_heatmaps(best_som)
    plot_u_matrix(best_som)
    plt.show()


if __name__ == "__main__":
    main()
