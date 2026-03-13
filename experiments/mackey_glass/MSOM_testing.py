import pickle
import cupy as cp
from joblib import Parallel, delayed
import pandas as pd
import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.MSOM_cp_vectorized import MSOM
from src.models.utils import *


def make_result_key(result):
    return (
        result["m"],
        result["n"],
        result["init"],
        result["metric"],
        result["kernel"],
        result["alpha"],
        result["beta"],
        result["epochs"],
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
    m, n, init, metric, kernel, alpha, beta, x, y, epochs = params
    msom = MSOM(
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
    msom.train(x, num_epochs=epochs)
    qe = msom.q_error_history[-1]

    prev_bmu = None
    hits = cp.zeros((m, n), dtype=cp.int32)

    for xi in cp.asarray(x):
        C_t = msom._compute_context_descriptor(prev_bmu)
        bmu = msom.find_bmu(xi, C_t)
        prev_bmu = bmu
        hits[bmu] += 1

    hits = hits.get()

    dead_neurons = (hits == 0).sum()
    utilization = hits / hits.sum()

    p = utilization[utilization > 0]
    entropy = -np.sum(p * np.log(p))

    return {
        "m": m,
        "n": n,
        "epochs": epochs,
        "init": init,
        "metric": metric,
        "kernel": kernel,
        "alpha": alpha,
        "beta": beta,
        "qe": qe,
        "entropy": entropy,
        "dead_neurons": dead_neurons,
        "msom": msom
    }


def main():
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "mackey_glass.xlsx"))
    x = df[["t", "t-taw"]].values
    x = cp.asarray(smart_normalize(x))
    y = df["t+1"].values.reshape(-1, 1)

    # dims = [(m, n) for m in range(8, 16) for n in range(m, 16) if 80 <= m*n <= 150]
    dims = [(10, 10)]
    inits = ["data_range"]
    metrics = ["euclid", "chebyshev", "toroidal"]
    kernels = ["bubble", "triangular"]

    alphas = [1.0, 0.5, 0.1]
    betas = [0.1, 0.3, 0.5, 0.65, 0.8]

    epochs = 150

    results_file = os.path.join(os.path.dirname(__file__), "msom_results.pkl")

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
    best_rsom = best["msom"]

    print(f"Best config: m={best['m']}, n={best['n']}, init={best['init']}, "
        f"metric={best['metric']}, kernel={best['kernel']}, "
        f"alpha={best['alpha']}, beta={best['beta']}, "
        f"QE={best['qe']}, Entropy={best['entropy']}, Dead neurons={best['dead_neurons']}")
    
    # load the old big dictionary if it exists
    all_results = load_results_dict(results_file)

    # update or overwrite only the configs from this run
    for result in results:
        key = make_result_key(result)
        all_results[key] = {
            "qe": result["qe"],
            "entropy": result["entropy"],
            "dead_neurons": result["dead_neurons"],
        }

    # save at the end
    pickle_dump(all_results, results_file)

    plot_quantization_error(best_rsom)
    plot_trajectory_map(best_rsom)
    # plot_recursive_state_evolution(best_rsom, 100)
    # plot_temporal_similarity(best_rsom)
    # plot_context_norms(best_rsom)



if __name__ == "__main__":
    main()