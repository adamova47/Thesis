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
        result["train_epochs"],
    )


def export_msom_state(msom):
    return {
        "weights": cp.asnumpy(msom.weights),
        "context_weights": cp.asnumpy(msom.context_weights),
        "bmu_trajectories": to_cpu(msom.bmu_trajectories),
        "sequence_lengths": list(msom.sequence_lengths)
    }


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
    qe = msom.temporal_q_error_history[-1]

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
        "train_epochs": epochs,
        "best_epoch": msom.best_epoch + 1,
        "init": init,
        "metric": metric,
        "kernel": kernel,
        "alpha": alpha,
        "beta": beta,
        "state": export_msom_state(msom),
        "qe": qe,
        "entropy": entropy,
        "dead_neurons": dead_neurons,
        "msom": msom
    }


def main():
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "mackey_glass.xlsx"))
    x = df[["t", "t-taw"]].values
    x = cp.asarray(smart_normalize(x), dtype=cp.float32)
    y = df["t+1"].values.reshape(-1, 1)

    # dims = [(m, n) for m in range(8, 16) for n in range(m, 16) if 80 <= m*n <= 150]
    dims = [(11,11), (11,12), (12,11), (12,12), (12,13), (13,12), (10,12), (10,13), (11,13)]
    """(11,11), (11,12), (12,11), (12,12), (12,13), (13,12), (10,12), (10,13), (11,13)"""
    inits = ["sample"]
    metrics = ["manhattan"]
    kernels = ["gaussian"]

    alphas = [0.05, 0.1, 0.15, 0.2, 0.75, 0.85, 0.9, 0.95, 0.99]
    betas = [0.8, 0.9, 0.95, 0.99]

    epochs = 250

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
    best_msom = best["msom"]

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
            "state": result["state"],
            "qe": result["qe"],
            "entropy": result["entropy"],
            "dead_neurons": result["dead_neurons"],
            "best_epoch": result["best_epoch"],
        }

    # save at the end
    pickle_dump(all_results, results_file)

    plot_quantization_error(best_msom)
    plot_temporal_quantization_error(best_msom)
    plot_trajectory_map(best_msom)
    # plot_recursive_state_evolution(best_msom, 100)
    # plot_temporal_similarity(best_msom)
    # plot_context_norms(best_msom)
    plt.show()



if __name__ == "__main__":
    main()