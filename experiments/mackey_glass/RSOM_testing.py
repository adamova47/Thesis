import pickle
import cupy as cp
from joblib import Parallel, delayed
import pandas as pd
import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.RSOM_cp_vectorized import RSOM
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
        42
    )


def export_rsom_state(rsom):
    return {
        "weights": cp.asnumpy(rsom.weights),
        "context_weights": cp.asnumpy(rsom.context_weights),
        "bmu_trajectories": to_cpu(rsom.bmu_trajectories),
        "activity_trajectories": to_cpu(rsom.activity_trajectories),
        "sequence_lengths": list(rsom.sequence_lengths)
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
    m, n, init, metric, kernel, alpha, beta, x, epochs = params
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
    qe = rsom.temporal_q_error_history[-1]

    hits = cp.zeros((m, n), dtype=cp.int32)
    Q = rsom.m * rsom.n
    rsom.context_vector = cp.zeros(Q, dtype=rsom.weights.dtype)

    for xi in cp.asarray(x):
        _, y, bmu = rsom._compute_energy_and_activity(xi)
        hits[bmu] += 1
        rsom.context_vector = y

    hits = hits.get()

    dead_neurons = (hits == 0).sum()
    utilization = hits / hits.sum()

    p = utilization[utilization > 0]
    entropy = -np.sum(p * np.log(p))

    return {
        "m": m,
        "n": n,
        "train_epochs": epochs,
        "best_epoch": rsom.best_epoch + 1,
        "init": init,
        "metric": metric,
        "kernel": kernel,
        "alpha": alpha,
        "beta": beta,
        "state": export_rsom_state(rsom),
        "qe": qe,
        "qe_history": to_cpu(rsom.temporal_q_error_history),
        "entropy": entropy,
        "dead_neurons": dead_neurons,
        "rsom": rsom
    }


def main():
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "mackey_glass.xlsx"))
    x = df[["t", "t-taw"]].values
    x = cp.asarray(smart_normalize(x), dtype=cp.float32)
    y = df["t+1"].values.reshape(-1, 1)

    dims = [(8,12), (9,9), (9,12), (10,10), (10,13), (11,11), (12,12)]
    inits = ["uniform", "sample", "pca"]
    metrics = ["euclid", "manhattan"]
    kernels = ["gaussian", "bubble"]

    alphas = [0.1, 0.3, 0.6, 0.9]
    betas = [0.1, 0.3, 0.6, 0.9]

    epochs = 250

    results_file = os.path.join(os.path.dirname(__file__), "rsom_results.pkl")

    configs = [
        (m, n, init, metric, kernel, a, b, x, epochs)
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
    
    # load the old big dictionary if it exists
    all_results = load_results_dict(results_file)

    # update or overwrite only the configs from this run
    for result in results:
        key = make_result_key(result)
        all_results[key] = {
            "state": result["state"],
            "qe": result["qe"],
            "qe_history": result["qe_history"],
            "entropy": result["entropy"],
            "dead_neurons": result["dead_neurons"],
            "best_epoch": result["best_epoch"],
        }

    # save at the end
    pickle_dump(all_results, results_file)

    plot_quantization_error(best_rsom)
    plot_temporal_quantization_error(best_rsom)
    plot_trajectory_map(best_rsom)

    plt.show()



if __name__ == "__main__":
    main()
