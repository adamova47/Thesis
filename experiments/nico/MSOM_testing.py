import pickle
import json
import cupy as cp
import numpy as np
from joblib import Parallel, delayed
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


def numeric_suffix(key):
    return int(key.split()[-1])


def load_nico_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    x = []
    y = []
    for grasp_type, sequences in data.items():
        for seq_key in sorted(sequences.keys(), key=numeric_suffix):
            positions = sequences[seq_key]
            seq = np.array(
                [
                    positions[pos_key]
                    for pos_key in sorted(positions.keys(), key=numeric_suffix)
                ],
                dtype=np.float32
            )
            x.append(seq)
            y.append(grasp_type)
    return x, y


def normalize_sequences(x, eps=1e-8):
    all_points = np.concatenate(x, axis=0)
    mean = all_points.mean(axis=0)
    std = all_points.std(axis=0) + eps
    x_norm = [(seq - mean) / std for seq in x]
    return x_norm


def compute_hits(msom, x, m, n):
    """
    same idea as your Mackey hit counting, but adapted for many sequences.
    prev_bmu resets at the start of each grasp sequence.
    """
    hits = cp.zeros((m, n), dtype=cp.int32)
    for seq in x:
        prev_bmu = None
        for xi in seq:
            C_t = msom._compute_context_descriptor(prev_bmu)
            bmu = msom.find_bmu(xi, C_t)
            prev_bmu = bmu
            hits[bmu] += 1
    return hits.get()


def run_config(params):
    m, n, init, metric, kernel, alpha, beta, x, y, epochs = params

    dim = x[0].shape[1]

    msom = MSOM(
        m=m,
        n=n,
        dim=dim,
        weight_init_method=init,
        grid_metric=metric,
        neighborhood_kernel=kernel,
        alpha=alpha,
        beta=beta,
        seed=42
    )

    msom.train(x, num_epochs=epochs)

    qe = float(msom.temporal_q_error_history[-1])
    static_qe = float(msom.q_error_history[-1])

    hits = compute_hits(msom, x, m, n)

    dead_neurons = int((hits == 0).sum())

    utilization = hits / hits.sum()
    p = utilization[utilization > 0]
    entropy = float(-np.sum(p * np.log(p)))

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
        "static_qe": static_qe,
        "entropy": entropy,
        "dead_neurons": dead_neurons,
        "msom": msom
    }


def main():
    script_dir = os.path.dirname(__file__)

    dataset_path = os.path.join(script_dir, "dataset.json")

    x, y = load_nico_dataset(dataset_path)
    x = normalize_sequences(x)
    x = [cp.asarray(seq, dtype=cp.float32) for seq in x]

    dims = [(4, 4)]
    """, (5, 5), (6, 6), (7, 7), (8, 8)"""

    inits = ["sample"]
    metrics = ["manhattan"]
    kernels = ["gaussian"]

    alphas = [0.1] 
    """, 0.2, 0.4, 0.6, 0.8"""
    betas = [0.2] 
    """, 0.5, 0.8"""

    epochs = 150

    results_file = os.path.join(script_dir, "msom_nico_results.pkl")

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

    print(
        f"Best config: m={best['m']}, n={best['n']}, init={best['init']}, "
        f"metric={best['metric']}, kernel={best['kernel']}, "
        f"alpha={best['alpha']}, beta={best['beta']}, "
        f"QE={best['qe']}, Static QE={best['static_qe']}, "
        f"Entropy={best['entropy']}, Dead neurons={best['dead_neurons']}"
    )

    all_results = load_results_dict(results_file)

    for result in results:
        key = make_result_key(result)
        all_results[key] = {
            "state": result["state"],
            "qe": result["qe"],
            "static_qe": result["static_qe"],
            "entropy": result["entropy"],
            "dead_neurons": result["dead_neurons"],
            "best_epoch": result["best_epoch"],
        }

    pickle_dump(all_results, results_file)

    plot_quantization_error(best_msom)
    plot_temporal_quantization_error(best_msom)
    plot_trajectory_map(best_msom)

    plt.show()


if __name__ == "__main__":
    main()