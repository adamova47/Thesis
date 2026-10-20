import pickle
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.utils import *


def pickle_dump(obj, filepath):
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(filepath)


def recompute_msom_temporal_qe(state, x, m, n, alpha, beta):
    W = np.asarray(state["weights"], dtype=np.float32)
    C = np.asarray(state["context_weights"], dtype=np.float32)

    prev_bmu = None
    temporal_err = 0.0
    total_steps = 0

    for xi in x:
        if prev_bmu is None:
            C_t = np.zeros(W.shape[2], dtype=np.float32)
        else:
            w_prev = W[prev_bmu]
            c_prev = C[prev_bmu]
            C_t = (1.0 - beta) * w_prev + beta * c_prev

        diff_x = W - xi
        d_x = np.sum(diff_x ** 2, axis=2)

        diff_c = C - C_t
        d_c = np.sum(diff_c ** 2, axis=2)

        d = (1.0 - alpha) * d_x + alpha * d_c

        flat_idx = int(np.argmin(d))
        bmu = (flat_idx // n, flat_idx % n)

        temporal_err += np.sqrt(d[bmu])

        prev_bmu = bmu
        total_steps += 1

    return float(temporal_err / total_steps)


def main():
    script_dir = Path(__file__).resolve().parent

    results_path = script_dir / "msom_results.pkl"
    fixed_path = script_dir / "msom_results_fixed_temporal_qe.pkl"
    backup_path = script_dir / "msom_results_original_backup.pkl"

    df = pd.read_excel(script_dir / "mackey_glass.xlsx")
    x = df[["t", "t-taw"]].values
    x = smart_normalize(x).astype(np.float32)

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    fixed = {}

    for key, val in results.items():
        m, n, init, metric, kernel, alpha, beta, train_epochs, seed = key

        temporal_qe = recompute_msom_temporal_qe(
            state=val["state"],
            x=x,
            m=m,
            n=n,
            alpha=alpha,
            beta=beta
        )

        new_val = dict(val)
        new_val["old_qe"] = val["qe"]
        new_val["qe"] = temporal_qe

        fixed[key] = new_val

    pickle_dump(results, backup_path)
    pickle_dump(fixed, fixed_path)

    print(f"Fixed {len(fixed)} experiments.")
    print(f"Original backup: {backup_path}")
    print(f"Fixed file: {fixed_path}")


if __name__ == "__main__":
    main()