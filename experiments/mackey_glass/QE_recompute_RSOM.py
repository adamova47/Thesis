import pickle
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.utils import smart_normalize


def load_results_dict(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def pickle_dump(obj, filepath):
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(filepath)


def recompute_rsom_temporal_qe(state, x, m, n, alpha, beta):
    W_grid = np.asarray(state["weights"], dtype=np.float32)
    C = np.asarray(state["context_weights"], dtype=np.float32)

    Q = m * n
    W = W_grid.reshape(Q, -1)

    context_vector = np.zeros(Q, dtype=np.float32)

    temporal_err = 0.0
    total_steps = 0

    for xi in x:
        d_input = np.sum((W - xi[None, :]) ** 2, axis=1)
        d_context = np.sum((C - context_vector[None, :]) ** 2, axis=1)

        d_comb = alpha * d_input + beta * d_context

        bmu_idx = int(np.argmin(d_comb))
        temporal_err += np.sqrt(d_comb[bmu_idx])

        context_vector = np.exp(-d_comb).astype(np.float32)
        total_steps += 1

    return float(temporal_err / total_steps)


def main():
    script_dir = Path(__file__).resolve().parent

    results_path = script_dir / "rsom_results.pkl"
    fixed_path = script_dir / "rsom_results_fixed_temporal_qe.pkl"
    backup_path = script_dir / "rsom_results_original_static_qe_backup.pkl"

    df = pd.read_excel(script_dir / "mackey_glass.xlsx")
    x = df[["t", "t-taw"]].values
    x = smart_normalize(x).astype(np.float32)

    results = load_results_dict(results_path)

    fixed = {}

    for key, val in results.items():
        m, n, init, metric, kernel, alpha, beta, train_epochs, seed = key

        temporal_qe = recompute_rsom_temporal_qe(
            state=val["state"],
            x=x,
            m=m,
            n=n,
            alpha=alpha,
            beta=beta
        )

        new_val = dict(val)
        new_val["static_qe_old"] = val["qe"]
        new_val["qe"] = temporal_qe

        fixed[key] = new_val

    pickle_dump(results, backup_path)
    pickle_dump(fixed, fixed_path)

    print(f"Fixed {len(fixed)} experiments.")
    print(f"Original backup: {backup_path}")
    print(f"Fixed file: {fixed_path}")


if __name__ == "__main__":
    main()