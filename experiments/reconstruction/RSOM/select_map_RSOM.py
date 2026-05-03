from pathlib import Path
import sys

MSOM_DIR = Path(__file__).resolve().parents[1] / "MSOM"
sys.path.append(str(MSOM_DIR))

from select_map_MSOM import (
    load_results_dict,
    shortlist_maps,
    pareto_front,
    choose_best_compromise,
)

import math
import numpy as np
import pandas as pd


def results_to_dataframe(results_dict):
    rows = []

    for key, val in results_dict.items():
        m, n, init, metric, kernel, alpha, beta, train_epochs, seed = key

        qe = float(val["qe"])
        entropy = float(val["entropy"])
        dead_neurons = int(val["dead_neurons"])
        best_epoch = int(val["best_epoch"])

        total_neurons = m * n
        active_neurons = total_neurons - dead_neurons
        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else np.nan

        if active_neurons > 1:
            norm_entropy = entropy / math.log(active_neurons)
        else:
            norm_entropy = 0.0

        state = val["state"]

        has_activity = (
            "activity_trajectories" in state
            and len(state["activity_trajectories"]) > 0
        )

        rows.append({
            "m": m,
            "n": n,
            "neurons": total_neurons,
            "init": init,
            "metric": metric,
            "kernel": kernel,
            "alpha": alpha,
            "beta": beta,
            "train_epochs": train_epochs,
            "best_epoch": best_epoch,
            "qe": qe,
            "entropy": entropy,
            "norm_entropy": norm_entropy,
            "dead_neurons": dead_neurons,
            "dead_ratio": dead_ratio,
            "state": state,
            "key": key,
        })

    df = pd.DataFrame(rows)

    return df.sort_values(
        ["qe", "neurons", "dead_ratio"],
        ascending=[True, True, True]
    )