import pickle
import math
import numpy as np
import pandas as pd
from pathlib import Path


def load_results_dict(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def results_to_dataframe(results_dict):
    rows = []
    for key, val in results_dict.items():
        m, n, init, metric, kernel, train_epochs, seed = key
        qe = float(val["qe"])
        entropy = float(val["entropy"])
        dead_neurons = int(val["dead_neurons"])
        total_neurons = m * n
        active_neurons = total_neurons - dead_neurons
        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else np.nan
        if active_neurons > 1:
            norm_entropy = entropy / math.log(active_neurons)
        else:
            norm_entropy = 0.0

        rows.append({
            "m": m,
            "n": n,
            "neurons": total_neurons,
            "init": init,
            "metric": metric,
            "kernel": kernel,
            "train_epochs": train_epochs,
            "seed": seed,
            "qe": qe,
            "entropy": entropy,
            "norm_entropy": norm_entropy,
            "dead_neurons": dead_neurons,
            "dead_ratio": dead_ratio,
            "weights": val["weighs"],
            "key": key,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["qe", "neurons", "dead_ratio"], ascending=[True, True, True])



def shortlist_maps(df, qe_tol=0.05, min_norm_entropy=0.70, max_dead_ratio=0.25):
    best_qe = df["qe"].min()
    qe_cutoff = best_qe * (1.0 + qe_tol)
    short = df[df["qe"] <= qe_cutoff].copy()
    short = short[short["norm_entropy"] >= min_norm_entropy]
    short = short[short["dead_ratio"] <= max_dead_ratio]
    return short.sort_values(
        ["neurons", "qe", "dead_ratio", "norm_entropy"],
        ascending=[True, True, True, False]
    )


def pareto_front(df):
    keep = []
    rows = df.to_dict("records")
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                b["qe"] <= a["qe"] and
                b["neurons"] <= a["neurons"] and
                b["dead_ratio"] <= a["dead_ratio"] and
                b["norm_entropy"] >= a["norm_entropy"]
            )
            strictly_better = (
                b["qe"] < a["qe"] or
                b["neurons"] < a["neurons"] or
                b["dead_ratio"] < a["dead_ratio"] or
                b["norm_entropy"] > a["norm_entropy"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return pd.DataFrame(keep).sort_values(
        ["qe", "neurons", "dead_ratio", "norm_entropy"],
        ascending=[True, True, True, False]
    )


def choose_best_compromise(df):
    out = df.copy()
    for col in ["qe", "neurons", "dead_ratio", "norm_entropy"]:
        std = out[col].std(ddof=0)
        if std == 0:
            out[col + "_z"] = 0.0
        else:
            out[col + "_z"] = (out[col] - out[col].mean()) / std
    out["score"] = (
        0.50 * out["qe_z"] +
        0.25 * out["neurons_z"] +
        0.15 * out["dead_ratio_z"] -
        0.10 * out["norm_entropy_z"]
    )
    return out.sort_values("score", ascending=True)


def main():
    results_path = Path(__file__).resolve().parent / "som_results.pkl"
    results_dict = load_results_dict(results_path)
    df = results_to_dataframe(results_dict)

    print("\nTop 10 by QE:")
    print(df[[
        "m", "n", "neurons", "qe", "entropy", "norm_entropy",
        "dead_neurons", "dead_ratio", "init", "metric", "kernel"
    ]].head(10).to_string(index=False))


    short = shortlist_maps(df, qe_tol=0.05, min_norm_entropy=0.70, max_dead_ratio=0.25)
    print("\nShortlist (near-best QE, decent utilization, not too many dead neurons):")
    if len(short) == 0:
        print("No models passed the shortlist filter.")
    else:
        print(short[[
            "m", "n", "neurons", "qe", "entropy", "norm_entropy",
            "dead_neurons", "dead_ratio", "init", "metric", "kernel"
        ]].head(15).to_string(index=False))


    front = pareto_front(df)
    print("\nPareto front:")
    print(front[[
        "m", "n", "neurons", "qe", "norm_entropy", "dead_ratio",
        "init", "metric", "kernel"
    ]].head(20).to_string(index=False))


    if len(short) > 0:
        candidate_pool = short.copy()
    else:
        candidate_pool = pareto_front(df)
    compromise = choose_best_compromise(candidate_pool)
    print("\nBest compromise candidates:")
    print(compromise[[
        "m", "n", "neurons", "qe", "norm_entropy", "dead_ratio",
        "init", "metric", "kernel", "score"
    ]].head(10).to_string(index=False))

    best = compromise.iloc[0]
    print("\nChosen map:")
    print(best[[
        "m", "n", "neurons", "qe", "entropy", "norm_entropy",
        "dead_neurons", "dead_ratio", "init", "metric", "kernel"
    ]].to_string())


if __name__ == "__main__":
    main()