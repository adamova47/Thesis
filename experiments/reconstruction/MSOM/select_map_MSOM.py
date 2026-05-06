import pickle
import math
import numpy as np
import pandas as pd
from pathlib import Path


"""
Opens the saved pickle (dictionary) file.
Keys are hyperparameter settings.
Each value contains qe, entropy, dead_neurons, state
(weights, context_weights, bmu_trajectories, sequence_lengths).
"""
def load_results_dict(filepath):
    filepath = Path(filepath)
    # stop if the file doesnt exist
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    # open the pickle file and return the whole dictionary
    with open(filepath, "rb") as f:
        return pickle.load(f)


"""
Turns the dictionary into a dataframe
Dataframe is much easier to sort, filter and inspect than a nested
Python dictionary.
"""
def results_to_dataframe(results_dict):
    rows = []
    # we loop over every saved experiment in the pickle file
    for key, val in results_dict.items():
        # the key is: (m, n, init, metric, kernel, alpha, beta, train_epochs)
        m, n, init, metric, kernel, alpha, beta, train_epochs, seed = key
        # pull out the performance metrics saved for this run
        qe = float(val["qe"])
        entropy = float(val["entropy"])
        dead_neurons = int(val["dead_neurons"])
        best_epoch = int(val["best_epoch"])
        # total amount of neurons in the map
        total_neurons = m * n
        # active neurons = all neurons - dead neurons
        active_neurons = total_neurons - dead_neurons
        # dead ration = what fraction of the map is unused
        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else np.nan
        
        # raw entropy grows with map size, so comparing raw entropy
        # across different maps is unfair. So we normalize it.
        # if active neurons = 1, log(1) = 0, so we avoid division by zero.  
        if active_neurons > 1:
            norm_entropy = entropy / math.log(active_neurons)
        else:
            norm_entropy = 0.0
        # store one experiment as one row
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
            "seed": seed,
            "best_epoch": best_epoch,
            "qe": qe,
            "entropy": entropy,
            "norm_entropy": norm_entropy,
            "dead_neurons": dead_neurons,
            "dead_ratio": dead_ratio,
            "state": val["state"], # keep the saved map state for reconstruction
            "key": key, # original key for reference if we want to re-run or inspect this specific config
        })

    # convert the list into a dataframe
    df = pd.DataFrame(rows)
    # initial sorting: best QE first, then fewer neurons, then lower dead ratio
    return df.sort_values(["qe", "neurons", "dead_ratio"], ascending=[True, True, True])


"""
This function shortlists good maps.
You dont want to blindly pick only the absolutely best QE,
because maybe it came from a huge map with poor utilization.
So instead we keep maps whose QE is close to the best,
remove maps with poor utilization, and remove maps with too
many dead neurons.
"""
def shortlist_maps(df, qe_tol=0.05, min_norm_entropy=0.70, max_dead_ratio=0.25):
    # best QE found among all experiments
    best_qe = df["qe"].min()
    # allow maps up to 5% worse than the very best QE
    qe_cutoff = best_qe * (1.0 + qe_tol)

    # keep only the  rows whose QE is close to the best
    short = df[df["qe"] <= qe_cutoff].copy()
    # keep only the maps with decent neuron usage
    short = short[short["norm_entropy"] >= min_norm_entropy]
    # remove maps with too many dead neurons
    short = short[short["dead_ratio"] <= max_dead_ratio]

    # among the survivors prefer fewer neurons, then lower QE,
    # then lower dead ratio, then higher normalized entropy
    return short.sort_values(
        ["neurons", "qe", "dead_ratio", "norm_entropy"],
        ascending=[True, True, True, False]
    )

"""
Pareto front: to find non-dominant maps.
A map A is dominant if there is another map B that is
no worse in all criteria and strictly better in at least one criterion

criteria here are:
QE - lower is better
neurons - lower amount is better
dead_ratio - lower is better
norm_entropy - higher is better

The pareto front are all the maps that are a reasonable compromise
"""
def pareto_front(df):
    keep = []
    rows = df.to_dict("records")

    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue

            # b is no worse than a in every metric
            no_worse = (
                b["qe"] <= a["qe"] and
                b["neurons"] <= a["neurons"] and
                b["dead_ratio"] <= a["dead_ratio"] and
                b["norm_entropy"] >= a["norm_entropy"]
            )

            # b is strcitly better than a in at least one metric
            strictly_better = (
                b["qe"] < a["qe"] or
                b["neurons"] < a["neurons"] or
                b["dead_ratio"] < a["dead_ratio"] or
                b["norm_entropy"] > a["norm_entropy"]
            )

            if no_worse and strictly_better:
                dominated = True
                break

        # keep the non-dominant maps
        if not dominated:
            keep.append(a)

    return pd.DataFrame(keep).sort_values(
        ["qe", "neurons", "dead_ratio", "norm_entropy"],
        ascending=[True, True, True, False]
    )

"""
Function to choose one best compromise.
Just one possible scoring rule. It turns each metric into a z-score
so the scales are comparable and then combines them with weights.
A lower score means a better compromise.
"""
def choose_best_compromise(df):
    # z-score based compromise: lower qe, fewer neurons, lower dead_ratio, higher norm_entropy
    out = df.copy()

    # z = (value - mean) / std
    # which tells us how unusually high/low something is
    for col in ["qe", "neurons", "dead_ratio", "norm_entropy"]:
        std = out[col].std(ddof=0)
        if std == 0:
            out[col + "_z"] = 0.0
        else:
            out[col + "_z"] = (out[col] - out[col].mean()) / std

    # weighted score - QE matters most, map size matters second, 
    # ..., normalized entropy helps so we subtract it
    out["score"] = (
        0.50 * out["qe_z"] +
        0.25 * out["neurons_z"] +
        0.15 * out["dead_ratio_z"] -
        0.10 * out["norm_entropy_z"]
    )

    return out.sort_values("score", ascending=True)