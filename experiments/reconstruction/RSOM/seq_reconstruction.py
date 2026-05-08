from pathlib import Path
from select_map_RSOM import *
from inverse_mapping import *
import json
import numpy as np

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
    return [(seq - mean) / std for seq in x]


def load_dataset_sequences(dataset, base_dir):
    if dataset == "mackey":
        data_path = base_dir / "mackey_glass" / "mackey_glass.xlsx"
        sequences = load_mackey_glass_sequence(data_path)
        labels = None
        results_path = base_dir / "mackey_glass" / "rsom_results.pkl"
    elif dataset == "nico":
        data_path = base_dir / "nico" / "dataset.json"
        sequences, labels = load_nico_dataset(data_path)
        sequences = normalize_sequences(sequences)
        results_path = base_dir / "nico" / "rsom_nico_results.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return results_path, sequences, labels


def main():
    dataset = "mackey" # change to "mackey" if needed
    show_plots = False
    base_dir = Path(__file__).resolve().parent.parent.parent
    results_path, sequences, labels = load_dataset_sequences(dataset, base_dir)
    results_dict = load_results_dict(results_path)
    df = results_to_dataframe(results_dict)

    print("\nTop 10 by temporal QE:")
    print(df[[
        "m", "n", "neurons", "qe", "entropy", "norm_entropy",
        "dead_neurons", "dead_ratio", "alpha", "beta",
        "init", "metric", "kernel"
    ]].head(10).to_string(index=False))

    short = shortlist_maps(
        df,
        qe_tol=0.05,
        min_norm_entropy=0.70,
        max_dead_ratio=0.25
    )

    print("\nShortlist:")
    if len(short) == 0:
        print("No models passed the shortlist filter.")
        candidate_pool = pareto_front(df)
    else:
        print(short[[
            "m", "n", "neurons", "qe", "entropy", "norm_entropy",
            "dead_neurons", "dead_ratio", "alpha", "beta",
            "init", "metric", "kernel"
        ]].head(15).to_string(index=False))
        candidate_pool = short.copy()

    compromise = choose_best_compromise(candidate_pool)

    print("\nBest compromise candidates:")
    print(compromise[[
        "m", "n", "neurons", "qe", "norm_entropy", "dead_ratio",
        "alpha", "beta", "init", "metric", "kernel", "score"
    ]].head(10).to_string(index=False))

    best = compromise.iloc[0]
    print("\nChosen map:")
    print(best[[
        "m", "n", "neurons", "qe", "entropy", "norm_entropy",
        "dead_neurons", "dead_ratio", "alpha", "beta", "init", "metric", "kernel"
    ]].to_string())
    

    state = best["state"]
    activities = get_activity_trajectories(state)

    print_data_check(activities, sequences)

    inverse_results = run_inverse_mapping(
        activities,
        sequences,
        train_ratio=0.8,
        ridge=1e-3
    )

    decoded = inverse_results["decoded_sequences"][0]
    original = inverse_results["input_sequences"][0]

    print("\nDecoded trajectory:")
    print(f"    original shape: {original.shape}")
    print(f"    decoded shape: {decoded.shape}")
    print("    first 5 decoded points:")
    print(np.round(decoded[:5], 4))

    if labels is not None:
        print("\nLabels:")
        for i, label in enumerate(labels[:10]):
            print(f"    seq {i}: {label}")

    if show_plots:
        plot_inverse_mapping_results(inverse_results)

        weakness = map_weakness_diagnostics(best, inverse_results)
        plot_map_weakness_diagnostics(weakness)


if __name__ == "__main__":
    main()