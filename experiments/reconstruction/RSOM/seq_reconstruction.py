from pathlib import Path
from select_map_RSOM import *
from inverse_mapping import *


def print_reconstruction_summary(best_row):
    state = best_row["state"]


def main():
    results_path = Path(__file__).resolve().parent.parent.parent / "mackey_glass" / "rsom_results.pkl"

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
    

    data_path = Path(__file__).resolve().parent.parent.parent / "mackey_glass" / "mackey_glass.xlsx"

    activities, sequences = prepare_inverse_mapping_data(best, data_path)

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

    plot_inverse_mapping_results(inverse_results)

    weakness = map_weakness_diagnostics(best, inverse_results)
    plot_map_weakness_diagnostics(weakness)


if __name__ == "__main__":
    main()