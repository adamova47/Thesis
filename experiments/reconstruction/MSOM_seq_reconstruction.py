from MSOM_select import *
from MSOM_backwards_decode import *


def print_reconstruction_summary(best_row, top_k_targets=5, depth=5, beam_width=3, proto_length=30, show_plots=True):
    state = best_row["state"]
    m = int(best_row["m"])
    n = int(best_row["n"])
    alpha = float(best_row["alpha"])
    beta = float(best_row["beta"])

    # overall map usage
    if show_plots:
        plot_hit_map(state, m, n)

    # choose target neurons = the most frequently visited BMUs
    targets, counts = top_hit_neurons(state, m, n, top_k=top_k_targets)

    print("\nReconstruction targets (most visited neurons):")
    for idx in targets:
        coord = flat_to_coord(int(idx), n)
        print(f"  neuron {coord} | flat idx={idx} | hits={counts[idx]}")

    for idx in targets:
        coord = flat_to_coord(int(idx), n)
        print(f"\nTarget neuron {coord} (flat idx={idx}, hits={counts[idx]})")
        real_preds = extract_real_predecessors(state, m, n, idx, k=5)
        print("Observed predecessors from saved BMU trajectories:")
        if len(real_preds) == 0:
            print("  none")

        beams = beam_decode_chains(
            state=state,
            m=m,
            n=n,
            alpha=alpha,
            beta=beta,
            target_idx=int(idx),
            depth=depth,
            beam_width=beam_width,
            margin_weight=2.0,
            hard_self_consistent=False,
            allow_self=True,
            avoid_cycles=True,
            add_start_score=True,
        )

        if len(beams) == 0:
            print("There are no decoded chains.")
            continue

        print("\nDecoded predecessor chains:")
        for rank, (chain, score, diag) in enumerate(beams, start=1):
            coords = [flat_to_coord(c, n) for c in chain]
            proto_seq = chain_to_prototype_sequence(state, m, n, chain)

            qe, replay_bmus = validate_prototype(state, m, n, alpha, beta, proto_seq)
            replay_coords = [flat_to_coord(c, n) for c in replay_bmus]

            print(f"    Candidate {rank}:")
            print(f"        chain coords = {coords}")
            print(f"        total score = {score:.6f}")
            print(f"        validation QE = {qe:.6f}")
            print(f"        replay coords = {replay_coords}")
            print(f"        replay match = {list(chain) == list(replay_bmus)}")
            print(f"        prototype sequence shape = {proto_seq.shape}")

            if proto_seq.shape[1] <= 4:
                print("        prototype vectors:")
                for t, vec in enumerate(proto_seq):
                    print(f"          t={t}: {np.round(vec, 4)}")

        best_chain, best_score, best_diag = beams[0]
        best_proto = chain_to_prototype_sequence(state, m, n, best_chain)
        best_qe, best_replay_bmus = validate_prototype(state, m, n, alpha, beta, best_proto)

        print(f"    best validation QE = {best_qe:.6f}")
        print(f"    best replay match = {list(best_chain) == list(best_replay_bmus)}")

        if show_plots:
            plot_real_predecessor_heatmap(state, m, n, idx)
            plot_decoded_chain_on_map(m, n, best_chain, title=f"Best decoded chain for target {coord}\nscore={best_score:.4f}")
            plot_prototype_sequence(best_proto, title=f"Prototype sequence for target {coord}")




def main():
    results_path = Path(__file__).resolve().parent.parent / "mackey_glass" / "msom_results.pkl"
    results_dict = load_results_dict(results_path)
    df = results_to_dataframe(results_dict)

    print("\nTop 10 by temporal QE:")
    print(df[[
        "m", "n", "neurons", "qe", "entropy", "norm_entropy",
        "dead_neurons", "dead_ratio", "alpha", "beta", "init", "metric", "kernel"
    ]].head(10).to_string(index=False))

    # show the filtered shortlist
    short = shortlist_maps(df, qe_tol=0.05, min_norm_entropy=0.70, max_dead_ratio=0.25)
    print("\nShortlist (near-best QE, decent utilization, not too many dead neurons):")
    if len(short) == 0:
        print("No models passed the shortlist filter.")
    else:
        print(short[[
            "m", "n", "neurons", "qe", "entropy", "norm_entropy",
            "dead_neurons", "dead_ratio", "alpha", "beta", "init", "metric", "kernel"
        ]].head(15).to_string(index=False))

    # show the pareto front
    front = pareto_front(df)
    print("\nPareto front:")
    print(front[[
        "m", "n", "neurons", "qe", "norm_entropy", "dead_ratio",
        "alpha", "beta", "init", "metric", "kernel"
    ]].head(20).to_string(index=False))

    if len(short) > 0:
        candidate_pool = short.copy()
    else:
        candidate_pool = pareto_front(df)

    # show the final compromise ranking
    compromise = choose_best_compromise(candidate_pool)
    print("\nBest compromise candidates:")
    print(compromise[[
        "m", "n", "neurons", "qe", "norm_entropy", "dead_ratio",
        "alpha", "beta", "init", "metric", "kernel", "score"
    ]].head(10).to_string(index=False))

    # pick the top row as the chosen map
    best = compromise.iloc[0]
    print("\nChosen map:")
    print(best[[
        "m", "n", "neurons", "qe", "entropy", "norm_entropy",
        "dead_neurons", "dead_ratio", "alpha", "beta", "init", "metric", "kernel"
    ]].to_string())


    """print_reconstruction_summary(
        best_row=best,
        top_k_targets=5, # decode the 5 most visited neurons
        depth=5, # how far back to decode
        beam_width=3 # how many candidate histories to keep
    )"""


if __name__ == "__main__":
    main()