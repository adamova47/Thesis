from backwards_decode import *
from select_map_MSOM import *


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

        # validate and soft-score every decoded candidate
        ranked_beams = []

        for chain, score, diag in beams:
            proto_seq = chain_to_prototype_sequence(state, m, n, chain)
            qe, replay_bmus = validate_prototype(state, m, n, alpha, beta, proto_seq)
            soft = soft_replay_match(chain, replay_bmus, n)

            ranked_beams.append({
                "chain": chain,
                "score": score,
                "diag": diag,
                "proto_seq": proto_seq,
                "qe": qe,
                "replay_bmus": replay_bmus,
                "soft": soft,
                "exact_match": list(chain) == list(replay_bmus),
            })

        # we rerank by what we actually care about: 
        # exact replay match > soft replay quality > validation QE > original decoder score
        ranked_beams.sort(
            key=lambda item: (
                not item["exact_match"],
                -item["soft"]["soft_score"],
                item["soft"]["mean_grid_error"],
                item["qe"],
                item["score"],
            )
        )

        seen = set()
        unique = []
        for item in ranked_beams:
            chain_tuple = tuple(item["chain"])
            if chain_tuple not in seen:
                seen.add(chain_tuple)
                unique.append(item)

        ranked_beams = unique  

        print("\nDecoded predecessor chains:")
        for rank, item in enumerate(ranked_beams, start=1):
            chain = item["chain"]
            score = item["score"]
            proto_seq = item["proto_seq"]
            qe = item["qe"]
            replay_bmus = item["replay_bmus"]
            soft = item["soft"]
            coords = [flat_to_coord(c, n) for c in chain]
            replay_coords = [flat_to_coord(c, n) for c in replay_bmus]
            print(f"    Candidate {rank}:")
            print(f"        chain coords = {coords}")
            print(f"        total score = {score:.6f}")
            print(f"        validation QE = {qe:.6f}")
            print(f"        replay coords = {replay_coords}")
            print(f"        replay match = {item['exact_match']}")
            print(f"        exact fraction = {soft['exact_fraction']:.3f}")
            print(f"        mean grid error = {soft['mean_grid_error']:.3f}")
            print(f"        max grid error = {soft['max_grid_error']}")
            print(f"        soft replay score = {soft['soft_score']:.3f}")
            print(f"        step grid errors = {soft['step_grid_errors']}")
            print(f"        prototype sequence shape = {proto_seq.shape}")
            if proto_seq.shape[1] <= 4:
                print("        prototype vectors:")
                for t, vec in enumerate(proto_seq):
                    print(f"          t={t}: {np.round(vec, 4)}")

        # best chain is the best replay-consistent candidate
        best_item = ranked_beams[0]
        best_chain = best_item["chain"]
        best_score = best_item["score"]
        best_proto = best_item["proto_seq"]
        best_qe = best_item["qe"]
        best_replay_bmus = best_item["replay_bmus"]
        best_soft = best_item["soft"]

        print(f"best validation QE = {best_qe:.6f}")
        print(f"best replay match = {best_item['exact_match']}")
        print(f"best exact fraction = {best_soft['exact_fraction']:.3f}")
        print(f"best mean grid error = {best_soft['mean_grid_error']:.3f}")
        print(f"best max grid error = {best_soft['max_grid_error']}")
        print(f"best soft replay score = {best_soft['soft_score']:.3f}")
        print(f"best step grid errors = {best_soft['step_grid_errors']}\n")


        if show_plots:
            plot_real_predecessor_heatmap(state, m, n, idx)
            plot_decoded_chain_on_map(m, n, best_chain, 
                                      title=f"Best decoded chain for target {coord}\nscore={best_score:.4f}")
            plot_prototype_sequence(best_proto, title=f"Prototype sequence for target {coord}")


def main():
    results_path = Path(__file__).resolve().parent.parent.parent / "mackey_glass" / "msom_results.pkl"
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


    print_reconstruction_summary(
        best_row=best,
        top_k_targets=5, # decode the 5 most visited neurons
        depth=6, # how far back to decode
        beam_width=3, # how many candidate histories to keep
        show_plots=False
    )


if __name__ == "__main__":
    main()