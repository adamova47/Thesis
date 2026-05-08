from backwards_decode import *
from select_map_MSOM import *

import json


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


def compare_to_real_windows(proto_seq, real_sequences, labels=None, top_k=3):
    L = len(proto_seq)
    matches = []
    for seq_id, seq in enumerate(real_sequences):
        if len(seq) < L:
            continue
        for start in range(len(seq) - L + 1):
            window = seq[start:start + L]
            dist = np.mean(np.linalg.norm(proto_seq - window, axis=1))
            matches.append({
                "seq_id": seq_id,
                "label": labels[seq_id] if labels is not None else None,
                "start": start,
                "end": start + L - 1,
                "mean_dist": float(dist),
            })
    matches.sort(key=lambda item: item["mean_dist"])
    return matches[:top_k]


def print_prototype_summary(proto_seq, max_dims=10):
    print(f"        prototype sequence shape = {proto_seq.shape}")
    dims = min(proto_seq.shape[1], max_dims)
    print(f"        prototype vectors first {dims} dims:")
    for t, vec in enumerate(proto_seq):
        print(f"        t={t}: {np.round(vec[:dims], 4)}")


def print_reconstruction_summary(
        best_row,
        top_k_targets=5,
        depth=5,
        beam_width=3,
        show_plots=True,
        print_prototypes=True,
        max_proto_dims=10,
        real_sequences=None,
        labels=None,
        top_real_matches=3,
        verbose_candidates=False,
    ):
    state = best_row["state"]
    m = int(best_row["m"])
    n = int(best_row["n"])
    alpha = float(best_row["alpha"])
    beta = float(best_row["beta"])

    if show_plots:
        plot_hit_map(state, m, n)
    
    targets, counts = top_successor_neurons(
        state,
        m,
        n,
        top_k=top_k_targets,
        min_predecessors=3,
    )

    print("\nReconstruction targets (neurons with most observed predecessors):")
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
        else:
            for pred_coord, count in real_preds:
                print(f"  predecessor {pred_coord} | hits={count}")

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

        if verbose_candidates:
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
                if print_prototypes:
                    print_prototype_summary(proto_seq, max_dims=max_proto_dims)

        best_item = ranked_beams[0]
        best_chain = best_item["chain"]
        best_score = best_item["score"]
        best_proto = best_item["proto_seq"]
        best_qe = best_item["qe"]
        best_soft = best_item["soft"]

        print("Best decoded chain:")
        print(f"  chain coords = {[flat_to_coord(c, n) for c in best_chain]}")
        print(f"  validation QE = {best_qe:.6f}")
        print(f"  replay match = {best_item['exact_match']}")
        print(f"  exact fraction = {best_soft['exact_fraction']:.3f}")
        print(f"  mean grid error = {best_soft['mean_grid_error']:.3f}")
        print(f"  soft replay score = {best_soft['soft_score']:.3f}")
        print(f"  prototype shape = {best_proto.shape}")
        if print_prototypes:
            print_prototype_summary(best_proto, max_dims=max_proto_dims)
        if real_sequences is not None:
            matches = compare_to_real_windows(
                best_proto,
                real_sequences,
                labels=labels,
                top_k=top_real_matches,
            )
            print("  closest real sequence windows:")
            for match in matches:
                print(
                    f"    label={match['label']} | seq={match['seq_id']} | "
                    f"window={match['start']}..{match['end']} | "
                    f"mean dist={match['mean_dist']:.6f}"
                )
        print()

        if show_plots:
            plot_decoded_vs_replay_chain(m, n, best_chain, best_item["replay_bmus"], title=f"Decoded vs replayed chain for target {coord}")
            plot_transition_heatmap(state=state, m=m, n=n, alpha=alpha, beta=beta, target_idx=int(idx))
            plot_real_predecessor_heatmap(state=state, m=m, n=n, target_idx=int(idx))
            if real_sequences is not None and len(matches) > 0:
                best_match = matches[0]
                real_window = real_sequences[best_match["seq_id"]][
                    best_match["start"]:best_match["end"] + 1
                ]
                plot_decoded_vs_closest_real_error(best_proto, real_window, label=best_match["label"], seq_id=best_match["seq_id"], start=best_match["start"],)


def run_reconstruction(
        results_path,
        dataset_name="dataset",
        top_k_targets=5,
        depth=6,
        beam_width=3,
        show_plots=False,
        print_prototypes=True,
        max_proto_dims=10,
        real_sequences=None,
        labels=None,
        top_real_matches=3,
        verbose_candidates=False,
    ):
    results_path = Path(results_path)

    print(f"\nDataset/result set: {dataset_name}")
    print(f"Loading results from: {results_path}")

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
        top_k_targets=top_k_targets,
        depth=depth,
        beam_width=beam_width,
        show_plots=show_plots,
        print_prototypes=print_prototypes,
        max_proto_dims=max_proto_dims,
        real_sequences=real_sequences,
        labels=labels,
        top_real_matches=top_real_matches,
        verbose_candidates=verbose_candidates,
    )


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent

    x_real, labels = load_nico_dataset(base_dir / "nico" / "dataset.json")
    x_real = normalize_sequences(x_real)

    run_reconstruction(
        results_path=base_dir / "nico" / "msom_nico_results.pkl",
        dataset_name="nico",
        top_k_targets=5,
        depth=6,
        beam_width=3,
        show_plots=True,
        print_prototypes=False,
        real_sequences=x_real,
        labels=labels,
        top_real_matches=3,
        verbose_candidates=False,
    )


if __name__ == "__main__":
    main()