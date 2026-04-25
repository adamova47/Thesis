import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def _to_numpy(x):
    # if x is a cupy array, move it from GPU memory to a numpy array on CPU
    # otherwise just convert it to a numpy array if needed
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def unpack_state(state, m, n):
    # reads the trained MSOMs state
    # weights and context weights are stored in grid form with shape (m, n, dim)
    W_grid = _to_numpy(state["weights"])
    C_grid = _to_numpy(state["context_weights"])
    # flatten the 2D map into shape (m * n, dim)
    # this makes later decoding steps easier, because they work over a flat list of neurons
    W = W_grid.reshape(m * n, -1).astype(np.float32, copy=False)
    C = C_grid.reshape(m * n, -1).astype(np.float32, copy=False)
    # we return both the grid and flat versions of the weights because some 
    # visualizations are easier with the grid form, and decoding is easier with the flat form
    return W_grid, C_grid, W, C


def flat_to_coord(idx, n_cols):
    # converts a flat neuron index into its 2D map coordinate (row, column)
    # echample: for n_cols=10, flat idx=23 -> coord=(2, 3) because it's in the 3rd row (0-based) and 4th column (0-based)
    return (idx // n_cols, idx % n_cols)


def coord_to_flat(coord, n_cols):
    # converts a 2D map coordinate (row, column) back into a flat neuron index
    # echample: for n_cols=10, coord=(2, 3) -> flat idx=23 because it's in the 3rd row (0-based) and 4th column (0-based)
    i, j = coord
    return i * n_cols + j


def pairwise_sqdist(A, B):
    # computes all pairwise squared Euclidean distances between rows of A and rows of B
    # if A has shape (NA, dim) and B has shape (NB, dim)
    # the result D has shape (NA, NB), where D[i, j] = ||A[i] - B[j]||^2
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    ab = A @ B.T
    D = aa + bb - 2.0 * ab
    # small negative values can appear from floating-point roundoff
    # so we clamp them to zero because squared distances should not be negative
    return np.maximum(D, 0.0)


def count_bmu_hits(state, m, n):
    # counts how many times each neuron appears in the saved BMU trajectories
    # result is a flat array of length m * n
    counts = np.zeros(m * n, dtype=np.int32)
    # each trajectory is is stored as a sequence of 2D neuron coordinates
    for seq in state["bmu_trajectories"]:
        for coord in seq:
            # converts each (row, col) coordinate into a flat neuron index
            # so we can count visits in a 1D array
            counts[coord_to_flat(coord, n)] += 1
    return counts


def precompute_transition_tables(state, m, n, beta):
    # loads the trained map state in flat form
    _, _, W, C = unpack_state(state, m, n)
    # build the generated context for every possible predecessor neuron
    # if neuron i was the previous BMU, this is the context it would produce
    # at the next step under the MSOM context rule
    G = (1.0 - beta) * W + beta * C
    # pairwise distance between all main prototypes
    # input_dists[j, k] = ||W[j] - W[k]||^2
    input_dists = pairwise_sqdist(W, W)
    # pairwise distances between generated predecessor contexts and stored contexts
    # ctx_dists[i, k] = ||G[i] - C[k]||^2
    ctx_dists = pairwise_sqdist(G, C)
    # since the tables are used a lot in decoding its better to precompute them
    return W, C, input_dists, ctx_dists


def transition_scores_to_target(
        target_j,
        input_dists,
        ctx_dists,
        alpha,
        margin_weight=2.0,
        hard_self_consistent=False
    ):
    # this scores all possible predecessor neurons for one fixed target neuron
    # for each possible predecessor i, we compute whether if i came before the current step, which neuron k
    # would win next under the merged MSOM distance
    # first term comes from input prototype similarity and second from context similarity
    all_costs = (1.0 - alpha) * input_dists[target_j][None, :] + alpha * ctx_dists
    # for each predecessor i, find the best next winner k
    winners = np.argmin(all_costs, axis=1)
    best_cost = np.min(all_costs, axis=1)
    # measure how well the chosen target fits as next state
    # this is the target-specific context term for each predecessor
    self_cost = alpha * ctx_dists[:, target_j]
    # if self_cost is much worse than the best achievable merged cost 
    # then target_j is a weak next-state choice for that predecessor
    gap = self_cost - best_cost
    # margin_weight dedcides how much we penalize predecessors
    # that do not make target_j competitive against other next winners
    scores = self_cost + margin_weight * gap
    # only keep predecessors that make target_j the actual best next winner
    if hard_self_consistent:
        scores[winners != target_j] = np.inf
    return scores, winners, best_cost, self_cost, gap


def start_scores(alpha, W, C, input_dists, margin_weight=2.0, hard_self_consistent=False):
    # scores each neuron as a possible start of a decoded chain 
    # at sequence start, MSOM uses a zero context vector
    # (if decoding begins from zero context. how plausible is each neuron j as the first winner?)
    zero_ctx = np.zeros(W.shape[1], dtype=W.dtype)
    # distance from zero start-context to each stored context prototype
    # start_ctx_dists[j] = ||C[j] - 0||^2
    start_ctx_dists = np.sum((C - zero_ctx) ** 2, axis=1)
    N = W.shape[0]
    scores = np.empty(N, dtype=np.float32)
    winners = np.empty(N, dtype=np.int32)
    for j in range(N):
        # merged cost for choosing neuron j as the start and comparing it against all possible winners
        all_costs = (1.0 - alpha) * input_dists[j] + alpha * start_ctx_dists
        # best winner under that start-state setup
        winner = int(np.argmin(all_costs))
        best = float(np.min(all_costs))
        # cost of making j itself the chosen start neuron
        self_cost = float(alpha * start_ctx_dists[j])
        # penalize j if some other neuron fits the start better
        gap = self_cost - best
        s = self_cost + margin_weight * gap
        # only keep j if it is its own best winner (is optional really)
        if hard_self_consistent and winner != j:
            s = np.inf
        scores[j] = s
        winners[j] = winner
    return scores, winners


def beam_decode_chains(
        state,
        m,
        n,
        alpha,
        beta,
        target_idx,
        depth=5,
        beam_width=5,
        margin_weight=2.0, # 1.0 = pure score, >1 penalises weak predecessors
        hard_self_consistent=False,
        allow_self=True,
        avoid_cycles=True,
        add_start_score=True,
    ):
    # precomputes the distance tables used for decoding
    W, C, input_dists, ctx_dists = precompute_transition_tables(state, m, n, beta)
    # cache target specific scores so we dont have to recompute
    cache = {}
    def get_scores(j):
        if j not in cache:
            cache[j] = transition_scores_to_target(
                target_j=j,
                input_dists=input_dists,
                ctx_dists=ctx_dists,
                alpha=alpha,
                margin_weight=margin_weight,
                hard_self_consistent=hard_self_consistent,
            )
        return cache[j]
    # each beam stores: (decoded chain, total score, diagnostics)
    beams = [([int(target_idx)], 0.0, [])]
    # grow the chain backwards one step at a time
    for _ in range(depth):
        new_beams = []
        for chain, total_score, diagnostics in beams:
            # first element is the current front of the chain
            current_j = chain[0]
            # score all possible predecessors for this target
            scores, winners, best_cost, self_cost, gap = get_scores(current_j)
            # try the best-scoring prededcessor candidates first
            candidate_preds = np.argsort(scores)[: max(beam_width * 10, 20)]
            for pred in candidate_preds:
                pred = int(pred)
                # block immediate self-transitions (also optional)
                if not allow_self and pred == current_j:
                    continue
                # block loops inside the decoded chain (also optional)
                if avoid_cycles and pred in chain:
                    continue
                # skip invalid candidates, for example filtered by hard_self_consistent 
                if not np.isfinite(scores[pred]):
                    continue
                # build the chain by preprending the predecessor
                new_chain = [pred] + chain
                # save extra info for inspection
                new_diag = diagnostics + [{
                    "pred": pred,
                    "next": current_j,
                    "score": float(scores[pred]),
                    "winner_for_next_proto": int(winners[pred]),
                    "self_cost": float(self_cost[pred]),
                    "best_cost": float(best_cost[pred]),
                    "gap": float(gap[pred]),
                }]
                new_beams.append((new_chain, total_score + float(scores[pred]), new_diag))
        # stop early if no valid expansion remains
        if not new_beams:
            break
        # keep only the best beam_width chains
        new_beams.sort(key=lambda x: x[1])
        beams = new_beams[:beam_width]

    # add a start-state score using zero context (once again optional)
    if add_start_score and beams:
        s0_all, w0_all = start_scores(
            alpha=alpha,
            W=W,
            C=C,
            input_dists=input_dists,
            margin_weight=margin_weight,
            hard_self_consistent=hard_self_consistent,
        )
        rescored = []
        for chain, total_score, diagnostics in beams:
            first = chain[0]
            s0 = float(s0_all[first])
            if np.isfinite(s0):
                diag2 = diagnostics + [{
                    "pred": None,
                    "next": first,
                    "score": s0,
                    "winner_for_next_proto": int(w0_all[first]),
                    "note": "START -> first",
                }]
                rescored.append((chain, total_score + s0, diag2))
        # re-rank after adding the start-state term
        if rescored:
            rescored.sort(key=lambda x: x[1])
            beams = rescored[:beam_width]
    return beams


def chain_to_prototype_sequence(state, m, n, chain):
    # converts a decoded neuron chain into the corresponding sequence of prototypes
    _, _, W, _ = unpack_state(state, m, n)
    return W[np.array(chain)]


def validate_prototype(state, m, n, alpha, beta, proto_seq):
    # replay a prototype sequence through the MSOM matching rule
    # return the temporal QE and the BMUs found along the way
    _, _, W, C = unpack_state(state, m, n)
    qe = 0.0
    bmus = []
    # sequence start uses zero context
    C_t = np.zeros(W.shape[1], dtype=W.dtype)
    for x in proto_seq:
        # input and context distance to every neuron
        d_x = np.sum((W - x) ** 2, axis=1)
        d_c = np.sum((C - C_t) ** 2, axis=1)
        # merged MSOM distance
        d = (1.0 - alpha) * d_x + alpha * d_c
        # best matching neuron for this prototype
        bmu = int(np.argmin(d))
        # accumulate temporal quantization error
        qe += np.sqrt((1.0 - alpha) * d_x[bmu] + alpha * d_c[bmu])
        bmus.append(bmu)
        # update context from the winning neuron
        C_t = (1.0 - beta) * W[bmu] + beta * C[bmu]
    return float(qe / len(proto_seq)), bmus


def top_hit_neurons(state, m, n, top_k=5):
    # returns the most frequently visited neurons from the saved BMU trajectories
    counts = count_bmu_hits(state, m, n)
    top = np.argsort(counts)[::-1][:top_k]
    return top, counts


def extract_real_predecessors(state, m, n, target_flat_idx, k=5):
    # looks through saved BMU trajectories and collect which neurons 
    # appeared immediately before the chosen target neuron
    predecessors = []
    for seq in state["bmu_trajectories"]:
        for t in range(1, len(seq)):
            if coord_to_flat(seq[t], n) == target_flat_idx:
                predecessors.append(seq[t - 1])
    # return the most common observed predecessors 
    return Counter(predecessors).most_common(k)


def plot_hit_map(state, m, n):
    hits = count_bmu_hits(state, m, n).reshape(m, n)
    plt.figure(figsize=(5, 4))
    plt.imshow(hits, origin="lower")
    plt.colorbar(label="BMU hits")
    plt.title("Neuron hit map")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.show()


def plot_transition_heatmap(
        state,
        m,
        n,
        alpha,
        beta,
        target_idx,
        margin_weight=2.0,
        hard_self_consistent=False,
    ):
    _, _, input_dists, ctx_dists = precompute_transition_tables(state, m, n, beta)
    scores, _, _, _, _ = transition_scores_to_target(
        target_j=target_idx,
        input_dists=input_dists,
        ctx_dists=ctx_dists,
        alpha=alpha,
        margin_weight=margin_weight,
        hard_self_consistent=hard_self_consistent,
    )
    heat = scores.reshape(m, n)
    plt.figure(figsize=(5, 4))
    plt.imshow(heat, origin="lower")
    plt.colorbar(label="Transition score")
    plt.title(f"Predecessor scores for target {flat_to_coord(target_idx, n)}")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.show()


def plot_real_predecessor_heatmap(state, m, n, target_idx):
    heat = np.zeros((m, n), dtype=np.int32)
    for seq in state["bmu_trajectories"]:
        for t in range(1, len(seq)):
            if coord_to_flat(seq[t], n) == target_idx:
                pi, pj = seq[t - 1]
                heat[pi, pj] += 1
    plt.figure(figsize=(5, 4))
    plt.imshow(heat, origin="lower")
    plt.colorbar(label="Observed predecessor count")
    plt.title(f"Observed predecessors for target {flat_to_coord(target_idx, n)}")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.show()


def plot_decoded_chain_on_map(m, n, chain, title="Decoded chain"):
    coords = [flat_to_coord(c, n) for c in chain]
    xs = [c[1] for c in coords]
    ys = [c[0] for c in coords]
    plt.figure(figsize=(5, 5))
    plt.xlim(-0.5, n - 0.5)
    plt.ylim(-0.5, m - 0.5)
    plt.grid(True, alpha=0.3)
    plt.plot(xs, ys, "o-")
    for t, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x + 0.05, y + 0.05, str(t), fontsize=8)
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.gca().set_aspect("equal")
    plt.show()


def plot_prototype_sequence(proto_seq, title="Decoded prototype sequence"):
    if proto_seq.shape[1] < 2:
        raise ValueError("plot_prototype_sequence needs dim >= 2")
    plt.figure(figsize=(5, 4))
    plt.plot(proto_seq[:, 0], proto_seq[:, 1], "o-")
    for t, (x, y) in enumerate(proto_seq[:, :2]):
        plt.text(x, y, str(t), fontsize=8)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()