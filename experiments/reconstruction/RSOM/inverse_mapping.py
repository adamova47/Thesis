from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.utils import smart_normalize


def to_numpy(x):
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def get_activity_trajectories(state):
    return [to_numpy(a).astype(np.float32) for a in state["activity_trajectories"]]


def load_mackey_glass_sequence(path):
    sheets = pd.read_excel(Path(path), sheet_name=None)
    sequences = []
    for _, df in sheets.items():
        x = df[["t", "t-taw"]].values
        x = smart_normalize(x)
        arr = np.asarray(x, dtype=np.float32)
        sequences.append(arr)
    return sequences


def print_data_check(activities, sequences):
    print("\nInverse mapping data:")
    print(f"    activity trajectories: {len(activities)}")
    print(f"    input sequences: {len(sequences)}")
    for i in range(min(len(activities), len(sequences))):
        print(f"    seq {i}: activity {activities[i].shape} | input {sequences[i].shape}")


def prepare_inverse_mapping_data(best_row, data_path):
    state = best_row["state"]
    activities = get_activity_trajectories(state)
    sequences = load_mackey_glass_sequence(data_path)
    print_data_check(activities, sequences)
    return activities, sequences


def stack_pairs(activities, sequences):
    Y = np.vstack(activities).astype(np.float64)
    X = np.vstack(sequences).astype(np.float64)
    if len(Y) != len(X):
        raise ValueError(f"Length mismatch: activities={len(Y)}, inputs={len(X)}")
    return Y, X


def train_test_split_time(Y, X, train_ratio=0.8):
    split = int(train_ratio * len(Y))
    return (
        Y[:split],
        Y[split:],
        X[:split],
        X[split:],
        split,
    )


def fit_ridge_decoder(Y_train, X_train, ridge=1e-3):
    """
    Linear inverse decoder: RSOM activity state -> original input vector
    Learns W such that: [Y_train, 1] @ W ≈ X_train
    """
    ones = np.ones((Y_train.shape[0], 1))
    A = np.hstack([Y_train, ones])
    I = np.eye(A.shape[1])
    I[-1, -1] = 0.0
    W = np.linalg.solve(A.T @ A + ridge * I, A.T @ X_train)
    return W


def predict_ridge_decoder(Y, W):
    ones = np.ones((Y.shape[0], 1))
    A = np.hstack([Y, ones])
    return A @ W


def reconstruction_errors(X_true, X_pred):
    err = X_true - X_pred
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))
    return mse, rmse, mae


def split_flat_predictions(flat_predictions, sequences):
    decoded_sequences = []
    start = 0
    for seq in sequences:
        end = start + len(seq)
        decoded_sequences.append(flat_predictions[start:end])
        start = end
    return decoded_sequences


def print_sequence_metrics(sequences, decoded_sequences):
    print("\nDecoded trajectory metrics per sequence:")
    for i, (x_true, x_pred) in enumerate(zip(sequences, decoded_sequences)):
        mse, rmse, mae = reconstruction_errors(x_true, x_pred)
        print(
            f"seq {i}: "
            f"MSE={mse:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f}"
        )


def signal_scale(X):
    """
    Returns simple scale information for the reconstruction target.
    """
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    ranges = maxs - mins
    return mins, maxs, ranges


def mean_baseline(X_train, X_test):
    """
    Baseline decoder:
    Always predicts the mean of the training data.
    """
    mean_x = np.mean(X_train, axis=0)
    return np.tile(mean_x, (len(X_test), 1))


def normalized_errors(rmse, mae, ranges):
    """
    Express RMSE and MAE relative to the average signal range.
    """
    avg_range = np.mean(ranges)
    nrmse = rmse / avg_range
    nmae = mae / avg_range
    return nrmse, nmae


def run_inverse_mapping(activities, sequences, train_ratio=0.8, ridge=1e-3):
    Y, X = stack_pairs(activities, sequences)

    Y_train, Y_test, X_train, X_test, split = train_test_split_time(
        Y, X, train_ratio=train_ratio
    )

    # Train ridge decoder
    W = fit_ridge_decoder(Y_train, X_train, ridge=ridge)

    # Held-out test prediction
    X_test_pred = predict_ridge_decoder(Y_test, W)
    test_mse, test_rmse, test_mae = reconstruction_errors(X_test, X_test_pred)

    # Mean baseline prediction
    X_base_pred = mean_baseline(X_train, X_test)
    base_mse, base_rmse, base_mae = reconstruction_errors(X_test, X_base_pred)

    # Scale information
    x_min, x_max, x_range = signal_scale(X)
    test_nrmse, test_nmae = normalized_errors(test_rmse, test_mae, x_range)
    base_nrmse, base_nmae = normalized_errors(base_rmse, base_mae, x_range)

    # Full decoded trajectory, useful for visualization and later analysis
    X_decoded_full = predict_ridge_decoder(Y, W)
    decoded_sequences = split_flat_predictions(X_decoded_full, sequences)

    print("\nSignal scale:")
    print(f"    input min:   {np.round(x_min, 6)}")
    print(f"    input max:   {np.round(x_max, 6)}")
    print(f"    input range: {np.round(x_range, 6)}")
    print(f"    avg range:   {np.mean(x_range):.6f}")

    print("\nBaseline results:")
    print("    baseline: training mean")
    print(f"    test MSE:   {base_mse:.6f}")
    print(f"    test RMSE:  {base_rmse:.6f}")
    print(f"    test MAE:   {base_mae:.6f}")
    print(f"    test NRMSE: {base_nrmse:.6f}")
    print(f"    test NMAE:  {base_nmae:.6f}")

    print("\nRSOM inverse mapping results:")
    print("    decoder: linear ridge regression")
    print(f"    state matrix Y: {Y.shape}")
    print(f"    input matrix X: {X.shape}")
    print(f"    train samples: {len(Y_train)}")
    print(f"    test samples:  {len(Y_test)}")
    print(f"    ridge: {ridge}")
    print(f"    test MSE:   {test_mse:.6f}")
    print(f"    test RMSE:  {test_rmse:.6f}")
    print(f"    test MAE:   {test_mae:.6f}")
    print(f"    test NRMSE: {test_nrmse:.6f}")
    print(f"    test NMAE:  {test_nmae:.6f}")

    print("\nImprovement over baseline:")
    print(f"    RMSE reduction: {(1.0 - test_rmse / base_rmse) * 100:.2f}%")
    print(f"    MAE reduction:  {(1.0 - test_mae / base_mae) * 100:.2f}%")

    print_sequence_metrics(sequences, decoded_sequences)

    return {
        "W": W,
        "Y": Y,
        "X": X,
        "split": split,

        "X_train": X_train,
        "X_test": X_test,

        "X_test_pred": X_test_pred,
        "X_base_pred": X_base_pred,

        "X_decoded_full": X_decoded_full,
        "input_sequences": sequences,
        "decoded_sequences": decoded_sequences,

        "x_min": x_min,
        "x_max": x_max,
        "x_range": x_range,

        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_nrmse": test_nrmse,
        "test_nmae": test_nmae,

        "base_mse": base_mse,
        "base_rmse": base_rmse,
        "base_mae": base_mae,
        "base_nrmse": base_nrmse,
        "base_nmae": base_nmae,
    }


def get_sequence_split_indices(sequences, global_split):
    split_indices = []
    start = 0
    for seq in sequences:
        end = start + len(seq)
        if global_split <= start:
            local_split = 0
        elif global_split >= end:
            local_split = len(seq)
        else:
            local_split = global_split - start
        split_indices.append(local_split)
        start = end
    return split_indices


def plot_sequence_reconstruction(original, decoded, seq_idx=0, split_idx=None):
    t = np.arange(len(original))
    dims = original.shape[1]
    for d in range(dims):
        plt.figure(figsize=(12, 4))
        plt.plot(t, original[:, d], label=f"original dim {d}")
        plt.plot(t, decoded[:, d], "--", label=f"decoded dim {d}")
        if split_idx is not None and 0 < split_idx < len(original):
            plt.axvline(split_idx, linestyle=":", label="train/test split")
        plt.title(f"Sequence {seq_idx} - original vs decoded (dim {d})")
        plt.xlabel("time step")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.show()
    abs_err = np.abs(original - decoded)
    for d in range(dims):
        plt.figure(figsize=(12, 4))
        plt.plot(t, abs_err[:, d], label=f"|error| dim {d}")
        if split_idx is not None and 0 < split_idx < len(original):
            plt.axvline(split_idx, linestyle=":", label="train/test split")
        plt.title(f"Sequence {seq_idx} - absolute error (dim {d})")
        plt.xlabel("time step")
        plt.ylabel("absolute error")
        plt.legend()
        plt.tight_layout()
        plt.show()
    if dims == 2:
        plt.figure(figsize=(6, 6))
        plt.plot(original[:, 0], original[:, 1], label="original trajectory")
        plt.plot(decoded[:, 0], decoded[:, 1], "--", label="decoded trajectory")
        plt.scatter(original[0, 0], original[0, 1], marker="o", label="start original")
        plt.scatter(decoded[0, 0], decoded[0, 1], marker="x", label="start decoded")
        plt.title(f"Sequence {seq_idx} - trajectory in input space")
        plt.xlabel("dim 0")
        plt.ylabel("dim 1")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_inverse_mapping_results(results):
    sequences = results["input_sequences"]
    decoded_sequences = results["decoded_sequences"]
    split = results["split"]
    split_indices = get_sequence_split_indices(sequences, split)
    for i, (original, decoded) in enumerate(zip(sequences, decoded_sequences)):
        plot_sequence_reconstruction(
            original,
            decoded,
            seq_idx=i,
            split_idx=split_indices[i]
        )

def coord_to_flat(coord, n_cols):
    return int(coord[0]) * n_cols + int(coord[1])


def flatten_bmu_trajectories(bmu_trajectories, n_cols):
    flat = []

    for traj in bmu_trajectories:
        flat.extend([coord_to_flat(coord, n_cols) for coord in traj])

    return np.asarray(flat, dtype=int)


def per_timestep_reconstruction_error(X_true, X_pred):
    """
    Euclidean reconstruction error per timestep.
    """
    return np.linalg.norm(X_true - X_pred, axis=1)


def per_timestep_input_qe(X_true, bmu_flat, weights):
    """
    Input quantization error per timestep:

        ||x(t) - w_BMU(t)||

    This checks whether high decoder error happens where the RSOM
    input prototype itself is weak.
    """
    prototypes = weights.reshape(-1, weights.shape[-1])
    bmu_prototypes = prototypes[bmu_flat]
    return np.linalg.norm(X_true - bmu_prototypes, axis=1)


def summarize_error_by_neuron(values, bmu_flat, n_neurons):
    """
    Groups timestep-level values by BMU.
    Example values:
        reconstruction error per timestep
        input QE per timestep
    """
    sums = np.zeros(n_neurons, dtype=np.float64)
    counts = np.zeros(n_neurons, dtype=np.int64)

    for bmu, value in zip(bmu_flat, values):
        sums[bmu] += value
        counts[bmu] += 1

    means = np.full(n_neurons, np.nan, dtype=np.float64)
    used = counts > 0
    means[used] = sums[used] / counts[used]

    return means, counts


def safe_corr(a, b):
    """
    Correlation ignoring NaNs.
    """
    mask = np.isfinite(a) & np.isfinite(b)

    if np.sum(mask) < 2:
        return np.nan

    return np.corrcoef(a[mask], b[mask])[0, 1]


def map_weakness_diagnostics(best_row, inverse_results):
    """
    Tests whether reconstruction errors correlate with map weakness.
    Map weakness indicators: low BMU hit count, high input quantization error, specific neurons having high mean reconstruction error
    """
    state = best_row["state"]
    m = int(best_row["m"])
    n = int(best_row["n"])
    n_neurons = m * n

    X = inverse_results["X"]
    X_decoded = inverse_results["X_decoded_full"]

    weights = to_numpy(state["weights"]).astype(np.float64)
    bmu_flat = flatten_bmu_trajectories(state["bmu_trajectories"], n)

    if len(bmu_flat) != len(X):
        raise ValueError(
            f"BMU length mismatch: bmus={len(bmu_flat)}, inputs={len(X)}"
        )

    recon_err = per_timestep_reconstruction_error(X, X_decoded)
    input_qe = per_timestep_input_qe(X, bmu_flat, weights)

    mean_recon_by_neuron, hit_counts = summarize_error_by_neuron(
        recon_err, bmu_flat, n_neurons
    )

    mean_qe_by_neuron, _ = summarize_error_by_neuron(
        input_qe, bmu_flat, n_neurons
    )

    # correlations over visited neurons
    corr_recon_qe = safe_corr(mean_recon_by_neuron, mean_qe_by_neuron)

    # hit count correlation uses only visited neurons
    hit_counts_float = hit_counts.astype(np.float64)
    hit_counts_float[hit_counts == 0] = np.nan
    corr_recon_hits = safe_corr(mean_recon_by_neuron, hit_counts_float)

    visited = hit_counts > 0
    dead = hit_counts == 0

    print("\nMap weakness diagnostics:")
    print(f"    visited neurons: {np.sum(visited)} / {n_neurons}")
    print(f"    dead neurons in saved trajectory: {np.sum(dead)} / {n_neurons}")
    print(f"    mean timestep reconstruction error: {np.mean(recon_err):.6f}")
    print(f"    max timestep reconstruction error:  {np.max(recon_err):.6f}")
    print(f"    mean timestep input QE:             {np.mean(input_qe):.6f}")
    print(f"    max timestep input QE:              {np.max(input_qe):.6f}")
    print(f"    corr(mean recon error, mean input QE): {corr_recon_qe:.6f}")
    print(f"    corr(mean recon error, hit count):     {corr_recon_hits:.6f}")

    # show highest-error neurons
    ranked = np.argsort(np.nan_to_num(mean_recon_by_neuron, nan=-1.0))[::-1]

    print("\nHighest mean reconstruction-error neurons:")
    shown = 0
    for idx in ranked:
        if hit_counts[idx] == 0:
            continue

        coord = (idx // n, idx % n)
        print(
            f"    neuron {coord} | flat={idx} | "
            f"hits={hit_counts[idx]} | "
            f"mean recon err={mean_recon_by_neuron[idx]:.6f} | "
            f"mean input QE={mean_qe_by_neuron[idx]:.6f}"
        )

        shown += 1
        if shown >= 10:
            break

    return {
        "bmu_flat": bmu_flat,
        "recon_err": recon_err,
        "input_qe": input_qe,
        "mean_recon_by_neuron": mean_recon_by_neuron,
        "mean_qe_by_neuron": mean_qe_by_neuron,
        "hit_counts": hit_counts,
        "corr_recon_qe": corr_recon_qe,
        "corr_recon_hits": corr_recon_hits,
        "m": m,
        "n": n,
    }


def plot_map_weakness_diagnostics(diag):
    m = diag["m"]
    n = diag["n"]

    hit_map = diag["hit_counts"].reshape(m, n)
    recon_map = diag["mean_recon_by_neuron"].reshape(m, n)
    qe_map = diag["mean_qe_by_neuron"].reshape(m, n)

    plt.figure(figsize=(6, 5))
    plt.imshow(hit_map)
    plt.title("BMU hit count per neuron")
    plt.colorbar(label="hits")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(recon_map)
    plt.title("Mean reconstruction error per BMU")
    plt.colorbar(label="mean reconstruction error")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(qe_map)
    plt.title("Mean input QE per BMU")
    plt.colorbar(label="mean input QE")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.show()

    # scatter: input QE vs reconstruction error
    x = diag["mean_qe_by_neuron"]
    y = diag["mean_recon_by_neuron"]
    hits = diag["hit_counts"]

    mask = np.isfinite(x) & np.isfinite(y) & (hits > 0)

    plt.figure(figsize=(6, 5))
    plt.scatter(x[mask], y[mask])
    plt.title("Neuron-wise QE vs reconstruction error")
    plt.xlabel("mean input QE")
    plt.ylabel("mean reconstruction error")
    plt.tight_layout()
    plt.show()

    # Scatter: hit count vs reconstruction error
    plt.figure(figsize=(6, 5))
    plt.scatter(hits[mask], y[mask])
    plt.title("Neuron hit count vs reconstruction error")
    plt.xlabel("hit count")
    plt.ylabel("mean reconstruction error")
    plt.tight_layout()
    plt.show()

    # timestep-level diagnostic
    t = np.arange(len(diag["recon_err"]))

    plt.figure(figsize=(12, 4))
    plt.plot(t, diag["recon_err"], label="reconstruction error")
    plt.plot(t, diag["input_qe"], "--", label="input QE")
    plt.title("Timestep reconstruction error vs input QE")
    plt.xlabel("time step")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.show()