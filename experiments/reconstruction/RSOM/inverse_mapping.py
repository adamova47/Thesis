from pathlib import Path
import numpy as np
import pandas as pd


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
        arr = df[["t", "t-taw"]].to_numpy(dtype=np.float32)
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


def run_inverse_mapping(activities, sequences, train_ratio=0.8, ridge=1e-3):
    Y, X = stack_pairs(activities, sequences)
    Y_train, Y_test, X_train, X_test, split = train_test_split_time(Y, X, train_ratio=train_ratio)
    W = fit_ridge_decoder(Y_train, X_train, ridge=ridge)
    X_test_pred = predict_ridge_decoder(Y_test, W)
    test_mse, test_rmse, test_mae = reconstruction_errors(X_test, X_test_pred)
    X_decoded_full = predict_ridge_decoder(Y, W)
    decoded_sequences = split_flat_predictions(X_decoded_full, sequences)

    print("\nRSOM inverse mapping results:")
    print("     decoder: linear ridge regression")
    print(f"    state matrix Y: {Y.shape}")
    print(f"    input matrix X: {X.shape}")
    print(f"    train samples: {len(Y_train)}")
    print(f"    test samples: {len(Y_test)}")
    print(f"    ridge: {ridge}")
    print(f"    test MSE: {test_mse:.6f}")
    print(f"    test RMSE: {test_rmse:.6f}")
    print(f"    test MAE: {test_mae:.6f}")

    print_sequence_metrics(sequences, decoded_sequences)

    return {
        "W": W,
        "Y": Y,
        "X": X,
        "split": split,
        "X_train": X_train,
        "X_test": X_test,
        "X_test_pred": X_test_pred,
        "X_decoded_full": X_decoded_full,
        "input_sequences": sequences,
        "decoded_sequences": decoded_sequences,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }