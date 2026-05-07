import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def smart_normalize(data):
    # Check for constant or near constant features: MinMax handles them well
    std_devs = np.std(data, axis=0)
    if np.any(std_devs < 1e-10):
        return MinMaxScaler().fit_transform(data)
    
    # Check for outliers using IQR: RobustScaler is robust to outliers
    Q1, Q3 = np.percentile(data, [25, 75], axis=0)
    IQR = Q3 - Q1
    valid_iqr = IQR > 0
    outlier_mask = np.zeros_like(data, dtype=bool)
    if np.any(valid_iqr):
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for i in range(data.shape[1]):
            if valid_iqr[i]:
                outlier_mask[:, i] = (data[:, i] < lower_bound[i]) | (data[:, i] > upper_bound[i])
        outlier_ratio = np.mean(outlier_mask)
        if outlier_ratio > 0.05:  # More than 5% outliers
            return RobustScaler().fit_transform(data)
    
    # Default to z-score normalization
    return StandardScaler().fit_transform(data)


def to_cpu(obj):
    if isinstance(obj, cp.ndarray):
        return cp.asnumpy(obj)
    if isinstance(obj, list):
        return [to_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


def plot_quantization_error(som):
    if hasattr(som.q_error_history[0], "get"):
        q_errors = [err.get() for err in som.q_error_history]
    else:
        q_errors = som.q_error_history

    epochs = range(1, len(q_errors) + 1)
    plt.figure()
    plt.plot(epochs, q_errors, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Quantization error")
    plt.title("QE vs. epoch")
    plt.grid(alpha=0.4)

def plot_temporal_quantization_error(xsom):
    if hasattr(xsom.temporal_q_error_history[0], "get"):
        q_errors = [err.get() for err in xsom.temporal_q_error_history]
    else:
        q_errors = xsom.temporal_q_error_history

    epochs = range(1, len(q_errors) + 1)
    plt.figure()
    plt.plot(epochs, q_errors, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Temporal quantization error")
    plt.title("TQE vs. epoch")
    plt.grid(alpha=0.4)

def plot_avg_adjustment(som):
    if hasattr(som.avg_adjust_history[0], 'get'):
        adjustments = [adj.get() for adj in som.avg_adjust_history]
    else:
        adjustments = som.avg_adjust_history
    
    epochs = range(1, len(adjustments) + 1)
    plt.figure()
    plt.plot(epochs, adjustments, linestyle='-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Mean |Δw|')
    plt.title('Avg. weight adjustment vs. epoch')
    plt.grid(alpha=0.4)

def plot_context_norms(rsom):
    vals = rsom.context_norms
    vals = vals.get() if hasattr(vals, "get") else vals
    plt.plot(vals, linestyle='-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('|Context vector|')
    plt.title('Context vector norm vs. epoch')
    plt.grid(alpha=0.4)

def plot_winner_map(som, X, y):
    m, n = som.m, som.n
    num_classes = len(np.unique(y))

    X_cp = X if hasattr(X, "device") else cp.asarray(X)

    class_counts = np.zeros((m, n, num_classes))

    for i in range(len(X_cp)):
        bi, bj = som.find_bmu(X_cp[i])
        bi = int(bi.get())
        bj = int(bj.get())
        class_counts[bi, bj, y[i]-1] += 1

    dominant_class = np.argmax(class_counts, axis=2) + 1
    class_sums = class_counts.sum(axis=2)
    class_proportions = np.divide(class_counts.max(axis=2), class_sums, where=class_sums != 0)
    class_proportions[class_sums == 0] = 0

    bubble_size = (class_proportions * class_sums) * 100

    x, y = np.meshgrid(np.arange(n), np.arange(m))
    x, y = x.flatten(), y.flatten()
    dominant_class_flat = dominant_class.flatten()
    bubble_size_flat = bubble_size.flatten()

    cmap = plt.get_cmap('tab10', 3)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(x, y, s=bubble_size_flat, c=dominant_class_flat, cmap=cmap, alpha=0.75, edgecolors='black')
    cbar = plt.colorbar(scatter, ticks=[1, 2, 3])
    cbar.set_label('Class label')
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['1', '2', '3'])
    plt.title('Class membership map')
    plt.xlabel('Grid X-axis')
    plt.ylabel('Grid Y-axis')
    plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.gca().set_aspect('equal', 'box')
    plt.xticks(np.arange(n))
    plt.yticks(np.arange(m))


def plot_feature_heatmaps(som):
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for k in range(som.dim):
        ax = axs.flat[k]
        mat = som.weights[..., k]
        mat = mat.get() if hasattr(mat, "get") else mat
        im = ax.imshow(mat, origin='lower')
        ax.set_title(f'Attr {k+1}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axs.flat[som.dim].axis('off')
    plt.suptitle('Attribute heatmaps')
    plt.tight_layout()


def compute_u_matrix(weights):
    m, n, _ = weights.shape
    U = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            nbrs = []
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < n:
                    nbrs.append(np.linalg.norm(weights[i,j] - weights[ii,jj]))
            U[i,j] = np.mean(nbrs)
    return U


def plot_u_matrix(som):
    U = compute_u_matrix(som.weights.get() if hasattr(som.weights, "get") else som.weights)
    plt.figure(figsize=(6,6))
    plt.imshow(U, cmap='viridis', origin='lower')
    plt.colorbar(label='Avg. neighbor dist.')
    plt.title('U-matrix')


def plot_trajectory_map(xsom):
    coords = np.array([
        (int(i), int(j)) for (i, j) in xsom.bmu_trajectory
    ])
    plt.figure(figsize=(6,6))
    plt.imshow(np.zeros((xsom.m, xsom.n)), cmap='gray_r', origin='lower')
    plt.plot(coords[:,1], coords[:,0], '-o', color='red', alpha=0.7)
    plt.title("BMU trajectory over time")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()


def plot_recursive_state_evolution(rsom, n_neurons_to_plot=5):
    ctx = np.array(rsom.context_history.get())
    plt.figure(figsize=(10,4))
    for i in range(min(n_neurons_to_plot, ctx.shape[1])):
        plt.plot(ctx[:, i], label=f'Neuron {i}')
    plt.xlabel('Time')
    plt.ylabel('Context activation')
    plt.title('Evolution of recursive states over time')
    plt.legend()
    plt.grid(alpha=0.4)


def plot_temporal_similarity(rsom):
    ctx = np.array(rsom.context_history.get() if hasattr(rsom.context_history, "get") else rsom.context_history)
    sim = np.corrcoef(ctx)
    plt.figure(figsize=(6,6))
    plt.imshow(sim, cmap='coolwarm', origin='lower')
    plt.colorbar(label='Context similarity')
    plt.title('Temporal similarity of RSOM context states')
    plt.xlabel('Time step')
    plt.ylabel('Time step')


def plot_context_norms(xsom):
    plt.figure()
    plt.plot(xsom.context_norms.get() if hasattr(xsom.context_norms, "get") else xsom.context_norms, linestyle='-', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("‖Context Vector‖")
    plt.title("Context Magnitude Over Training")
    plt.grid(alpha=0.4)


def plot_merged_input_evolution(msom, dim=0):
    """Plot how one merged feature evolves over time."""
    z = np.array(msom.merged_inputs.get() if hasattr(msom.merged_inputs, "get") else msom.merged_inputs)
    half = z.shape[1] // 2
    input_part = z[:, :half]
    context_part = z[:, half:]
    plt.figure()
    plt.plot(input_part[:, dim], label='Input feature')
    plt.plot(context_part[:, dim], label='Context feature', alpha=0.7)
    plt.title(f"Merged Input Evolution (feature {dim})")
    plt.xlabel("Time step")
    plt.legend()


def plot_recurrence(msom):
    coords = np.array(msom.bmu_trajectory.get() if hasattr(msom.bmu_trajectory, "get") else msom.bmu_trajectory)
    dists = np.sqrt(np.sum((coords[:,None,:] - coords[None,:,:])**2, axis=2))
    sim = np.exp(-dists)
    plt.figure(figsize=(6,6))
    plt.imshow(sim, cmap='viridis', origin='lower')
    plt.title("BMU Recurrence (Temporal Similarity)")
    plt.xlabel("Time")
    plt.ylabel("Time")
    plt.colorbar(label="Similarity")