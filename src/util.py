import numpy as np
import matplotlib.pyplot as plt


def plot_quantization_error(som):
    epochs = range(1, len(som.q_error_history) + 1)
    plt.figure()
    plt.plot(epochs, som.q_error_history, linestyle='-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Quantization error')
    plt.title('QE vs. epoch')
    plt.grid(alpha=0.4)


def plot_avg_adjustment(som):
    epochs = range(1, len(som.avg_adjust_history) + 1)
    plt.figure()
    plt.plot(epochs, som.avg_adjust_history, linestyle='-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Mean |Δw|')
    plt.title('Avg. weight adjustment vs. epoch')
    plt.grid(alpha=0.4)


def plot_winner_map(som, X, y):
    m, n = som.m, som.n
    num_classes = len(np.unique(y))
    class_counts = np.zeros((m, n, num_classes))

    for sample, label in zip(X, y):
        bmu = som.find_bmu(sample)
        class_counts[bmu][label - 1] += 1

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
    U = compute_u_matrix(som.weights)
    plt.figure(figsize=(6,6))
    plt.imshow(U, cmap='viridis', origin='lower')
    plt.colorbar(label='Avg. neighbor dist.')
    plt.title('U-matrix')