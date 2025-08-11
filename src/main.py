from SOM import SOM
from util import *
from joblib import Parallel, delayed


def run_config(params):
    m, n, init_method, grid_metric, kernel, x, y, epochs = params
    som = SOM(
        m,
        n,
        dim=x.shape[1],
        weight_init_method=init_method,
        grid_metric=grid_metric,
        neighborhood_kernel=kernel,
        seed=42,
    )
    som.train(x, num_epochs=epochs)
    q = som.q_error_history[-1]
    return {
        "m": m,
        "n": n,
        "init": init_method,
        "metric": grid_metric,
        "kernel": kernel,
        "q": q,
        "som": som,
    }


def main():
    # load data
    data = np.loadtxt("seeds.txt")
    x = data[:, :-1]
    y = data[:, -1].astype(int)

    # generate map dimensions - number of neurons between 80 and 150
    dims = [(m, n) for m in range(8, 16) for n in range(m, 16) if 80 <= m * n <= 150]

    # other hyperparameters
    inits = ["uniform", "data_range", "sample", "sample", "pca", "kmeans"]
    metrics = ["euclid", "manhattan", "chebyshev", "cosine", "toroidal"]
    kernels = ["gaussian", "bubble", "epanechnikov", "triangular", "inverse"]
    epochs = 150

    configs = [
        (m, n, init_method, grid_metric, kernel, x, y, epochs)
        for m, n in dims
        for init_method in inits
        for grid_metric in metrics
        for kernel in kernels
    ]

    results = Parallel(n_jobs=-1)(delayed(run_config)(cfg) for cfg in configs)

    best = min(results, key=lambda r: r["q"])
    best_som = best["som"]
    print(
        f"Best config: m={best['m']}, n={best['n']}, init={best['init']},"
        f" metric={best['metric']}, kernel={best['kernel']}, QE={best['q']}"
    )

    plot_quantization_error(best_som)
    plot_avg_adjustment(best_som)
    plot_winner_map(best_som, x, y)
    plot_feature_heatmaps(best_som)
    plot_u_matrix(best_som)
    plt.show()


if __name__ == "__main__":
    main()
