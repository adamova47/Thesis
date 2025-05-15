from MSOM import MSOM
from joblib import Parallel, delayed
from util import *


def run_config(params):
    m, n, init_method, grid_metric, kernel, x, y, epochs = params
    som = MSOM(
        m,
        n,
        dim=x.shape[1],
        weight_init_method=init_method,
        grid_metric=grid_metric,
        neighborhood_kernel=kernel,
        seed=42
    )
    som.train(x, num_epochs=epochs)
    return {'m': m, 'n': n, 'init': init_method, 'metric': grid_metric, 'kernel': kernel, 'som': som}


def main():
    # load data
    data = np.loadtxt('seeds.txt')
    x = data[:, :-1]
    y = data[:, -1].astype(int)

    # generate map dimensions - number of neurons between 80 and 150
    dims = [(10, 10)]

    # other hyperparameters
    inits = ['uniform']
    metrics = ['euclid']
    kernels = ['gaussian']
    epochs = 150

    configs = [
        (m, n, init_method, grid_metric, kernel, x, y, epochs)
        for m, n in dims
        for init_method in inits
        for grid_metric in metrics
        for kernel in kernels
    ]

    results = Parallel(n_jobs=-1)(
        delayed(run_config)(cfg) for cfg in configs
    )

    best = min(results, key=lambda r: r['m'])
    best_som = best['som']
    print(f"Best config: m={best['m']}, n={best['n']}, init={best['init']},"
          f" metric={best['metric']}, kernel={best['kernel']}")

    plot_winner_map(best_som, x, y)
    plt.show()


if __name__ == '__main__':
    main()