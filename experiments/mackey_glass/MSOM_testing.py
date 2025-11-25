import numpy as np

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.models.MSOM import MSOM
from src.models.utils import *

np.random.seed(42)


def main():
    np.random.seed(42)
    T = 200
    noise = 0.05
    data = []
    for t in range(T):
        c1 = np.array([np.sin(t/20), np.cos(t/25)]) + np.array([-1,0])
        c2 = np.array([np.sin(t/15), np.cos(t/30)]) + np.array([1,0])
        x1 = c1 + np.random.normal(0,noise,(10,2))
        x2 = c2 + np.random.normal(0,noise,(10,2))
        data.append(np.vstack([x1,x2]))
    data = np.array(data)

    msom = MSOM(m=10, n=10, dim=2)
    msom.train(data, 100)

    plot_quantization_error(msom)
    plot_trajectory_map(msom)
    plot_context_norms(msom)
    plot_merged_input_evolution(msom)
    plot_recurrence(msom)

if __name__ == "__main__":
    main()
