import numpy as np
from SOM import SOM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import Parallel, delayed

data = np.loadtxt('seeds.txt')
x = data[:, :-1]
y = data[:, -1].astype(int)

# this is called stratified split (if you want to find it): 50 training and 20 test samples per class basically
np.random.seed(42)
train_idx = []
test_idx = []
for cls in np.unique(y):
    cls_idx = np.where(y == cls)[0]
    np.random.shuffle(cls_idx)
    train_idx.extend(cls_idx[:50])
    test_idx.extend(cls_idx[50:70])
train_idx = np.array(train_idx)
test_idx = np.array(test_idx)
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

# hyperparameters for testing
dims = [(m, n) for m in range(8, 16) for n in range(m, 16) if 80 <= m * n <= 150]
inits = ['uniform', 'data_range', 'sample', 'pca', 'kmeans']
metrics = ['euclid', 'manhattan', 'chebyshev', 'cosine', 'toroidal']
kernels = ['gaussian', 'bubble', 'epanechnikov', 'triangular', 'inverse']
epochs = 150

def run_config(m, n, init_method, grid_metric, kernel):
    # trains the SOM on training features only
    som = SOM(
        m, n,
        dim=x_train.shape[1],
        weight_init_method=init_method,
        grid_metric=grid_metric,
        neighborhood_kernel=kernel,
        seed=42
    )
    som.train(x_train, num_epochs=epochs)

    # assign a class to each neuron based on training labels
    num_classes = len(np.unique(y_train))
    class_counts = np.zeros((m, n, num_classes))
    for sample, label in zip(x_train, y_train):
        bmu = som.find_bmu(sample)
        class_counts[bmu][label - 1] += 1
    neuron_classes = np.argmax(class_counts, axis=2) + 1

    # classify by BMU neuron classes on the test set
    preds = []
    for sample in x_test:
        bmu = som.find_bmu(sample)
        preds.append(neuron_classes[bmu])

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return {
        'm': m, 'n': n,
        'init': init_method,
        'metric': grid_metric,
        'kernel': kernel,
        'accuracy': acc,
        'confusion_matrix': cm
    }

# grid search over the map parameters
configs = [
    (m, n, init, metric, kernel)
    for m, n in dims
    for init in inits
    for metric in metrics
    for kernel in kernels
]

# parallel evaluation of all configs
results = Parallel(n_jobs=-1)(
    delayed(run_config)(m, n, init, metric, kernel) for m, n, init, metric, kernel in configs
)

# best model selected based on test accuracy
best = max(results, key=lambda r: r['accuracy'])
print(f"Best config: m={best['m']}, n={best['n']}, init={best['init']}, metric={best['metric']}, kernel={best['kernel']}, Accuracy={best['accuracy']:.4f}")
print("Confusion matrix:")
print(best['confusion_matrix'])

# plot the confusion matrix for the best model
plt.figure(figsize=(8, 6))
plt.imshow(best['confusion_matrix'], cmap='viridis', interpolation='nearest')
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.colorbar()
plt.show()