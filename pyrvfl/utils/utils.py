import numpy as np


@staticmethod
def get_random_vectors(m, n, scale_range):
    x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
    return x


@staticmethod
def one_hot(x, n_class):
    y = np.zeros([len(x), n_class])
    for i in range(len(x)):
        y[i, x[i]] = 1
    return y


@staticmethod
def standardize_rvfl(x):

    data_std = np.maximum(np.std(x, axis=0), 1 / np.sqrt(len(x)))
    data_mean = np.mean(x, axis=0)
    return (x - data_mean) / data_std


@staticmethod
def softmax(x):
    return np.exp(x) / np.repeat(
        (np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1
    )


@staticmethod
def minmax_normalize(x, feature_range=(0, 1)):
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)

    # avoid zero division
    range_val = np.maximum(max_val - min_val, 1e-10)

    scale = (feature_range[1] - feature_range[0]) / range_val
    shift = feature_range[0] - min_val * scale

    return x * scale + shift


@staticmethod
def prepare_data_classify(dataset, proportion):

    label = dataset["target"]
    data = dataset["data"]

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val)


@staticmethod
def stratified_shuffle_split_indices(X, y, n_splits=5, test_size=0.2, random_state=42):
    """
    Generate stratified shuffle split indices for cross-validation.

    Parameters:
    - X: Features dataset.
    - y: Labels dataset.
    - n_splits: Number of splits for cross-validation.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed for random number generator.

    Yields:
    - train_indices: Indices for the training set.
    - test_indices: Indices for the test set.
    """
    n_samples = len(X)
    n_test = int(np.ceil(test_size * n_samples))

    if random_state is not None:
        np.random.seed(random_state)

    for _ in range(n_splits):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        yield train_indices, test_indices


@staticmethod
def stratify_split(X, y):
    class_counts = np.sum(y, axis=0)

    # Calcula la proporción de muestras que quieres en el conjunto de entrenamiento
    train_ratio = 0.8

    # Calcula la cantidad de muestras para el conjunto de entrenamiento
    train_size = int(train_ratio * len(X))

    # Hacemos un shuffle de los índices
    indices = np.random.permutation(len(X))

    # Selecciona los índices para el conjunto de entrenamiento
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Divide los datos en el conjunto de entrenamiento y prueba
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
