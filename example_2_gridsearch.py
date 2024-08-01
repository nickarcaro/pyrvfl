from pyrvfl.rvfl import RVFL
from pyrvfl.metrics.gridSearch import gridSearch
import numpy as np
import sklearn.datasets as sk_dataset
from pyrvfl.utils.utils import standardize_rvfl


def prepare_data_classify(proportion):
    dataset = sk_dataset.load_breast_cancer()
    label = dataset["target"]
    data = dataset["data"]
    n_class = len(dataset["target_names"])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


# Define the parameter grid
param_grid = [
    {
        "n_nodes": num_neurons,
        "lam": reg_val,
        "w_random_vec_range": [-1, 1],
        "b_random_vec_range": [0, 1],
        "activation": activations,
        "task_type": "classification",
    }
    for activations in [
        "relu",
        "sigmoid",
        "sine",
        "hardlim",
        "tribas",
        "radbas",
        "sign",
        "softmax",
    ]
    for num_neurons in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for reg_val in [2**exp for exp in range(-6, 13, 2)]
]

(train_data, train_labels), (val_data, val_labels), num_class = prepare_data_classify(
    0.8
)

train_data = standardize_rvfl(train_data)
val_data = standardize_rvfl(val_data)

result = gridSearch(
    RVFL,
    train_data,
    train_labels,
    param_grid,
    n_splits=5,
    test_size=0.2,
    random_state=42,
)
