<<<<<<< HEAD:example_2_gridsearch.py
from pyrvfl.Deep_RVFL import DeepRVFL
from pyrvfl.metrics.gridSearch import gridSearch
import numpy as np
from pyrvfl.utils.utils import standardize_rvfl
import sklearn.datasets as sk_dataset
=======
import numpy as np
import sklearn.datasets as sk_dataset
from pyrvfl.rvfl import RVFL
from pyrvfl.metrics.gridSearch import gridSearch
from pyrvfl.utils.utils import standardize_rvfl
from pyrvfl.RVFLK import RVFLK
>>>>>>> dec345de5b0c4d81f826443d507e27f1fb378c7d:pyrvfl/example_1_gridsearch.py


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
        "n_layer": 2,
        "w_random_vec_range": [-1, 1],
        "b_random_vec_range": [0, 1],
        "activation": activations,
    }
    for activations in [
        "relu",
        "sigmoid",
    ]
    for num_neurons in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for reg_val in [2**exp for exp in range(-6, 13, 2)]
]

(train_data, train_labels), (test_data, test_labels), num_class = prepare_data_classify(
    0.8
)

train_data = standardize_rvfl(train_data)
test_data = standardize_rvfl(test_data)

<<<<<<< HEAD:example_2_gridsearch.py
=======

"""
train_data = minmax_normalize(train_data)
test_data = minmax_normalize(test_data)
"""
param_grid = [
    {
        "n_nodes": num_neurons,
        "lam": reg_val,
        "w_random_vec_range": [-1, 1],
        "b_random_vec_range": [0, 1],
        "activation": "relu",
    }
    for num_neurons in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for reg_val in [2**exp for exp in range(-6, 13, 2)]
]

>>>>>>> dec345de5b0c4d81f826443d507e27f1fb378c7d:pyrvfl/example_1_gridsearch.py

# Conduct grid search and evaluate on test data with specified metrics
metrics_to_evaluate = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
result = gridSearch(
<<<<<<< HEAD:example_2_gridsearch.py
    DeepRVFL,
=======
    RVFLK,
>>>>>>> dec345de5b0c4d81f826443d507e27f1fb378c7d:pyrvfl/example_1_gridsearch.py
    train_data,
    train_labels,
    param_grid,
    test_data,
    test_labels,
    metrics=metrics_to_evaluate,
    n_iterations=100,
    n_splits=5,
    test_size=0.2,
    random_state=42,
    generate_plot=True,  # Set to True to generate the plot
    save_csv=True,  # Set to True to save the results to a CSV
)
