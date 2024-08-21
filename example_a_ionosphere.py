from ucimlrepo import fetch_ucirepo
import numpy as np
import sklearn.datasets as sk_dataset
from pyrvfl.rvfl import RVFL
from pyrvfl.Deep_RVFL import DeepRVFL
from pyrvfl.Ensemble_Deep_RVFL import EnsembleDeepRVFL
from pyrvfl.metrics.gridSearch import gridSearch
from pyrvfl.utils.utils import standardize_rvfl

# fetch dataset
ionosphere = fetch_ucirepo(id=52)

# Print the shape of the labels to confirm successful encoding


def prepare_data_classify(proportion, dataset):
    label = dataset.data.targets.copy()
    data = dataset.data.features

    # Define the encoding dictionary to map letter labels to numbers
    encoding = {"g": 1, "b": 0}

    # Replace the letter labels with the corresponding numbers
    label["Class"].replace(encoding, inplace=True)
    label = label.squeeze()

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]

    data_train = data.iloc[train_index]  # Use .iloc for row selection
    label_train = label.iloc[train_index]  # Use .iloc for row selection
    data_val = data.iloc[val_index]  # Use .iloc for row selection
    label_val = label.iloc[val_index]  # Use .iloc for row selection
    return (data_train.to_numpy(), label_train.to_numpy()), (
        data_val.to_numpy(),
        label_val.to_numpy(),
    )


(train_data, train_labels), (test_data, test_labels) = prepare_data_classify(
    0.8, ionosphere
)


train_data = standardize_rvfl(train_data)
test_data = standardize_rvfl(test_data)

# Define the parameter grid

param_grid = [
    {
        "n_nodes": num_neurons,
        "lam": reg_val,
        # "n_layer": 2,  layers to deep rvfl and ensemble deep rvfl
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

# Conduct grid search and evaluate on test data with specified metrics
metrics_to_evaluate = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
result = gridSearch(
    RVFL,  # RVFL, DeepRVFL, EnsembleDeepRVFL
    train_data,  # Convert to numpy array if needed
    train_labels,  # Convert to numpy array if needed
    param_grid,
    test_data,  # Convert to numpy array if needed
    test_labels,  # Convert to numpy array if needed
    metrics=metrics_to_evaluate,
    n_iterations=100,
    n_splits=5,
    test_size=0.2,
    random_state=42,
    generate_plot=True,  # Set to True to generate the plot
    save_csv=True,  # Set to True to save the results to a CSV
)
