import numpy as np
from pyrvfl.models.rvfl import RVFL
from pyrvfl.models.SPRVFL import SPRVFL

from pyrvfl.utils.utils import standardize_rvfl, stratified_shuffle_split_indices
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo


def run_classification_experiment():
    # Cargar el conjunto de datos "Ionosphere"
    ionosphere = fetch_ucirepo(id=52)

    def prepare_data_classify(proportion, dataset):
        label = dataset.data.targets.copy()
        data = dataset.data.features

        encoding = {"g": 1, "b": 0}
        label["Class"].replace(encoding, inplace=True)
        label = label.squeeze()

        shuffle_index = np.arange(len(label))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(label))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]

        data_train = data.iloc[train_index]
        label_train = label.iloc[train_index]
        data_val = data.iloc[val_index]
        label_val = label.iloc[val_index]
        return (data_train.to_numpy(), label_train.to_numpy()), (
            data_val.to_numpy(),
            label_val.to_numpy(),
        )

    (X_train, y_train), (X_test, y_test) = prepare_data_classify(0.8, ionosphere)

    X_train = standardize_rvfl(X_train)
    X_test = standardize_rvfl(X_test)

    metrics_to_evaluate = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]

    num_neurons_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    regularization_values = [2**exp for exp in range(-6, 13, 2)]
