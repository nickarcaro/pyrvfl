from pyrvfl.models.rvfl import RVFL
from pyrvfl.models.SPRVFL import SPRVFL
from pyrvfl.models.Deep_RVFL import DeepRVFL
from pyrvfl.models.Ensemble_Deep_RVFL import EnsembleDeepRVFL
from pyrvfl.metrics.gridSearch import gridSearch
import sklearn.datasets as sk_dataset
import numpy as np
from pyrvfl.utils.utils import standardize_rvfl, minmax_normalize


def prepare_data_regression(proportion):
    dataset = sk_dataset.load_diabetes()
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


(train_data, train_labels), (test_data, test_labels) = prepare_data_regression(0.8)


train_data = standardize_rvfl(train_data)
test_data = standardize_rvfl(test_data)


# Conduct grid search and evaluate on test data with specified metrics


model = SPRVFL(n_nodes=100, lam=1, activation="relu", task_type="regression")
model.fit(train_data, train_labels)
predictions = model.predict(test_data)
results = model.eval(test_data, test_labels)
print(results)
