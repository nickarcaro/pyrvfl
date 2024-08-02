from pyrvfl.rvfl import RVFL

import numpy as np
import sklearn.datasets as sk_dataset


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


# Classification
num_nodes = 2  # Number of enhancement nodes.
regular_para = 1  # Regularization parameter.
weight_random_range = [-1, 1]  # Range of random weights.
bias_random_range = [0, 1]  # Range of random weights.

train, val, num_class = prepare_data_classify(0.8)
rvfl = RVFL(
    n_nodes=num_nodes,
    lam=regular_para,
    w_random_vec_range=weight_random_range,
    b_random_vec_range=bias_random_range,
    activation="relu",
    task_type="classification",
)
rvfl.fit(train[0], train[1])


accuracy = rvfl.eval(val[0], val[1])
print(accuracy)
