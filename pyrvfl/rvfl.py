import numpy as np
from pyrvfl.utils.activation import Activation
from pyrvfl.utils.utils import get_random_vectors, one_hot, softmax
from pyrvfl.metrics.metrics import f1_score, roc_auc, accuracy_score

import numpy as np


class RVFL:
    """A simple RVFL classifier or regression.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
        task_type: A string of ML task type, 'classification' or 'regression'.
    """

    def __init__(
        self,
        n_nodes,
        lam,
        w_random_vec_range,
        b_random_vec_range,
        activation,
    ):

        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None

    def fit(self, data, label):
        """
        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """

        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        n_sample = len(data)
        n_feature = len(data[0])
        n_class = len(np.unique(label))
        self.random_weights = get_random_vectors(
            n_feature, self.n_nodes, self.w_random_range
        )
        self.random_bias = get_random_vectors(1, self.n_nodes, self.b_random_range)

        h = self.activation_function(
            np.dot(data, self.random_weights)
            + np.dot(np.ones([n_sample, 1]), self.random_bias)
        )
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

        y = one_hot(label, n_class)

        self.beta = (
            np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
            .dot(d.T)
            .dot(y)
        )

    def predict(self, data):
        """

        :param data: Predict data.
        :return: When classification, return Prediction result and probability.
                  When regression, return the output of rvfl.
        """

        # data = self.standardize(data)  # Normalization data
        h = self.activation_function(
            np.dot(data, self.random_weights) + self.random_bias
        )
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)

        proba = softmax(output)
        result = np.argmax(proba, axis=1)
        return result, proba

    def eval(self, data, label, metrics=None):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return accuracy.
                  When regression return MAE.
        """

        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        result, proba = self.predict(data)

        """
        h = self.activation_function(
            np.dot(data, self.random_weights) + self.random_bias
        )
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)

        result = np.argmax(output, axis=1)

        proba = softmax(output)
"""
        if metrics is None:
            metrics = ["accuracy"]  # Métrica predeterminada

        results = {}

        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(label, result)
        if "f1_score" in metrics:
            results["f1_score"] = f1_score(label, result)["f1_score"]
        if "precision" in metrics:
            results["precision"] = f1_score(label, result)["precision"]
        if "recall" in metrics:
            results["recall"] = f1_score(label, result)["recall"]
        if "tpr" in metrics:
            results["tpr"] = f1_score(label, result)["tpr"]
        if "fnr" in metrics:
            results["fnr"] = f1_score(label, result)["fnr"]
        if "fpr" in metrics:
            results["fpr"] = f1_score(label, result)["fpr"]
        if "roc_auc" in metrics and len(np.unique(label)) == 2:
            results["roc_auc"] = roc_auc(label, proba[:, 1])

        return results
