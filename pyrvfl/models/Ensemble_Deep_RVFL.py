import numpy as np
from pyrvfl.utils.activation import Activation
from pyrvfl.utils.utils import get_random_vectors, one_hot, softmax
from pyrvfl.metrics.metrics import (
    f1_score,
    roc_auc,
    accuracy_score,
    mae,
    mse,
    rmse,
    r2_score,
)


class EnsembleDeepRVFL:
    """A ensemble deep RVFL classifier or regression model.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        n_layer: A integer, N=number of hidden layers.
        gamma: A list, store the scale parameters for batch normalization for each layer.
        beta_bn: A list, store the shift parameters for batch normalization for each layer.
        task_type: A string of ML task type, 'classification' or 'regression'.
    """

    def __init__(
        self,
        n_nodes=100,
        lam=1e-6,
        activation="relu",
        n_layer=2,
        task_type="classification",
    ):
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = [-1, 1]
        self.b_random_range = [0, 1]
        self.random_weights = []
        self.random_bias = []
        self.beta = []
        self.gamma = [np.ones(n_nodes) for _ in range(n_layer)]
        self.beta_bn = [np.zeros(n_nodes) for _ in range(n_layer)]
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer
        self.task_type = task_type

    def batch_normalization(self, X, layer_index, epsilon=1e-5):
        mean = np.mean(X, axis=0)
        variance = np.var(X, axis=0)
        X_normalized = (X - mean) / np.sqrt(variance + epsilon)
        out = self.gamma[layer_index] * X_normalized + self.beta_bn[layer_index]
        return out

    def fit(self, data, label):
        """
        Train the ensemble deep RVFL model.

        :param data: Training data.
        :param label: Training label.
        :return: No return
        """
        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        n_sample = len(data)
        h = data.copy()

        if self.task_type == "classification":
            y = one_hot(label, len(np.unique(label)))
        else:
            y = label

        for i in range(self.n_layer):
            self.random_weights.append(
                get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range)
            )
            self.random_bias.append(
                get_random_vectors(1, self.n_nodes, self.b_random_range)
            )
            h = np.dot(h, self.random_weights[i]) + np.dot(
                np.ones([n_sample, 1]), self.random_bias[i]
            )
            h = self.batch_normalization(h, i)
            h = self.activation_function(h)

            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            self.beta.append(
                np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                .dot(d.T)
                .dot(y)
            )

    def predict(self, data, output_prob=True):
        """
        Predict using the ensemble deep RVFL model.

        :param data: Predict data.
        :return: When classification, return vote result, addition result and probability.
                 When regression, return the mean output of edrvfl.
        """
        n_sample = len(data)
        h = data.copy()
        outputs = []

        for i in range(self.n_layer):
            h = np.dot(h, self.random_weights[i]) + np.dot(
                np.ones([n_sample, 1]), self.random_bias[i]
            )
            h = self.batch_normalization(h, i)
            h = self.activation_function(h)

            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))

        if self.task_type == "classification":
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))

            add_proba = softmax(np.sum(outputs, axis=0))
            add_res = np.argmax(add_proba, axis=1)
            if output_prob:
                return vote_res, (add_res, add_proba)
            return vote_res, add_res

        elif self.task_type == "regression":
            return np.mean(
                outputs, axis=0
            )  # Para regresión, devolver la media de las salidas

    def eval(self, data, label, metrics=None):
        """
        Evaluate the ensemble deep RVFL model.

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return vote and addition accuracy.
                 When regression return metrics results.
        """
        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        if metrics is None:
            metrics = ["accuracy"] if self.task_type == "classification" else ["mae"]

        results = {}

        if self.task_type == "classification":
            vote_res, (add_res, add_proba) = self.predict(data)
            if "accuracy" in metrics:
                results["accuracy"] = accuracy_score(label, add_res)
            if "f1_score" in metrics:
                results["f1_score"] = f1_score(label, add_res)["f1_score"]
            if "precision" in metrics:
                results["precision"] = f1_score(label, add_res)["precision"]
            if "recall" in metrics:
                results["recall"] = f1_score(label, add_res)["recall"]
            if "tpr" in metrics:
                results["tpr"] = f1_score(label, add_res)["tpr"]
            if "fnr" in metrics:
                results["fnr"] = f1_score(label, add_res)["fnr"]
            if "fpr" in metrics:
                results["fpr"] = f1_score(label, add_res)["fpr"]
            if "roc_auc" in metrics and len(np.unique(label)) == 2:
                results["roc_auc"] = roc_auc(label, add_proba[:, 1])

        elif self.task_type == "regression":
            predictions = self.predict(data, output_prob=False)
            if "mse" in metrics:
                results["mse"] = mse(label, predictions)
            if "rmse" in metrics:
                results["rmse"] = rmse(label, predictions)
            if "mae" in metrics:
                results["mae"] = mae(label, predictions)
            if "r2" in metrics:
                results["r2"] = r2_score(label, predictions)

        return results
