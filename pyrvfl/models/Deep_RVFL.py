import numpy as np
from pyrvfl.utils.activation import Activation
from pyrvfl.utils.utils import get_random_vectors, one_hot, standardize_deep, softmax


class DeepRVFL:
    """A deep RVFL classifier or regression.

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
        n_layer,
        task_type="classification",
    ):
        assert task_type in [
            "classification",
            "regression",
        ], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = []
        self.random_bias = []
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer

        self.task_type = task_type

    def train(self, data, label, n_class):
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
        d = standardize_deep(data, 0)  # Normalization data
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize_deep(h, i)  # Normalization data
            self.random_weights.append(
                get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range)
            )
            self.random_bias.append(
                get_random_vectors(1, self.n_nodes, self.b_random_range)
            )
            h = self.activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_sample, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, d], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        if self.task_type == "classification":
            y = one_hot(label, n_class)
        else:
            y = label
        if n_sample > (self.n_nodes * self.n_layer + n_feature):
            self.beta = (
                np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                .dot(d.T)
                .dot(y)
            )
        else:
            self.beta = d.T.dot(
                np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))
            ).dot(y)

    def predict(self, data):
        """

        :param data: Predict data.
        :return: When classification, return Prediction result and probability.
                 When regression, return the output of rvfl.
        """
        n_sample = len(data)
        d = standardize_deep(data, 0)
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize_deep(h, i)
            h = self.activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_sample, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, d], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == "classification":
            proba = self.softmax(output)
            result = np.argmax(proba, axis=1)
            return result, proba
        elif self.task_type == "regression":
            return output

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return accuracy.
                 When regression return MAE.
        """

        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        n_sample = len(data)
        d = standardize_deep(data, 0)
        h = data.copy()
        for i in range(self.n_layer):
            h = standardize_deep(h, i)
            h = self.activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_sample, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, d], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == "classification":
            result = np.argmax(output, axis=1)
            acc = np.sum(np.equal(result, label)) / len(label)
            return acc
        elif self.task_type == "regression":
            mae = np.mean(np.abs(output - label))
            return mae
