import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans


class EnsembleDeepRVFLK(BaseEstimator, ClassifierMixin):
    """An ensemble deep RVFL classifier or regression."""

    def __init__(
        self,
        n_nodes=100,
        k=3,
        lam=1e-6,
        w_random_vec_range=[-1, 1],
        b_random_vec_range=[0, 1],
        activation="relu",
        n_layer=2,
        same_feature=False,
        task_type="classification",
    ):
        assert task_type in [
            "classification",
            "regression",
        ], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.k = k  # Number of clusters
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.activation = activation
        self.n_layer = n_layer
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        self.same_feature = same_feature
        self.task_type = task_type
        self.random_weights = []
        self.random_bias = []
        self.beta = []
        self.classes_ = None
        self.is_fitted_ = False

    def _activation_function(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "linear":
            return x
        else:
            raise ValueError("Unsupported activation function")

    @staticmethod
    def get_random_vectors(m, n, scale_range):
        return (scale_range[1] - scale_range[0]) * np.random.random(
            [m, n]
        ) + scale_range[0]

    @staticmethod
    def one_hot(x, n_class):
        encoder = OneHotEncoder(sparse_output=False)
        return encoder.fit_transform(x.reshape(-1, 1))

    def standardize(self, x, index):
        if self.same_feature:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x), 1 / np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(
                    np.std(x, axis=0), 1 / np.sqrt(len(x))
                )
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]

    def train(self, data, label, n_class):
        data, label = check_X_y(data, label)
        n_sample, n_feature = data.shape
        h = data.copy()
        data = self.standardize(data, 0)

        if self.task_type == "classification":
            y = self.one_hot(label, n_class)
            self.classes_ = unique_labels(label)
        else:
            y = label

        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalize the data

            # Apply KMeans
            kmeans = KMeans(n_clusters=self.k, n_init=10)
            cluster_labels = kmeans.fit_predict(h)

            self.random_weights.append(
                self.get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range)
            )
            self.random_bias.append(
                self.get_random_vectors(1, self.n_nodes, self.b_random_range)
            )

            # Process each cluster
            h_list = []
            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self._activation_function(
                    np.dot(cluster_data, self.random_weights[i])
                    + np.dot(np.ones([n_cluster_sample, 1]), self.random_bias[i])
                )
                h_list.append(h_cluster)

            # Concatenate cluster results
            final_h = np.concatenate(h_list, axis=0)
            h = final_h

            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            if n_sample > (self.n_nodes + n_feature):
                self.beta.append(
                    np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                    @ d.T
                    @ y
                )
            else:
                self.beta.append(
                    d.T
                    @ np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))
                    @ y
                )

        self.is_fitted_ = True

    def predict(self, data, output_prob=False):
        check_is_fitted(self, "is_fitted_")
        data = check_array(data)
        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)  # Normalize the data
        outputs = []

        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalize the data

            # Apply KMeans
            kmeans = KMeans(n_clusters=self.k, n_init=10)
            cluster_labels = kmeans.fit_predict(h)

            h_list = []
            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self._activation_function(
                    np.dot(cluster_data, self.random_weights[i])
                    + np.dot(np.ones([n_cluster_sample, 1]), self.random_bias[i])
                )
                h_list.append(h_cluster)

            # Concatenate cluster results
            final_h = np.concatenate(h_list, axis=0)
            h = final_h

            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))

        if self.task_type == "classification":
            votes = [np.argmax(item, axis=1) for item in outputs]
            votes = np.array(votes).T
            final_vote = np.array([np.bincount(vote).argmax() for vote in votes])
            return final_vote
        elif self.task_type == "regression":
            return np.mean(outputs, axis=0)

    def eval(self, data, label):
        data, label = check_X_y(data, label)
        outputs = self.predict(data)
        if self.task_type == "classification":
            return accuracy_score(label, outputs)
        elif self.task_type == "regression":
            return mean_absolute_error(label, outputs)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x):
        x[x >= 0] = x[x >= 0]
        x[x < 0] = x[x < 0] / 10.0
        return x
