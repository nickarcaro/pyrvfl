import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans


class DeepRVFLK(BaseEstimator, ClassifierMixin):
    """A deep RVFL classifier or regression."""

    def __init__(
        self,
        n_nodes=100,
        k=3,
        lam=1e-3,
        activation="relu",
        n_layer=2,
        task_type="classification",
    ):
        assert task_type in [
            "classification",
            "regression",
        ], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.k = k  # Number of clusters
        self.lam = lam
        self.activation = activation
        self.n_layer = n_layer
        self.task_type = task_type
        self.random_weights = []
        self.random_bias = []
        self.beta = None
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        self.w_random_range = [-1, 1]
        self.b_random_range = [0, 1]

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
    def _get_random_vectors(m, n, scale_range):
        return np.random.uniform(scale_range[0], scale_range[1], (m, n))

    @staticmethod
    def _one_hot(labels, n_classes):
        encoder = OneHotEncoder(sparse_output=False)
        labels = labels.reshape(-1, 1)
        return encoder.fit_transform(labels)

    @staticmethod
    def _softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def fit(self, X, y):
        """Train the Deep RVFL model."""
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # Initialize random weights and biases
        h = X.copy()

        # Apply KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        for i in range(self.n_layer):
            self.random_weights.append(
                self._get_random_vectors(h.shape[1], self.n_nodes, self.w_random_range)
            )
            self.random_bias.append(
                self._get_random_vectors(1, self.n_nodes, self.b_random_range)
            )

            # List to store h for each cluster
            h_list = []

            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self._activation_function(
                    np.dot(cluster_data, self.random_weights[i])
                    + np.dot(np.ones((n_cluster_sample, 1)), self.random_bias[i])
                )
                h_list.append(h_cluster)

            # Concatenate all h to get a unique h matrix
            final_h = np.concatenate(h_list, axis=0)
            h = final_h
            d = np.concatenate([h, X], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

        if self.task_type == "classification":
            y_one_hot = self._one_hot(y, len(np.unique(y)))
            self.beta = (
                np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                @ d.T
                @ y_one_hot
            )
        else:
            self.beta = (
                np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                @ d.T
                @ y
            )

        return self

    def predict(self, X):
        """Make predictions."""
        check_is_fitted(self, "beta")  # Ensure the model is fitted
        X = check_array(X)
        n_samples = len(X)

        h = X.copy()
        cluster_labels = KMeans(n_clusters=self.k, n_init=10).fit_predict(X)

        for i in range(self.n_layer):
            # List to store h for each cluster
            h_list = []

            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self._activation_function(
                    np.dot(cluster_data, self.random_weights[i])
                    + np.dot(np.ones((n_cluster_sample, 1)), self.random_bias[i])
                )
                h_list.append(h_cluster)

            # Concatenate all h to get a unique h matrix
            final_h = np.concatenate(h_list, axis=0)
            h = final_h
            d = np.concatenate([h, X], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = d @ self.beta

        if self.task_type == "classification":
            proba = self._softmax(output)
            return np.argmax(proba, axis=1)
        else:
            return output

    def predict_proba(self, X):
        """Predict probabilities for each class."""
        check_is_fitted(self, "beta")  # Ensure the model is fitted
        X = check_array(X)
        n_samples = len(X)

        h = X.copy()
        cluster_labels = KMeans(n_clusters=self.k, n_init=10).fit_predict(X)

        for i in range(self.n_layer):
            # List to store h for each cluster
            h_list = []

            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self._activation_function(
                    np.dot(cluster_data, self.random_weights[i])
                    + np.dot(np.ones((n_cluster_sample, 1)), self.random_bias[i])
                )
                h_list.append(h_cluster)

            # Concatenate all h to get a unique h matrix
            final_h = np.concatenate(h_list, axis=0)
            h = final_h
            d = np.concatenate([h, X], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = d @ self.beta

        if self.task_type == "classification":
            return self._softmax(output)
        else:
            raise ValueError(
                "Probability predictions are not available for regression tasks."
            )

    def score(self, X, y):
        """Calculate the score of the model."""
        if self.task_type == "classification":
            return accuracy_score(y, self.predict(X)[0])
        else:
            return mean_absolute_error(y, self.predict(X))
