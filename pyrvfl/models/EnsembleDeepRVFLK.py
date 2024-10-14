import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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
        self.w_random_range = [-1, 1]
        self.b_random_range = [0, 1]
        self.activation = activation
        self.n_layer = n_layer
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer

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

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_sample, n_feature = X.shape
        h = X.copy()

        if self.task_type == "classification":
            self.classes_ = unique_labels(y)
            y = self.one_hot(y, len(self.classes_))
        else:
            y = y

        for i in range(self.n_layer):
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

            d = np.concatenate([h, X], axis=1)
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
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        data = check_array(X)
        n_sample = len(X)
        h = data.copy()

        outputs = []

        for i in range(self.n_layer):
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

            d = np.concatenate([h, X], axis=1)
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

    def predict_proba(self, X):
        """Predict probabilities for each class."""
        check_is_fitted(self, "is_fitted_")
        data = check_array(X)
        n_sample = len(X)
        h = data.copy()

        outputs = []

        for i in range(self.n_layer):
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

            d = np.concatenate([h, X], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))

        if self.task_type == "classification":
            return self._softmax(np.sum(outputs, axis=0))
        else:
            raise ValueError(
                "Probability predictions are not available for regression tasks."
            )

    def eval(self, data, label):
        data, label = check_X_y(data, label)
        outputs = self.predict(data)
        if self.task_type == "classification":
            return accuracy_score(label, outputs)
        elif self.task_type == "regression":
            return mean_absolute_error(label, outputs)
