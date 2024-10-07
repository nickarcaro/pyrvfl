import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans


class RVFLK(BaseEstimator, ClassifierMixin):
    """A simple RVFL classifier or regression."""

    def __init__(
        self, n_nodes=100, k=3, lam=1e-3, activation="relu", task_type="classification"
    ):
        assert task_type in [
            "classification",
            "regression",
        ], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.k = k
        self.activation = activation
        self.task_type = task_type
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        self.is_fitted_ = False
        self.classes_ = None

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

    def _get_random_vectors(self, input_size, output_size, range_vals):
        return np.random.uniform(
            range_vals[0], range_vals[1], (input_size, output_size)
        )

    def _one_hot(self, labels, n_classes):
        encoder = OneHotEncoder(sparse_output=False)
        labels = labels.reshape(-1, 1)
        return encoder.fit_transform(labels)

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def fit(self, X, y):
        """Train the RVFL model."""
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # Initialize random weights and biases
        self.random_weights = self._get_random_vectors(
            n_features, self.n_nodes, [-1, 1]
        )
        self.random_bias = self._get_random_vectors(1, self.n_nodes, [0, 1])

        # Apply KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10)
        labels = kmeans.fit_predict(X)

        # List to store h for each cluster
        h_list = []

        # Iterate over each cluster
        for i in range(self.k):
            cluster_data = X[labels == i]
            h = self._activation_function(
                np.dot(cluster_data, self.random_weights) + self.random_bias
            )
            h_list.append(h)

        # Concatenate all h to get a single h matrix
        final_h = np.concatenate(h_list, axis=0)

        # Prepare the design matrix d
        d = np.concatenate([final_h, X], axis=1)
        d = np.concatenate([d, np.ones((d.shape[0], 1))], axis=1)

        if self.task_type == "classification":
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            y_one_hot = self._one_hot(y, n_classes)
            self.beta = (
                np.linalg.inv(self.lam * np.eye(d.shape[1]) + d.T @ d) @ d.T @ y_one_hot
            )
        else:
            self.beta = np.linalg.inv(self.lam * np.eye(d.shape[1]) + d.T @ d) @ d.T @ y

        self.is_fitted_ = True

    def predict(self, X):
        """Make predictions."""
        check_is_fitted(self, "is_fitted_")  # Ensure the model is fitted
        X = check_array(X)

        # Apply KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10)
        labels = kmeans.fit_predict(X)

        # List to store h for each cluster
        h_list = []

        # Iterate over each cluster
        for i in range(self.k):
            cluster_data = X[labels == i]
            h = self._activation_function(
                np.dot(cluster_data, self.random_weights) + self.random_bias
            )
            h_list.append(h)

        # Concatenate all h to get a single h matrix
        final_h = np.concatenate(h_list, axis=0)

        # Prepare the design matrix d
        d = np.concatenate([final_h, X], axis=1)
        d = np.concatenate([d, np.ones((d.shape[0], 1))], axis=1)

        output = d @ self.beta
        if self.task_type == "classification":
            proba = self._softmax(output)
            return np.argmax(proba, axis=1)
        else:
            return output

    def predict_proba(self, X):
        """Predict probabilities for each class."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        # Apply KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10)
        labels = kmeans.fit_predict(X)

        # List to store h for each cluster
        h_list = []

        # Iterate over each cluster
        for i in range(self.k):
            cluster_data = X[labels == i]
            h = self._activation_function(
                np.dot(cluster_data, self.random_weights) + self.random_bias
            )
            h_list.append(h)

        # Concatenate all h to get a single h matrix
        final_h = np.concatenate(h_list, axis=0)

        # Prepare the design matrix d
        d = np.concatenate([final_h, X], axis=1)
        d = np.concatenate([d, np.ones((d.shape[0], 1))], axis=1)

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
            return accuracy_score(y, self.predict(X))
        else:
            return mean_absolute_error(y, self.predict(X))
