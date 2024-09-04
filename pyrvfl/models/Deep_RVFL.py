import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.utils.multiclass import unique_labels


class DeepRVFL(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_nodes=100,
        lam=1e-3,
        activation="relu",
        n_layer=3,
        task_type="classification",
    ):
        assert task_type in [
            "classification",
            "regression",
        ], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.activation = activation
        self.n_layer = n_layer
        self.task_type = task_type
        self.random_weights = []
        self.random_bias = []
        self.beta = None
        self.is_fitted_ = False
        self.classes_ = None

        # Para Batch Normalization
        self.gamma = [np.ones(self.n_nodes) for _ in range(n_layer)]
        self.beta_bn = [np.zeros(self.n_nodes) for _ in range(n_layer)]

    def _activation_function(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "linear":
            return x
        elif self.activation == "hardlim":
            return np.where(x >= 0, 1, 0)
        elif self.activation == "softlim":
            return np.where(x >= 0, 1, -1)
        elif self.activation == "sin":
            return np.sin(x)
        elif self.activation == "hardlims":
            return np.where(x >= 0, 1, -1)
        elif self.activation == "tribas":
            return np.maximum(0, 1 - np.abs(x))
        elif self.activation == "radbas":
            return np.exp(-(x**2))
        else:
            raise ValueError("Unsupported activation function")

    # Funciones auxiliares
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

    def batch_normalization(self, X, layer_index, epsilon=1e-5):
        mean = np.mean(X, axis=0)
        variance = np.var(X, axis=0)
        X_normalized = (X - mean) / np.sqrt(variance + epsilon)
        out = self.gamma[layer_index] * X_normalized + self.beta_bn[layer_index]
        return out

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        h = X
        for i in range(self.n_layer):
            self.random_weights.append(
                self._get_random_vectors(
                    n_features if i == 0 else self.n_nodes, self.n_nodes, [-1, 1]
                )
            )
            self.random_bias.append(self._get_random_vectors(1, self.n_nodes, [0, 1]))
            h = np.dot(h, self.random_weights[i]) + self.random_bias[i]
            h = self.batch_normalization(h, i)
            h = self._activation_function(h)

        # Para la clasificación, definir las clases y one-hot encoding
        if self.task_type == "classification":
            self.classes_ = unique_labels(y)
            n_classes = len(self.classes_)
            y = self._one_hot(y, n_classes)

        self.beta = np.dot(
            np.linalg.pinv(np.dot(h.T, h) + np.eye(self.n_nodes) * self.lam),
            np.dot(h.T, y),
        )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        h = X
        for i in range(self.n_layer):
            h = np.dot(h, self.random_weights[i]) + self.random_bias[i]
            h = self.batch_normalization(h, i)
            h = self._activation_function(h)

        output = np.dot(h, self.beta)

        if self.task_type == "classification":
            return np.argmax(self._softmax(output), axis=1)
        elif self.task_type == "regression":
            return output.flatten()

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        h = X
        for i in range(self.n_layer):
            h = np.dot(h, self.random_weights[i]) + self.random_bias[i]
            h = self.batch_normalization(h, i)
            h = self._activation_function(h)

        output = np.dot(h, self.beta)
        return self._softmax(output)

    def score(self, X, y):
        if self.task_type == "classification":
            return accuracy_score(y, self.predict(X))
        elif self.task_type == "regression":
            return mean_absolute_error(y, self.predict(X))
