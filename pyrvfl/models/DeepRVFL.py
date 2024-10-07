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
        lam=1e-6,
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
        self.lam = lam
        self.activation = activation
        self.n_layer = n_layer
        self.same_feature = same_feature
        self.task_type = task_type
        self.random_weights = []
        self.random_bias = []
        self.beta = None
        self.data_mean = [None] * self.n_layer
        self.data_std = [None] * self.n_layer
        self.w_random_range = [-1, 1]
        self.b_random_range = [0, 1]
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
    def _get_random_vectors(self, m, n, scale_range):
        return np.random.uniform(scale_range[0], scale_range[1], (m, n))

    def _one_hot(self, labels, n_classes):
        encoder = OneHotEncoder(sparse_output=False)
        labels = labels.reshape(-1, 1)
        return encoder.fit_transform(labels)

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # Inicializar random weights y biases

        h = X.copy()

        # Configurar etiquetas
        if self.task_type == "classification":
            self.classes_ = unique_labels(y)
            y = self._one_hot(y, len(self.classes_))
        else:
            y = y

        # Estandarizar y propagar hacia adelante
        for i in range(self.n_layer):
            # h = self._standardize(h, i)  # Normalizar en cada capa
            self.random_weights.append(
                self._get_random_vectors(h.shape[1], self.n_nodes, self.w_random_range)
            )
            self.random_bias.append(
                self._get_random_vectors(1, self.n_nodes, self.b_random_range)
            )

            h = self._activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_samples, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, X], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

        # Calcular beta
        if n_samples > (self.n_nodes * self.n_layer + n_features):
            self.beta = (
                np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                .dot(d.T)
                .dot(y)
            )
        else:
            self.beta = d.T.dot(
                np.linalg.inv(self.lam * np.identity(n_samples) + np.dot(d, d.T))
            ).dot(y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        n_samples = len(X)
        h = X.copy()

        # Propagación hacia adelante
        for i in range(self.n_layer):
            # h = self._standardize(h, i)
            h = self._activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_samples, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, X], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)

        if self.task_type == "classification":
            proba = self._softmax(output)
            return np.argmax(proba, axis=1)
        elif self.task_type == "regression":
            return output

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        n_samples = len(X)
        h = X.copy()

        for i in range(self.n_layer):

            h = self._activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_samples, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, X], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)

        if self.task_type == "classification":
            return self._softmax(output)
        else:
            raise ValueError(
                "Probability predictions are not available for regression tasks."
            )

    def score(self, X, y):
        if self.task_type == "classification":
            return accuracy_score(y, self.predict(X))
        elif self.task_type == "regression":
            return mean_absolute_error(y, self.predict(X))
