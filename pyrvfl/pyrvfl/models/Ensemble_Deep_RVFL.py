import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.utils.multiclass import unique_labels

# Funciones auxiliares


class EnsembleDeepRVFL(BaseEstimator, ClassifierMixin):
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
        self.beta = []
        self.classes_ = None
        self.is_fitted_ = False
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

    def _standardize(self, X, layer_index):
        """Normalizar los datos"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-5)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        h = X.copy()

        if self.task_type == "classification":
            self.classes_ = unique_labels(y)
            y = self._one_hot(y, len(self.classes_))
        else:
            y = y

        # Estandarizar los datos de entrada
        X = self._standardize(X, 0)

        for i in range(self.n_layer):
            h = self._standardize(h, i)

            # Generar pesos y sesgos aleatorios
            self.random_weights.append(
                self._get_random_vectors(h.shape[1], self.n_nodes, self.w_random_range)
            )
            self.random_bias.append(
                self._get_random_vectors(1, self.n_nodes, self.b_random_range)
            )

            # Propagación hacia adelante
            h = self._activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_samples, 1]), self.random_bias[i])
            )

            # Concatenar h con X
            d = np.concatenate([h, X], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            # Calcular beta
            if n_samples > (self.n_nodes + n_features):
                self.beta.append(
                    np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                    .dot(d.T)
                    .dot(y)
                )
            else:
                self.beta.append(
                    d.T.dot(
                        np.linalg.inv(
                            self.lam * np.identity(n_samples) + np.dot(d, d.T)
                        )
                    ).dot(y)
                )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        n_samples = len(X)
        h = X.copy()
        X = self._standardize(X, 0)
        outputs = []

        for i in range(self.n_layer):
            h = self._standardize(h, i)
            h = self._activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_samples, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, X], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))

        if self.task_type == "classification":
            add_proba = self._softmax(np.sum(outputs, axis=0))
            return np.argmax(add_proba, axis=1)
        elif self.task_type == "regression":
            return np.mean(outputs, axis=0)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        n_samples = len(X)
        h = X.copy()
        X = self._standardize(X, 0)
        outputs = []

        for i in range(self.n_layer):
            h = self._standardize(h, i)
            h = self._activation_function(
                np.dot(h, self.random_weights[i])
                + np.dot(np.ones([n_samples, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, X], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))

        if self.task_type == "classification":
            add_proba = self._softmax(np.sum(outputs, axis=0))
            return add_proba
        else:
            raise ValueError(
                "Probability predictions are not available for regression tasks."
            )

    def score(self, X, y):
        if self.task_type == "classification":
            return accuracy_score(y, self.predict(X))
        elif self.task_type == "regression":
            return mean_absolute_error(y, self.predict(X))
