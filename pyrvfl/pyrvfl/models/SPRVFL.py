import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from numpy import linalg as LA


class SPRVFL(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_nodes=100,
        lam=1e-3,
        activation="relu",
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
        self.same_feature = same_feature
        self.task_type = task_type
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        self.is_fitted_ = False  # Verificar si el modelo está ajustado
        self.classes_ = None  # Almacenar las clases para clasificación

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

    def fista1(self, A, b, l, maxit):
        J_save = np.zeros((maxit, 1))
        AA = np.dot(A.T, A)
        Lf = np.max(LA.eigvals(AA))
        Li = 1 / Lf
        alp = l * Li
        m = A.shape[1]
        x = np.zeros((m, b.shape[1]))
        yk = x
        tk = 1
        L1 = 2 * Li * AA
        L2 = 2 * Li * np.dot(A.T, b)

        for _ in range(maxit):
            med = np.dot(L1, yk)
            ck = yk - med + L2
            x1 = (np.maximum(abs(ck) - alp, 0)) * np.sign(ck)

            sqrt = math.sqrt(1 + 4 * tk**2)
            tk1 = 0.5 + 0.5 * sqrt
            tt = (tk - 1) / tk1
            yk = x1 + tt * (x - x1)
            tk = tk1
            x = x1

        return x

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        self.random_weights = self._get_random_vectors(
            n_features, self.n_nodes, [-1, 1]
        )
        self.random_bias = self._get_random_vectors(1, self.n_nodes, [0, 1])

        h = self._activation_function(np.dot(X, self.random_weights) + self.random_bias)

        rbeta = self.fista1(h, X, 0.01, 100)
        rbeta = np.transpose(rbeta)
        rbias = np.sum(rbeta, axis=0) / self.n_nodes
        rbias = rbias.reshape(-1, self.n_nodes)

        h1 = np.dot(X, rbeta) + np.dot(np.ones((n_samples, 1)), rbias)
        h = h1

        d = np.concatenate([h, X], axis=1)
        d = np.concatenate([d, np.ones((n_samples, 1))], axis=1)

        if self.task_type == "classification":
            self.classes_ = np.unique(
                y
            )  # Definir las clases encontradas en los datos de entrenamiento
            n_classes = len(self.classes_)
            y_one_hot = self._one_hot(y, n_classes)
            self.beta = (
                np.linalg.inv(self.lam * np.eye(d.shape[1]) + d.T @ d) @ d.T @ y_one_hot
            )
        elif self.task_type == "regression":
            self.beta = np.linalg.inv(self.lam * np.eye(d.shape[1]) + d.T @ d) @ d.T @ y

        self.is_fitted_ = True  # Marcar el modelo como ajustado
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")  # Verificar si el modelo está ajustado
        X = check_array(X)
        h = self._activation_function(np.dot(X, self.random_weights) + self.random_bias)
        d = np.concatenate([h, X], axis=1)
        d = np.concatenate([d, np.ones((X.shape[0], 1))], axis=1)
        output = d @ self.beta

        if self.task_type == "classification":
            proba = self._softmax(output)
            return np.argmax(proba, axis=1)
        elif self.task_type == "regression":
            return output

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        h = self._activation_function(np.dot(X, self.random_weights) + self.random_bias)
        d = np.concatenate([h, X], axis=1)
        d = np.concatenate([d, np.ones((X.shape[0], 1))], axis=1)
        output = d @ self.beta
        return self._softmax(output)

    def score(self, X, y):
        if self.task_type == "classification":
            return accuracy_score(y, self.predict(X))
        elif self.task_type == "regression":
            return mean_absolute_error(y, self.predict(X))
