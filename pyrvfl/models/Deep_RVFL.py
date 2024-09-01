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


class DeepRVFL:

    def __init__(
        self,
        n_nodes,
        lam,
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
        self.w_random_range = [-1, 1]
        self.b_random_range = [0, 1]
        self.random_weights = []
        self.random_bias = []
        self.beta = None
        self.n_layer = n_layer
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.gamma = [np.ones(self.n_nodes) for _ in range(n_layer)]
        self.beta_bn = [np.zeros(self.n_nodes) for _ in range(n_layer)]
        self.task_type = task_type

    def batch_normalization(self, X, layer_index, epsilon=1e-5):
        mean = np.mean(X, axis=0)
        variance = np.var(X, axis=0)
        X_normalized = (X - mean) / np.sqrt(variance + epsilon)
        out = self.gamma[layer_index] * X_normalized + self.beta_bn[layer_index]
        return out

    def fit(self, data, label):
        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        h = data
        for i in range(self.n_layer):
            # Generar pesos y sesgos aleatorios para cada capa
            self.random_weights.append(
                get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range)
            )
            self.random_bias.append(
                get_random_vectors(1, self.n_nodes, self.b_random_range)
            )
            # Propagación hacia adelante
            h = np.dot(h, self.random_weights[i]) + self.random_bias[i]
            # Aplicar Normalización por Lote
            h = self.batch_normalization(h, i)
            # Aplicar función de activación
            h = self.activation_function(h)

        # Calcular beta para la capa de salida
        if self.task_type == "classification":
            n_class = len(np.unique(label))
            label = one_hot(label, n_class)
        elif self.task_type == "regression":
            label = label

        self.beta = np.dot(
            np.linalg.pinv(np.dot(h.T, h) + np.eye(self.n_nodes) * self.lam),
            np.dot(h.T, label),
        )

    def predict(self, data):
        h = data
        for i in range(self.n_layer):
            h = np.dot(h, self.random_weights[i]) + self.random_bias[i]
            h = self.batch_normalization(h, i)
            h = self.activation_function(h)

        output = np.dot(h, self.beta)

        if self.task_type == "classification":
            result = np.argmax(softmax(output), axis=1)
            proba = softmax(output)
            return result, proba
        elif self.task_type == "regression":
            return output.flatten()  # Devolver las predicciones para regresión

    def eval(self, data, label, metrics=None):
        if metrics is None:
            metrics = ["accuracy"] if self.task_type == "classification" else ["mae"]

        if self.task_type == "classification":
            result, proba = self.predict(data)
            results = {}

            if "accuracy" in metrics:
                results["accuracy"] = accuracy_score(label, result)
            if "f1_score" in metrics:
                results["f1_score"] = f1_score(label, result)["f1_score"]
            if "precision" in metrics:
                results["precision"] = f1_score(label, result)["precision"]
            if "recall" in metrics:
                results["recall"] = f1_score(label, result)["recall"]
            if "tpr" in metrics:
                results["tpr"] = f1_score(label, result)["tpr"]
            if "fnr" in metrics:
                results["fnr"] = f1_score(label, result)["fnr"]
            if "fpr" in metrics:
                results["fpr"] = f1_score(label, result)["fpr"]
            if "roc_auc" in metrics and len(np.unique(label)) == 2:
                results["roc_auc"] = roc_auc(label, proba[:, 1])

            return results

        elif self.task_type == "regression":
            result = self.predict(data)
            if metrics is None:
                metrics = ["mae"]  # Métrica predeterminada

            results = {}

            if "mae" in metrics:
                results["mae"] = mae(label, result)
            if "mse" in metrics:
                results["mse"] = mse(label, result)
            if "rmse" in metrics:
                results["rmse"] = rmse(label, result)
            if "r2" in metrics:
                results["r2"] = r2_score(label, result)

            return results
