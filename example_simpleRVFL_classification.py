import numpy as np
import sklearn.datasets as sk_dataset


from pyrvfl.models.RVFL import RVFL
from pyrvfl.models.RVFLK import RVFLK
from pyrvfl.models.SPRVFL import SPRVFL
from pyrvfl.models.DeepRVFL import DeepRVFL
from pyrvfl.models.DeepRVFLK import DeepRVFLK
from pyrvfl.models.EnsembleDeepRVFL import EnsembleDeepRVFL
from pyrvfl.models.EnsembleDeepRVFLK import EnsembleDeepRVFLK
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)


iris = load_iris()
X, y = iris.data, iris.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear una instancia del modelo RVFL para clasificación
"""
rvfl_clf = DeepRVFL(
    n_nodes=100, lam=1e-3, n_layer=3, activation="relu", task_type="classification"
)

"""
rvfl_clf = EnsembleDeepRVFLK(
    n_nodes=100, lam=1e-3, activation="relu", task_type="classification"
)

# Entrenar el modelo
rvfl_clf.fit(X_train, y_train)

# Hacer predicciones
y_pred = rvfl_clf.predict(X_test)
y_proba = rvfl_clf.predict_proba(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")
precision = precision_score(y_test, y_pred, average="micro")
roc = roc_auc_score(y_test, y_proba, multi_class="ovo")
print(f"Accuracy en clasificación: {accuracy * 100:.2f}%")
print(f"Precisión en clasificación: {precision * 100:.2f}%")
print(f"Recall en clasificación: {recall * 100:.2f}%")
print(f"f1 en clasificación: {f1 * 100:.2f}%")
print(f"roc en clasificación: {roc * 100:.2f}%")
