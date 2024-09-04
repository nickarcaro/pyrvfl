import numpy as np
import sklearn.datasets as sk_dataset


# from pyrvfl.models.rvfl import RVFL
from pyrvfl.models.rvfl import RVFL
from pyrvfl.models.SPRVFL import SPRVFL
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# from pyrvfl.utils.utils import standardize_rvfl, prepare_data_classify


# (train_data, train_labels), (test_data, test_labels) = prepare_data_classify(
#    sk_dataset.load_breast_cancer(), 0.8
# )

# train_data = standardize_rvfl(train_data)
# test_data = standardize_rvfl(test_data)

# Cargar el conjunto de datos iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear una instancia del modelo RVFL para clasificación
rvfl_clf = SPRVFL(n_nodes=100, lam=1e-3, activation="relu", task_type="classification")

# Entrenar el modelo
rvfl_clf.fit(X_train, y_train)

# Hacer predicciones
y_pred = rvfl_clf.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en clasificación: {accuracy * 100:.2f}%")


from sklearn.model_selection import cross_validate

# Realizar validación cruzada con múltiples métricas
scoring = ["accuracy", "precision_macro", "recall_macro"]
cv_results = cross_validate(rvfl_clf, X, y, cv=5, scoring=scoring)

print(f"Accuracy en cada pliegue: {cv_results['test_accuracy']}")
print(f"Precisión promedio: {cv_results['test_accuracy'].mean() * 100:.2f}%")
print(
    f"Precisión macro promedio: {cv_results['test_precision_macro'].mean() * 100:.2f}%"
)
print(f"Recall macro promedio: {cv_results['test_recall_macro'].mean() * 100:.2f}%")
