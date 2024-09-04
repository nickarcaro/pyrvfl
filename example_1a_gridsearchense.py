from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from pyrvfl.models.Ensemble_Deep_RVFL import EnsembleDeepRVFL

# Cargar el dataset de iris
iris = load_iris()
X, y = iris.data, iris.target

# Crear una instancia del modelo EnsembleDeepRVFL
ensemble_rvfl_clf = EnsembleDeepRVFL(task_type="classification")

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    "n_nodes": [50, 100],
    "lam": [1e-6, 1e-3],
    "activation": ["relu", "sigmoid"],
    "n_layer": [2, 3],
}

# Crear el GridSearchCV
grid_search = GridSearchCV(ensemble_rvfl_clf, param_grid, cv=5, scoring="accuracy")

# Ajustar el grid search a los datos
grid_search.fit(X, y)

# Obtener los mejores parámetros encontrados
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor precisión: {grid_search.best_score_ * 100:.2f}%")
