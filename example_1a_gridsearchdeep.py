from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from pyrvfl.models.Deep_RVFL import DeepRVFL
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import numpy as np

# Cargar el dataset de iris
iris = load_iris()
X, y = iris.data, iris.target

# Crear una instancia del modelo DeepRVFL
deep_rvfl_clf = DeepRVFL(task_type="classification")

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    "n_nodes": [50, 100, 150],
    "lam": [1e-3, 1e-2, 1e-1],
    "activation": ["relu", "sigmoid", "tanh"],
    "n_layer": [2, 3, 4],
}

# Crear el GridSearchCV
grid_search = GridSearchCV(deep_rvfl_clf, param_grid, cv=5, scoring="accuracy")

# Ajustar el grid search a los datos
grid_search.fit(X, y)

# Obtener los mejores parámetros encontrados
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor precisión: {grid_search.best_score_ * 100:.2f}%")


### The right part ###
# define the pipeline to include scaling and the model.
# This pipeline will be the input to cross_val_score, instead of the model.
steps = list()
steps.append(("scaler", MinMaxScaler()))
steps.append(("model", deep_rvfl_clf))
pipeline = Pipeline(steps=steps)


# prepare the cross-validation procedure
cv = KFold(n_splits=5, random_state=42, shuffle=True)


# evaluate model
scores2 = cross_val_score(pipeline, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

for score in scores2:
    print("Accuracy for this fold is: ", score)

# Mean accuracy
print(" Mean accuracy over all folds is: ", (np.mean(scores2)))
