import numpy as np
import pandas as pd
from tqdm import tqdm
import time


def stratified_shuffle_split_indices(X, y, n_splits=5, test_size=0.2, random_state=42):
    """
    Generate stratified shuffle split indices for cross-validation.

    Parameters:
    - X: Features dataset.
    - y: Labels dataset.
    - n_splits: Number of splits for cross-validation.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed for random number generator.

    Yields:
    - train_indices: Indices for the training set.
    - test_indices: Indices for the test set.
    """
    n_samples = len(X)
    n_test = int(np.ceil(test_size * n_samples))

    if random_state is not None:
        np.random.seed(random_state)

    for _ in range(n_splits):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        yield train_indices, test_indices


def gridSearch(
    model_class,
    X,
    y,
    param_grid,
    test_data,
    test_labels,
    metrics=None,
    n_iterations=100,
    n_splits=5,
    test_size=0.2,
    random_state=42,
    generate_plot=False,
    save_csv=False,
    task_type=None,  # Nuevo parámetro para el tipo de tarea
):
    """
    Realiza grid search con validación cruzada para encontrar los mejores hiperparámetros para un modelo.

    task_type: str
        Tipo de tarea, puede ser 'classification' o 'regression'.
    """

    # Selección de métricas dependiendo del tipo de tarea
    if task_type == "classification":
        if metrics is None:
            metrics = ["accuracy"]
    elif task_type == "regression":
        if metrics is None:
            metrics = ["mae"]

    best_score = -np.inf if task_type == "classification" else np.inf
    best_params = None
    final_model = None

    # Iterar sobre cada combinación de hiperparámetros
    for params in tqdm(param_grid):
        # Crear una instancia del modelo con los hiperparámetros actuales
        model = model_class(**params)

        scores = []

        # Validación cruzada
        for train_indices, val_indices in stratified_shuffle_split_indices(
            X, y, n_splits=n_splits, test_size=test_size, random_state=random_state
        ):
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            if task_type == "classification":
                current_score = np.mean(
                    [metric(y_val, predictions) for metric in metrics]
                )
            elif task_type == "regression":
                current_score = np.mean(
                    [metric(y_val, predictions) for metric in metrics]
                )

            scores.append(current_score)

        mean_score = np.mean(scores)

        # Actualizar el mejor modelo si se encuentra uno con mejor rendimiento
        if (task_type == "classification" and mean_score > best_score) or (
            task_type == "regression" and mean_score < best_score
        ):
            best_score = mean_score
            best_params = params
            final_model = model_class(task_type=task_type, **params)
            final_model.fit(X, y)

    print(f"Mejor combinación de hiperparámetros: {best_params}")
    print(f"Mejor puntuación: {best_score}")

    # Evaluar el modelo final en el conjunto de prueba
    test_predictions = final_model.predict(test_data)

    if task_type == "classification":
        test_eval_results = {
            metric.__name__: metric(test_labels, test_predictions) for metric in metrics
        }
    elif task_type == "regression":
        test_eval_results = {
            metric.__name__: metric(test_labels, test_predictions) for metric in metrics
        }

    print("Resultados de la evaluación en el conjunto de prueba:", test_eval_results)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "final_model": final_model,
        "test_eval_results": test_eval_results,
    }
