import numpy as np
from pyrvfl.models.rvfl import RVFL
import time
from pyrvfl.utils.utils import standardize_rvfl, stratified_shuffle_split_indices
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from ucimlrepo import fetch_ucirepo


# Función para ejecutar un experimento con un conjunto de hiperparámetros
def run_experiment(model, X_train, y_train, X_test, y_test):
    # Estandarizar los datos de entrenamiento y prueba
    # X_train_std, X_test_std = standardize_rvfl(X_train, X_test)

    X_train_std = standardize_rvfl(X_train)
    X_test_std = standardize_rvfl(X_test)

    # Entrenar el modelo
    start_time = time.time()
    model.fit(X_train_std, y_train)
    exec_time = time.time() - start_time

    # Predecir y calcular las métricas
    train_predictions = model.predict(X_train_std)
    test_predictions = model.predict(X_test_std)

    train_acc = np.mean(train_predictions == y_train)
    test_acc = np.mean(test_predictions == y_test)

    return {"train_acc": train_acc, "test_acc": test_acc, "exec_time": exec_time}


# Función principal para optimizar y ejecutar experimentos
def optimize_and_run_experiments(X, y, n_experiments=10):
    # Definir la cuadrícula de hiperparámetros
    hyperparams = [
        {"n_nodes": n, "lam": l} for n in [50, 100, 150] for l in [0.01, 0.1, 1.0]
    ]

    results = []

    # Usar multiprocesamiento para ejecutar experimentos en paralelo
    with Pool() as pool:
        for i in tqdm(range(n_experiments)):
            # División estratificada
            train_idx, test_idx = stratified_shuffle_split_indices(X, y, test_size=0.3)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            for hp in hyperparams:
                model = RVFL(n_nodes=hp["n_nodes"], lam=hp["lam"])
                result = pool.apply_async(
                    run_experiment, (model, X_train, y_train, X_test, y_test)
                )
                results.append((hp, result))

        # Recoger resultados
        final_results = []
        for hp, result in results:
            res = result.get()
            final_results.append(
                {
                    "n_nodes": hp["n_nodes"],
                    "lam": hp["lam"],
                    "train_acc": res["train_acc"],
                    "test_acc": res["test_acc"],
                    "exec_time": res["exec_time"],
                }
            )

    return pd.DataFrame(final_results)


# Cargar el conjunto de datos "Ionosphere"
def load_ionosphere_data():
    ionosphere = fetch_ucirepo(id=52)  # ID del conjunto de datos "Ionosphere" en UCI
    X = ionosphere.data.features

    label = ionosphere.data.targets.copy()

    # Define the encoding dictionary to map letter labels to numbers
    encoding = {"g": 1, "b": 0}

    # Replace the letter labels with the corresponding numbers
    label["Class"].replace(encoding, inplace=True)
    y = label.squeeze()

    return X, y


# Visualizar los resultados
def plot_results(results):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = results["n_nodes"]
    y = results["lam"]
    z = results["test_acc"]

    ax.scatter(x, y, z, c="r", marker="o")

    ax.set_xlabel("n_nodes")
    ax.set_ylabel("lambda")
    ax.set_zlabel("Test Accuracy")

    plt.show()


# Ejecución del código
if __name__ == "__main__":
    X, y = load_ionosphere_data()
    optimized_results = optimize_and_run_experiments(X, y, n_experiments=10)

    print(optimized_results)
    plot_results(optimized_results)
