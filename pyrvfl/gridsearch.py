import numpy as np
from pyrvfl.models.rvfl import RVFL
from pyrvfl.models.SPRVFL import SPRVFL
import time
from pyrvfl.utils.utils import standardize_rvfl, stratified_shuffle_split_indices
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo
from scipy.interpolate import griddata


def gridsearch(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    n_iterations,
    n_splits,
    neurons,
    reg_values,
    metrics_to_evaluate,
):
    best_accuracy = 0.0
    best_num_neurons = 0
    best_regularization_value = 0.0

    pass


def run_classification_experiment():
    # Cargar el conjunto de datos "Ionosphere"
    ionosphere = fetch_ucirepo(id=52)

    def prepare_data_classify(proportion, dataset):
        label = dataset.data.targets.copy()
        data = dataset.data.features

        encoding = {"g": 1, "b": 0}
        label["Class"].replace(encoding, inplace=True)
        label = label.squeeze()

        shuffle_index = np.arange(len(label))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(label))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]

        data_train = data.iloc[train_index]
        label_train = label.iloc[train_index]
        data_val = data.iloc[val_index]
        label_val = label.iloc[val_index]
        return (data_train.to_numpy(), label_train.to_numpy()), (
            data_val.to_numpy(),
            label_val.to_numpy(),
        )

    (X_train, y_train), (X_test, y_test) = prepare_data_classify(0.8, ionosphere)

    X_train = standardize_rvfl(X_train)
    X_test = standardize_rvfl(X_test)

    metrics_to_evaluate = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]

    num_neurons_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    regularization_values = [2**exp for exp in range(-6, 13, 2)]

    n_iterations = 100
    test_size = 0.2
    n_splits = 5
    best_accuracy = 0.0
    best_num_neurons = 0
    best_regularization_value = 0.0

    accuracy_data = []
    num_neurons_data = []
    regularization_data = []

    f1_data = []
    roc_data = []
    precision_data = []
    recall_data = []

    init_time = time.time()

    for _ in tqdm(range(n_iterations)):
        for regularization_value in regularization_values:
            for num_neurons in num_neurons_list:
                model = SPRVFL(
                    n_nodes=num_neurons,
                    lam=regularization_value,
                    activation="relu",
                    task_type="classification",
                )

                errors = []
                f1 = []
                precision = []
                recall = []
                roc_auc = []

                for train_indices, test_indices in stratified_shuffle_split_indices(
                    X_train,
                    y_train,
                    n_splits=n_splits,
                    test_size=test_size,
                    random_state=42,
                ):
                    X_train_split, X_val_split = (
                        X_train[train_indices],
                        X_train[test_indices],
                    )
                    y_train_split, y_val_split = (
                        y_train[train_indices],
                        y_train[test_indices],
                    )

                    model.fit(X_train_split, y_train_split)

                    metric = model.eval(
                        X_val_split, y_val_split, metrics=metrics_to_evaluate
                    )

                    errors.append(metric["accuracy"])
                    f1.append(metric["f1_score"])
                    roc_auc.append(metric["roc_auc"])
                    precision.append(metric["precision"])
                    recall.append(metric["recall"])

                mean_accuracy = np.mean(errors)
                mean_f1 = np.mean(f1)
                mean_roc = np.mean(roc_auc)
                mean_precision = np.mean(precision)
                mean_recall = np.mean(recall)

                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_num_neurons = num_neurons
                    best_regularization_value = regularization_value

                roc_data.append(mean_roc)
                precision_data.append(mean_precision)
                recall_data.append(mean_recall)
                f1_data.append(mean_f1)
                accuracy_data.append(mean_accuracy)
                num_neurons_data.append(num_neurons)
                regularization_data.append(regularization_value)

    print(f"Best Accuracy: {(best_accuracy):.1f}")
    print("Best Number of Neurons:", best_num_neurons)
    print("Best Regularization Value:", best_regularization_value)

    num_neurons_data = np.array(num_neurons_data)
    regularization_data = np.array(regularization_data)
    accuracy_data = np.array(accuracy_data)

    num_neurons_data = num_neurons_data.reshape(len(regularization_values), -1)
    regularization_data = regularization_data.reshape(len(regularization_values), -1)
    accuracy_data = accuracy_data.reshape(len(regularization_values), -1)

    finish_time = time.time()
    train_time = finish_time - init_time

    num_neurons_mesh, regularization_mesh = np.meshgrid(
        num_neurons_list, regularization_values
    )

    accuracy_mesh = griddata(
        (num_neurons_data.ravel(), regularization_data.ravel()),
        accuracy_data.ravel(),
        (num_neurons_mesh, regularization_mesh),
        method="linear",
    )

    hparams = (
        f"neurons: {best_num_neurons} Regularization Value {best_regularization_value}"
    )

    int_mean_acc = f"{(np.mean(accuracy_data)):.3f}"
    int_std_acc = f" {(np.std(accuracy_data)):.3f}"
    int_mean_f1 = f"{(np.mean(f1_data)):.3f}"
    int_std_f1 = f"{(np.std(f1_data)):.3f}"
    int_mean_roc = f"{(np.mean(roc_data)):.3f}"
    int_std_roc = f"{(np.std(roc_data)):.3f}"
    int_mean_p = f"{(np.mean(precision_data)):.3f}"
    int_std_p = f"{(np.std(precision_data)):.3f}"
    int_mean_r = f"{(np.mean(recall_data)):.3f}"
    int_std_r = f"{(np.std(recall_data)):.3f}"

    print(f"Mean Accuracy: {int_mean_acc} - Std Accuracy: {int_std_acc}")
    print(f"Mean F1: {int_mean_f1} - Std F1: {int_std_f1}")
    print(f"Mean ROC: {int_mean_roc} - Std ROC: {int_std_roc}")
    print(f"Mean Precision: {int_mean_p} - Std Precision: {int_std_p}")
    print(f"Mean Recall: {int_mean_r} - Std Recall: {int_std_r}")
    print(f"Training Time: {train_time:.3f} seconds")

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        num_neurons_mesh, np.log2(regularization_mesh), accuracy_mesh, cmap="inferno"
    )

    ax.set_xlabel("Number of Neurons")
    ax.set_ylabel("Regularization Value")
    ax.set_zlabel("Accuracy")

    plt.savefig(f"RVFL.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"RVFL.eps", format="eps", bbox_inches="tight")
    plt.show()

    test_errors = []
    test_f_score = []
    test_roc_score = []
    test_p_score = []
    test_r_score = []

    start_time = time.time()
    for _ in tqdm(range(n_iterations)):
        model = SPRVFL(
            n_nodes=best_num_neurons,
            lam=best_regularization_value,
            activation="relu",
            task_type="classification",
        )

        model.fit(X_train, y_train)

        metric = model.eval(X_test, y_test, metrics=metrics_to_evaluate)
        test_errors.append(metric["accuracy"])
        test_f_score.append(metric["f1_score"])
        test_p_score.append(metric["precision"])
        test_r_score.append(metric["recall"])
        test_roc_score.append(metric["roc_auc"])

    end_time = time.time()
    test_execution_time = end_time - start_time

    print("total:", len(test_errors))
    print(f"acc: {(np.mean(test_errors)):.3f} ({(np.std(test_errors)):.3f}) &")
    print(f"roc: {(np.mean(test_roc_score)):.3f} ({(np.std(test_roc_score)):.3f}) &")
    print(f"f1: {(np.mean(test_f_score)):.3f} ({(np.std(test_f_score)):.3f}) &")
    print(f"precision: {(np.mean(test_p_score)):.3f} ({(np.std(test_p_score)):.3f}) &")
    print(f"recall: {(np.mean(test_r_score)):.3f} ({(np.std(test_r_score)):.3f}) &")
    print(f"Time exec: & {(test_execution_time):.1f} seconds")

    ex_mean_acc = f"{(np.mean(test_errors)):.3f}"
    ex_std_acc = f"{(np.std(test_errors)):.3f}"
    ex_mean_f1 = f"{(np.mean(test_f_score)):.3f}"
    ex_std_f1 = f"{(np.std(test_f_score)):.3f}"
    ex_mean_roc = f"{(np.mean(test_roc_score)):.3f}"
    ex_std_roc = f"{(np.std(test_roc_score)):.3f}"
    ex_mean_p = f"{(np.mean(test_p_score)):.3f}"
    ex_std_p = f"{(np.std(test_p_score)):.3f}"
    ex_mean_r = f"{(np.mean(test_r_score)):.3f}"
    ex_std_r = f"{(np.std(test_r_score)):.3f}"

    Time_test_rvfl = test_execution_time

    reporter_rvfl = pd.DataFrame(
        {
            "experiment": ["Ionosphere-rvfl"],
            "model": ["RVFL"],
            "hyparams": [hparams],
            "train mean acc": [int_mean_acc],
            "train std acc": [int_std_acc],
            "train mean f1": [int_mean_f1],
            "train std f1": [int_std_f1],
            "train mean roc": [int_mean_roc],
            "train std roc": [int_std_roc],
            "train mean precision": [int_mean_p],
            "train std precision": [int_std_p],
            "train mean recall": [int_mean_r],
            "train std recall": [int_std_r],
            "train exec time": [train_time],
            "test mean acc": [ex_mean_acc],
            "test std acc": [ex_std_acc],
            "test mean f1": [ex_mean_f1],
            "test std f1": [ex_std_f1],
            "test mean roc": [ex_mean_roc],
            "test std roc": [ex_std_roc],
            "test mean precision": [ex_mean_p],
            "test std precision": [ex_std_p],
            "test mean recall": [ex_mean_r],
            "test std recall": [ex_std_r],
            "test exec time": [Time_test_rvfl],
        }
    )

    reporter = pd.DataFrame()
    reporter = reporter.concat(reporter_rvfl, ignore_index=True)
    reporter.to_csv(f"rvfl_report_final.csv")


# Ejecución del experimento
if __name__ == "__main__":
    run_classification_experiment()
