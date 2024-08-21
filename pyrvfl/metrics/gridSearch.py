import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
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
):
    """
    Perform grid search with cross-validation to find the best hyperparameters for a model.

    Parameters:
    - model_class: The model class to be instantiated.
    - X: Features dataset.
    - y: Labels dataset.
    - param_grid: Dictionary of hyperparameters to search.
    - test_data: Features of the test dataset.
    - test_labels: Labels of the test dataset.
    - metrics: List of metrics to evaluate.
    - n_iterations: Number of iterations for grid search.
    - n_splits: Number of splits for cross-validation.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed for random number generator.
    - generate_plot: Whether to generate a 3D surface plot.
    - save_csv: Whether to save results to CSV files.

    Returns:
    - Dictionary with best hyperparameters, best score, final model, and test evaluation results.
    """
    if metrics is None:
        metrics = ["accuracy"]

    best_score = -np.inf
    best_params = {}

    # Arrays to store all iteration data
    accuracy_data = []
    num_neurons_data = []
    regularization_data = []
    f1_data = []
    precision_data = []
    recall_data = []
    roc_data = []

    # To store metrics over all iterations
    metrics_summary = {metric: [] for metric in metrics}

    hyperparam_search_times = []

    # Begin iterating
    for _ in tqdm(range(n_iterations), desc="Iterations"):
        start_time = time.time()
        for params in param_grid:
            scores = {metric: [] for metric in metrics}

            for train_indices, test_indices in stratified_shuffle_split_indices(
                X, y, n_splits=n_splits, test_size=test_size, random_state=random_state
            ):
                X_train, X_val = X[train_indices], X[test_indices]
                y_train, y_val = y[train_indices], y[test_indices]

                # Instantiate the model with current parameters
                model = model_class(**params)

                model.fit(X_train, y_train)
                eval_results = model.eval(X_val, y_val, metrics=metrics)

                for metric in metrics:
                    scores[metric].append(eval_results[metric])

            # Calculate the mean score for each metric
            mean_scores = {metric: np.mean(scores[metric]) for metric in metrics}

            # Store scores in metrics_summary
            for metric in metrics:
                metrics_summary[metric].append(mean_scores[metric])

            # Determine if this is the best score based on the primary metric (assumed to be the first one)
            primary_metric = metrics[0]
            if mean_scores[primary_metric] > best_score:
                best_score = mean_scores[primary_metric]
                best_params = params

            # Collect data for 3D plot analysis
            accuracy_data.append(mean_scores["accuracy"])
            num_neurons_data.append(params.get("n_nodes", None))
            regularization_data.append(params.get("lam", None))
            if "f1_score" in metrics:
                f1_data.append(mean_scores["f1_score"])
            if "precision" in metrics:
                precision_data.append(mean_scores["precision"])
            if "recall" in metrics:
                recall_data.append(mean_scores["recall"])
            if "roc_auc" in metrics:
                roc_data.append(mean_scores["roc_auc"])

        end_time = time.time()
        hyperparam_search_times.append(end_time - start_time)

    # Calculate mean and std for each metric over all iterations
    metrics_means = {metric: np.mean(metrics_summary[metric]) for metric in metrics}
    metrics_stds = {metric: np.std(metrics_summary[metric]) for metric in metrics}

    print("\nTraining results:")

    print("\nMean of metrics over iterations:")
    for metric in metrics:
        print(f"{metric.capitalize()} Mean: {metrics_means[metric]:.2f}")
    print("\nStandard deviation of metrics over iterations:")
    for metric in metrics:
        print(f"{metric.capitalize()} Std: {metrics_stds[metric]:.2f}")

    # Train the final model with the best parameters on the entire training set
    test_eval_times = []
    for _ in tqdm(range(n_iterations), desc="Test Evaluations"):
        test_start_time = time.time()
        final_model = model_class(**best_params)
        final_model.fit(X, y)

        # Evaluate the final model on the test set
        test_eval_results = final_model.eval(test_data, test_labels, metrics=metrics)
        test_end_time = time.time()
        test_eval_times.append(test_end_time - test_start_time)

    print("\nTest Set Evaluation:")
    for metric in metrics:
        print(f"Test {metric.capitalize()}: {test_eval_results[metric]:.2f}")

    print(
        f"\nMean Hyperparameter Search Time: {np.mean(hyperparam_search_times):.2f} seconds"
    )
    print(
        f"Std of Hyperparameter Search Time: {np.std(hyperparam_search_times):.2f} seconds"
    )

    print(f"\nMean Test Evaluation Time: {np.mean(test_eval_times):.2f} seconds")
    print(f"Std of Test Evaluation Time: {np.std(test_eval_times):.2f} seconds")

    # Save results to CSV if requested
    if save_csv:
        results_df = pd.DataFrame(
            {
                "num_neurons": num_neurons_data,
                "regularization": regularization_data,
                "accuracy": accuracy_data,
                "f1_score": f1_data,
                "precision": precision_data,
                "recall": recall_data,
                "roc_auc": roc_data,
            }
        )
        results_df.to_csv("grid_search_results.csv", index=False)
        print("\nResults saved to grid_search_results.csv")

        # Save mean and std to a separate CSV
        summary_df = pd.DataFrame(
            {
                "Metric": metrics,
                "Mean": [metrics_means[m] for m in metrics],
                "Std": [metrics_stds[m] for m in metrics],
            }
        )
        summary_df.to_csv("summary_grid_search_results.csv", index=False)
        print("Summary saved to summary_grid_search_results.csv")

        # Save Hyperparams to a separate CSV
        summary_df = pd.DataFrame(
            {
                "best_params": best_params,
                "best_score": best_score,
            }
        )
        summary_df.to_csv("summary_grid_search_best_params.csv", index=False)
        print("Summary saved to summary_grid_best_params.csv")

    # Plot 3D surface if requested
    if generate_plot:
        plot_3d_surface(num_neurons_data, regularization_data, accuracy_data)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "final_model": final_model,
        "test_eval_results": test_eval_results,
    }


def plot_3d_surface(num_neurons_data, regularization_data, accuracy_data):
    """
    Generate a 3D surface plot to visualize hyperparameter effects on model accuracy.

    Parameters:
    - num_neurons_data: List of number of neurons used in the model.
    - regularization_data: List of regularization values used in the model.
    - accuracy_data: List of accuracy values corresponding to the hyperparameters.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    num_neurons_data = np.array(num_neurons_data)
    regularization_data = np.array(regularization_data)
    accuracy_data = np.array(accuracy_data)

    # Create meshgrid for num_neurons and regularization_values
    num_neurons_mesh, regularization_mesh = np.meshgrid(
        np.unique(num_neurons_data), np.unique(regularization_data)
    )

    # Interpolate the accuracy values on the meshgrid
    accuracy_mesh = griddata(
        (num_neurons_data, regularization_data),
        accuracy_data,
        (num_neurons_mesh, regularization_mesh),
        method="linear",
    )

    # Corresponding exponents for the labels
    exponents = [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
    labels = [
        f"$2^{{{exp}}}$" for exp in exponents
    ]  # Convert exponents to formatted strings

    ax.plot_surface(
        num_neurons_mesh, np.log2(regularization_mesh), accuracy_mesh, cmap="inferno"
    )
    ax.set_yticks(exponents, labels)
    ax.set_xlabel("Number of Neurons")
    ax.set_ylabel("Regularization Value")
    ax.set_zlabel("Accuracy")
    plt.title("3D Surface Plot of Hyperparameter Effects")

    plt.show()
