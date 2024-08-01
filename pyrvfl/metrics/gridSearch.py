import numpy as np

from sklearn.model_selection import StratifiedKFold


def stratified_shuffle_split_indices(X, y, n_splits=5, test_size=0.2, random_state=42):
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
    model_class, X, y, param_grid, n_splits=5, test_size=0.2, random_state=42
):
    """
    model_class: The class of the model to be optimized (e.g., RVFL).
    X: Features of the dataset.
    y: Target labels of the dataset.
    param_grid: List of dictionaries with hyperparameters to search over.
    n_splits: Number of splits for cross-validation.
    test_size: Proportion of the dataset to include in the test split.
    random_state: Random seed for reproducibility.
    """
    best_accuracy = 0.0
    best_params = {}

    accuracy_data = []
    num_neurons_data = []
    regularization_data = []
    f1_data = []
    precision_data = []
    recall_data = []
    roc_data = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for params in param_grid:
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        roc_aucs = []

        for train_indices, test_indices in stratified_shuffle_split_indices(
            X, y, n_splits=n_splits, test_size=test_size, random_state=random_state
        ):
            X_train, X_val = X[train_indices], X[test_indices]
            y_train, y_val = y[train_indices], y[test_indices]

            # Instantiate the model with current parameters
            model = model_class(**params)

            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            accuracies.append(model.eval(X_val, y_val)["acc"])
            f1_scores.append(model.eval(X_val, y_val)["f1_score"])
            precisions.append(model.eval(X_val, y_val)["precision"])
            recalls.append(model.eval(X_val, y_val)["recall"])
            roc_aucs.append(model.eval(X_val, y_val)["roc"])

        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1_scores)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_roc_auc = np.mean(roc_aucs)

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params

        # Store metrics for analysis
        accuracy_data.append(mean_accuracy)
        num_neurons_data.append(params.get("n_nodes", None))
        regularization_data.append(params.get("lam", None))
        f1_data.append(mean_f1)
        precision_data.append(mean_precision)
        recall_data.append(mean_recall)
        roc_data.append(mean_roc_auc)

    print(f"Best Accuracy: {best_accuracy:.2f}")
    print("Best Parameters:", best_params)

    return {
        "best_params": best_params,
        "best_accuracy": best_accuracy,
        "accuracy_data": accuracy_data,
        "num_neurons_data": num_neurons_data,
        "regularization_data": regularization_data,
        "f1_data": f1_data,
        "precision_data": precision_data,
        "recall_data": recall_data,
        "roc_data": roc_data,
    }
