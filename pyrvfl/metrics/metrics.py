import numpy as np


@staticmethod
def f1_score(y_true, y_pred):
    # Calculate True Positives, False Positives, and False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1-score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics: dict = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    return metrics


@staticmethod
def acc_score(y_true, y_pred):
    return np.sum(np.equal(y_pred, y_true)) / len(y_true)


@staticmethod
def roc_auc(y_true, y_score):
    """
    Calculate the ROC AUC (Area Under the Receiver Operating Characteristic Curve).

    Parameters:
        y_true (array-like): The true labels.
        y_score (array-like): The predicted scores/probabilities for the positive class.

    Returns:
        roc_auc_score (float): The ROC AUC score.
    """

    # Sort y_true and y_score in descending order of y_score
    descending_order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[descending_order]
    y_score_sorted = y_score[descending_order]

    # Count the number of positive samples
    num_positives = np.sum(y_true_sorted == 1)

    # Initialize variables to calculate TPR and FPR
    tpr = 0.0
    fpr = 0.0

    # Initialize the previous true positive rate
    prev_tpr = 0.0
    fpr_prev = 0.0

    # Initialize the area under the ROC curve
    roc_auc_score = 0.0

    # Iterate through the sorted y_true and y_score
    for i in range(len(y_score_sorted)):
        if y_true_sorted[i] == 1:
            tpr += 1
        else:
            fpr += 1

        # Calculate the true positive rate and false positive rate at the current threshold
        tpr_current = tpr / num_positives
        fpr_current = fpr / (len(y_score_sorted) - num_positives)

        # Calculate the trapezoidal area under the ROC curve (using the trapezoidal rule)
        roc_auc_score += 0.5 * (fpr_current - fpr_prev) * (tpr_current + prev_tpr)

        prev_tpr = tpr_current
        fpr_prev = fpr_current

    return roc_auc_score
