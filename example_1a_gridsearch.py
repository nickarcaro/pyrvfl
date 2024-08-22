import numpy as np
import sklearn.datasets as sk_dataset
from pyrvfl.rvfl import RVFL
from pyrvfl.SPRVFL import SPRVFL
from pyrvfl.utils.utils import standardize_rvfl, prepare_data_classify


(train_data, train_labels), (test_data, test_labels) = prepare_data_classify(
    sk_dataset.load_breast_cancer(), 0.8
)

train_data = standardize_rvfl(train_data)
test_data = standardize_rvfl(test_data)


"""
train_data = minmax_normalize(train_data)
test_data = minmax_normalize(test_data)
"""


# Conduct grid search and evaluate on test data with specified metrics
metrics_to_evaluate = ["accuracy", "recall", "roc_auc"]

model = SPRVFL(n_nodes=10, lam=0.01, activation="relu", task_type="classification")
model.fit(train_data, train_labels)
predictions, _ = model.predict(test_data)
results = model.eval(test_data, test_labels, metrics=metrics_to_evaluate)
print(results)
