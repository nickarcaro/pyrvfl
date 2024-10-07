import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


@staticmethod
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
