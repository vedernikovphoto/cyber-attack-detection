from sklearn.svm import SVC

def initialize_model(kernel, gamma, C, random_state):
    """
    Initialize the SVM model with specified hyperparameters.

    This function initializes an SVM (Support Vector Machine) model with the given hyperparameters.
    The hyperparameters are passed as arguments to the function:
        - kernel: Specifies the kernel type to be used in the algorithm.
        - gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - C: Regularization parameter.
        - random_state: Controls the randomness of the estimator.

    Args:
        kernel (str): Specifies the kernel type to be used in the algorithm.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        C (float): Regularization parameter.
        random_state (int): Controls the randomness of the estimator.

    Returns:
        SVC: An instance of the SVC class from scikit-learn with the specified parameters.
    """
    return SVC(kernel=kernel, gamma=gamma, C=C, random_state=random_state)
