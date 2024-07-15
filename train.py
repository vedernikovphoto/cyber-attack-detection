from imblearn.over_sampling import SMOTE

def handle_class_imbalance(train_data_scaled, train_labels, random_state):
    """
    Handle class imbalance using SMOTE.

    This function applies SMOTE (Synthetic Minority Over-sampling Technique) to the training data
    to address class imbalance by generating synthetic samples for the minority class.

    Args:
        train_data_scaled (ndarray): The scaled training data.
        train_labels (DataFrame or Series): The training labels.

    Returns:
        tuple: A tuple containing:
            - X_resampled (ndarray): The resampled training data.
            - y_resampled (ndarray): The resampled training labels.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(train_data_scaled, train_labels.values.ravel())
    return X_resampled, y_resampled

def train_model(model, X_resampled, y_resampled):
    """
    Train the given model on the resampled data.

    This function trains the specified machine learning model using the resampled training data.

    Args:
        model (sklearn.base.BaseEstimator): The machine learning model to be trained.
        X_resampled (ndarray): The resampled training data.
        y_resampled (ndarray): The resampled training labels.

    Returns:
        sklearn.base.BaseEstimator: The trained model.
    """
    print("Training the model.")
    model.fit(X_resampled, y_resampled)
    return model
