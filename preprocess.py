from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(train_data, test_data):
    """
    Preprocess the training and test data by aligning columns and normalizing features.

    This function aligns the columns of the training and testing datasets by retaining only the common columns,
    ensuring a consistent feature set. It then normalizes the data so that all features have a mean of 0 and a
    standard deviation of 1.

    Args:
        train_data (DataFrame): The training data.
        test_data (DataFrame): The test data.

    Returns:
        tuple: A tuple containing two arrays:
            - train_data_scaled (ndarray): The scaled training data.
            - test_data_scaled (ndarray): The scaled test data.
    """
    # Align the columns by finding and retaining only the common columns between the training and testing datasets
    common_columns = train_data.columns.intersection(test_data.columns)

    # Keep only the common columns in both datasets to ensure consistent feature sets
    train_data = train_data[common_columns]
    test_data = test_data[common_columns]

    # Normalize the data to ensure all features have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    print("Data normalization completed.")
    return train_data_scaled, test_data_scaled

def convert_labels(train_labels):
    """
    Convert labels from -1 to 0 for compatibility with XGBoost and other models.

    This function replaces all occurrences of the label -1 with 0 in the training labels,
    making them compatible with models that require binary labels in the range [0, 1].

    Args:
        train_labels (DataFrame or Series): The training labels to be converted.

    Returns:
        DataFrame or Series: The converted training labels with -1 replaced by 0.
    """
    return train_labels.replace(-1, 0)
