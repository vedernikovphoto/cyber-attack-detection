import numpy as np
import pandas as pd

def generate_predictions(model, test_data_scaled):
    """
    Generate predictions using the trained model on the scaled test data.

    This function generates predictions from the test data using the given trained model.
    It converts the binary predictions (0 and 1) back to the original format (-1 and 1).

    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        test_data_scaled (ndarray): The scaled test data.

    Returns:
        ndarray: The predictions in the original format (-1 and 1).
    """
    print("Generating the predictions.")
    predictions = model.predict(test_data_scaled)
    predictions = np.where(predictions == 0, -1, 1)
    return predictions

def save_predictions(predictions, filename='test_labels.csv'):
    """
    Save the predictions to a CSV file.

    This function saves the predictions to a CSV file with a single column named 'label'.

    Args:
        predictions (ndarray): The predictions to be saved.
        filename (str): The name of the file to save the predictions to. Default is 'test_labels.csv'.

    Returns:
        None
    """
    pd.DataFrame(predictions).to_csv(filename, index=False, header=False)
