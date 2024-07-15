import json
from data_loader import load_data
from preprocess import preprocess_data, convert_labels
from model import initialize_model
from train import handle_class_imbalance, train_model
from predict import generate_predictions, save_predictions


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    # Load the configuration
    config = load_config('config.json')

    # Load the data
    train_data, test_data, train_labels = load_data()

    # Preprocess the data
    train_data_scaled, test_data_scaled = preprocess_data(train_data, test_data)

    # Convert labels from -1 to 0 for compatibility with SMOTE and SVM
    train_labels_binary = convert_labels(train_labels)

    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(train_data_scaled, 
                                                      train_labels_binary, 
                                                      random_state=config["random_state"])

    # Initialize and train the model
    model = initialize_model(kernel=config["kernel"],
                             gamma=config["gamma"],
                             C=config["C"],
                             random_state=config["random_state"])
    trained_model = train_model(model, X_resampled, y_resampled)

    # Generate predictions
    predictions = generate_predictions(trained_model, test_data_scaled)

    # Save predictions to a CSV file
    save_predictions(predictions)

    print("Predictions have been generated and saved to test_labels.csv")
