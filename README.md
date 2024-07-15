# Cyber Attack Detection with Machine Learning

## Overview
This project develops a machine learning model to detect cyber attacks using a high-dimensional dataset. The process involves data preprocessing, feature engineering, handling class imbalance, model training, and evaluation.

## Project Structure

```
repository
    ├── data/                       # Contains a small fraction of data files (first 10 rows) for illustration purposes
    │   ├── train_data.csv          # Training data file (first 10 rows)
    │   ├── test_data.csv           # Testing data file (first 10 rows)
    │   └── train_labels.csv        # Training labels file (first 10 rows)
    ├── data_loader.py              # Functions for loading the dataset
    ├── preprocess.py               # Functions for preprocessing data and converting labels
    ├── model.py                    # Functions for initializing the model
    ├── train.py                    # Functions for handling class imbalance and training the model
    ├── predict.py                  # Functions for generating and saving predictions
    ├── generate_predictions.py     # Main script to orchestrate data loading, preprocessing, model training, and prediction generation
    ├── config.json                 # Configuration file for model parameters
    ├── requirements.txt            # Lists the dependencies required to run the project
    ├── test_labels.csv             # Output file containing predicted labels for the test data
    ├── report_notebook.ipynb       # Jupyter notebook explaining the problem, solution, and justifications
    ├── .gitignore                  # Specifies files and directories to be ignored by git
    └── README.md                   # Provides an overview, installation instructions, and usage information for the project


```

## Important Note
The data files provided in the `data/` folder contain only the first 10 rows of the dataset for illustration purposes. These small sample sizes are insufficient for training the SVM model effectively. To run the model properly, you will need to use the complete dataset.


## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
To generate predictions, ensure that your data files (`train_data.csv`, `test_data.csv`, and `train_labels.csv`) are in the in the folder `data\`:

```
python generate_predictions.py
```

## Future work
- Exploring additional feature engineering techniques and advanced machine learning algorithms.
- Implementing real-time data processing and model deployment for continuous monitoring of cyber threats.
- Conducting further validation using more diverse datasets to enhance the model's generalizability.
- Investigating deep learning models, such as neural networks and recurrent neural networks (RNNs), which could capture more complex patterns in the data and potentially improve detection performance.
- Exploring state-of-the-art models like transformers and other SOTA architectures that have shown superior performance in various domains. These models could be adapted for cybersecurity tasks to further enhance detection capabilities.

## Contributing
Feel free to submit issues or pull requests if you have any improvements or bug fixes.


