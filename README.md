# Hospital-LOS-Predictor-TBI

This project predicts prolonged hospital length of stay (LOS) in patients with traumatic brain injury (TBI) using ensemble learning. It includes data preprocessing, feature engineering, model training, evaluation, and ensemble techniques to improve prediction accuracy. The project and associated manuscript are under review. 

## Features

- **Initial Statistical Analysis**: Perform feature significance tests, LOS summaries, and generate baseline characteristics.
- **Machine Learning Models**:
    - XGBoost
    - Support Vector Machine (SVM)
    - Artificial Neural Network (ANN)
- **Ensemble Techniques**:
    - Weighted averaging ensembles
    - Snapshot ensembles
- **Evaluation Metrics**: ROC-AUC, F1-score, accuracy, sensitivity, specificity.
    - Bias-Variance-Diversity Decomposition
    - Critical Difference Analysis (Friedman's test with Conover post-hoc analysis)

## Project Structure

```
Hospital-LOS-Predictor-TBI/
├── modules/                     # Core modules for data processing, modeling, and evaluation
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and feature engineering
│   ├── diversity.py             # Diversity and bias-variance analysis
│   ├── ensemble.py              # Ensemble methods
│   ├── ensemble_utils.py        # Utility functions for ensembles
│   ├── evaluation.py            # Model evaluation metrics
│   ├── model_utils.py           # Model pipeline utilities
│   ├── statistics.py            # Statistical tests and summaries
│   ├── train.py                 # Training scripts
├── saved_models/                # Directory for saving trained models
│   ├── xgb_model.pkl
│   ├── svm_model.pkl
│   ├── ann_model.pkl
│   ├── XGB+ANN_ensemble_model_weighted.pkl
│   ├── XGB_ANN_SVM_ensemble_model_weighted.pkl
│   ├── snapshot_ensemble_model.pkl
├── preliminary_statistical_analysis.ipynb  # Statistical analysis notebook
├── supervised_learning.ipynb               # Machine learning and ensemble modeling notebook
├── streamlit_app.py                         # Streamlit app for interactive predictions (UNDER CONSTRUCTION!)
├── updated_dataset (1).csv                  # Dataset for training and evaluation
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # License file
└── README.md                                # Project documentation
```

## Installation

1. Clone the repository:
     ```bash
     git clone https://github.com/your-repo/Hospital-LOS-Predictor-TBI.git
     cd Hospital-LOS-Predictor-TBI
     ```

2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

- **Statistical Analysis**: Run the `preliminary_statistical_analysis.ipynb` notebook to perform feature significance tests and generate LOS summaries.
- **Model Training**: Use the `supervised_learning.ipynb` notebook to train machine learning models and evaluate their performance.
- **Interactive Predictions**: Launch the Streamlit app for predictions:
     ```bash
     streamlit run streamlit_app.py
     ```

## License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.

## Contact
shrinitbabel at gmail dot com



