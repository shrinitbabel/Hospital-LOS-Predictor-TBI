import os
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.base import clone
import matplotlib.pyplot as plt

# Bootstrap sampling function
def bootstrap_sample(X, y, n_samples=5):
    np.random.seed(42)
    indices = [np.random.choice(range(len(y)), size=len(y), replace=True) for _ in range(n_samples)]
    return indices

# Bias-Variance Calculation
def calculate_bias_variance_with_risk(models, X, y, n_samples=5):
    indices = bootstrap_sample(X, y, n_samples)
    predictions = {model_name: [] for model_name in models}

    for idx_set in indices:
        X_train, y_train = X[idx_set], y[idx_set]
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions[model_name].append(model.predict(X))

    bias_variance_risk = {}
    for model_name, preds in predictions.items():
        preds = np.array(preds)
        avg_pred = preds.mean(axis=0)
        bias = np.mean((avg_pred - y) ** 2)
        variance = np.mean(np.var(preds, axis=0))
        diversity = np.mean([np.var([pred for pred in preds]) for pred in preds.T])
        expected_risk = bias + variance - diversity

        bias_variance_risk[model_name] = {
            'bias': bias,
            'variance': variance,
            'diversity': diversity,
            'expected_risk': expected_risk,
        }
    return bias_variance_risk

# Diversity Calculation
def calculate_diversity(models, X, y):
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X)
    
    diversity_metrics = {}
    model_names = list(models.keys())
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                # Disagreement Measure
                disagree = np.mean(predictions[model1] != predictions[model2])
                # Correlation Coefficient
                corr, _ = pearsonr(predictions[model1], predictions[model2])
                # Q-Statistic
                q_stat = (np.sum(predictions[model1] == y) * np.sum(predictions[model2] == y) - 
                          np.sum(predictions[model1] != y) * np.sum(predictions[model2] != y))
                q_stat /= len(y) ** 2

                diversity_metrics[f"{model1} vs {model2}"] = {
                    'disagreement': disagree,
                    'correlation': corr,
                    'q_statistic': q_stat
                }
    return diversity_metrics



# Visualization Function
def plot_decomposition(results, title):
    categories = list(results.keys())
    bias = [results[cat]['bias'] for cat in categories]
    variance = [results[cat]['variance'] for cat in categories]
    diversity = [results[cat]['diversity'] for cat in categories]
    risk = [results[cat]['expected_risk'] for cat in categories]

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, bias, width, label='Bias')
    ax.bar(x, variance, width, label='Variance')
    ax.bar(x + width, diversity, width, label='Diversity')
    ax.plot(x, risk, label='Expected Risk', color='red', marker='o')

    ax.set_xlabel('Models')
    ax.set_ylabel('Metrics')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Updated calculate_decomposition to handle raw predictions
def calculate_decomposition(models, X, y, n_samples=5):
    indices = bootstrap_sample(X, y, n_samples)
    predictions = {model_name: [] for model_name in models}

    for idx_set in indices:
        X_train, y_train = X[idx_set], y[idx_set]
        for model_name, model in models.items():
            if isinstance(model, list):  # Snapshot Ensemble Case
                # Directly use the precomputed snapshot ensemble predictions
                snapshots = model
                snapshot_preds = snapshot_predict(snapshots, X)
                predictions[model_name].append(snapshot_preds)
            elif isinstance(model, np.ndarray):  # Precomputed Predictions (e.g., snapshot_probs)
                predictions[model_name].append(model)
            else:  # Regular scikit-learn model
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                predictions[model_name].append(model_clone.predict(X))

    results = {}
    for model_name, preds in predictions.items():
        preds = np.array(preds)
        avg_pred = preds.mean(axis=0)
        bias = np.mean((avg_pred - y) ** 2)
        variance = np.mean(np.var(preds, axis=0))
        diversity = np.mean([np.var([pred for pred in preds]) for pred in preds.T])
        expected_risk = bias + variance - diversity

        results[model_name] = {
            'bias': bias,
            'variance': variance,
            'diversity': diversity,
            'expected_risk': expected_risk
        }
    return results