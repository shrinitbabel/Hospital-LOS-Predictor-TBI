import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from scikit_posthocs import critical_difference_diagram


def evaluate_model(y_true, y_pred, y_prob):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"AUC Score: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")



def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    cm = confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else 0.0

    return [auc, f1, acc, precision, recall, specificity]


def evaluate_models_auc(y, model_probs):
    scores = {model: roc_auc_score(y, probs) for model, probs in model_probs.items()}
    print("AUC Scores:", scores)

    matrix = np.array(list(model_probs.values())).T
    stat, p = friedmanchisquare(*matrix.T)
    print(f"Friedman Test: {stat:.4f}, P-value: {p:.4f}")

    posthoc = sp.posthoc_conover_friedman(matrix)
    return scores, posthoc


def evaluate_models_f1(y, model_preds):
    scores = {model: f1_score(y, preds, average='weighted') for model, preds in model_preds.items()}
    print("F1 Scores:", scores)

    matrix = np.array(list(model_preds.values())).T
    stat, p = friedmanchisquare(*matrix.T)
    print(f"Friedman Test: {stat:.4f}, P-value: {p:.4f}")

    posthoc = sp.posthoc_conover_friedman(matrix)
    return scores, posthoc


def evaluate_models_accuracy(y, model_preds):
    scores = {model: accuracy_score(y, preds) for model, preds in model_preds.items()}
    print("Accuracy Scores:", scores)

    matrix = np.array(list(model_preds.values())).T
    stat, p = friedmanchisquare(*matrix.T)
    print(f"Friedman Test: {stat:.4f}, P-value: {p:.4f}")

    posthoc = sp.posthoc_conover_friedman(matrix)
    return scores, posthoc


def evaluate_all_metrics(y_true, model_probs: dict, verbose=True):
    metrics = ['AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'Specificity']
    scores = {}

    for model, probs in model_probs.items():
        scores[model] = calculate_metrics(y_true, probs)

    df_scores = pd.DataFrame(scores, index=metrics)
    df_ranks = df_scores.rank(axis=1, ascending=False)
    avg_ranks = df_ranks.mean(axis=0)

    # Friedman test
    stat, p = friedmanchisquare(*df_ranks.values.T)
    if verbose:
        print(f"\n📈 Friedman Test: Statistic={stat:.4f}, P-value={p:.4f}")

    # Posthoc Conover test
    posthoc = sp.posthoc_conover_friedman(df_ranks.values)
    posthoc.index = df_scores.columns
    posthoc.columns = df_scores.columns

    if verbose:
        print("\n📊 Conover Post-Hoc Test Results:")
        print(posthoc.round(6).to_string())

    return df_scores, df_ranks, avg_ranks.to_dict(), posthoc

def plot_cd_diagram(rank_dict, posthoc_matrix, title="CD Diagram"):
    plt.figure(figsize=(14, 6))
    critical_difference_diagram(
        ranks=rank_dict,
        sig_matrix=posthoc_matrix,
        ax=None,
        label_fmt_left="{label} ({rank:.2g})",
        label_fmt_right="",
        label_props={'fontsize': 12, 'weight': 'bold'},
        marker_props={'s': 100},
        crossbar_props={'linewidth': 2.0},
        color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        text_h_margin=0.03,
        left_only=True
    )
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Average Rank", fontsize=14)
    plt.tight_layout()
    plt.show()
