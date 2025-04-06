import os
from sklearn.base import clone
import numpy as np

# Snapshot + Dropout Ensemble
def snapshot_dropout_ensemble(base_model, X, y, n_snapshots=5, dropout_rate=0.3):
    snapshots = []
    for i in range(n_snapshots):
        # Clone and apply dropout for each snapshot
        model = clone(base_model)
        if hasattr(model, 'dropout_rate'):
            model.dropout_rate = dropout_rate  # Ensure ANN model adjusts dropout rate
        model.fit(X, y)
        snapshots.append(model)
    return snapshots

# Predict using Snapshot Ensemble
def snapshot_predict(snapshots, X):
    predictions = np.array([model.predict(X) for model in snapshots])
    return predictions.mean(axis=0)

# Weighted Averaging Ensemble
def weighted_average_ensemble(models, weights, X):
    predictions = np.array([model.predict_proba(X)[:, 1] for model in models])
    return np.dot(weights, predictions)

