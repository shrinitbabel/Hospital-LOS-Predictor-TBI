# modules/ensemble_utils.py

import numpy as np

class WeightedAverageEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        probs = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        weighted_avg = np.average(probs, axis=0, weights=self.weights)
        return weighted_avg

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

class SnapshotANNEnsemble:
    def __init__(self, base_model, n_snapshots=30, dropout_rate=0.5):
        self.base_model = base_model
        self.n_snapshots = n_snapshots
        self.dropout_rate = dropout_rate
        self.snapshots = []

    def fit(self, X, y):
        from modules.ensemble import snapshot_dropout_ensemble
        self.snapshots = snapshot_dropout_ensemble(
            self.base_model,
            X,
            y,
            n_snapshots=self.n_snapshots,
            dropout_rate=self.dropout_rate
        )

    def predict_proba(self, X):
        from modules.ensemble import snapshot_predict
        return snapshot_predict(self.snapshots, X)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)
