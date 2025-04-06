import os
import os
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from modules.model_utils import build_pipeline
from modules.data_loader import build_preprocessor


def train_xgb(X, y, numerical, categorical, save_path="saved_models/xgb_model.pkl"):
    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
    pipeline = build_pipeline(build_preprocessor(numerical, categorical), model)

    param_grid = {
        'model__n_estimators': [50, 100, 200, 500],  # 4 options
        'model__learning_rate': [0.001, 0.01, 0.1, 0.3],  # 4
        'model__max_depth': [3, 5, 7, 10],  # 4
        'model__gamma': [0, 0.1, 0.2, 0.3],  # 4
        'model__subsample': [0.6, 1.0],  # 2
        'model__reg_alpha': [0, 0.01, 0.1, 1],  # 4
        'model__reg_lambda': [0.1, 1, 10]  # 3
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid_search.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)

    return grid_search.best_estimator_


def train_svm(X, y, numerical, categorical, save_path="saved_models/svm_model.pkl"):
    model = SVC(probability=True, random_state=42)
    pipeline = build_pipeline(build_preprocessor(numerical, categorical), model)

    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto', 0.01, 0.001],
        'model__class_weight': [None, 'balanced']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid_search.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)

    return grid_search.best_estimator_


def create_ann_model(hidden_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units // 2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_ann(X, y, numerical, categorical, save_path="saved_models/ann_model.pkl"):
    preprocessor = build_preprocessor(numerical, categorical)
    X_transformed = preprocessor.fit_transform(X)

    ann_model = KerasClassifier(
        model=create_ann_model,
        hidden_units=64,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        verbose=0,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'model__hidden_units': [32, 64, 128],
        'model__dropout_rate': [0.2, 0.3, 0.4],
        'model__learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'epochs': [50, 100]
    }

    grid_search = GridSearchCV(ann_model, param_grid, cv=cv, scoring="roc_auc", verbose=0, n_jobs=-1)
    grid_search.fit(X_transformed, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)

    return grid_search.best_estimator_
