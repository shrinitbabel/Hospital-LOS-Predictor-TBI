import streamlit as st
import pandas as pd
import numpy as np
import pickle
from modules.data_loader import build_preprocessor
from modules.ensemble_utils import SnapshotANNEnsemble, WeightedAverageEnsemble


# === Constants ===
FEATURES = ['Age', 'Gender', 'Race', 'BMI', 'GCS', 'ISS', 'Charlson_Comorbidity_Index']
NUMERICAL = ['Age', 'BMI', 'GCS', 'ISS', 'Charlson_Comorbidity_Index']
CATEGORICAL = ['Gender', 'Race']
MODEL_PATHS = {
    'XGBoost': "saved_models/xgb_model.pkl",
    'SVM': "saved_models/svm_model.pkl",
    'ANN': "saved_models/ann_model.pkl",
    'Weighted XGB+ANN': "saved_models/XGB+ANN_ensemble_model_weighted.pkl",
    'Weighted XGB+ANN+SVM': "saved_models/XGB+ANN+SVM_ensemble_model_weighted.pkl",
    'XGB + Snapshot ANN': "saved_models/snapshot_ensemble_model.pkl",
}

# === Load Models ===
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    return models

# === Build UI ===
st.set_page_config(page_title="LOS Predictor", layout="wide")
st.title("🏥 Hospital LOS Prediction")
st.markdown("Enter patient characteristics to predict **Prolonged Length of Stay (PLOS)**.")

# === Sidebar Inputs ===
st.sidebar.header("Patient Features")
user_input = {}
user_input['Age'] = st.sidebar.slider("Age", 0, 100, 45)
user_input['Gender'] = st.sidebar.selectbox("Gender", ['MALE', 'FEMALE'])
user_input['Race'] = st.sidebar.selectbox("Race", ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER'])
user_input['BMI'] = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
user_input['GCS'] = st.sidebar.slider("GCS Score", 3, 15, 12)
user_input['ISS'] = st.sidebar.slider("ISS", 0, 75, 16)
user_input['Charlson_Comorbidity_Index'] = st.sidebar.slider("Charlson Comorbidity Index", 0, 30, 2)

input_df = pd.DataFrame([user_input])

# === Preprocess ===
preprocessor = build_preprocessor(NUMERICAL, CATEGORICAL)
X_input = preprocessor.fit_transform(input_df)

# === Predict ===
models = load_models()
results = []

for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_input)[0][1]
    else:
        # Snapshot model assumed to output probabilities directly via .predict()
        prob = model.predict(X_input)[0]
    pred = int(prob > 0.5)
    results.append({
        "Model": name,
        "Probability (PLOS)": f"{prob:.4f}",
        "Prediction": "PLOS" if pred == 1 else "Normal"
    })

# === Display ===
st.subheader("🔮 Prediction Results")
st.dataframe(pd.DataFrame(results), use_container_width=True)
