"""
Streamlit App - Credit Card Default Prediction
ML Assignment 2

Features:
- Upload test CSV data
- Select from 5 trained classification models
- View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- View confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Credit Card Default Predictor", layout="wide")

TARGET = 'default_payment_next_month'
MODEL_DIR = 'model'

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'kNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest (Ensemble)': 'random_forest_ensemble.pkl'
}
NEEDS_SCALING = {'Logistic Regression', 'kNN'}


@st.cache_resource
def load_artifacts():
    with open(f'{MODEL_DIR}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{MODEL_DIR}/feature_names.json') as f:
        feature_names = json.load(f)
    with open(f'{MODEL_DIR}/categorical_cols.json') as f:
        categorical_cols = json.load(f)
    with open(f'{MODEL_DIR}/numeric_cols.json') as f:
        numeric_cols = json.load(f)

    models = {}
    for name, filename in MODEL_FILES.items():
        with open(f'{MODEL_DIR}/{filename}', 'rb') as f:
            models[name] = pickle.load(f)

    return models, scaler, feature_names, categorical_cols, numeric_cols


def preprocess(df, feature_names, categorical_cols, numeric_cols):
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    present_cat = [c for c in categorical_cols if c in df.columns]
    present_num = [c for c in numeric_cols if c in df.columns]

    if present_num:
        df[present_num] = SimpleImputer(strategy='median').fit_transform(df[present_num])
    if present_cat:
        df[present_cat] = SimpleImputer(strategy='most_frequent').fit_transform(df[present_cat])

    df_encoded = pd.get_dummies(df, columns=present_cat, drop_first=True)

    # Align columns to match training feature set exactly
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    extra_cols = [c for c in df_encoded.columns if c not in feature_names and c != TARGET]
    df_encoded = df_encoded.drop(columns=extra_cols)
    df_encoded = df_encoded[feature_names]

    return df_encoded


def main():
    st.title("💳 Credit Card Default Prediction")
    st.caption("ML Assignment 2 — Compare 5 classification models on the UCI Credit Card Default dataset")

    models, scaler, feature_names, categorical_cols, numeric_cols = load_artifacts()

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))

    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=['csv'])
    st.sidebar.caption("Upload the provided `test_data.csv`, or your own CSV with the same columns "
                        "(must include the target column `default_payment_next_month` to compute metrics).")

    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar to get started. "
                "You can use the included `test_data.csv`.")
        st.subheader("Expected input format")
        st.write("Your CSV should contain these columns:")
        st.code(", ".join(numeric_cols + categorical_cols + [TARGET]))
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(10))
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    has_target = TARGET in df.columns

    X_processed = preprocess(df, feature_names, categorical_cols, numeric_cols)

    model = models[selected_model_name]
    if selected_model_name in NEEDS_SCALING:
        X_input = scaler.transform(X_processed)
    else:
        X_input = X_processed

    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]

    st.subheader(f"🔮 Predictions — {selected_model_name}")
    result_df = df.copy()
    result_df['Predicted_Default'] = y_pred
    result_df['Default_Probability'] = np.round(y_proba, 4)
    st.dataframe(result_df.head(20))

    if has_target:
        y_true = df[TARGET]

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader("📊 Evaluation Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Default', 'Default'],
                        yticklabels=['No Default', 'Default'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_true, y_pred, target_names=['No Default', 'Default'],
                                             output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4))

        # --- Compare all models on this uploaded data ---
        st.subheader("🏆 Compare All Models on This Data")
        if st.checkbox("Run all 5 models and compare"):
            comparison = []
            for name, mdl in models.items():
                Xi = scaler.transform(X_processed) if name in NEEDS_SCALING else X_processed
                yp = mdl.predict(Xi)
                ypr = mdl.predict_proba(Xi)[:, 1]
                comparison.append({
                    'Model': name,
                    'Accuracy': round(accuracy_score(y_true, yp), 4),
                    'AUC': round(roc_auc_score(y_true, ypr), 4),
                    'Precision': round(precision_score(y_true, yp), 4),
                    'Recall': round(recall_score(y_true, yp), 4),
                    'F1': round(f1_score(y_true, yp), 4),
                    'MCC': round(matthews_corrcoef(y_true, yp), 4)
                })
            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df.style.highlight_max(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], color='lightgreen'))
    else:
        st.warning("No target column found in uploaded data — showing predictions only "
                    "(metrics require the true `default_payment_next_month` column).")


if __name__ == "__main__":
    main()
