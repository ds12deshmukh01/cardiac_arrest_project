import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload test dataset (CSV)", type=["csv"])

# Model selection
model_choice = st.selectbox(
    "Choose a model:",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Load model
model_path = f"model/{model_choice.replace(' ', '_')}.pkl"
model = joblib.load(model_path)

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(test_data.head())

    if "target" in test_data.columns:
        X_test = test_data.drop("target", axis=1)
        y_test = test_data["target"]
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("Evaluation Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Classification Report as DataFrame
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
    else:
        st.error("Uploaded CSV must contain a 'target' column for evaluation.")
else:
    st.info("ℹPlease upload a test dataset CSV to evaluate the model.")
