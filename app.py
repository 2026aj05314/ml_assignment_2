import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Taiwanese Bankruptcy Classifier", layout="wide")

st.title("Taiwanese Company Bankruptcy Classification")
st.write("Demonstration of pre-trained classification models evaluated on test data.")

# Model registry mapping user-facing labels to serialized pipeline filenames
MODEL_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

target_col = "Bankrupt?"

# Sidebar controls for test data upload and model selection
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_MAP.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()
    
    if target_col not in test_df.columns:
        st.error(f"Target column '{target_col}' not found in uploaded CSV.")
    else:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Load pre-trained model pipeline artifact without retraining
        model_path = os.path.join("model", MODEL_MAP[selected_model_name])
        
        if not os.path.exists(model_path):
            st.error(f"Model artifact not found at {model_path}")
        else:
            pipeline = joblib.load(model_path)
            
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Compute the 6 mandatory metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.subheader(f"Evaluation Metrics: {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")
            
            st.divider()
            
            # Diagnostic outputs: Confusion Matrix & Classification Report
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                            xticklabels=["Non-Bankrupt", "Bankrupt"],
                            yticklabels=["Non-Bankrupt", "Bankrupt"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close(fig)
                
            with col_right:
                st.subheader("Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
else:
    st.info("Upload test_data.csv using the sidebar to view evaluation metrics.")