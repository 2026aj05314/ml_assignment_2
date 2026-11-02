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

test_df = None
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        st.stop()
        
    if test_df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    test_df.columns = test_df.columns.str.strip()
    
    if target_col not in test_df.columns:
        st.error(f"Target column '{target_col}' not found in uploaded CSV. Please ensure the CSV contains '{target_col}'.")
    else:
        y_test = test_df[target_col]
        if y_test.isna().any():
            st.error(f"Target column '{target_col}' contains missing values (NaN).")
            st.stop()
            
        try:
            y_test = y_test.astype(int)
        except Exception:
            st.error(f"Target column '{target_col}' must contain binary numeric values (0 or 1).")
            st.stop()
            
        X_test = test_df.drop(columns=[target_col])
        
        # Check for missing values in features
        if X_test.isna().any().any():
            st.warning("Warning: Features contain missing values (NaN). This may cause model prediction to fail.")
        
        # Load pre-trained model pipeline artifact without retraining
        model_path = os.path.join("model", MODEL_MAP[selected_model_name])
        
        if not os.path.exists(model_path):
            st.error(f"Model artifact not found at {model_path}")
        else:
            try:
                pipeline = joblib.load(model_path)
            except Exception as e:
                st.error(f"Failed to load model artifact: {e}")
                st.stop()
            
            try:
                y_pred = pipeline.predict(X_test)
            except Exception as e:
                st.error(f"Model prediction failed on uploaded data: {e}")
                st.info("Please verify that the uploaded CSV contains the same 95 numerical features the models were trained on.")
                st.stop()
                
            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = float("nan")
            
            # Compute the 6 mandatory metrics
            try:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
            except Exception as e:
                st.error(f"Error computing evaluation metrics: {e}")
                st.stop()
            
            st.subheader(f"Evaluation Metrics: {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}" if not pd.isna(auc) else "N/A")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")
            
            st.divider()
            
            # Diagnostic outputs: Confusion Matrix & Classification Report
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Confusion Matrix")
                try:
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                                xticklabels=["Non-Bankrupt", "Bankrupt"],
                                yticklabels=["Non-Bankrupt", "Bankrupt"])
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Could not render confusion matrix: {e}")
                
            with col_right:
                st.subheader("Classification Report")
                try:
                    report_dict = classification_report(
                        y_test,
                        y_pred,
                        labels=[0, 1],
                        target_names=["Non-Bankrupt", "Bankrupt"],
                        output_dict=True,
                        zero_division=0
                    )
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate classification report: {e}")
else:
    st.info("Upload test_data.csv using the sidebar to view evaluation metrics.")