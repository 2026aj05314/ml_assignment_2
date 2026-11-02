# 2026AJ05314 - Machine Learning (S1-26_AIMLZG565) Assignment 2
## Taiwanese Company Bankruptcy Prediction [Input](https://www.kaggle.com/code/bhavanjuli/taiwanese-company-bankruptcy-prediction/input)

## a. Problem Statement
The objective of this project is to develop and evaluate multiple supervised machine learning classification models to predict corporate bankruptcy among Taiwanese firms based on historical financial metrics. Early identification of financial distress enables financial institutions, investors, and regulatory bodies to mitigate default risks and allocate capital prudently. The challenge is characterized by extreme class imbalance, where bankrupt entities represent a small fraction of the overall sample, demanding evaluation metrics beyond raw classification accuracy.

## b. Dataset Description
The analysis utilizes the **Taiwanese Bankruptcy Prediction Dataset** sourced from the UCI Machine Learning Repository / Kaggle:
- **Instances (Rows):** 6,819 corporate filings
- **Features (Columns):** 95 financial ratio features covering operational profitability, leverage, cash flow coverage, liquidity, and growth indices
- **Target Variable:** `Bankrupt?` (Binary classification: `1` = Bankrupt, `0` = Non-bankrupt)
- **Data Preprocessing:** Column headers were trimmed of whitespace. Pipelines incorporate `VarianceThreshold(threshold=0.0)` to eliminate zero-variance constants (`Net Income Flag`) and `StandardScaler` to normalize disparate financial scales prior to classification.
- **Partitioning:** The dataset was split into training and evaluation sets, with the full 96-column evaluation slice preserved in `test_data.csv` for independent application inference.

## c. Github Repository Link
- **Repository URL:** [https://github.com/2026aj05314/ml_assignment_2](https://github.com/2026aj05314/ml_assignment_2)
- **Live Streamlit App:** [https://2026aj05314-aimlzg565-assignment2.streamlit.app/](https://2026aj05314-aimlzg565-assignment2.streamlit.app/)

## d. Models Used
Five distinct classification algorithms were trained and evaluated on `test_data.csv` across six performance metrics: Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9663 | 0.8639 | 0.1806 | 0.3939 | 0.2476 | 0.2515 |
| Decision Tree | 0.9420 | 0.6719 | 0.1008 | 0.3939 | 0.1605 | 0.1776 |
| KNN | 0.9720 | 0.7652 | 0.1845 | 0.2879 | 0.2249 | 0.2168 |
| Naive Bayes | 0.9516 | 0.7965 | 0.1221 | 0.3939 | 0.1864 | 0.2000 |
| Support Vector Machine | 0.9806 | 0.8626 | 0.2642 | 0.2121 | 0.2353 | 0.2270 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates the highest discriminatory capability across the full threshold spectrum (AUC = 0.8639) while tying for peak minority class detection (Recall = 0.3939). The linear decision boundary, stabilized via standard scaling, achieves the highest balanced performance on skewed data (F1 = 0.2476, MCC = 0.2515). |
| **Decision Tree** | Matches the maximum Recall rate (0.3939) by identifying orthogonal feature thresholds, but exhibits high false-positive susceptibility due to greedy partitioning, leading to the lowest Precision (0.1008) and lower separation quality (AUC = 0.6719). |
| **KNN** | Yields high overall Accuracy (0.9720) and competitive Precision (0.1845), but suffers in positive instance retrieval (Recall = 0.2879) as local neighbor voting is diluted by majority dominance in high-dimensional Euclidean space. |
| **Naive Bayes** | Recovers 39.39% of true insolvencies with acceptable discriminative power (AUC = 0.7965). However, violations of the conditional independence assumption among interdependent accounting ratios induce probability miscalibration and an elevated false-alarm rate (Precision = 0.1221). |
| **Support Vector Machine** | Attains the highest raw Accuracy (0.9806) and Precision (0.2642) using an RBF kernel. The maximal-margin objective favors the dominant majority class, yielding conservative positive predictions that limit Recall to 0.2121. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the definitive overall winner. In credit risk and insolvency modeling, failure to flag a defaulting firm (false negative) is significantly costlier than manual review of a solvent firm (false positive). Logistic Regression maximizes AUC (0.8639), F1 Score (0.2476), and MCC (0.2515) while maintaining top-tier Recall (0.3939), delivering the most robust risk-scoring performance under extreme class imbalance. |

## Project Structure
```text
.
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Environment build dependencies
├── README.md                   # Project documentation and evaluation report
├── data.csv                    # Full source dataset (local / gitignored)
├── test_data.csv               # Test evaluation split with target column
├── script.ipynb                # Data preprocessing, training, and evaluation notebook
└── model/                      # Serialized scikit-learn pipeline artifacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```
