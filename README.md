# Cardiac Arrest Prediction App

## a. Problem Statement
Cardiac arrest is a life‑threatening condition and remains one of the most common causes of sudden mortality worldwide. Detecting cardiac risks early and predicting outcomes accurately can significantly improve survival rates through timely medical care.  
This project aims to develop a streamlit application that enables users to experiment with multiple machine learning models on cardiac data, assess their performance, and visualize the results.  
The application provides an interactive interface for uploading test datasets, selecting models, and reviewing evaluation metrics, confusion matrices, and classification reports.

---

## b. Dataset Description
The dataset used is the **UCI Heart Disease dataset (heart.csv)**, which contains patient health records with attributes relevant to cardiac health:

- **Age**: Patient’s age in years  
- **Sex**: Male (1) or Female (0)  
- **Chest Pain Type (cp)**: 0–3 categorical values  
- **Resting Blood Pressure (trestbps)**: mm Hg  
- **Cholesterol (chol)**: mg/dl  
- **Fasting Blood Sugar (fbs)**: >120 mg/dl (1 = true, 0 = false)  
- **Resting ECG (restecg)**: 0–2 categorical values  
- **Max Heart Rate Achieved (thalach)**  
- **Exercise Induced Angina (exang)**: 1 = yes, 0 = no  
- **ST Depression (oldpeak)**  
- **Slope of ST Segment (slope)**: 0–2 categorical values  
- **Number of Major Vessels (ca)**: 0–3  
- **Thalassemia (thal)**: 1 = normal, 2 = fixed defect, 3 = reversible defect  
- **Target**: 1 = cardiac arrest risk present, 0 = no risk  

This dataset is widely recognized for benchmarking classification models in healthcare research.

---

## c. Models Used
Six machine learning models were trained and evaluated to predict cardiac arrest risk:

- Logistic Regression  
- Decision Tree  
- k‑Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

## d. Comparison of Evaluation Metrics

| ML Model Name        | Accuracy  | AUC       | Precision | Recall   | F1       | MCC      |
|----------------------|-----------|-----------|-----------|----------|----------|----------|
| Logistic Regression  | 0.809756  | 0.929810  | 0.761905  | 0.914286 | 0.831169 | 0.630908 |
| Decision Tree        | 0.985366  | 0.985714  | 1.000000  | 0.971429 | 0.985507 | 0.971151 |
| kNN                  | 0.863415  | 0.962905  | 0.873786  | 0.857143 | 0.865385 | 0.726935 |
| Naive Bayes          | 0.829268  | 0.904286  | 0.807018  | 0.876190 | 0.840183 | 0.660163 |
| Random Forest        | 1.000000  | 1.000000  | 1.000000  | 1.000000 | 1.000000 | 1.000000 |
| XGBoost              | 1.000000  | 1.000000  | 1.000000  | 1.000000 | 1.000000 | 1.000000 |

---

## e. Observations on Model Performance

| ML Model Name        | Observation |
|----------------------|-------------|
| Logistic Regression  | Provides balanced accuracy and recall. It is interpretable and useful for understanding cardiac risk factors, though slightly weaker than tree‑based models. |
| Decision Tree        | Delivers very high accuracy and precision. However, it can overfit cardiac data if not pruned carefully. |
| kNN                  | Performs reasonably well but is sensitive to the choice of k and feature scaling. Recall is slightly lower compared to Logistic Regression. |
| Naive Bayes          | Shows decent performance with strong recall, but assumes independence among features, which may not fully apply to cardiac datasets. |
| Random Forest        | Achieves perfect scores thanks to ensemble averaging, reducing overfitting and improving generalization. Highly reliable for cardiac risk prediction. |
| XGBoost              | Also achieves perfect scores, leveraging gradient boosting for superior performance. Very effective but computationally more demanding. |
