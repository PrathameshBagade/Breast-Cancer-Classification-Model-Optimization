# **Model Optimization & Hyperparameter Tuning**

This project focuses on the *professional workflow* for machine learning. Instead of just building a single model, the goal is to **systematically find the *best possible* model** for a problem. We do this by introducing Pipelines and GridSearchCV.

## **1\. Project Overview & Problem Statement**

Problem: Given a set of 30 numerical features from a breast tumor biopsy, can we build a model to accurately predict if the tumor is malignant (cancerous) or benign (not cancerous)?  
Task: This is a binary classification problem.  
Dataset: The Wisconsin Breast Cancer dataset, loaded directly from scikit-learn.

## **2\. Key Professional Skills**

This project introduces three core concepts used by all professional data scientists:

1. **Support Vector Machine (SVM):** A new, powerful, and highly effective classification model. SVMs are a cornerstone of "classic" machine learning.  
2. **Pipeline (from Scikit-learn):** A tool that chains together multiple processing steps (like data scaling and modeling) into a single object. This makes the code cleaner, more robust, and **prevents a common error called "data leakage."**  
3. **GridSearchCV (Hyperparameter Tuning):** The method for *optimizing* a model. GridSearchCV automatically tests dozens (or hundreds) of different combinations of model settings (hyperparameters) to find the single best-performing combination.

## **3\. Project Workflow**

1. **Data Loading:** Loaded the breast cancer dataset from scikit-learn.  
2. **EDA:** Visualized the class distribution (Malignant vs. Benign).  
3. **Baseline Model:** A simple RandomForestClassifier was trained to get a baseline accuracy score (typically \~96.5%).  
4. **Pipeline Creation:** A Pipeline was created containing two steps:  
   1. StandardScaler(): To scale all features.  
   2. SVC(): Our Support Vector Machine model.  
5. **Hyperparameter Grid:** A "grid" of settings for the SVM was defined to test different C values (regularization) and kernel types (linear vs. rbf).  
6. **Grid Search:** GridSearchCV was used to run all combinations of parameters on the Pipeline using 5-fold cross-validation.  
7. **Evaluation:** The single best model found by GridSearchCV was then evaluated on the final, unseen test set.

## **4\. Model & Results**

* **Baseline Model:** RandomForestClassifier  
* **Optimized Model:** SVC (tuned with GridSearchCV)

The GridSearchCV process automatically found the best settings for the SVC model (e.g., kernel='rbf', C=10, gamma=0.01).

* **Baseline RF Accuracy:** \~96.49%  
* **Tuned SVM Accuracy:** \~98.25%

The tuned SVM outperformed the baseline Random Forest, demonstrating the power of proper model selection and optimization. The final confusion matrix shows the optimized model made very few errors on the test set.

## **5\. How to Run This Project**

1. Ensure you have Python 3+ installed.  
2. Download the project\_4\_optimization.py script.  
3. Install the required libraries:  
   pip install pandas matplotlib seaborn scikit-learn

4. Run the Python script from your terminal:  
   python project\_4\_optimization.py

   The script will print all steps, including the GridSearchCV progress, and show the final evaluation plots.
