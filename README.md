# Diabetes Prediction
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Wael-Messaoudi/Diabet)

## Overview
This repository contains a machine learning project for predicting the onset of diabetes based on diagnostic medical measurements. The project utilizes the Pima Indians Diabetes Database to build and evaluate several classification models. The primary goal is to accurately classify a patient as either diabetic or non-diabetic.

The analysis is contained within a single Jupyter Notebook (`Diabet.ipynb`) that covers all steps from data loading and exploration to model training, evaluation, and persistence.

## Dataset
The project uses the **Pima Indians Diabetes Database**, which is publicly available from the UCI Machine Learning Repository.

The dataset consists of 768 records of female patients of at least 21 years of age of Pima Indian heritage. It includes 8 independent features and one target variable:

*   **Features:**
    *   `Pregnancies`: Number of times pregnant
    *   `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    *   `BloodPressure`: Diastolic blood pressure (mm Hg)
    *   `SkinThickness`: Triceps skin fold thickness (mm)
    *   `Insulin`: 2-Hour serum insulin (mu U/ml)
    *   `BMI`: Body mass index (weight in kg/(height in m)^2)
    *   `DiabetesPedigreeFunction`: A function that scores the likelihood of diabetes based on family history
    *   `Age`: Age in years
*   **Target Variable:**
    *   `Outcome`: Class variable (0 = non-diabetic, 1 = diabetic)

## Methodology
The project follows a standard machine learning workflow:

1.  **Data Loading and Exploration:** The dataset is loaded into a pandas DataFrame. An initial exploratory data analysis (EDA) is performed to understand its structure, statistics, and distributions. A correlation heatmap is generated to visualize relationships between features.

2.  **Data Preprocessing:** The analysis reveals a class imbalance in the target variable (`Outcome`). To address this, the **Synthetic Minority Over-sampling Technique (SMOTE)** is applied to the training data, creating a balanced dataset for model training.

3.  **Model Training and Tuning:** The resampled data is split into training (80%) and testing (20%) sets. Three different classification models are trained and evaluated:
    *   Random Forest Classifier
    *   K-Nearest Neighbors (KNN)
    *   CatBoost Classifier

    `GridSearchCV` is used to find the optimal hyperparameters for the Random Forest and KNN models based on the `roc_auc` scoring metric.

4.  **Model Evaluation:** The performance of each tuned model is assessed on the unseen test set using the following metrics:
    *   Accuracy
    *   Precision
    *   Recall
    *   F1-Score

## Results
The performance of the models on the test set is summarized below. The RandomForestClassifier achieved the best overall performance.

| Model                  | Accuracy | Precision | Recall | F1-score |
| ---------------------- | :------: | :-------: | :----: | :------: |
| RandomForestClassifier |  0.8200  |  0.7963   | 0.8600 |  0.8269  |
| KNeighborsClassifier   |  0.7950  |  0.7398   | 0.9100 |  0.8161  |
| CatBoostClassifier     |  0.8050  |  0.7798   | 0.8500 |  0.8134  |

#### Best Hyperparameters for Random Forest:
*   `max_depth`: 20
*   `min_samples_leaf`: 1
*   `min_samples_split`: 2
*   `n_estimators`: 100

## Installation and Usage
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Wael-Messaoudi/Diabet.git
    cd Diabet
    ```

2.  **Install the dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn catboost joblib jupyter
    ```

3.  **Run the notebook:**
    Launch Jupyter Notebook or JupyterLab and open the `Diabet.ipynb` file to explore the code and results.
    ```bash
    jupyter notebook Diabet.ipynb
    ```

## Saved Model
The best-performing model, the tuned `RandomForestClassifier`, has been saved to the file `best_rf_model.pkl` using `joblib`. This model can be loaded and used for inference without needing to retrain.
