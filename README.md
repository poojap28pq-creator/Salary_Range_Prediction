# Salary_Range_Prediction

## Project Overview

This project uses machine learning to predict the minimum and maximum annual salary ranges for job postings based on job-related information.

The project compares multiple regression models and uses XGBoost to develop the final salary prediction models. An interactive Streamlit application was also developed for real-time salary estimation.

## Objectives

- Predict minimum annual salary.
- Predict maximum annual salary.
- Identify the key factors influencing salary.
- Compare different machine learning models.
- Deploy the final solution through a Streamlit application.

## Dataset

The project was developed using a historical NYC job postings dataset. Due to data confidentiality and privacy considerations, the complete dataset is not included in this repository. A sample dataset (sample_nyc_jobs.csv) is provided to demonstrate the data structure and features used in the project.

The dataset contains historical job posting information, including:

- Agency
- Business Title
- Civil Service Title
- Career Level
- Education
- Experience
- Job Category
- Work Location
- Division/Work Unit
- Posting Year and Month

The final cleaned dataset contains **5,075 job postings and 21 features**.

## Data Preparation

- Removed duplicate records
- Handled missing values
- Cleaned categorical and text-based features
- Extracted education and experience information
- Created posting year and month features
- Applied categorical encoding
- Selected important features using XGBoost feature importance

## Models Tested

- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

XGBoost achieved the strongest overall performance and was selected for the final prediction models.

## Model Deployment

The final models are integrated into a **Streamlit web application**.

Users can enter job-related information and receive:

- Predicted Minimum Salary
- Predicted Maximum Salary

## Technologies Used

Python | Pandas | NumPy | Scikit-learn | XGBoost | LightGBM | CatBoost | Matplotlib | Seaborn | Streamlit | Joblib

## Key Outcome

The project demonstrates how machine learning can be used to provide consistent, data-driven salary estimates based on historical job posting information.
