Telecom Customer Churn Prediction App

This project is a Streamlit-based machine learning application that predicts telecom customer churn using an XGBoost model trained on the IBM Telco Customer Churn dataset. The app provides real-time predictions, explains the factors influencing each prediction, and suggests possible retention actions. It is designed to be simple, clear, and practical for business use cases.

Features
1. Real-Time Churn Prediction

Users can input customer details through a clean interface. The model then outputs whether the customer is likely to churn and the associated probability.

2. Machine Learning Model (XGBoost)

The application uses an XGBoost classifier with feature engineering steps such as tenure grouping, average charge per month, and simplified payment method categories. Both numerical and categorical features are handled appropriately.

3. Explainability with SHAP

To ensure transparency, the app provides SHAP-based explanations showing which features contributed most to the prediction. It also displays global feature importance for the model.

4. PDF Report Generation

The application generates a downloadable PDF summarizing the customer details, prediction outcome, probability score, top contributing features, and recommended retention actions.

5. Retention Recommendations

Based on the customer’s profile, the system provides practical suggestions to reduce churn risk, such as offering discounts, improving support, or adjusting contract terms.

6. Clean and Simple User Interface

The app presents information clearly using structured layouts, sectioned explanations, and readable formatting.

Summary

This project offers an end-to-end solution for telecom churn prediction, combining machine learning, interpretability, practical recommendations, and a user-friendly interface. It is suitable for academic use, demonstrations, or integration into a business workflow.