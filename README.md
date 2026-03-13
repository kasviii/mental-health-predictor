# Mental Health Treatment Predictor

A machine learning web application that predicts whether an individual may benefit from mental health treatment based on lifestyle, occupational, and psychological indicators.

Live Demo: [mental-health-predictor-kasviii.streamlit.app](https://mental-health-predictor-kasviii.streamlit.app)

## Dataset
- Source: Kaggle — Mental Health Dataset
- Size: 292,364 records
- Features: 14 indicators including gender, occupation, stress levels, mental health history, and behavioral patterns
- Target: Whether the individual sought mental health treatment (Yes/No)

## Models Trained
| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 69.1% | 0.752 | 0.697 |
| Decision Tree | 68.9% | 0.716 | 0.702 |
| Random Forest | 69.1% | 0.719 | 0.708 |
| Gradient Boosting | **71.5%** | **0.773** | **0.729** |
| KNN | 65.0% | 0.687 | 0.657 |

Gradient Boosting was selected as the final model based on highest accuracy, F1, and ROC-AUC scores.

## Explainability
The app uses SHAP (SHapley Additive exPlanations) to explain individual predictions — showing which factors pushed the model toward or away from recommending treatment. This makes the model transparent and interpretable, which is critical for healthcare applications.

## Features
- Interactive assessment form with 14 clinical and lifestyle indicators
- Real-time prediction with confidence breakdown
- SHAP-based explanation of individual predictions
- Model analysis page with confusion matrix, ROC curves, and feature importance

## Stack
- Python, Scikit-learn, Pandas, NumPy
- SHAP, Plotly, Streamlit
- Deployed on Streamlit Community Cloud

## Disclaimer
This tool is for educational purposes only. It does not constitute medical advice. Always consult a qualified mental health professional for diagnosis and treatment.
