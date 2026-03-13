import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="Mental Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .result-positive {
        background: linear-gradient(135deg, #1a3a2a, #1e4d35);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2ecc71;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #3a1a1a, #4d1e1e);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e74c3c;
        text-align: center;
    }
    .section-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('mental_health_model.pkl')



@st.cache_data
def load_features():
    with open('feature_names.json') as f:
        feature_names = json.load(f)
    with open('feature_options.json') as f:
        feature_options = json.load(f)
    return feature_names, feature_options

model = load_model()
feature_names, feature_options = load_features()

encoding_maps = {
    'Gender': {'Female': 0, 'Male': 1},
    'Occupation': {'Business': 0, 'Corporate': 1, 'Housewife': 2, 'Others': 3, 'Student': 4},
    'self_employed': {'No': 0, 'Yes': 1},
    'family_history': {'No': 0, 'Yes': 1},
    'Days_Indoors': {'1-14 days': 0, '15-30 days': 1, '31-60 days': 2, 'Go out Every day': 3, 'More than 2 months': 4},
    'Growing_Stress': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'Changes_Habits': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'Mental_Health_History': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'Mood_Swings': {'High': 0, 'Low': 1, 'Medium': 2},
    'Coping_Struggles': {'No': 0, 'Yes': 1},
    'Work_Interest': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'Social_Weakness': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'mental_health_interview': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'care_options': {'No': 0, 'Not sure': 1, 'Yes': 2},
}

feature_labels = {
    'Gender': 'Gender',
    'Occupation': 'Occupation',
    'self_employed': 'Self Employed',
    'family_history': 'Family History of Mental Health Issues',
    'Days_Indoors': 'Days Spent Indoors',
    'Growing_Stress': 'Growing Stress',
    'Changes_Habits': 'Changes in Habits',
    'Mental_Health_History': 'Mental Health History',
    'Mood_Swings': 'Mood Swings',
    'Coping_Struggles': 'Coping Struggles',
    'Work_Interest': 'Interest in Work',
    'Social_Weakness': 'Social Weakness',
    'mental_health_interview': 'Comfortable Discussing Mental Health at Work',
    'care_options': 'Aware of Mental Health Care Options',
}

st.sidebar.title("Mental Health Predictor")
st.sidebar.markdown("An ML-powered tool to assess mental health treatment likelihood based on lifestyle and psychological indicators.")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Predict", "Model Analysis", "About"])

# ══════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════
if page == "Predict":
    st.title("Mental Health Treatment Predictor")
    st.markdown("Answer the questions below. The model will predict whether mental health treatment may be beneficial and explain the key factors driving that prediction.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Personal & Lifestyle</p>', unsafe_allow_html=True)
        gender = st.selectbox(feature_labels['Gender'], feature_options['Gender'])
        occupation = st.selectbox(feature_labels['Occupation'], feature_options['Occupation'])
        self_employed = st.radio(feature_labels['self_employed'], feature_options['self_employed'], horizontal=True)
        family_history = st.radio(feature_labels['family_history'], feature_options['family_history'], horizontal=True)
        days_indoors = st.selectbox(feature_labels['Days_Indoors'], feature_options['Days_Indoors'])
        growing_stress = st.radio(feature_labels['Growing_Stress'], feature_options['Growing_Stress'], horizontal=True)
        changes_habits = st.radio(feature_labels['Changes_Habits'], feature_options['Changes_Habits'], horizontal=True)

    with col2:
        st.markdown('<p class="section-header">Mental Health Indicators</p>', unsafe_allow_html=True)
        mental_health_history = st.radio(feature_labels['Mental_Health_History'], feature_options['Mental_Health_History'], horizontal=True)
        mood_swings = st.select_slider(feature_labels['Mood_Swings'], options=['Low', 'Medium', 'High'])
        coping_struggles = st.radio(feature_labels['Coping_Struggles'], feature_options['Coping_Struggles'], horizontal=True)
        work_interest = st.radio(feature_labels['Work_Interest'], feature_options['Work_Interest'], horizontal=True)
        social_weakness = st.radio(feature_labels['Social_Weakness'], feature_options['Social_Weakness'], horizontal=True)
        mental_health_interview = st.radio(feature_labels['mental_health_interview'], feature_options['mental_health_interview'], horizontal=True)
        care_options = st.radio(feature_labels['care_options'], feature_options['care_options'], horizontal=True)

    st.divider()

    if st.button("Run Prediction", type="primary", use_container_width=True):
        input_data = {
            'Gender': encoding_maps['Gender'][gender],
            'Occupation': encoding_maps['Occupation'][occupation],
            'self_employed': encoding_maps['self_employed'][self_employed],
            'family_history': encoding_maps['family_history'][family_history],
            'Days_Indoors': encoding_maps['Days_Indoors'][days_indoors],
            'Growing_Stress': encoding_maps['Growing_Stress'][growing_stress],
            'Changes_Habits': encoding_maps['Changes_Habits'][changes_habits],
            'Mental_Health_History': encoding_maps['Mental_Health_History'][mental_health_history],
            'Mood_Swings': encoding_maps['Mood_Swings'][mood_swings],
            'Coping_Struggles': encoding_maps['Coping_Struggles'][coping_struggles],
            'Work_Interest': encoding_maps['Work_Interest'][work_interest],
            'Social_Weakness': encoding_maps['Social_Weakness'][social_weakness],
            'mental_health_interview': encoding_maps['mental_health_interview'][mental_health_interview],
            'care_options': encoding_maps['care_options'][care_options],
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if prediction == 1:
                st.markdown("""
                <div class="result-negative">
                    <h2>Treatment Recommended</h2>
                    <p>Based on your responses, seeking mental health support may be beneficial.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-positive">
                    <h2>Low Risk</h2>
                    <p>Based on your responses, your mental health indicators appear stable.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability[1]*100, 1),
                title={'text': "Treatment Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if probability[1] > 0.5 else "#2ecc71"},
                    'steps': [
                        {'range': [0, 40], 'color': "#1a3a2a"},
                        {'range': [40, 60], 'color': "#3a3a1a"},
                        {'range': [60, 100], 'color': "#3a1a1a"},
                    ],
                }
            ))
            fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col_res3:
            fig2 = px.bar(
                x=['No Treatment', 'Treatment'],
                y=[round(probability[0]*100,1), round(probability[1]*100,1)],
                color=['No Treatment', 'Treatment'],
                color_discrete_map={'No Treatment': '#2ecc71', 'Treatment': '#e74c3c'},
                title="Confidence Breakdown"
            )
            fig2.update_layout(
                height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # SHAP explanation
        st.divider()
        st.subheader("Why this prediction?")
        st.markdown("The chart below shows which factors contributed most to this assessment, based on SHAP (SHapley Additive exPlanations) values.")

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            fig3 = go.Figure(go.Bar(
                x=shap_values[0],
                y=list(feature_labels.values()),
                orientation='h',
                marker_color=['#e74c3c' if v > 0 else '#2ecc71' for v in shap_values[0]]
            ))
            fig3.update_layout(
                title="Feature Contribution to Prediction (SHAP Values)",
                xaxis_title="Impact on Prediction",
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Red bars push toward Treatment Recommended. Green bars push toward Low Risk.")
        except Exception as e:
            st.warning("SHAP explanation unavailable.")
# ══════════════════════════════════════
# PAGE 2 — MODEL ANALYSIS
# ══════════════════════════════════════
elif page == "Model Analysis":
    st.title("Model Analysis")
    st.markdown("Performance metrics and visualizations for all trained models.")
    st.divider()

    st.subheader("Model Performance Comparison")
    metrics_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'KNN', 'XGBoost'],
        'Accuracy': [0.6909, 0.6886, 0.6905, 0.7146, 0.6498, 0.7139],
        'Precision': [0.6910, 0.6805, 0.6766, 0.7014, 0.6509, 0.7014],
        'Recall': [0.7035, 0.7247, 0.7433, 0.7587, 0.6635, 0.7314],
        'F1': [0.6972, 0.7019, 0.7084, 0.7289, 0.6571, 0.7314],
        'ROC-AUC': [0.7517, 0.7161, 0.7192, 0.7729, 0.6867, 0.7694],
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#2ecc71'), use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        st.image('plots/confusion_matrix.png', use_container_width=True)
    with col2:
        st.subheader("ROC Curves")
        st.image('plots/roc_curves.png', use_container_width=True)

    st.subheader("SHAP Feature Importance (Global)")
    col3, col4 = st.columns(2)
    with col3:
        st.image('plots/shap_summary.png', use_container_width=True)
    with col4:
        st.image('plots/shap_bar.png', use_container_width=True)

    st.subheader("Feature Importance")
    st.image('plots/feature_importance.png', use_container_width=True)

    st.subheader("Model Comparison")
    st.image('plots/model_comparison.png', use_container_width=True)

# ══════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════
elif page == "About":
    st.title("About This Project")
    st.divider()

    st.markdown("""
    ## Mental Health Treatment Predictor

    This project uses machine learning to predict whether an individual may benefit from
    mental health treatment based on lifestyle, occupational, and psychological indicators.

    ## Dataset
    - **Source:** Kaggle — Mental Health Dataset
    - **Size:** 292,364 records
    - **Features:** 14 features including gender, occupation, stress indicators, and mental health history
    - **Target:** Whether the individual sought mental health treatment (Yes/No)

    ## Models Trained
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting (Best Model — 71.5% accuracy, 0.773 ROC-AUC)
    - K-Nearest Neighbors
    - XGBoost

    ## Explainability
    This app uses SHAP (SHapley Additive exPlanations) to explain individual predictions,
    showing which features contributed most to each prediction and in which direction.

    ## Tech Stack
    - Python, Scikit-learn, XGBoost, SHAP, Pandas, NumPy
    - Streamlit, Plotly
    - Deployed on Streamlit Community Cloud

    ## Disclaimer
    This tool is for educational purposes only. It does not constitute medical advice.
    Always consult a qualified mental health professional for diagnosis and treatment.
    """)
