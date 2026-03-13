import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
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
    'Growing_Stress': {'No': 0, 'Yes': 1},
    'Changes_Habits': {'No': 0, 'Yes': 1},
    'Mental_Health_History': {'No': 0, 'Yes': 1},
    'Mood_Swings': {'High': 0, 'Low': 1, 'Medium': 2},
    'Coping_Struggles': {'No': 0, 'Yes': 1},
    'Work_Interest': {'No': 0, 'Yes': 1},
    'Social_Weakness': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'mental_health_interview': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'care_options': {'No': 0, 'Not sure': 1, 'Yes': 2},
}

st.sidebar.title("Mental Health Predictor")
st.sidebar.markdown("An ML-powered tool to assess mental health treatment likelihood.")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Predict", "Model Analysis", "About"])

if page == "Predict":
    st.title("Mental Health Treatment Predictor")
    st.markdown("Fill in the details below to assess whether mental health treatment may be beneficial.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", feature_options['Gender'])
        occupation = st.selectbox("Occupation", feature_options['Occupation'])
        self_employed = st.selectbox("Self Employed", feature_options['self_employed'])
        family_history = st.selectbox("Family History of Mental Health Issues", feature_options['family_history'])
        days_indoors = st.selectbox("Days Spent Indoors", feature_options['Days_Indoors'])
        growing_stress = st.selectbox("Growing Stress", feature_options['Growing_Stress'])
        changes_habits = st.selectbox("Changes in Habits", feature_options['Changes_Habits'])

    with col2:
        st.subheader("Mental Health Indicators")
        mental_health_history = st.selectbox("Mental Health History", feature_options['Mental_Health_History'])
        mood_swings = st.selectbox("Mood Swings", feature_options['Mood_Swings'])
        coping_struggles = st.selectbox("Coping Struggles", feature_options['Coping_Struggles'])
        work_interest = st.selectbox("Interest in Work", feature_options['Work_Interest'])
        social_weakness = st.selectbox("Social Weakness", feature_options['Social_Weakness'])
        mental_health_interview = st.selectbox("Would Discuss Mental Health in Interview", feature_options['mental_health_interview'])
        care_options = st.selectbox("Aware of Care Options", feature_options['care_options'])

    st.divider()

    if st.button("Predict", type="primary", use_container_width=True):
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

        st.info("This tool is for educational purposes only and does not constitute medical advice. Please consult a qualified mental health professional for proper diagnosis and treatment.")

elif page == "Model Analysis":
    st.title("Model Analysis")
    st.markdown("Performance metrics and visualizations for all trained models.")
    st.divider()

    st.subheader("Model Performance Comparison")
    metrics_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'KNN'],
        'Accuracy': [0.6909, 0.6886, 0.6905, 0.7146, 0.6498],
        'Precision': [0.6910, 0.6805, 0.6766, 0.7014, 0.6509],
        'Recall': [0.7035, 0.7247, 0.7433, 0.7587, 0.6635],
        'F1': [0.6972, 0.7019, 0.7084, 0.7289, 0.6571],
        'ROC-AUC': [0.7517, 0.7161, 0.7192, 0.7729, 0.6867],
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

    st.subheader("Feature Importance")
    st.image('plots/feature_importance.png', use_container_width=True)

    st.subheader("Model Comparison")
    st.image('plots/model_comparison.png', use_container_width=True)

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
    
    ## Tech Stack
    - Python, Scikit-learn, Pandas, NumPy
    - Streamlit, Plotly
    - Deployed on Streamlit Community Cloud
    
    ## Disclaimer
    This tool is for educational purposes only. It does not constitute medical advice. 
    Always consult a qualified mental health professional for diagnosis and treatment.
    """)