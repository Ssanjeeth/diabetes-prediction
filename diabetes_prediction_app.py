import streamlit as st
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess dataset with caching
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("diabetes.csv")
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
        df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
    df.dropna(inplace=True)
    X = df[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure', 'Pregnancies']]
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y), X

# Train the model with cached results
@st.cache_resource
def train_model(train_X, train_Y):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid.fit(train_X, train_Y)
    return grid.best_estimator_

# Load and process data
(train_X, test_X, train_Y, test_Y), X_full = load_and_preprocess()
best_rf = train_model(train_X, train_Y)

# Styling
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; font-family: 'Arial', sans-serif; }
    .stSidebar { background-color: #ffffff; padding: 20px; color: black; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); }
    .stButton button { background-color: #006400; color: white; font-size: 16px; padding: 10px 20px; border-radius: 5px; border: none; width: 100%; }
    .stButton button:hover { background-color: #45a049; }
    .card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
    .card h3 { margin-top: 0; color: #4CAF50; }
    .explanation { font-size: 14px; color: #555555; margin-bottom: 20px; }
    .highlight { color: #4CAF50; font-weight: bold; }
    .stSidebar label { color: #4CAF50 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; color: white;">
    <h1 style="margin: 0;">Diabetes Prediction</h1>
    <p style="margin: 0;">Predict the likelihood of diabetes using advanced machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar input
st.sidebar.header("User Input")
glucose = st.sidebar.number_input("Glucose Level", value=85)
bmi = st.sidebar.number_input("BMI", value=28.0)
age = st.sidebar.number_input("Age", value=25)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", value=0.3)
blood_pressure = st.sidebar.number_input("Blood Pressure", value=70)
pregnancies = st.sidebar.number_input("Pregnancies", value=1)

# Predict button
if st.sidebar.button("Predict Diabetes"):
    raw_df = pd.DataFrame([{
        'Glucose': glucose,
        'BMI': bmi,
        'Age': age,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'BloodPressure': blood_pressure,
        'Pregnancies': pregnancies
    }])

    prediction = best_rf.predict(raw_df)
    prediction_proba = best_rf.predict_proba(raw_df)[0][1]

    st.markdown(f"""
    <div class="card">
        <h3>Prediction: {'Has Diabetes' if prediction[0] == 1 else 'No Diabetes'}</h3>
        <p>Confidence Score: {prediction_proba:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation">
        The <span class="highlight">Confidence Score</span> indicates the model's certainty in its prediction.
    </div>
    """, unsafe_allow_html=True)

    if st.checkbox("Show Advanced Analysis"):
        st.markdown("<h2 style='color: black;'>Feature Importance</h2>", unsafe_allow_html=True)
        fig_feature_importance = px.bar(
            x=X_full.columns,
            y=best_rf.feature_importances_,
            labels={'x': 'Features', 'y': 'Importance'},
            title="Feature Importance"
        )
        st.plotly_chart(fig_feature_importance)

        st.markdown("""
        <div class="explanation">
            Higher importance = more influence on predictions.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h2 style='color: black;'>Confusion Matrix</h2>", unsafe_allow_html=True)
        predictions_test_rf = best_rf.predict(test_X)
        conf_matrix = confusion_matrix(test_Y, predictions_test_rf)
        fig_conf_matrix = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=["No Diabetes", "Has Diabetes"], y=["No Diabetes", "Has Diabetes"],
                                    title="Confusion Matrix (Test Data)")
        st.plotly_chart(fig_conf_matrix)

        st.markdown("<h2 style='color: black;'>ROC Curve</h2>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(test_Y, best_rf.predict_proba(test_X)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})',
                          labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        st.plotly_chart(fig_roc)

        st.markdown("<h2 style='color: black;'>Model Accuracy</h2>", unsafe_allow_html=True)
        accuracy_train_rf = metrics.accuracy_score(train_Y, best_rf.predict(train_X))
        accuracy_test_rf = metrics.accuracy_score(test_Y, predictions_test_rf)

        st.markdown(f"""
        <div class="card">
            <h3>Model Accuracy</h3>
            <div class="explanation">
            <p>Training Data: {accuracy_train_rf:.2f}</p>
            <p>Test Data: {accuracy_test_rf:.2f}</p></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="explanation">
            Accuracy = % of correct predictions.
        </div>
        """, unsafe_allow_html=True)
