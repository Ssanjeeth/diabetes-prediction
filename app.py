import streamlit as st
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tabulate import tabulate

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

# Data preprocessing
# Replace 0 values in certain columns with NaN
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

# Drop rows with NaN values to get a cleaner dataset
df.dropna(inplace=True)

# Select features based on feature importance analysis
X = df[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure', 'Pregnancies']]
y = df['Outcome']

# Split the dataset into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Setting up the hyperparameter grid for Random Forest tuning
param_grid = {
    'n_estimators': [100, 150, 200],  # Number of trees in the forest
    'max_depth': [5, 10, 15],         # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]     # Minimum samples required at each leaf node
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Apply Grid Search with Cross Validation to find the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_X, train_Y)

# Retrieve the best model from grid search
best_rf = grid_search.best_estimator_

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stSidebar {
        background-color: #ffffff;
        padding: 20px;
        color: black;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #006400;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card h3 {
        margin-top: 0;
        color: #4CAF50;
    }
    .explanation {
        font-size: 14px;
        color: #555555;
        margin-bottom: 20px;
    }
    .highlight {
        color: #4CAF50;
        font-weight: bold;
    }
    /* Custom CSS for sidebar input labels */
    .stSidebar label {
        color: #4CAF50 !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
st.markdown(
    """
    <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; color: white;">
        <h1 style="margin: 0;">Diabetes Prediction</h1>
        <p style="margin: 0;">Predict the likelihood of diabetes using advanced machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input
st.sidebar.header("User Input")
st.sidebar.write("Enter patient details:")

# Simulate raw patient data input
glucose = st.sidebar.number_input(
    "Glucose Level (mg/dL) - Blood sugar level after fasting", 
    value=85
)
bmi = st.sidebar.number_input(
    "BMI (kg/m²) - Body Mass Index, a measure of body fat", 
    value=28.0
)
age = st.sidebar.number_input(
    "Age (years) - Age of the patient", 
    value=25
)
diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function - Genetic influence of diabetes", 
    value=0.3
)
blood_pressure = st.sidebar.number_input(
    "Blood Pressure (mmHg) - Diastolic blood pressure", 
    value=70
)
pregnancies = st.sidebar.number_input(
    "Number of Pregnancies - Total number of pregnancies", 
    value=1
)

# Add an "Enter" button to trigger prediction
if st.sidebar.button("Predict Diabetes"):
    # Create raw data DataFrame
    raw_data = {
        'Glucose': [glucose],
        'BMI': [bmi],
        'Age': [age],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'BloodPressure': [blood_pressure],
        'Pregnancies': [pregnancies],
    }

    raw_df = pd.DataFrame(raw_data)

    # Predict the outcome for the new test dataset
    prediction = best_rf.predict(raw_df)
    prediction_proba = best_rf.predict_proba(raw_df)[0][1]

    # Display prediction and confidence score in a card
    st.markdown(
        f"""
        <div class="card">
            <h3>Prediction: {'Has Diabetes' if prediction[0] == 1 else 'No Diabetes'}</h3>
            <p>Confidence Score: {prediction_proba:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Explanation for Confidence Score
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">Confidence Score</span> represents the model's certainty in its prediction. 
            A score closer to <span class="highlight">1</span> indicates high confidence, while a score closer to <span class="highlight">0</span> indicates low confidence.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature Importance Plot
    st.markdown("<h2 style='color: black;'>Feature Importance</h2>", unsafe_allow_html=True)
    feature_importances = best_rf.feature_importances_
    features = X.columns
    fig_feature_importance = px.bar(x=features, y=feature_importances, labels={'x': 'Features', 'y': 'Importance'}, title="Feature Importance")
    st.plotly_chart(fig_feature_importance)

    # Explanation for Feature Importance
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">Feature Importance</span> plot shows the contribution of each feature to the model's predictions. 
            Features with higher importance have a greater impact on the model's decisions.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confusion Matrix
    st.markdown("<h2 style='color: black;'>Confusion Matrix</h2>", unsafe_allow_html=True)
    predictions_test_rf = best_rf.predict(test_X)
    conf_matrix = confusion_matrix(test_Y, predictions_test_rf)
    fig_conf_matrix = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual", color="Count"), 
                                x=["No Diabetes", "Has Diabetes"], y=["No Diabetes", "Has Diabetes"], 
                                title="Confusion Matrix (Test Data)")
    st.plotly_chart(fig_conf_matrix)

    # Explanation for Confusion Matrix
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">Confusion Matrix</span> shows the number of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). 
            It helps evaluate the model's performance in classifying patients.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ROC Curve
    st.markdown("<h2 style='color: black;'>ROC Curve</h2>", unsafe_allow_html=True)
    fpr, tpr, thresholds = roc_curve(test_Y, best_rf.predict_proba(test_X)[:,1])
    roc_auc = auc(fpr, tpr)
    fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
    st.plotly_chart(fig_roc)

    # Explanation for ROC Curve
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">ROC Curve</span> shows the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR). 
            A curve closer to the top-left corner indicates better model performance.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Model Accuracy
    st.markdown("<h2 style='color: black;'>Model Accuracy</h2>", unsafe_allow_html=True)
    accuracy_train_rf = metrics.accuracy_score(train_Y, best_rf.predict(train_X))
    accuracy_test_rf = metrics.accuracy_score(test_Y, predictions_test_rf)

    # Display Model Accuracy in a card
    st.markdown(
        f"""
        <div class="card">
            <h3>Model Accuracy</h3>
            <div class="explanation">
            <p>Training Data: {accuracy_train_rf:.2f}</p>
            <p>Test Data: {accuracy_test_rf:.2f}</p></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Explanation for Model Accuracy
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">Model Accuracy</span> represents the percentage of correct predictions made by the model on the training and test datasets.
        </div>
        """,
        unsafe_allow_html=True
    )
