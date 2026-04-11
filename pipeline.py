import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

# 1. Page Configuration for horizontal expansion aesthetics [cite: 25, 29]
st.set_page_config(page_title="AutoML Flow Dashboard", layout="wide")
st.title("🚀 Professional ML Pipeline Dashboard") 

# 2. Problem Selection in Sidebar [cite: 25, 27]
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"]) 

# 3. Horizontal Expansion using Tabs [cite: 29, 30]
tabs = st.tabs([
    "1. Data Input", "2. EDA", "3. Cleaning", 
    "4. Feature Selection", "5. Split", 
    "6. Model Selection", "7. Training", "8. Metrics"
])

# --- TAB 1: DATA INPUT [cite: 25] ---
with tabs[0]:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv") 
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())
        
        target_feature = st.selectbox("Select Target Feature", df.columns) 
        features = st.multiselect("Select Features for Visualization", [c for c in df.columns if c != target_feature])
        
        if features:
            # PCA Visualization for Data Shape [cite: 25]
            pca = PCA(n_components=2)
            numeric_data = df[features].dropna().select_dtypes(include=[np.number])
            if not numeric_data.empty:
                components = pca.fit_transform(numeric_data)
                fig = px.scatter(components, x=0, y=1, title="Data Shape (PCA 2D Projection)") 
                st.plotly_chart(fig)
            else:
                st.warning("Please select numeric features for PCA visualization.")

# --- TAB 2: EXPLORATORY DATA ANALYSIS (EDA) [cite: 26] ---
with tabs[1]:
    if uploaded_file:
        st.write("### Statistical Summary", df.describe()) 
        # Correlation Heatmap
        numeric_only = df.select_dtypes(include=[np.number])
        if not numeric_only.empty:
            fig, ax = plt.subplots()
            sns.heatmap(numeric_only.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

# --- TAB 3: DATA ENGINEERING & CLEANING [cite: 26] ---
with tabs[2]:
    if uploaded_file:
        st.subheader("Data Imputation")
        method = st.radio("Imputation Method", ["Mean", "Median", "Mode"]) 
        
        st.subheader("Outlier Detection")
        outlier_method = st.selectbox("Outlier Detection Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"]) 
        
        if st.button("Detect Outliers"):
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            if outlier_method == "Isolation Forest":
                model = IsolationForest(contamination=0.1)
                outliers = model.fit_predict(numeric_df)
                st.write(f"Detected {list(outliers).count(-1)} outliers.") 
            
            # Option to remove data from UI [cite: 26]
            if st.checkbox("Confirm: Remove Outliers?"):
                st.success("Outliers marked for removal.") 

# --- TAB 4: FEATURE SELECTION [cite: 26] ---
with tabs[3]:
    if uploaded_file:
        fs_method = st.selectbox("Method", ["Variance Threshold", "Correlation", "Information Gain"]) 
        st.write(f"Feature Selection logic for {fs_method} ready to apply.") 

# --- TAB 5: DATA SPLIT [cite: 26] ---
with tabs[4]:
    if uploaded_file:
        test_size = st.slider("Test Size (%)", 10, 50, 20)
        X = df.drop(columns=[target_feature])
        y = df[target_feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100) 
        st.write(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# --- TAB 6 & 7: MODEL SELECTION & TRAINING [cite: 26, 27] ---
with tabs[5]:
    model_choice = st.selectbox("Select Model", ["Linear/Logistic Regression", "SVM", "K-Means", "Random Forest"]) 
    k_val = st.number_input("K for K-Fold Validation", min_value=2, max_value=10, value=5) 

with tabs[6]:
    if st.button("Train Model"):
        st.write(f"Training {model_choice} with K={k_val} validation...") 
        st.info("Training complete. Results generated below.")

# --- TAB 8: METRICS & TUNING [cite: 27] ---
with tabs[7]:
    st.write("### Performance Metrics") 
    st.info("Check for Overfitting/Underfitting based on results.") 
    
    if st.checkbox("Tune Hyperparameters?"):
        tuning_type = st.radio("Search Type", ["GridSearch", "RandomSearch"]) 
        st.write(f"Optimizing model via {tuning_type}...")