import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="CrimeAI Neural Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# -------------------------------------------------------
# 🎨 PREMIUM FUTURISTIC CSS THEME
# -------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap" rel="stylesheet">
 
<style>
 
/* ================================================
   🌌  GLOBAL RESET & BACKGROUND
   ================================================ */
html, body, [data-testid="stAppViewContainer"] {
    background: #000814 !important;
    color: #cce9f7 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
 
/* Animated grid overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,200,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    pointer-events: none;
    z-index: 0;
}
 
/* Scanline effect */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 200, 0.008) 2px,
        rgba(0, 255, 200, 0.008) 4px
    );
    pointer-events: none;
    z-index: 1;
}
 
/* ================================================
   📦  SIDEBAR
   ================================================ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000d1a 0%, #00040f 100%) !important;
    border-right: 1px solid rgba(0, 255, 200, 0.12) !important;
    position: relative;
}
 
[data-testid="stSidebar"]::after {
    content: '';
    position: absolute;
    right: 0;
    top: 15%;
    height: 70%;
    width: 1px;
    background: linear-gradient(180deg, transparent, #00ffe5 50%, transparent);
    pointer-events: none;
}
 
[data-testid="stSidebar"] * {
    color: #a0cce8 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.5px;
}
 
[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: rgba(160, 204, 232, 0.7) !important;
    transition: color 0.2s;
}
 
[data-testid="stSidebar"] .stRadio label:hover {
    color: #00ffe5 !important;
}
 
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: #00ffe5 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
}
 
/* ================================================
   🔠  TITLES & HEADINGS
   ================================================ */
h1, .stTitle {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    font-size: 28px !important;
    letter-spacing: 5px !important;
    text-transform: uppercase !important;
    background: linear-gradient(90deg, #00ffe5, #38bdf8, #818cf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: titleFlicker 4s infinite alternate;
}
 
h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: #00ffe5 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(0,255,200,0.15) !important;
    padding-bottom: 6px !important;
    margin-bottom: 12px !important;
}
 
@keyframes titleFlicker {
    0%   { filter: brightness(1); }
    90%  { filter: brightness(1); }
    91%  { filter: brightness(0.6); }
    92%  { filter: brightness(1); }
    94%  { filter: brightness(0.8); }
    95%  { filter: brightness(1); }
    100% { filter: brightness(1); }
}
 
/* ================================================
   💎  METRIC CARDS
   ================================================ */
[data-testid="metric-container"] {
    background: rgba(0, 20, 40, 0.9) !important;
    border: 1px solid rgba(0, 255, 200, 0.15) !important;
    border-radius: 4px !important;
    padding: 18px 20px !important;
    position: relative;
    overflow: hidden;
    box-shadow: inset 0 0 30px rgba(0, 255, 200, 0.03), 0 0 20px rgba(0, 0, 0, 0.5) !important;
}
 
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffe5, transparent);
}
 
[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,200,0.3), transparent);
}
 
[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: rgba(0, 255, 200, 0.5) !important;
}
 
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #00ffe5 !important;
    text-shadow: 0 0 20px rgba(0, 255, 200, 0.4) !important;
}
 
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    color: #22c55e !important;
}
 
/* ================================================
   🧊  DATAFRAME / TABLE
   ================================================ */
[data-testid="stDataFrame"],
[data-testid="stDataFrameGlideDataEditor"] {
    border: 1px solid rgba(0, 255, 200, 0.12) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
    background: rgba(0, 15, 30, 0.9) !important;
}
 
/* ================================================
   🔘  BUTTONS
   ================================================ */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ffe5 !important;
    color: #00ffe5 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    border-radius: 3px !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
 
.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, #00ffe5, #38bdf8);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.25s ease;
    z-index: -1;
}
 
.stButton > button:hover {
    color: #000814 !important;
    box-shadow: 0 0 25px rgba(0, 255, 200, 0.4) !important;
    transform: translateY(-1px) !important;
}
 
.stButton > button:hover::before {
    transform: scaleX(1);
}
 
/* ================================================
   📋  SELECT BOX & INPUTS
   ================================================ */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(0, 20, 40, 0.8) !important;
    border: 1px solid rgba(0, 255, 200, 0.2) !important;
    border-radius: 3px !important;
    color: #00ffe5 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
}
 
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: #00ffe5 !important;
    box-shadow: 0 0 10px rgba(0, 255, 200, 0.15) !important;
}
 
.stSelectbox label,
.stNumberInput label,
.stTextInput label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: rgba(0, 255, 200, 0.5) !important;
}
 
/* ================================================
   📁  FILE UPLOADER
   ================================================ */
[data-testid="stFileUploader"] {
    background: rgba(0, 255, 200, 0.02) !important;
    border: 1px dashed rgba(0, 255, 200, 0.25) !important;
    border-radius: 4px !important;
    padding: 8px !important;
}
 
[data-testid="stFileUploader"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    color: rgba(0, 255, 200, 0.5) !important;
    text-transform: uppercase !important;
}
 
/* ================================================
   ✅  ALERTS / INFO BOXES
   ================================================ */
[data-testid="stSuccess"] {
    background: rgba(0, 40, 20, 0.7) !important;
    border: 1px solid rgba(0, 255, 100, 0.3) !important;
    border-radius: 3px !important;
    color: rgba(0, 255, 150, 0.9) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.5px;
}
 
[data-testid="stWarning"] {
    background: rgba(40, 20, 0, 0.7) !important;
    border: 1px solid rgba(255, 180, 0, 0.3) !important;
    border-radius: 3px !important;
    color: rgba(255, 200, 50, 0.9) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
}
 
[data-testid="stError"] {
    background: rgba(40, 0, 10, 0.7) !important;
    border: 1px solid rgba(255, 50, 80, 0.3) !important;
    border-radius: 3px !important;
    color: rgba(255, 80, 100, 0.9) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
}
 
[data-testid="stInfo"] {
    background: rgba(0, 10, 40, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 3px !important;
    color: rgba(100, 200, 255, 0.9) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
}
 
/* ================================================
   📊  MATPLOTLIB CHART CONTAINER
   ================================================ */
[data-testid="stImage"],
.stPlotlyChart {
    border: 1px solid rgba(0, 255, 200, 0.12) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
    background: rgba(0, 10, 22, 0.9) !important;
}
 
/* ================================================
   ✨  SCROLLBAR
   ================================================ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #000814; }
::-webkit-scrollbar-thumb { background: #00ffe5; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #38bdf8; }
 
/* ================================================
   🃏  GLASS CARD HELPER CLASS (use in st.markdown)
   ================================================ */
.nx-card {
    background: rgba(0, 20, 40, 0.85);
    border: 1px solid rgba(0, 255, 200, 0.12);
    border-top: 1px solid rgba(0, 255, 200, 0.35);
    border-radius: 4px;
    padding: 20px 24px;
    margin: 8px 0;
    position: relative;
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    line-height: 1.7;
    color: #a0cce8;
}
 
.nx-card-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00ffe5;
    margin-bottom: 10px;
}
 
/* Progress / loading bar */
.nx-progress {
    height: 2px;
    background: rgba(0,255,200,0.1);
    border-radius: 2px;
    overflow: hidden;
    margin: 8px 0;
}
.nx-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00ffe5, #38bdf8);
    animation: progressPulse 2s ease-in-out infinite;
}
@keyframes progressPulse {
    0%   { width: 0%; }
    50%  { width: 100%; }
    100% { width: 0%; }
}
 
</style>
""", unsafe_allow_html=True)
 
 
# -------------------------------------------------------
# 🎨  MATPLOTLIB DARK THEME (matches the UI)
# -------------------------------------------------------
def apply_nx_plot_style():
    """Call once before any plt.figure() to get matching chart aesthetics."""
    mpl.rcParams.update({
        'figure.facecolor':  '#000d1a',
        'axes.facecolor':    '#000d1a',
        'axes.edgecolor':    'rgba(0,255,200,0.2)',
        'axes.labelcolor':   '#00ffe5',
        'axes.titlecolor':   '#00ffe5',
        'axes.titlesize':    11,
        'axes.labelsize':    10,
        'xtick.color':       'rgba(0,255,200,0.4)',
        'ytick.color':       'rgba(0,255,200,0.4)',
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'grid.color':        'rgba(0,255,200,0.07)',
        'grid.linestyle':    '--',
        'grid.linewidth':    0.5,
        'text.color':        '#a0cce8',
        'font.family':       'monospace',
        'legend.facecolor':  '#000d1a',
        'legend.edgecolor':  'rgba(0,255,200,0.2)',
    })
 
 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="CrimeAI Dashboard", layout="wide")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("⬡ CrimeAI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Explorer",
        "📊 EDA",
        "📉 Visualization",
        "🔥 Heatmap",
        "🤖 Train Model",
        "📊 Model Comparison",
        "📈 Evaluation",
        "🧠 AI Insights",
        "🎯 Prediction"
    ]
)

file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.session_state.df = df

# ------------------------------
# HOME
# ------------------------------
if page == "🏠 Home":
    st.title("🚀 CrimeAI Dashboard")
    st.info("Upload dataset to start")

# ------------------------------
# DATA EXPLORER
# ------------------------------
elif page == "📊 Data Explorer":
    if "df" in st.session_state:
        df = st.session_state.df
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
    else:
        st.warning("Upload dataset first")

# ------------------------------
# EDA
# ------------------------------
elif page == "📊 EDA":
    if "df" in st.session_state:
        df = st.session_state.df

        st.subheader("Summary")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        col = st.selectbox("Column", df.columns)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Upload dataset first")

# ------------------------------
# VISUALIZATION
# ------------------------------
elif page == "📉 Visualization":
    if "df" in st.session_state:
        df = st.session_state.df

        cols = st.multiselect("Select Columns", df.columns)
        chart = st.selectbox("Chart", ["Scatter", "Line", "Bar"])

        if len(cols) >= 2:
            fig, ax = plt.subplots()

            if chart == "Scatter":
                ax.scatter(df[cols[0]], df[cols[1]])
            elif chart == "Line":
                ax.plot(df[cols[0]], df[cols[1]])
            else:
                ax.bar(df[cols[0]], df[cols[1]])

            st.pyplot(fig)

    else:
        st.warning("Upload dataset first")

# ------------------------------
# HEATMAP
# ------------------------------
elif page == "🔥 Heatmap":
    if "df" in st.session_state:
        df = st.session_state.df

        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Upload dataset first")

# ------------------------------
# TRAIN MODEL
# ------------------------------
elif page == "🤖 Train Model":
    if "df" in st.session_state:
        df = st.session_state.df

        target = st.selectbox("Select Target", df.columns)
        st.session_state.target = target

        X = df.drop(columns=[target])
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.session_state.scaler = scaler
        st.session_state.X_columns = X.columns

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        if st.button("Train Model"):
            model = RandomForestRegressor(max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.session_state.model = model
            st.session_state.data = (X_test, y_test, y_pred)

            st.success("Model trained successfully")

    else:
        st.warning("Upload dataset first")

# ------------------------------
# MODEL COMPARISON
# ------------------------------
elif page == "📊 Model Comparison":
    if "df" in st.session_state and "target" in st.session_state:
        df = st.session_state.df
        target = st.session_state.target

        X = df.drop(columns=[target])
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        if st.button("Compare Models"):
            models = {
                "Random Forest": RandomForestRegressor(max_depth=5),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "SVM": SVR()
            }

            results = []

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append([name, r2, rmse])

            result_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE"])

            st.dataframe(result_df)

            best = result_df.sort_values(by="R2", ascending=False).iloc[0]
            st.success(f"Best Model: {best['Model']}")

    else:
        st.warning("Train model first")

# ------------------------------
# EVALUATION
# ------------------------------
elif page == "📈 Evaluation":
    if "data" in st.session_state:
        X_test, y_test, y_pred = st.session_state.data

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.metric("R2", round(r2, 3))
        st.metric("RMSE", round(rmse, 3))

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())

        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')
        st.pyplot(fig)

    else:
        st.warning("Train model first")

# ------------------------------
# AI INSIGHTS
# ------------------------------
elif page == "🧠 AI Insights":
    if "model" in st.session_state and "target" in st.session_state:
        model = st.session_state.model
        df = st.session_state.df
        target = st.session_state.target

        X = df.drop(columns=[target])

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(feat_df)

            top = feat_df.iloc[0]["Feature"]
            st.success(f"Top factor affecting crime: {top}")

    else:
        st.warning("Train model first")

# ------------------------------
# PREDICTION
# ------------------------------
elif page == "🎯 Prediction":
    if "model" in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.scaler
        cols = st.session_state.X_columns

        input_data = []

        for col in cols:
            val = st.number_input(col, value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            scaled = scaler.transform([input_data])
            pred = model.predict(scaled)

            st.success(f"Predicted Crime Rate: {round(pred[0],2)}")

    else:
        st.warning("Train model first")
