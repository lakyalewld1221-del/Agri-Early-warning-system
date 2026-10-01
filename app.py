# =====================================================
# AGRICULTURAL EARLY WARNING SYSTEM AND
# Explainable AI-Based Food Security Risk Mapping
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import io
from contextlib import redirect_stdout
import json
import seaborn as sns

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Agricultural Early Warning System",
    page_icon="🌾",
    layout="wide",
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("labeled_agri_risk_data.csv")

@st.cache_data
def load_risk_map():
    return pd.read_csv("ethiopia_risk_map.csv")

@st.cache_data
def load_X_test():
    return pd.read_csv("X_test.csv")

@st.cache_data
def load_geojson():
    with open('ethiopia_regions.geojson') as f:
        geojson_data = json.load(f)
    return geojson_data

df = load_dataset()
risk_df = load_risk_map()
X_test = load_X_test()
ethiopia_geojson = load_geojson()

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_xgboost():
    return joblib.load("xgboost_model.pkl")

@st.cache_resource
def load_random_forest():
    return joblib.load("random_forest_model.pkl")

@st.cache_resource
def load_mlp_model():
    return joblib.load("mlp_model.pkl")

xgb_model = load_xgboost()
rf_model = load_random_forest()
mlp_model = load_mlp_model()

st.title("🌾 Agricultural Early Warning System")

st.success("Application loaded successfully!")

st.write("Dataset shape:", df.shape)
st.write("Risk map shape:", risk_df.shape)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌾 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    (
        "🏠 Home",
        "📂 Dataset",
        "🤖 Early Warning Prediction",
        "🔍 Explainable AI",
        "📊 Risk Analysis",
        "🗺️ Ethiopia Risk Map",
        "ℹ️ About"
    )
)

st.sidebar.info(
"""
Agricultural Early Warning System

This application predicts food security risk using:

- 🌳 Random Forest
- 🚀 XGBoost
- 🔍 SHAP Explainability

Country: Ethiopia
"""
)

st.sidebar.markdown("---")

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🌾 Agricultural Early Warning System")

    st.subheader(
        "Explainable AI-Based Food Security Risk Mapping for Ethiopia"
    )

    st.markdown("---")

    st.write("""
Welcome to the Agricultural Early Warning System.

This dashboard helps predict food security risk using machine learning and provides explainable predictions through SHAP.
    """)

    st.markdown("### Project Objectives")

    st.markdown("""
- Predict agricultural food security risk
- Compare XGBoost and Random Forest models
- Explain predictions using SHAP
- Visualize regional food security risk across Ethiopia
""")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", len(df))
    col2.metric("Regions", df["Region"].nunique())
    col3.metric("Crop Types", df["crop type"].nunique())

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "📂 Dataset":

    st.title("📂 Agricultural Dataset")

    st.write("Explore the agricultural dataset used for training the machine learning models.")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Description")
    st.write(df.describe().T)

    st.subheader("Column Information")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        df.info()
    st.text(buffer.getvalue())

# =====================================================
# EARLY WARNING PREDICTION PAGE
# =====================================================

elif page == "🤖 Early Warning Prediction":
    st.title("🤖 Early Warning Prediction")
    st.write("Input agricultural parameters to get a food security risk prediction.")

    # Re-initialize encoders with the full dataset for prediction consistency
    region_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    region_encoder.fit(df['Region'].unique())
    crop_encoder.fit(df['crop type'].unique())

    # Input form
    with st.form("prediction_form"):
        st.header("Input Features")

        # Categorical Inputs
        region = st.selectbox("Region", df['Region'].unique())
        crop_type = st.selectbox("Crop Type", df['crop type'].unique())

        # Numerical Inputs
        year = st.number_input("Year", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=int(df['Year'].max()))
        area_cultivated = st.number_input("Area cultivated(Ha)", min_value=0.0, value=float(df['Area cultivated(Ha)'].median()))
        production_kg = st.number_input("Production(kg)", min_value=0.0, value=float(df['Production(kg)'].median()))

        yield_growth_rate = st.number_input("Yield Growth Rate", value=float(df['Yield_Growth_Rate'].median()))
        production_growth_rate = st.number_input("Production Growth Rate", value=float(df['Production_Growth_Rate'].median()))
        area_efficiency = st.number_input("Area Efficiency", value=float(df['Area_Efficiency'].median()))
        regional_avg_yield = st.number_input("Regional Average Yield", value=float(df['Regional_Avg_Yield'].median()))
        crop_avg_yield = st.number_input("Crop Average Yield", value=float(df['Crop_Avg_Yield'].median()))
        yield_anomaly = st.number_input("Yield Anomaly", value=float(df['Yield_Anomaly'].median()))
        rolling_yield_trend = st.number_input("Rolling Yield Trend", value=float(df['Rolling_Yield_Trend'].median()))
        yield_stability = st.number_input("Yield Stability", value=float(df['Yield_Stability'].median()))
        production_area_ratio = st.number_input("Production Area Ratio", value=float(df['Production_Area_Ratio'].median()))
        early_warning_score = st.number_input("Early Warning Score", value=float(df['Early_Warning_Score'].median()))

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        # Encode categorical inputs
        region_code = region_encoder.transform([region])[0]
        crop_code = crop_encoder.transform([crop_type])[0]

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[
            region_code, crop_code, year, area_cultivated, production_kg,
            yield_growth_rate, production_growth_rate, area_efficiency,
            regional_avg_yield, crop_avg_yield, yield_anomaly,
            rolling_yield_trend, yield_stability, production_area_ratio,
            early_warning_score
        ]], columns=[
            'Region_Code', 'Crop_Code', 'Year', 'Area cultivated(Ha)', 'Production(kg)',
            'Yield_Growth_Rate', 'Production_Growth_Rate', 'Area_Efficiency',
            'Regional_Avg_Yield', 'Crop_Avg_Yield', 'Yield_Anomaly',
            'Rolling_Yield_Trend', 'Yield_Stability', 'Production_Area_Ratio',
            'Early_Warning_Score'
        ])

        prediction = xgb_model.predict(input_data)[0]

        def map_risk_to_name(risk_level):
            risk_names = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            return risk_names.get(risk_level, "Unknown Risk")

        def assign_color(risk_level):
            if risk_level == 0:
                return "Green"
            elif risk_level == 1:
                return "Yellow"
            else:
                return "Red"

        risk_name = map_risk_to_name(prediction)
        alert_color = assign_color(prediction)

        st.subheader("Prediction Result:")
        st.markdown(
            f"<div style='background-color:{alert_color.lower()}; padding: 10px; border-radius: 5px;'>"
            f"<h4>Predicted Risk: {risk_name}</h4>"
            f"</div>",
            unsafe_allow_html=True
        )

# =====================================================
# EXPLAINABLE AI PAGE
# =====================================================

elif page == "🔍 Explainable AI":
    st.title("🔍 Explainable AI (SHAP)")
    st.write("Understand the factors driving the food security risk predictions.")

    st.subheader("Global Feature Importance")
    st.markdown("This plot shows the overall importance of each feature across the entire dataset.")

    if len(X_test) > 1000:
        sample_X_test = X_test.sample(1000, random_state=42)
    else:
        sample_X_test = X_test

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_X_test)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_X_test, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("Individual Prediction Explanation")
    st.markdown("Select an instance to see how each feature contributes to its specific risk prediction.")

    instance_index = st.number_input(
        "Select an instance index from the test set (0 to {})".format(len(X_test) - 1),
        min_value=0, max_value=len(X_test) - 1, value=0, step=1
    )

    if instance_index is not None:
        selected_instance = X_test.iloc[[instance_index]]
        selected_shap_values = explainer.shap_values(selected_instance)

        st.write(f"Showing explanation for instance {instance_index}:")

        predicted_risk_level = xgb_model.predict(selected_instance)[0]
        risk_names = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        predicted_risk_name = risk_names.get(predicted_risk_level, "Unknown Risk")
        st.write(f"Predicted Risk: **{predicted_risk_name}** (Level {predicted_risk_level})")

        if isinstance(explainer.expected_value, list):
            expected_value_for_plot = explainer.expected_value[predicted_risk_level]
            shap_values_for_plot = selected_shap_values[predicted_risk_level]
        else:
            expected_value_for_plot = explainer.expected_value
            shap_values_for_plot = selected_shap_values[0]

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.force_plot(expected_value_for_plot, shap_values_for_plot, selected_instance)
        st.pyplot(bbox_inches='tight')
        st.set_option('deprecation.showPyplotGlobalUse', True)

# =====================================================
# RISK ANALYSIS PAGE
# =====================================================

elif page == "📊 Risk Analysis":
    st.title("📊 Food Security Risk Analysis")
    st.write("Detailed analysis and visualization of food security risk.")

    st.subheader("Regional Risk Summary")
    st.write("Average predicted risk and categories by region.")
    st.dataframe(risk_df)

    st.subheader("Average Risk Score by Region")
    fig = px.bar(
        risk_df,
        x='Region',
        y='Predicted_Risk',
        color='Risk_Category',
        color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'yellow', 'High Risk': 'red'},
        title='Average Food Security Risk by Region'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Food Security Risk Trend Over Time")
    st.write("This heatmap shows the average predicted risk for each region across different years.")

    temp_df = df.copy()
    model_features = [
        'Region_Code', 'Crop_Code', 'Year', 'Area cultivated(Ha)', 'Production(kg)',
        'Yield_Growth_Rate', 'Production_Growth_Rate', 'Area_Efficiency',
        'Regional_Avg_Yield', 'Crop_Avg_Yield', 'Yield_Anomaly',
        'Rolling_Yield_Trend', 'Yield_Stability', 'Production_Area_Ratio',
        'Early_Warning_Score'
    ]

    temp_df[model_features] = temp_df[model_features].replace([np.inf, -np.inf], np.nan)
    for col in model_features:
        if temp_df[col].isnull().any():
            temp_df[col] = temp_df[col].fillna(temp_df[col].median())

    temp_df['Predicted_Risk'] = xgb_model.predict(temp_df[model_features])

    pivot_table = temp_df.pivot_table(
        values='Predicted_Risk',
        index='Region',
        columns='Year',
        aggfunc='mean'
    )

    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_table, annot=False, cmap='YlOrRd')
    plt.title("Regional Food Security Risk Trend")
    plt.xlabel("Year")
    plt.ylabel("Region")
    st.pyplot(plt)

    st.subheader("High Risk Regions")
    high_risk_regions = risk_df[risk_df['Risk_Category'] == 'High Risk']
    if not high_risk_regions.empty:
        st.dataframe(high_risk_regions)
    else:
        st.info("No regions currently categorized as 'High Risk' based on the defined thresholds.")

# =====================================================
# ETHIOPIA RISK MAP PAGE
# =====================================================

elif page == "🗺️ Ethiopia Risk Map":
    st.title("🗺️ Ethiopia Food Security Risk Map")
    st.write("Geographical visualization of food security risk across Ethiopian regions.")

    fig = px.choropleth_mapbox(
        risk_df,
        geojson=ethiopia_geojson,
        featureidkey="properties.ADM1_EN",
        locations='Region',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'yellow',
            'High Risk': 'red'
        },
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": 9.145, "lon": 40.4897},
        opacity=0.7,
        title="Food Security Risk Map of Ethiopia by Region"
    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")

    st.markdown("""
### Agricultural Early Warning System

This application was built to help predict food security risk across Ethiopian regions using machine learning.

**Models Used:**
- 🌳 Random Forest
- 🚀 XGBoost
- 🧠 MLP (Multi-Layer Perceptron)

**Explainability:**
- 🔍 SHAP (SHapley Additive exPlanations) is used to explain individual and global predictions.

**Data:**
- Historical agricultural data including crop yield, production, and regional statistics for Ethiopia.

**Version:** 1.0  
**Country:** Ethiopia  
""")
