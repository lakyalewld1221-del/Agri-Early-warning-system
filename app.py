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
    return joblib.load("random_forest_model (1).pkl")

xgb_model = load_xgboost()
rf_model = load_random_forest()

# =====================================================
# SIDEBAR - Navigation + Global Filters
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

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Global Filters")

# --- Three main filters used across all pages ---
selected_region = st.sidebar.selectbox(
    "🌍 Region",
    options=sorted(df["Region"].unique()),
    index=0
)

selected_crop = st.sidebar.selectbox(
    "🌾 Crop Type",
    options=sorted(df["crop type"].unique()),
    index=0
)

selected_year = st.sidebar.selectbox(
    "📅 Year",
    options=sorted(df["Year"].unique(), reverse=True),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
"""
Agricultural Early Warning System

Predicts food security risk using:
- 🌳 Random Forest
- 🚀 XGBoost
- 🔍 SHAP Explainability

Country: Ethiopia
"""
)

# =====================================================
# FILTER DATA BASED ON SIDEBAR SELECTIONS
# =====================================================

filtered_df = df[
    (df["Region"] == selected_region) &
    (df["crop type"] == selected_crop) &
    (df["Year"] == selected_year)
]

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🌾 Agricultural Early Warning System")
    st.subheader("Explainable AI-Based Food Security Risk Mapping for Ethiopia")
    st.markdown("---")

    st.write("""
Welcome to the Agricultural Early Warning System.

This dashboard helps predict food security risk using machine learning
and provides explainable predictions through SHAP.
    """)

    st.markdown("### 🔎 Currently Selected Filters")
    c1, c2, c3 = st.columns(3)
    c1.info(f"🌍 Region: **{selected_region}**")
    c2.info(f"🌾 Crop: **{selected_crop}**")
    c3.info(f"📅 Year: **{selected_year}**")

    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Filtered Records", len(filtered_df))
    col3.metric("Total Regions", df["Region"].nunique())
    col4.metric("Crop Types", df["crop type"].nunique())

    st.markdown("---")
    st.markdown("### Project Objectives")
    st.markdown("""
- Predict agricultural food security risk
- Compare XGBoost and Random Forest models
- Explain predictions using SHAP
- Visualize regional food security risk across Ethiopia
""")

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "📂 Dataset":

    st.title("📂 Agricultural Dataset")
    st.markdown(f"Showing data for **{selected_region}** | **{selected_crop}** | **{selected_year}**")
    st.markdown("---")

    st.subheader("Filtered Dataset Preview")
    if filtered_df.empty:
        st.warning("No data found for the selected Region, Crop Type, and Year combination.")
    else:
        st.dataframe(filtered_df, use_container_width=True)
        st.markdown(f"**{len(filtered_df)} records** found.")

    st.markdown("---")
    st.subheader("Full Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

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
    st.markdown(f"Predicting risk for **{selected_region}** | **{selected_crop}** | **{selected_year}**")
    st.markdown("---")

    # Encoders
    region_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    region_encoder.fit(df['Region'].unique())
    crop_encoder.fit(df['crop type'].unique())

    # Encode the sidebar selections
    region_code = region_encoder.transform([selected_region])[0]
    crop_code = crop_encoder.transform([selected_crop])[0]

    # Use median values from filtered data if available, else full dataset
    base_df = filtered_df if not filtered_df.empty else df

    st.subheader("📋 Auto-filled Features")
    st.info("The features below are auto-filled from your selected Region, Crop Type and Year. You can adjust them if needed.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            area_cultivated = st.number_input("Area cultivated (Ha)", min_value=0.0, value=float(base_df['Area cultivated(Ha)'].median()))
            production_kg = st.number_input("Production (kg)", min_value=0.0, value=float(base_df['Production(kg)'].median()))
            yield_growth_rate = st.number_input("Yield Growth Rate", value=float(base_df['Yield_Growth_Rate'].median()))
            production_growth_rate = st.number_input("Production Growth Rate", value=float(base_df['Production_Growth_Rate'].median()))
            area_efficiency = st.number_input("Area Efficiency", value=float(base_df['Area_Efficiency'].median()))
            regional_avg_yield = st.number_input("Regional Average Yield", value=float(base_df['Regional_Avg_Yield'].median()))
            crop_avg_yield = st.number_input("Crop Average Yield", value=float(base_df['Crop_Avg_Yield'].median()))

        with col2:
            yield_anomaly = st.number_input("Yield Anomaly", value=float(base_df['Yield_Anomaly'].median()))
            rolling_yield_trend = st.number_input("Rolling Yield Trend", value=float(base_df['Rolling_Yield_Trend'].median()))
            yield_stability = st.number_input("Yield Stability", value=float(base_df['Yield_Stability'].median()))
            production_area_ratio = st.number_input("Production Area Ratio", value=float(base_df['Production_Area_Ratio'].median()))
            early_warning_score = st.number_input("Early Warning Score", value=float(base_df['Early_Warning_Score'].median()))

        submitted = st.form_submit_button("🔍 Predict Risk")

    if submitted:
        input_data = pd.DataFrame([[
            region_code, crop_code, selected_year,
            area_cultivated, production_kg,
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

        risk_names = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        colors = {0: "green", 1: "orange", 2: "red"}

        # XGBoost prediction
        xgb_pred = xgb_model.predict(input_data)[0]
        xgb_risk = risk_names.get(xgb_pred, "Unknown")
        xgb_color = colors.get(xgb_pred, "gray")

        # Random Forest prediction
        rf_pred = rf_model.predict(input_data)[0]
        rf_risk = risk_names.get(rf_pred, "Unknown")
        rf_color = colors.get(rf_pred, "gray")

        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<div style='background-color:{xgb_color}; padding:20px; border-radius:10px; text-align:center;'>"
                f"<h3 style='color:white;'>🚀 XGBoost</h3>"
                f"<h2 style='color:white;'>{xgb_risk}</h2>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div style='background-color:{rf_color}; padding:20px; border-radius:10px; text-align:center;'>"
                f"<h3 style='color:white;'>🌳 Random Forest</h3>"
                f"<h2 style='color:white;'>{rf_risk}</h2>"
                f"</div>",
                unsafe_allow_html=True
            )

# =====================================================
# EXPLAINABLE AI PAGE
# =====================================================

elif page == "🔍 Explainable AI":
    st.title("🔍 Explainable AI (SHAP)")
    st.markdown(f"Showing SHAP explanations for **{selected_region}** | **{selected_crop}** | **{selected_year}**")
    st.markdown("---")

    st.subheader("Global Feature Importance")

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

    instance_index = st.number_input(
        "Select an instance index (0 to {})".format(len(X_test) - 1),
        min_value=0, max_value=len(X_test) - 1, value=0, step=1
    )

    selected_instance = X_test.iloc[[instance_index]]
    selected_shap_values = explainer.shap_values(selected_instance)

    predicted_risk_level = xgb_model.predict(selected_instance)[0]
    risk_names = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    st.write(f"Predicted Risk: **{risk_names.get(predicted_risk_level, 'Unknown')}** (Level {predicted_risk_level})")

    if isinstance(explainer.expected_value, list):
        expected_value_for_plot = explainer.expected_value[predicted_risk_level]
        shap_values_for_plot = selected_shap_values[predicted_risk_level]
    else:
        expected_value_for_plot = explainer.expected_value
        shap_values_for_plot = selected_shap_values[0]

    fig2, ax2 = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_for_plot[0],
        base_values=expected_value_for_plot,
        data=selected_instance.iloc[0],
        feature_names=X_test.columns.tolist()
    ), show=False)
    st.pyplot(fig2)
    plt.close(fig2)

# =====================================================
# RISK ANALYSIS PAGE
# =====================================================

elif page == "📊 Risk Analysis":
    st.title("📊 Food Security Risk Analysis")
    st.markdown(f"Analysis for **{selected_region}** | **{selected_crop}** | **{selected_year}**")
    st.markdown("---")

    st.subheader("Regional Risk Summary")
    st.dataframe(risk_df, use_container_width=True)

    st.subheader("Average Risk Score by Region")
    fig = px.bar(
        risk_df,
        x='Region',
        y='Predicted_Risk',
        color='Risk_Category',
        color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'},
        title='Average Food Security Risk by Region'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Trend for Selected Region Over Years")
    region_trend = df[
        (df["Region"] == selected_region) &
        (df["crop type"] == selected_crop)
    ]

    if not region_trend.empty:
        model_features = [
            'Region_Code', 'Crop_Code', 'Year', 'Area cultivated(Ha)', 'Production(kg)',
            'Yield_Growth_Rate', 'Production_Growth_Rate', 'Area_Efficiency',
            'Regional_Avg_Yield', 'Crop_Avg_Yield', 'Yield_Anomaly',
            'Rolling_Yield_Trend', 'Yield_Stability', 'Production_Area_Ratio',
            'Early_Warning_Score'
        ]
        region_trend = region_trend.copy()
        region_trend[model_features] = region_trend[model_features].replace([np.inf, -np.inf], np.nan)
        for col in model_features:
            if region_trend[col].isnull().any():
                region_trend[col] = region_trend[col].fillna(region_trend[col].median())

        region_trend['Predicted_Risk'] = xgb_model.predict(region_trend[model_features])
        fig2 = px.line(
            region_trend,
            x='Year',
            y='Predicted_Risk',
            title=f"Risk Trend for {selected_region} - {selected_crop}",
            markers=True
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available for this Region and Crop Type combination.")

    st.subheader("High Risk Regions")
    high_risk_regions = risk_df[risk_df['Risk_Category'] == 'High Risk']
    if not high_risk_regions.empty:
        st.dataframe(high_risk_regions, use_container_width=True)
    else:
        st.info("No regions currently categorized as High Risk.")

# =====================================================
# ETHIOPIA RISK MAP PAGE
# =====================================================

elif page == "🗺️ Ethiopia Risk Map":
    st.title("🗺️ Ethiopia Food Security Risk Map")
    st.markdown(f"Showing risk map | Selected Region: **{selected_region}** | Year: **{selected_year}**")
    st.markdown("---")

    fig = px.choropleth_mapbox(
        risk_df,
        geojson=ethiopia_geojson,
        featureidkey="properties.ADM1_EN",
        locations='Region',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'orange',
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

This application predicts food security risk across Ethiopian regions using machine learning.

**Models Used:**
- 🌳 Random Forest
- 🚀 XGBoost

**Explainability:**
- 🔍 SHAP (SHapley Additive exPlanations)

**Key Inputs:**
- 🌍 Region
- 🌾 Crop Type
- 📅 Year

**Data:**
- Historical agricultural data including crop yield, production, and regional statistics for Ethiopia.

**Version:** 1.0
**Country:** Ethiopia
""")
