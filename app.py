# =====================================================
# AGRICULTURAL EARLY WARNING SYSTEM
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
# ENCODERS
# =====================================================

region_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
region_encoder.fit(sorted(df['Region'].unique()))
crop_encoder.fit(sorted(df['crop type'].unique()))

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_risk_label(code):
    return {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}.get(int(code), "Unknown")

def get_risk_color(code):
    return {0: "#28a745", 1: "#fd7e14", 2: "#dc3545"}.get(int(code), "#6c757d")

def predict_risk(region, crop_type, year):
    """Given Region, Crop Type and Year, fill remaining features from dataset medians
    for that combination and return predictions from both models."""

    # Filter dataset for selected combination
    mask = (
        (df["Region"] == region) &
        (df["crop type"] == crop_type) &
        (df["Year"] == year)
    )
    base = df[mask] if df[mask].shape[0] > 0 else df[
        (df["Region"] == region) & (df["crop type"] == crop_type)
    ]
    if base.empty:
        base = df  # fallback to full dataset medians

    region_code = region_encoder.transform([region])[0]
    crop_code = crop_encoder.transform([crop_type])[0]

    input_data = pd.DataFrame([[
        region_code,
        crop_code,
        year,
        float(base['Area cultivated(Ha)'].median()),
        float(base['Production(kg)'].median()),
        float(base['Yield_Growth_Rate'].median()),
        float(base['Production_Growth_Rate'].median()),
        float(base['Area_Efficiency'].median()),
        float(base['Regional_Avg_Yield'].median()),
        float(base['Crop_Avg_Yield'].median()),
        float(base['Yield_Anomaly'].median()),
        float(base['Rolling_Yield_Trend'].median()),
        float(base['Yield_Stability'].median()),
        float(base['Production_Area_Ratio'].median()),
        float(base['Early_Warning_Score'].median()),
    ]], columns=[
        'Region_Code', 'Crop_Code', 'Year',
        'Area cultivated(Ha)', 'Production(kg)',
        'Yield_Growth_Rate', 'Production_Growth_Rate', 'Area_Efficiency',
        'Regional_Avg_Yield', 'Crop_Avg_Yield', 'Yield_Anomaly',
        'Rolling_Yield_Trend', 'Yield_Stability', 'Production_Area_Ratio',
        'Early_Warning_Score'
    ])

    xgb_pred = xgb_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    return xgb_pred, rf_pred, input_data

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌾 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    (
        "🏠 Home",
        "🤖 Prediction",
        "📂 Dataset",
        "🔍 Explainable AI",
        "📊 Risk Analysis",
        "🗺️ Ethiopia Risk Map",
        "ℹ️ About"
    )
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Agricultural Early Warning System**

Models:
- 🚀 XGBoost
- 🌳 Random Forest
- 🔍 SHAP Explainability

Country: 🇪🇹 Ethiopia
""")

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🌾 Agricultural Early Warning System")
    st.subheader("Explainable AI-Based Food Security Risk Mapping for Ethiopia")
    st.markdown("---")

    st.markdown("""
Welcome to the **Agricultural Early Warning System**.

This dashboard predicts food security risk for Ethiopian regions using machine learning.

👈 **Use the sidebar to navigate between pages.**

### How to use:
1. Go to **🤖 Prediction** page
2. Select **Region**, **Crop Type**, and **Year**
3. Click **Predict Risk**
4. View results compared between XGBoost and Random Forest
5. Explore other pages for deeper analysis
""")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Total Records", len(df))
    col2.metric("🌍 Regions", df["Region"].nunique())
    col3.metric("🌾 Crop Types", df["crop type"].nunique())

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🤖 Prediction":

    st.title("🤖 Early Warning Prediction")
    st.write("Select Region, Crop Type, and Year to predict food security risk.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_region = st.selectbox(
            "🌍 Region",
            options=sorted(df["Region"].unique())
        )
    with col2:
        selected_crop = st.selectbox(
            "🌾 Crop Type",
            options=sorted(df["crop type"].unique())
        )
    with col3:
        selected_year = st.selectbox(
            "📅 Year",
            options=sorted(df["Year"].unique(), reverse=True)
        )

    st.markdown("---")

    if st.button("🔍 Predict Risk", use_container_width=True):

        xgb_pred, rf_pred, input_data = predict_risk(
            selected_region, selected_crop, selected_year
        )

        # Store in session state so other pages can use it
        st.session_state['region'] = selected_region
        st.session_state['crop'] = selected_crop
        st.session_state['year'] = selected_year
        st.session_state['xgb_pred'] = xgb_pred
        st.session_state['rf_pred'] = rf_pred
        st.session_state['input_data'] = input_data
        st.session_state['predicted'] = True

        st.subheader("🎯 Prediction Results")
        st.markdown(f"**Region:** {selected_region} | **Crop:** {selected_crop} | **Year:** {selected_year}")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            xgb_label = get_risk_label(xgb_pred)
            xgb_color = get_risk_color(xgb_pred)
            st.markdown(
                f"""<div style='background-color:{xgb_color}; padding:30px;
                border-radius:12px; text-align:center;'>
                <h2 style='color:white; margin:0;'>🚀 XGBoost</h2>
                <h1 style='color:white; margin:10px 0 0 0;'>{xgb_label}</h1>
                </div>""",
                unsafe_allow_html=True
            )

        with col2:
            rf_label = get_risk_label(rf_pred)
            rf_color = get_risk_color(rf_pred)
            st.markdown(
                f"""<div style='background-color:{rf_color}; padding:30px;
                border-radius:12px; text-align:center;'>
                <h2 style='color:white; margin:0;'>🌳 Random Forest</h2>
                <h1 style='color:white; margin:10px 0 0 0;'>{rf_label}</h1>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Agreement check
        if xgb_pred == rf_pred:
            st.success(f"✅ Both models agree: **{get_risk_label(xgb_pred)}** for {selected_region} - {selected_crop} in {selected_year}.")
        else:
            st.warning(f"⚠️ Models disagree. XGBoost says **{get_risk_label(xgb_pred)}**, Random Forest says **{get_risk_label(rf_pred)}**.")

    elif 'predicted' not in st.session_state:
        st.info("👆 Select Region, Crop Type, and Year above then click **Predict Risk**.")

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "📂 Dataset":

    st.title("📂 Agricultural Dataset")

    # Show filtered data if prediction was made
    if 'predicted' in st.session_state:
        region = st.session_state['region']
        crop = st.session_state['crop']
        year = st.session_state['year']

        st.markdown(f"Filtered for: **{region}** | **{crop}** | **{year}**")

        filtered = df[
            (df["Region"] == region) &
            (df["crop type"] == crop) &
            (df["Year"] == year)
        ]

        st.subheader("Filtered Data")
        if filtered.empty:
            st.warning("No exact records found for this combination.")
        else:
            st.dataframe(filtered, use_container_width=True)
            st.markdown(f"**{len(filtered)} records** found.")
        st.markdown("---")
    else:
        st.info("💡 Go to **🤖 Prediction** page first and make a prediction to see filtered data here.")

    st.subheader("Full Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Statistics")
    st.write(df.describe().T)

# =====================================================
# EXPLAINABLE AI PAGE
# =====================================================

elif page == "🔍 Explainable AI":

    st.title("🔍 Explainable AI (SHAP)")
    st.markdown("---")

    if 'predicted' in st.session_state:
        region = st.session_state['region']
        crop = st.session_state['crop']
        year = st.session_state['year']
        input_data = st.session_state['input_data']
        xgb_pred = st.session_state['xgb_pred']

        st.markdown(f"Showing SHAP for: **{region}** | **{crop}** | **{year}**")
        st.markdown(f"XGBoost Predicted: **{get_risk_label(xgb_pred)}**")
        st.markdown("---")
    else:
        st.info("💡 Go to **🤖 Prediction** page first and make a prediction to see SHAP explanations.")
        st.stop()

    # Global Feature Importance
    st.subheader("🌍 Global Feature Importance")
    sample_X = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Individual explanation for the predicted input
    st.subheader("🔎 Explanation for Your Prediction")
    shap_single = explainer.shap_values(input_data)

    try:
        # shap_single can be:
        # 1. list of arrays (one per class) — multiclass
        # 2. 2D array — binary or single output
        # 3. 3D array — (samples, features, classes)

        if isinstance(shap_single, list):
            # list of arrays: one per class
            idx = min(int(xgb_pred), len(shap_single) - 1)
            sv = shap_single[idx][0]
            ev = explainer.expected_value[idx] if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value

        elif isinstance(shap_single, np.ndarray) and shap_single.ndim == 3:
            # shape: (samples, features, classes)
            idx = min(int(xgb_pred), shap_single.shape[2] - 1)
            sv = shap_single[0, :, idx]
            ev = explainer.expected_value[idx] if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value

        else:
            # 2D array: shape (samples, features)
            sv = shap_single[0]
            ev = explainer.expected_value[0] if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=float(ev),
                data=input_data.iloc[0].values,
                feature_names=input_data.columns.tolist()
            ),
            show=False
        )
        st.pyplot(fig2)
        plt.close(fig2)

    except Exception as e:
        # Fallback: bar chart of feature importance for this prediction
        st.warning("Waterfall plot could not be rendered. Showing bar chart instead.")
        if isinstance(shap_single, list):
            sv = shap_single[0][0]
        elif isinstance(shap_single, np.ndarray) and shap_single.ndim == 3:
            sv = shap_single[0, :, 0]
        else:
            sv = shap_single[0]

        shap_df = pd.DataFrame({
            "Feature": input_data.columns.tolist(),
            "SHAP Value": sv
        }).sort_values("SHAP Value", key=abs, ascending=False)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        colors = ["#dc3545" if v > 0 else "#28a745" for v in shap_df["SHAP Value"]]
        ax3.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
        ax3.set_xlabel("SHAP Value")
        ax3.set_title("Feature Contributions to Prediction")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

# =====================================================
# RISK ANALYSIS PAGE
# =====================================================

elif page == "📊 Risk Analysis":

    st.title("📊 Food Security Risk Analysis")
    st.markdown("---")

    if 'predicted' in st.session_state:
        region = st.session_state['region']
        crop = st.session_state['crop']
        year = st.session_state['year']
        xgb_pred = st.session_state['xgb_pred']
        rf_pred = st.session_state['rf_pred']

        st.markdown(f"Analysis for: **{region}** | **{crop}** | **{year}**")
        st.markdown("---")

        # Model comparison
        st.subheader("🆚 Model Comparison")
        comp_df = pd.DataFrame({
            "Model": ["🚀 XGBoost", "🌳 Random Forest"],
            "Predicted Risk": [get_risk_label(xgb_pred), get_risk_label(rf_pred)],
            "Risk Level": [int(xgb_pred), int(rf_pred)]
        })
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")

        # Risk trend for selected region and crop over all years
        st.subheader(f"📈 Risk Trend: {region} - {crop} Over Years")
        trend_df = df[
            (df["Region"] == region) &
            (df["crop type"] == crop)
        ].copy()

        if not trend_df.empty:
            model_features = [
                'Region_Code', 'Crop_Code', 'Year',
                'Area cultivated(Ha)', 'Production(kg)',
                'Yield_Growth_Rate', 'Production_Growth_Rate', 'Area_Efficiency',
                'Regional_Avg_Yield', 'Crop_Avg_Yield', 'Yield_Anomaly',
                'Rolling_Yield_Trend', 'Yield_Stability', 'Production_Area_Ratio',
                'Early_Warning_Score'
            ]
            trend_df[model_features] = trend_df[model_features].replace([np.inf, -np.inf], np.nan)
            for col in model_features:
                trend_df[col] = trend_df[col].fillna(trend_df[col].median())

            trend_df['XGBoost_Risk'] = xgb_model.predict(trend_df[model_features])
            trend_df['RF_Risk'] = rf_model.predict(trend_df[model_features])

            yearly = trend_df.groupby('Year')[['XGBoost_Risk', 'RF_Risk']].mean().reset_index()

            fig = px.line(
                yearly, x='Year',
                y=['XGBoost_Risk', 'RF_Risk'],
                title=f"Risk Trend for {region} - {crop}",
                markers=True,
                labels={"value": "Risk Level (0=Low, 1=Med, 2=High)", "variable": "Model"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for this Region and Crop combination.")

    else:
        st.info("💡 Go to **🤖 Prediction** page first and make a prediction.")

    st.markdown("---")
    st.subheader("🌍 Regional Risk Summary")
    st.dataframe(risk_df, use_container_width=True)

    fig2 = px.bar(
        risk_df,
        x='Region',
        y='Predicted_Risk',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk': '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk': '#dc3545'
        },
        title='Average Food Security Risk by Region'
    )
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# ETHIOPIA RISK MAP PAGE
# =====================================================

elif page == "🗺️ Ethiopia Risk Map":

    st.title("🗺️ Ethiopia Food Security Risk Map")
    st.markdown("---")

    if 'predicted' in st.session_state:
        st.markdown(f"Selected: **{st.session_state['region']}** | **{st.session_state['crop']}** | **{st.session_state['year']}**")

    fig = px.choropleth(
        risk_df,
        geojson=ethiopia_geojson,
        featureidkey="properties.ADM1_EN",
        locations='Region',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk': '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk': '#dc3545'
        },
        title="Food Security Risk Map of Ethiopia by Region"
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Application")
    st.markdown("""
### Agricultural Early Warning System

This application predicts food security risk across Ethiopian regions using machine learning.

**How it works:**
1. User selects **Region**, **Crop Type**, and **Year**
2. The system fills in remaining features from historical data medians
3. Both **XGBoost** and **Random Forest** models predict the risk level
4. Results are shown as **Low**, **Medium**, or **High** risk
5. SHAP explains which features drove the prediction

**Models:**
- 🚀 XGBoost
- 🌳 Random Forest

**Risk Levels:**
- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

**Version:** 1.0
**Country:** 🇪🇹 Ethiopia
""")
