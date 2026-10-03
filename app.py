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
# CROP IMAGES (using free Unsplash URLs)
# =====================================================

CROP_IMAGES = {
    "Maize":        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_corn_on_the_cob.jpg/320px-Simple_corn_on_the_cob.jpg",
    "Wheat":        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Wheat_field_in_Dorset%2C_England_-_July_2009.jpg/320px-Wheat_field_in_Dorset%2C_England_-_July_2009.jpg",
    "Teff":         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eragrostis_tef_-_teff_%28aka%29.jpg/320px-Eragrostis_tef_-_teff_%28aka%29.jpg",
    "Sorghum":      "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Sorghum_crop.jpg/320px-Sorghum_crop.jpg",
    "Barley":       "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Barley_field.jpg/320px-Barley_field.jpg",
    "Coffee":       "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/320px-A_small_cup_of_coffee.JPG",
    "Sesame":       "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Sesamum_indicum_2.jpg/320px-Sesamum_indicum_2.jpg",
    "Chickpea":     "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Chickpea_crop.jpg/320px-Chickpea_crop.jpg",
    "Millet":       "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Panicum_miliaceum_USDA.jpg/320px-Panicum_miliaceum_USDA.jpg",
    "Bean":         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Haricot_beans.jpg/320px-Haricot_beans.jpg",
    "Lentil":       "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Lens_culinaris.jpg/320px-Lens_culinaris.jpg",
    "Sunflower":    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/320px-Sunflower_sky_backdrop.jpg",
    "Potato":       "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Potato_and_cross_section.jpg/320px-Potato_and_cross_section.jpg",
    "Sweet Potato": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Ipomoea_batatas_006.jpg/320px-Ipomoea_batatas_006.jpg",
    "Rice":         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/White_rice.jpg/320px-White_rice.jpg",
    "Cassava":      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Manihot_esculenta_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-090.jpg/320px-Manihot_esculenta_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-090.jpg",
    "Groundnut":    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Peanut_cluster_2.jpg/320px-Peanut_cluster_2.jpg",
    "Pea":          "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Peas_in_pods_-_Studio.jpg/320px-Peas_in_pods_-_Studio.jpg",
    "Faba Bean":    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/forty/Vicia_faba_habito.jpg/320px-Vicia_faba_habito.jpg",
    "Oat":          "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Avena_sativa_-_Köhler–s_Medizinal-Pflanzen-016.jpg/320px-Avena_sativa_-_Köhler–s_Medizinal-Pflanzen-016.jpg",
    "Enset":        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Enset_2.jpg/320px-Enset_2.jpg",
    "Sugar Cane":   "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Sugarcane_field.jpg/320px-Sugarcane_field.jpg",
    "Cotton":       "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Cotton_plant.jpg/320px-Cotton_plant.jpg",
}

DEFAULT_CROP_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Farmland_at_sunset.jpg/320px-Farmland_at_sunset.jpg"

def get_crop_image(crop_name):
    # Try exact match first, then partial match
    if crop_name in CROP_IMAGES:
        return CROP_IMAGES[crop_name]
    for key in CROP_IMAGES:
        if key.lower() in crop_name.lower() or crop_name.lower() in key.lower():
            return CROP_IMAGES[key]
    return DEFAULT_CROP_IMAGE

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_risk_label(code):
    return {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}.get(int(code), "Unknown")

def get_risk_color(code):
    return {0: "#28a745", 1: "#fd7e14", 2: "#dc3545"}.get(int(code), "#6c757d")

def get_risk_reasons(region, crop, year, risk_code, input_data):
    """Generate human-readable reasons based on input features and risk level."""
    reasons = []
    rec = []

    yield_growth = float(input_data['Yield_Growth_Rate'].iloc[0])
    prod_growth  = float(input_data['Production_Growth_Rate'].iloc[0])
    yield_anom   = float(input_data['Yield_Anomaly'].iloc[0])
    area_eff     = float(input_data['Area_Efficiency'].iloc[0])
    ews          = float(input_data['Early_Warning_Score'].iloc[0])
    stability    = float(input_data['Yield_Stability'].iloc[0])

    risk_code = int(risk_code)

    # --- Reasons ---
    if yield_growth < 0:
        reasons.append(f"📉 Yield growth rate is **negative ({yield_growth:.2f})**, indicating declining crop yields.")
    elif yield_growth > 0.1:
        reasons.append(f"📈 Yield growth rate is **positive ({yield_growth:.2f})**, indicating improving crop yields.")

    if prod_growth < 0:
        reasons.append(f"📉 Production growth rate is **negative ({prod_growth:.2f})**, suggesting reduced food production.")

    if yield_anom < -0.1:
        reasons.append(f"⚠️ Yield anomaly is **below average ({yield_anom:.2f})**, meaning this year's yield is worse than historical norms.")
    elif yield_anom > 0.1:
        reasons.append(f"✅ Yield anomaly is **above average ({yield_anom:.2f})**, meaning this year's yield exceeds historical norms.")

    if area_eff < 0.5:
        reasons.append(f"🌾 Area efficiency is **low ({area_eff:.2f})**, indicating poor land utilization for {crop}.")

    if ews > 0.6:
        reasons.append(f"🚨 Early Warning Score is **high ({ews:.2f})**, signaling a strong food security alert for {region}.")
    elif ews < 0.3:
        reasons.append(f"✅ Early Warning Score is **low ({ews:.2f})**, suggesting stable food security conditions.")

    if stability < 0.4:
        reasons.append(f"📊 Yield stability is **low ({stability:.2f})**, meaning crop production is highly variable year to year.")

    if not reasons:
        reasons.append(f"ℹ️ The model assessed historical patterns for {crop} in {region} ({year}) and determined the risk level.")

    # --- Recommendations ---
    if risk_code == 2:  # High Risk
        rec = [
            "🆘 **Immediate action required** — alert local agricultural authorities.",
            "💧 Improve irrigation infrastructure to mitigate drought impact.",
            "🌱 Introduce drought-resistant or high-yield crop varieties.",
            "🏦 Activate emergency food reserves and support programs.",
            "📋 Conduct field assessments to verify crop conditions on the ground.",
            "🤝 Coordinate with NGOs and government agencies for humanitarian support.",
        ]
    elif risk_code == 1:  # Medium Risk
        rec = [
            "⚠️ **Monitor closely** — situation may worsen without intervention.",
            "💧 Optimize water usage and promote water conservation techniques.",
            "🌾 Diversify crop types to reduce dependency on a single crop.",
            "📊 Strengthen early warning monitoring for the coming season.",
            "🧑‍🌾 Provide farmers with training on improved agricultural practices.",
            "📦 Build buffer food stocks as a precautionary measure.",
        ]
    else:  # Low Risk
        rec = [
            "✅ **Conditions are stable** — maintain current agricultural practices.",
            "📈 Continue investing in yield improvement programs.",
            "🌱 Expand cultivation areas where land is available.",
            "📊 Keep monitoring seasonal trends to detect early changes.",
            "🧑‍🌾 Share best practices with neighboring regions.",
        ]

    return reasons, rec

def predict_risk(region, crop_type, year):
    mask = (
        (df["Region"] == region) &
        (df["crop type"] == crop_type) &
        (df["Year"] == year)
    )
    base = df[mask] if df[mask].shape[0] > 0 else df[
        (df["Region"] == region) & (df["crop type"] == crop_type)
    ]
    if base.empty:
        base = df

    region_code = region_encoder.transform([region])[0]
    crop_code   = crop_encoder.transform([crop_type])[0]

    input_data = pd.DataFrame([[
        region_code, crop_code, year,
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

    # Clean input: replace inf/-inf with NaN, then fill NaN with global median
    input_data = input_data.replace([np.inf, -np.inf], np.nan)
    for col in input_data.columns:
        if input_data[col].isnull().any():
            global_median = df[col].replace([np.inf, -np.inf], np.nan).median()
            input_data[col] = input_data[col].fillna(global_median if not np.isnan(global_median) else 0)

    xgb_pred = xgb_model.predict(input_data)[0]
    rf_pred  = rf_model.predict(input_data)[0]

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
        selected_region = st.selectbox("🌍 Region", options=sorted(df["Region"].unique()))
    with col2:
        selected_crop = st.selectbox("🌾 Crop Type", options=sorted(df["crop type"].unique()))
    with col3:
        selected_year = st.selectbox("📅 Year", options=sorted(df["Year"].unique(), reverse=True))

    # Show crop image immediately when crop is selected
    st.markdown("---")
    img_col, info_col = st.columns([1, 2])
    with img_col:
        crop_img_url = get_crop_image(selected_crop)
        st.image(crop_img_url, caption=f"🌾 {selected_crop}", use_container_width=True)
    with info_col:
        st.markdown(f"### Selected Input")
        st.markdown(f"- 🌍 **Region:** {selected_region}")
        st.markdown(f"- 🌾 **Crop Type:** {selected_crop}")
        st.markdown(f"- 📅 **Year:** {selected_year}")
        crop_data = df[df["crop type"] == selected_crop]
        if not crop_data.empty:
            st.markdown(f"- 📋 **Records for this crop:** {len(crop_data)}")
            st.markdown(f"- 📅 **Years available:** {int(crop_data['Year'].min())} – {int(crop_data['Year'].max())}")
            st.markdown(f"- 🌍 **Regions with this crop:** {crop_data['Region'].nunique()}")

    st.markdown("---")

    if st.button("🔍 Predict Risk", use_container_width=True):

        xgb_pred, rf_pred, input_data = predict_risk(
            selected_region, selected_crop, selected_year
        )

        st.session_state['region']     = selected_region
        st.session_state['crop']       = selected_crop
        st.session_state['year']       = selected_year
        st.session_state['xgb_pred']   = xgb_pred
        st.session_state['rf_pred']    = rf_pred
        st.session_state['input_data'] = input_data
        st.session_state['predicted']  = True

        st.subheader("🎯 Prediction Results")
        st.markdown(f"**Region:** {selected_region} | **Crop:** {selected_crop} | **Year:** {selected_year}")
        st.markdown("---")

        # Show crop image in results
        res_img_col, res_results_col = st.columns([1, 3])
        with res_img_col:
            st.image(
                get_crop_image(selected_crop),
                caption=f"🌾 {selected_crop}",
                use_container_width=True
            )

        with res_results_col:
            col1, col2 = st.columns(2)
            with col1:
                xgb_color = get_risk_color(xgb_pred)
                st.markdown(
                    f"""<div style='background-color:{xgb_color}; padding:30px;
                    border-radius:12px; text-align:center;'>
                    <h2 style='color:white; margin:0;'>🚀 XGBoost</h2>
                    <h1 style='color:white; margin:10px 0 0 0;'>{get_risk_label(xgb_pred)}</h1>
                    </div>""",
                    unsafe_allow_html=True
                )
            with col2:
                rf_color = get_risk_color(rf_pred)
                st.markdown(
                    f"""<div style='background-color:{rf_color}; padding:30px;
                    border-radius:12px; text-align:center;'>
                    <h2 style='color:white; margin:0;'>🌳 Random Forest</h2>
                    <h1 style='color:white; margin:10px 0 0 0;'>{get_risk_label(rf_pred)}</h1>
                    </div>""",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        if xgb_pred == rf_pred:
            st.success(f"✅ Both models agree: **{get_risk_label(xgb_pred)}** for {selected_region} - {selected_crop} in {selected_year}.")
        else:
            st.warning(f"⚠️ Models disagree. XGBoost: **{get_risk_label(xgb_pred)}** | Random Forest: **{get_risk_label(rf_pred)}**.")

    elif 'predicted' not in st.session_state:
        st.info("👆 Select Region, Crop Type, and Year above then click **Predict Risk**.")

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "📂 Dataset":

    st.title("📂 Agricultural Dataset")

    if 'predicted' in st.session_state:
        region = st.session_state['region']
        crop   = st.session_state['crop']
        year   = st.session_state['year']

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
        region     = st.session_state['region']
        crop       = st.session_state['crop']
        year       = st.session_state['year']
        input_data = st.session_state['input_data']
        xgb_pred   = st.session_state['xgb_pred']

        st.markdown(f"Showing SHAP for: **{region}** | **{crop}** | **{year}**")
        st.markdown(f"XGBoost Predicted: **{get_risk_label(xgb_pred)}**")
        st.markdown("---")
    else:
        st.info("💡 Go to **🤖 Prediction** page first and make a prediction to see SHAP explanations.")
        st.stop()

    # --- Reasons & Recommendations ---
    reasons, recommendations = get_risk_reasons(region, crop, year, xgb_pred, input_data)

    col_r, col_rec = st.columns(2)

    with col_r:
        st.subheader("📌 Why This Risk Level?")
        for r in reasons:
            st.markdown(f"- {r}")

    with col_rec:
        st.subheader("💡 Recommendations")
        for r in recommendations:
            st.markdown(f"- {r}")

    st.markdown("---")

    # --- Global Feature Importance ---
    st.subheader("🌍 Global Feature Importance (XGBoost)")
    sample_X  = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # --- Individual SHAP waterfall ---
    st.subheader("🔎 Feature Contributions for Your Prediction")
    shap_single = explainer.shap_values(input_data)

    try:
        if isinstance(shap_single, list):
            idx = min(int(xgb_pred), len(shap_single) - 1)
            sv  = shap_single[idx][0]
            ev  = explainer.expected_value[idx] if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value

        elif isinstance(shap_single, np.ndarray) and shap_single.ndim == 3:
            idx = min(int(xgb_pred), shap_single.shape[2] - 1)
            sv  = shap_single[0, :, idx]
            ev  = explainer.expected_value[idx] if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value

        else:
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

    except Exception:
        st.warning("Waterfall plot unavailable. Showing bar chart instead.")
        if isinstance(shap_single, list):
            sv = shap_single[0][0]
        elif isinstance(shap_single, np.ndarray) and shap_single.ndim == 3:
            sv = shap_single[0, :, 0]
        else:
            sv = shap_single[0]

        shap_df = pd.DataFrame({
            "Feature":    input_data.columns.tolist(),
            "SHAP Value": sv
        }).sort_values("SHAP Value", key=abs, ascending=False)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bar_colors = ["#dc3545" if v > 0 else "#28a745" for v in shap_df["SHAP Value"]]
        ax3.barh(shap_df["Feature"], shap_df["SHAP Value"], color=bar_colors)
        ax3.set_xlabel("SHAP Value (Red = increases risk, Green = decreases risk)")
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
        region   = st.session_state['region']
        crop     = st.session_state['crop']
        year     = st.session_state['year']
        xgb_pred = st.session_state['xgb_pred']
        rf_pred  = st.session_state['rf_pred']

        st.markdown(f"Analysis for: **{region}** | **{crop}** | **{year}**")
        st.markdown("---")

        st.subheader("🆚 Model Comparison")
        comp_df = pd.DataFrame({
            "Model":          ["🚀 XGBoost", "🌳 Random Forest"],
            "Predicted Risk": [get_risk_label(xgb_pred), get_risk_label(rf_pred)],
            "Risk Level":     [int(xgb_pred), int(rf_pred)]
        })
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")

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
            trend_df['RF_Risk']      = rf_model.predict(trend_df[model_features])

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
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
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
        region = st.session_state['region']
        crop   = st.session_state['crop']
        year   = st.session_state['year']
        xgb_pred = st.session_state['xgb_pred']

        st.markdown(
            f"Selected: **{region}** | **{crop}** | **{year}** → "
            f"Predicted Risk: **{get_risk_label(xgb_pred)}**"
        )
        st.markdown("---")

    # --- Full Ethiopia Map ---
    st.subheader("🗺️ Ethiopia Regional Risk Map")
    st.markdown("Each region is colored by its food security risk category.")

    fig_map = px.choropleth(
        risk_df,
        geojson=ethiopia_geojson,
        featureidkey="properties.ADM1_EN",
        locations='Region',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
        },
        hover_data={'Region': True, 'Risk_Category': True, 'Predicted_Risk': True},
        title="Food Security Risk Map of Ethiopia"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=550
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # --- Regional Bar Chart ---
    st.subheader("📊 Risk Level by Region")
    fig_bar = px.bar(
        risk_df.sort_values("Predicted_Risk", ascending=False),
        x='Region',
        y='Predicted_Risk',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
        },
        title='Predicted Risk Score by Region',
        labels={'Predicted_Risk': 'Risk Score (0=Low, 1=Med, 2=High)'}
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # --- Highlight selected region ---
    if 'predicted' in st.session_state:
        region = st.session_state['region']
        st.subheader(f"📍 Selected Region: {region}")

        region_risk = risk_df[risk_df['Region'] == region]
        if not region_risk.empty:
            risk_cat  = region_risk['Risk_Category'].values[0]
            risk_val  = region_risk['Predicted_Risk'].values[0]
            risk_code = 0 if "Low" in risk_cat else (1 if "Medium" in risk_cat else 2)
            color     = get_risk_color(risk_code)

            st.markdown(
                f"""<div style='background-color:{color}; padding:20px;
                border-radius:12px; text-align:center; max-width:400px;'>
                <h3 style='color:white; margin:0;'>📍 {region}</h3>
                <h2 style='color:white; margin:10px 0 0 0;'>{get_risk_label(risk_code)}</h2>
                <p style='color:white; margin:5px 0 0 0;'>Risk Score: {risk_val:.2f}</p>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.info(f"No risk map data available for {region}.")

    st.markdown("---")
    st.subheader("📋 All Regions Risk Table")
    st.dataframe(
        risk_df.sort_values("Predicted_Risk", ascending=False),
        use_container_width=True
    )

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
2. The system fills remaining features from historical data medians
3. Both **XGBoost** and **Random Forest** models predict the risk level
4. Results are shown as **Low**, **Medium**, or **High** risk
5. SHAP explains which features drove the prediction with reasons and recommendations

**Models:**
- 🚀 XGBoost
- 🌳 Random Forest

**Risk Levels:**
- 🟢 Low Risk — stable food security
- 🟡 Medium Risk — monitor closely
- 🔴 High Risk — immediate action required

**Version:** 1.0
**Country:** 🇪🇹 Ethiopia
""")
