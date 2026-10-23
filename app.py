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
# CROP IMAGES (Botanical Reference Close-ups)
# =====================================================

CROP_IMAGES = {
    "Maize":   "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_corn_on_the_cob.jpg/320px-Simple_corn_on_the_cob.jpg",
    "Wheat":   "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/WheatBread_transparency.png/320px-WheatBread_transparency.png",
    "Teff":    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eragrostis_tef_-_teff_%28aka%29.jpg/320px-Eragrostis_tef_-_teff_%28aka%29.jpg",
    "Barley":  "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Barleyleaf.jpg/320px-Barleyleaf.jpg",
    "Sorghum": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Sorghum_crop.jpg/320px-Sorghum_crop.jpg",
    "Millet":  "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Panicum_miliaceum_USDA.jpg/320px-Panicum_miliaceum_USDA.jpg",
    "Oats":    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg/320px-Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg",
}

DEFAULT_CROP_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Farmland_at_sunset.jpg/320px-Farmland_at_sunset.jpg"

def get_crop_image(crop_name):
    """Return close-up reference image of the crop."""
    if crop_name in CROP_IMAGES:
        return CROP_IMAGES[crop_name]
    for key in CROP_IMAGES:
        if key.lower() in crop_name.lower() or crop_name.lower() in key.lower():
            return CROP_IMAGES[key]
    return DEFAULT_CROP_IMAGE

# =====================================================
# CROP FIELD / LAND IMAGES — DYNAMIC BY CROP & RISK LEVEL
# Spacing:
# 0 (Low Risk):    Dense, lush, closely spaced healthy crop stand
# 1 (Medium Risk): Moderate density, patchy or partially spaced crops
# 2 (High Risk):   Sparse, widely spaced plants, stunted / drought-stressed field
# =====================================================

CROP_LAND_IMAGES = {
    "Maize": {
        0: "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Cornfield_banner.jpg/640px-Cornfield_banner.jpg",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Sweet_corn_field.jpg/640px-Sweet_corn_field.jpg",
        2: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Drought_stress_on_young_corn.jpg/640px-Drought_stress_on_young_corn.jpg",
    },
    "Wheat": {
        0: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Wheat_field_in_Dorset%2C_England_-_July_2009.jpg/640px-Wheat_field_in_Dorset%2C_England_-_July_2009.jpg",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Barley_field.jpg/640px-Barley_field.jpg",
        2: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Drought_in_Australia_2006.jpg/640px-Drought_in_Australia_2006.jpg",
    },
    "Teff": {
        0: "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eragrostis_tef_-_teff_%28aka%29.jpg/640px-Eragrostis_tef_-_teff_%28aka%29.jpg",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Panicum_miliaceum_USDA.jpg/640px-Panicum_miliaceum_USDA.jpg",
        2: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Drought_in_Australia_2006.jpg/640px-Drought_in_Australia_2006.jpg",
    },
    "Barley": {
        0: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Barley_field.jpg/640px-Barley_field.jpg",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Barleyleaf.jpg/640px-Barleyleaf.jpg",
        2: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Drought_in_Australia_2006.jpg/640px-Drought_in_Australia_2006.jpg",
    },
    "Sorghum": {
        0: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Sorghum_crop_Andhra_Pradesh_India.jpg/640px-Sorghum_crop_Andhra_Pradesh_India.jpg",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Sorghum_crop.jpg/640px-Sorghum_crop.jpg",
        2: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Drought_stress_on_young_corn.jpg/640px-Drought_stress_on_young_corn.jpg",
    },
    "Millet": {
        0: "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Panicum_miliaceum_USDA.jpg/640px-Panicum_miliaceum_USDA.jpg",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Sorghum_crop.jpg/640px-Sorghum_crop.jpg",
        2: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Drought_in_Australia_2006.jpg/640px-Drought_in_Australia_2006.jpg",
    },
    "Oats": {
        0: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg/640px-Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Barley_field.jpg/640px-Barley_field.jpg",
        2: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Drought_in_Australia_2006.jpg/640px-Drought_in_Australia_2006.jpg",
    },
}

DEFAULT_LAND_BY_RISK = {
    0: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Wheat_field_in_Dorset%2C_England_-_July_2009.jpg/640px-Wheat_field_in_Dorset%2C_England_-_July_2009.jpg",
    1: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Farmland_at_sunset.jpg/640px-Farmland_at_sunset.jpg",
    2: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Drought_in_Australia_2006.jpg/640px-Drought_in_Australia_2006.jpg",
}

DEFAULT_LAND_IMAGE = DEFAULT_LAND_BY_RISK[0]

def get_crop_land_image(crop_name, risk_code=None):
    """Return field image filled with selected crop type, adjusted dynamically by risk level."""
    code = int(risk_code) if risk_code is not None else 0
    if code not in [0, 1, 2]:
        code = 0

    for key, risk_dict in CROP_LAND_IMAGES.items():
        if key.lower() in crop_name.lower() or crop_name.lower() in key.lower():
            if isinstance(risk_dict, dict):
                return risk_dict.get(code, DEFAULT_LAND_BY_RISK.get(code, DEFAULT_LAND_IMAGE))
            return risk_dict

    return DEFAULT_LAND_BY_RISK.get(code, DEFAULT_LAND_IMAGE)

def get_field_density_description(risk_code):
    """Return human-readable agronomic description of crop spacing according to risk level."""
    code = int(risk_code) if risk_code is not None else 0
    if code == 2:
        return {
            "title": "🔴 Sparse Stand (Wide Spacing)",
            "desc": "Crops are far apart with visible dry soil gaps, reflecting severe moisture deficit, poor germination, and high food security risk.",
            "badge_color": "#dc3545"
        }
    elif code == 1:
        return {
            "title": "🟡 Moderate Stand (Medium Spacing)",
            "desc": "Crops show moderate canopy coverage with intermittent thinning, signaling emerging moisture stress and medium risk.",
            "badge_color": "#fd7e14"
        }
    else:
        return {
            "title": "🟢 Dense Stand (Optimal Close Spacing)",
            "desc": "Crops exhibit full canopy closure, robust vegetative density, and thriving stand establishment under low risk conditions.",
            "badge_color": "#28a745"
        }

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_risk_label(code):
    return {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}.get(int(code), "Unknown")

def get_risk_color(code):
    return {0: "#28a745", 1: "#fd7e14", 2: "#dc3545"}.get(int(code), "#6c757d")

def get_risk_reasons(region, crop, year, risk_code, input_data):
    """Generate human-readable reasons and crop-specific recommendations."""
    reasons = []
    rec = []

    yield_growth = float(input_data['Yield_Growth_Rate'].iloc[0])
    prod_growth  = float(input_data['Production_Growth_Rate'].iloc[0])
    yield_anom   = float(input_data['Yield_Anomaly'].iloc[0])
    area_eff     = float(input_data['Area_Efficiency'].iloc[0])
    ews          = float(input_data['Early_Warning_Score'].iloc[0])
    stability    = float(input_data['Yield_Stability'].iloc[0])

    risk_code = int(risk_code)

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
        reasons.append(f"📊 Yield stability is **low ({stability:.2f})**, meaning {crop} production is highly variable year to year.")

    if not reasons:
        reasons.append(f"ℹ️ The model assessed historical patterns for {crop} in {region} ({year}) and determined the risk level.")

    crop_lower = crop.lower()

    if risk_code == 2:
        base_rec = [
            "🆘 **Immediate action required** — alert local agricultural authorities.",
            "🏦 Activate emergency food reserves and support programs.",
            "📋 Conduct urgent field assessments to verify crop conditions.",
            "🤝 Coordinate with NGOs and government agencies for humanitarian support.",
        ]
    elif risk_code == 1:
        base_rec = [
            "⚠️ **Monitor closely** — situation may worsen without intervention.",
            "📊 Strengthen early warning monitoring for the coming season.",
            "📦 Build buffer food stocks as a precautionary measure.",
            "🧑‍🌾 Provide farmers with training on improved agricultural practices.",
        ]
    else:
        base_rec = [
            "✅ **Conditions are stable** — maintain current agricultural practices.",
            "📈 Continue investing in yield improvement programs.",
            "📊 Keep monitoring seasonal trends to detect early changes.",
        ]

    if "maize" in crop_lower or "corn" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌽 Use drought-tolerant maize varieties (e.g., DROUGHT-TEFF, DK8031).",
                "💧 Apply supplemental irrigation during critical silking and tasseling stages.",
                "🐛 Monitor for fall armyworm — spray with recommended pesticides immediately.",
                "🌱 Apply nitrogen fertilizer in split doses to improve grain filling.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌽 Consider intercropping maize with legumes to improve soil fertility.",
                "💧 Ensure consistent soil moisture especially during pollination.",
                "🐛 Scout fields weekly for pest activity and diseases like gray leaf spot.",
                "🌿 Apply mulching to retain soil moisture and reduce weed pressure.",
            ]
        else:
            crop_rec = [
                "🌽 Expand maize cultivation to underutilized areas in the region.",
                "🌱 Test improved hybrid maize varieties to boost yield potential.",
                "📦 Invest in proper post-harvest storage to minimize losses.",
            ]
    elif "wheat" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌾 Switch to drought-resistant wheat varieties like KAKABA or DANDA'A.",
                "💧 Apply irrigation at critical growth stages — tillering and heading.",
                "🍄 Monitor for wheat rust (stem, leaf, yellow) — apply fungicides early.",
                "🌡️ Consider early planting to avoid terminal heat stress.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌾 Apply phosphorus fertilizer at planting to improve root development.",
                "💧 Optimize irrigation scheduling using soil moisture monitoring.",
                "🍄 Use disease-resistant wheat varieties to reduce fungicide dependency.",
                "🌿 Practice crop rotation with legumes to improve soil nitrogen.",
            ]
        else:
            crop_rec = [
                "🌾 Promote use of certified wheat seed to ensure high germination.",
                "📈 Explore market linkages for wheat surplus in stable years.",
                "🌱 Maintain soil health through balanced fertilization programs.",
            ]
    else:
        if risk_code == 2:
            crop_rec = [
                f"🌱 Use improved {crop} varieties adapted to local conditions.",
                f"💧 Apply supplemental irrigation during critical growth stages of {crop}.",
                f"🐛 Increase pest and disease monitoring for {crop} fields.",
                f"🧑‍🌾 Provide emergency agronomic support to {crop} farmers in {region}.",
            ]
        elif risk_code == 1:
            crop_rec = [
                f"🌱 Apply recommended fertilizers for {crop} production in {region}.",
                f"💧 Optimize water management practices for {crop} cultivation.",
                f"🌿 Practice crop rotation to improve soil health for {crop}.",
                f"📊 Monitor {crop} fields closely for pest and disease outbreaks.",
            ]
        else:
            crop_rec = [
                f"🌱 Continue good agricultural practices for {crop} in {region}.",
                f"📈 Explore yield improvement opportunities for {crop}.",
                f"📦 Invest in post-harvest handling to reduce {crop} losses.",
            ]

    rec = base_rec + crop_rec
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
5. Observe how the crop field condition and spacing dynamically adapt to the predicted risk level
6. Explore other pages for deeper analysis
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

    st.markdown("---")
    img_col, land_col, info_col = st.columns([1, 1, 2])
    with img_col:
        crop_img_url = get_crop_image(selected_crop)
        st.image(crop_img_url, caption=f"🌾 {selected_crop} (Reference)", use_container_width=True)
    with land_col:
        land_img_url = get_crop_land_image(selected_crop, risk_code=0)
        st.image(land_img_url, caption=f"🌱 {selected_crop} Field (Baseline Stand)", use_container_width=True)
        st.caption("ℹ️ *Note: Field stand and plant spacing will dynamically update to reflect the predicted risk.*")
    with info_col:
        st.markdown(f"### Selected Input")
        st.markdown(f"- 🌍 **Region:** {selected_region}")
        st.markdown(f"- 🌾 **Crop Type:** {selected_crop}")
        st.markdown(f"- 📅 **Year:** {selected_year}")
        crop_data = df[df["crop type"] == selected_crop]
        if not crop_data.empty:
            region_crop_data = df[
                (df["crop type"] == selected_crop) &
                (df["Region"] == selected_region)
            ]
            avg_area = region_crop_data['Area cultivated(Ha)'].mean() if not region_crop_data.empty else crop_data['Area cultivated(Ha)'].mean()
            avg_prod = region_crop_data['Production(kg)'].mean() if not region_crop_data.empty else crop_data['Production(kg)'].mean()
            st.markdown(f"- 📋 **Records for this crop:** {len(crop_data)}")
            st.markdown(f"- 📅 **Years available:** {int(crop_data['Year'].min())} – {int(crop_data['Year'].max())}")
            st.markdown(f"- 🌍 **Regions with this crop:** {crop_data['Region'].nunique()}")
            st.markdown(f"- 🌱 **Avg Area Cultivated:** {avg_area:,.1f} Ha")
            st.markdown(f"- 📦 **Avg Production:** {avg_prod:,.1f} kg")

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

        # Show crop image and dynamic land image based on predicted risk level
        res_img_col, res_land_col, res_results_col = st.columns([1, 1, 2])
        with res_img_col:
            st.image(
                get_crop_image(selected_crop),
                caption=f"🌾 {selected_crop} Reference",
                use_container_width=True
            )
        with res_land_col:
            # Field image selected according to predicted risk
            risk_land_img = get_crop_land_image(selected_crop, risk_code=xgb_pred)
            density_info = get_field_density_description(xgb_pred)
            st.image(
                risk_land_img,
                caption=f"🌱 {selected_crop} Field — {density_info['title']}",
                use_container_width=True
            )
            st.markdown(
                f"""<div style='background-color:{density_info["badge_color"]}15; border-left: 4px solid {density_info["badge_color"]}; padding: 8px 12px; border-radius: 6px; margin-top: -6px;'>
                <b style='color: {density_info["badge_color"]}; font-size: 13px;'>{density_info["title"]}</b><br>
                <span style='color: #444; font-size: 12px;'>{density_info["desc"]}</span>
                </div>""",
                unsafe_allow_html=True
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

    st.subheader("🌍 Global Feature Importance (XGBoost)")
    sample_X  = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("🔎 Feature Contributions for Your Prediction")
    st.caption("Red bars increase risk | Green bars decrease risk | Sorted by importance")
    shap_single = explainer.shap_values(input_data)

    try:
        if isinstance(shap_single, list):
            idx = min(int(xgb_pred), len(shap_single) - 1)
            sv  = shap_single[idx][0]
        elif isinstance(shap_single, np.ndarray) and shap_single.ndim == 3:
            idx = min(int(xgb_pred), shap_single.shape[2] - 1)
            sv  = shap_single[0, :, idx]
        else:
            sv = shap_single[0]
    except Exception:
        sv = shap_single[0] if not isinstance(shap_single, list) else shap_single[0][0]

    shap_df = pd.DataFrame({
        "Feature":    input_data.columns.tolist(),
        "SHAP Value": sv
    }).sort_values("SHAP Value", key=abs, ascending=True)

    bar_colors = ["#dc3545" if v > 0 else "#28a745" for v in shap_df["SHAP Value"]]

    fig_bar, ax_bar = plt.subplots(figsize=(10, 7))

    bars = ax_bar.barh(
        shap_df["Feature"],
        shap_df["SHAP Value"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.6
    )

    for bar, val in zip(bars, shap_df["SHAP Value"]):
        x_pos = bar.get_width()
        ax_bar.text(
            x_pos + (0.001 if x_pos >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if x_pos >= 0 else "right",
            fontsize=9,
            color="#333333"
        )

    ax_bar.axvline(x=0, color="black", linewidth=0.8, linestyle="--")

    ax_bar.set_xlabel("SHAP Value  (positive = increases risk,  negative = decreases risk)",
                      fontsize=10)
    ax_bar.set_title(
        f"Feature Contributions — {get_risk_label(xgb_pred)} "
        f"({region} | {crop} | {year})",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#dc3545", label="Increases Risk"),
        Patch(facecolor="#28a745", label="Decreases Risk"),
    ]
    ax_bar.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.markdown("---")

    st.subheader("📋 Feature Contributions Table")
    shap_table = shap_df.copy().sort_values("SHAP Value", key=abs, ascending=False)
    shap_table["Direction"] = shap_table["SHAP Value"].apply(
        lambda v: "🔴 Increases Risk" if v > 0 else "🟢 Decreases Risk"
    )
    shap_table["SHAP Value"] = shap_table["SHAP Value"].apply(lambda v: f"{v:+.4f}")
    shap_table["Feature Value"] = input_data.iloc[0].values
    shap_table["Feature Value"] = shap_table["Feature Value"].apply(lambda v: f"{v:.4f}")
    st.dataframe(
        shap_table[["Feature", "Feature Value", "SHAP Value", "Direction"]].reset_index(drop=True),
        use_container_width=True
    )

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
        region   = st.session_state['region']
        crop     = st.session_state['crop']
        year     = st.session_state['year']
        xgb_pred = st.session_state['xgb_pred']
        st.markdown(
            f"Selected: **{region}** | **{crop}** | **{year}** → "
            f"Predicted Risk: **{get_risk_label(xgb_pred)}**"
        )
        st.markdown("---")

    st.markdown("""
    <div style='display:flex; gap:20px; margin-bottom:16px;'>
        <span style='background:#28a745;color:white;padding:6px 16px;border-radius:8px;font-weight:700;'>🟢 Low Risk</span>
        <span style='background:#fd7e14;color:white;padding:6px 16px;border-radius:8px;font-weight:700;'>🟡 Medium Risk</span>
        <span style='background:#dc3545;color:white;padding:6px 16px;border-radius:8px;font-weight:700;'>🔴 High Risk</span>
    </div>
    """, unsafe_allow_html=True)

    REGION_COORDS = {
        "Tigray":            (14.0323, 38.3142),
        "Afar":              (11.7557, 40.9087),
        "Amhara":            (11.3400, 37.9783),
        "Oromia":            ( 7.5460, 40.6348),
        "Somali":            ( 6.6613, 43.7908),
        "Benishangul-Gumuz": (10.7795, 35.5657),
        "SNNPR":             ( 6.5000, 37.5000),
        "Gambela":           ( 7.9000, 34.5833),
        "Harari":            ( 9.3147, 42.1184),
        "Dire Dawa":         ( 9.5931, 41.8661),
        "Addis Ababa":       ( 9.0200, 38.7468),
        "Sidama":            ( 6.7792, 38.4521),
    }

    def get_region_coords(region_name):
        if region_name in REGION_COORDS:
            return REGION_COORDS[region_name]
        for key, coords in REGION_COORDS.items():
            if key.lower() in region_name.lower() or region_name.lower() in key.lower():
                return coords
        return (9.145, 40.489)

    map_df = risk_df.copy()
    map_df['lat'] = map_df['Region'].apply(lambda r: get_region_coords(r)[0])
    map_df['lon'] = map_df['Region'].apply(lambda r: get_region_coords(r)[1])
    map_df['color'] = map_df['Risk_Category'].map({
        'Low Risk':    '#28a745',
        'Medium Risk': '#fd7e14',
        'High Risk':   '#dc3545'
    }).fillna('#6c757d')
    map_df['risk_code'] = map_df['Risk_Category'].apply(
        lambda c: 0 if 'Low' in c else (1 if 'Medium' in c else 2)
    )
    map_df['size'] = 18

    st.subheader("🗺️ Ethiopia — Full Country Risk Map")

    fig_main = px.scatter_geo(
        map_df,
        lat='lat', lon='lon',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
        },
        size='size',
        size_max=28,
        hover_name='Region',
        hover_data={'Risk_Category': True, 'Predicted_Risk': ':.2f',
                    'lat': False, 'lon': False, 'size': False},
        text='Region',
        labels={'Risk_Category': 'Risk Level', 'Predicted_Risk': 'Risk Score'},
        title="Ethiopia Food Security Risk — All Regions",
    )

    fig_main.update_traces(
        textposition='top center',
        textfont=dict(size=11, color='black'),
        marker=dict(opacity=0.9, line=dict(width=1, color='white'))
    )

    fig_main.update_geos(
        visible=True,
        resolution=50,
        scope="africa",
        showcoastlines=True,  coastlinecolor="#888",
        showland=True,        landcolor="#f5f4e8",
        showocean=True,       oceancolor="#cce5ff",
        showlakes=True,       lakecolor="#cce5ff",
        showrivers=True,      rivercolor="#aad4f5",
        showsubunits=True,    subunitcolor="#cccccc",
        showcountries=True,   countrycolor="#999999",
        lonaxis_range=[32.0, 48.5],
        lataxis_range=[ 3.0, 15.5],
    )

    fig_main.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=0.01,
                    xanchor="right", x=1),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig_main, use_container_width=True)

    st.markdown("---")

    st.subheader("📍 Individual Region Risk Cards")

    all_regions = sorted(map_df['Region'].unique())
    cards_per_row = 3

    for row_start in range(0, len(all_regions), cards_per_row):
        cols = st.columns(cards_per_row)
        for col_idx, col in enumerate(cols):
            r_idx = row_start + col_idx
            if r_idx >= len(all_regions):
                break
            reg = all_regions[r_idx]
            row_data = map_df[map_df['Region'] == reg].iloc[0]
            color     = row_data['color']
            risk_cat  = row_data['Risk_Category']
            risk_val  = row_data['Predicted_Risk']
            risk_code_r = int(row_data['risk_code'])
            lat, lon  = row_data['lat'], row_data['lon']

            with col:
                reg_df = pd.DataFrame({
                    'Region':       [reg],
                    'lat':          [lat],
                    'lon':          [lon],
                    'Risk_Category':[risk_cat],
                    'size':         [20],
                })

                fig_mini = px.scatter_geo(
                    reg_df,
                    lat='lat', lon='lon',
                    color='Risk_Category',
                    color_discrete_map={
                        'Low Risk':    '#28a745',
                        'Medium Risk': '#fd7e14',
                        'High Risk':   '#dc3545'
                    },
                    size='size', size_max=22,
                    hover_name='Region',
                    text='Region',
                )
                fig_mini.update_traces(
                    textposition='top center',
                    textfont=dict(size=10, color='black'),
                    marker=dict(opacity=0.9, line=dict(width=1, color='white'))
                )
                fig_mini.update_geos(
                    visible=True,
                    resolution=50,
                    scope="africa",
                    showland=True,   landcolor="#f5f4e8",
                    showocean=True,  oceancolor="#cce5ff",
                    showcoastlines=True, coastlinecolor="#aaa",
                    showcountries=True,  countrycolor="#bbb",
                    lonaxis_range=[lon - 3, lon + 3],
                    lataxis_range=[lat - 3, lat + 3],
                )
                fig_mini.update_layout(
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=180,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_mini, use_container_width=True,
                                key=f"mini_{reg}")

                st.markdown(
                    f"""<div style='background:{color};padding:8px;
                    border-radius:8px;text-align:center;margin-top:-8px;'>
                    <b style='color:white;font-size:13px;'>📍 {reg}</b><br>
                    <span style='color:white;font-size:12px;'>{get_risk_label(risk_code_r)}</span><br>
                    <span style='color:white;font-size:11px;'>Score: {risk_val:.2f}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
                st.markdown(" ")

    st.markdown("---")

    if 'predicted' in st.session_state:
        sel_region = st.session_state['region']
        st.subheader(f"📌 Your Selected Region: {sel_region}")

        sel_data = map_df[map_df['Region'] == sel_region]
        if not sel_data.empty:
            s         = sel_data.iloc[0]
            color     = s['color']
            risk_val  = s['Predicted_Risk']
            risk_code_s = int(s['risk_code'])
            s_lat, s_lon = s['lat'], s['lon']

            hl1, hl2 = st.columns([1, 1])
            with hl1:
                sel_df = pd.DataFrame({
                    'Region':       [sel_region],
                    'lat':          [s_lat],
                    'lon':          [s_lon],
                    'Risk_Category':[s['Risk_Category']],
                    'size':         [25],
                })
                fig_sel = px.scatter_geo(
                    sel_df,
                    lat='lat', lon='lon',
                    color='Risk_Category',
                    color_discrete_map={
                        'Low Risk':    '#28a745',
                        'Medium Risk': '#fd7e14',
                        'High Risk':   '#dc3545'
                    },
                    size='size', size_max=30,
                    hover_name='Region',
                    text='Region',
                )
                fig_sel.update_traces(
                    textposition='top center',
                    textfont=dict(size=12, color='black'),
                    marker=dict(opacity=1.0, line=dict(width=2, color='white'))
                )
                fig_sel.update_geos(
                    visible=True,
                    resolution=50,
                    scope="africa",
                    showland=True,      landcolor="#f5f4e8",
                    showocean=True,     oceancolor="#cce5ff",
                    showcoastlines=True, coastlinecolor="#aaa",
                    showcountries=True,  countrycolor="#bbb",
                    lonaxis_range=[s_lon - 4, s_lon + 4],
                    lataxis_range=[s_lat - 4, s_lat + 4],
                )
                fig_sel.update_layout(
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=300, showlegend=False
                )
                st.plotly_chart(fig_sel, use_container_width=True,
                                key="selected_region_highlight")

            with hl2:
                st.markdown(
                    f"""<div style='background:{color};padding:35px;
                    border-radius:14px;text-align:center;margin-top:20px;'>
                    <h2 style='color:white;margin:0;'>📍 {sel_region}</h2>
                    <h1 style='color:white;margin:12px 0 0 0;'>{get_risk_label(risk_code_s)}</h1>
                    <p style='color:white;margin:8px 0 0 0;font-size:18px;'>
                    Risk Score: {risk_val:.2f}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
        else:
            st.info(f"No risk data available for {sel_region}.")

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
5. Crop field canopy density and spacing dynamically adapt to the risk level
6. SHAP explains which features drove the prediction with reasons and recommendations

**Models:**
- 🚀 XGBoost
- 🌳 Random Forest

**Risk Levels:**
- 🟢 Low Risk — stable food security (dense, thriving stand)
- 🟡 Medium Risk — monitor closely (moderate stand spacing)
- 🔴 High Risk — immediate action required (sparse, drought-affected stand)

**Version:** 1.0  
**Country:** 🇪🇹 Ethiopia
""")
