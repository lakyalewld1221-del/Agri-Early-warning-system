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
static_risk_df = load_risk_map()
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
# DYNAMIC CROP FIELD SIMULATOR (Matplotlib Generated)
# =====================================================

def render_crop_field_simulation(crop_name, risk_code):
    """
    Dynamically generates an aerial crop field visualization.
    If risk is HIGH -> crops are generated FAR APART with dry cracked soil.
    If risk is MEDIUM -> crops are generated with MODERATE SPACING.
    If risk is LOW -> crops are generated DENSE and closely packed.
    """
    code = int(risk_code) if risk_code is not None else 0
    if code not in [0, 1, 2]:
        code = 0

    fig, ax = plt.subplots(figsize=(6, 4.3), dpi=100)

    if code == 2:  # High Risk — Plants are far apart, dry soil, severe drought
        grid_step = 2.4
        plant_size = 140
        plant_color = "#b2973b"
        plant_edge  = "#5d4037"
        soil_color  = "#d7ccc8"
        spacing_text = "FAR APART (>150 cm spacing)"
        density_pct = "18% Canopy Cover"
        status_label = "🔴 HIGH RISK — SPARSE / WIDE SPACING"
        marker_shape = "v"
        noise_level = 0.35
        num_plants_approx = "~15,000 plants/Ha"

        np.random.seed(42)
        for _ in range(7):
            cx = np.random.uniform(1.0, 9.0)
            cy = np.random.uniform(1.0, 9.0)
            ax.plot([cx, cx + np.random.uniform(-1.8, 1.8)],
                    [cy, cy + np.random.uniform(-1.8, 1.8)],
                    color="#8d6e63", linewidth=1.2, alpha=0.55, linestyle=":")

    elif code == 1:  # Medium Risk — Moderate spacing
        grid_step = 1.35
        plant_size = 180
        plant_color = "#8bc34a"
        plant_edge  = "#33691e"
        soil_color  = "#a1887f"
        spacing_text = "MODERATE (75-90 cm spacing)"
        density_pct = "52% Canopy Cover"
        status_label = "🟡 MEDIUM RISK — MODERATE SPACING"
        marker_shape = "^"
        noise_level = 0.16
        num_plants_approx = "~42,000 plants/Ha"

    else:  # Low Risk (code 0) — Dense, lush, thriving
        grid_step = 0.8
        plant_size = 230
        plant_color = "#2e7d32"
        plant_edge  = "#1b5e20"
        soil_color  = "#4e342e"
        spacing_text = "DENSE (30-40 cm close spacing)"
        density_pct = "94% Canopy Cover"
        status_label = "🟢 LOW RISK — OPTIMAL DENSE STAND"
        marker_shape = "^"
        noise_level = 0.08
        num_plants_approx = "~78,000 plants/Ha"

    ax.set_facecolor(soil_color)
    fig.patch.set_facecolor("#ffffff")

    xs = np.arange(0.6, 9.6, grid_step)
    ys = np.arange(0.6, 9.6, grid_step)
    x_coords, y_coords = [], []

    np.random.seed(15)
    for x in xs:
        for y in ys:
            if code == 2 and np.random.rand() < 0.28:
                continue
            elif code == 1 and np.random.rand() < 0.08:
                continue
            jitter_x = x + np.random.uniform(-noise_level, noise_level)
            jitter_y = y + np.random.uniform(-noise_level, noise_level)
            x_coords.append(jitter_x)
            y_coords.append(jitter_y)

    ax.scatter(
        x_coords, y_coords,
        s=plant_size,
        c=plant_color,
        edgecolors=plant_edge,
        linewidths=1.2,
        marker=marker_shape,
        alpha=0.92,
        zorder=3
    )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color="#ffffff", alpha=0.18, linestyle="--")

    ax.set_title(
        f"🌱 {crop_name} Field Canopy Simulation\n{status_label}",
        fontsize=11, fontweight="bold", color="#111111", pad=8
    )

    ax.text(
        5, 0.45,
        f"Spacing: {spacing_text} | {density_pct} | {num_plants_approx}",
        fontsize=8.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#999")
    )

    plt.tight_layout()
    return fig

def get_field_density_description(risk_code):
    """Return human-readable agronomic description of crop spacing."""
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
# DYNAMIC REGIONAL RISK COMPUTATION
# Relates crop type, year, and features across all regions
# =====================================================

def get_dynamic_regional_risk(crop_type, year):
    """
    Computes model-predicted food security risk and aggregates key features
    across all Ethiopian regions for a specific crop and year.
    """
    regional_rows = []
    crop_code = crop_encoder.transform([crop_type])[0]

    for reg in sorted(df['Region'].unique()):
        reg_code = region_encoder.transform([reg])[0]

        mask = (df["Region"] == reg) & (df["crop type"] == crop_type) & (df["Year"] == year)
        base = df[mask]

        if base.empty:
            base = df[(df["Region"] == reg) & (df["crop type"] == crop_type)]
        if base.empty:
            base = df[df["Region"] == reg]
        if base.empty:
            base = df

        row_data = {
            'Region': reg,
            'crop type': crop_type,
            'Year': year,
            'Region_Code': reg_code,
            'Crop_Code': crop_code,
            'Area cultivated(Ha)': float(base['Area cultivated(Ha)'].median()),
            'Production(kg)': float(base['Production(kg)'].median()),
            'Yield_Growth_Rate': float(base['Yield_Growth_Rate'].median()),
            'Production_Growth_Rate': float(base['Production_Growth_Rate'].median()),
            'Area_Efficiency': float(base['Area_Efficiency'].median()),
            'Regional_Avg_Yield': float(base['Regional_Avg_Yield'].median()),
            'Crop_Avg_Yield': float(base['Crop_Avg_Yield'].median()),
            'Yield_Anomaly': float(base['Yield_Anomaly'].median()),
            'Rolling_Yield_Trend': float(base['Rolling_Yield_Trend'].median()),
            'Yield_Stability': float(base['Yield_Stability'].median()),
            'Production_Area_Ratio': float(base['Production_Area_Ratio'].median()),
            'Early_Warning_Score': float(base['Early_Warning_Score'].median()),
        }
        regional_rows.append(row_data)

    reg_df = pd.DataFrame(regional_rows)

    feature_cols = [
        'Region_Code', 'Crop_Code', 'Year',
        'Area cultivated(Ha)', 'Production(kg)',
        'Yield_Growth_Rate', 'Production_Growth_Rate', 'Area_Efficiency',
        'Regional_Avg_Yield', 'Crop_Avg_Yield', 'Yield_Anomaly',
        'Rolling_Yield_Trend', 'Yield_Stability', 'Production_Area_Ratio',
        'Early_Warning_Score'
    ]

    reg_df[feature_cols] = reg_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        if reg_df[col].isnull().any():
            g_med = df[col].replace([np.inf, -np.inf], np.nan).median()
            reg_df[col] = reg_df[col].fillna(g_med if not np.isnan(g_med) else 0)

    reg_df['Predicted_Risk'] = xgb_model.predict(reg_df[feature_cols])
    reg_df['RF_Risk']        = rf_model.predict(reg_df[feature_cols])

    cat_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    reg_df['Risk_Category'] = reg_df['Predicted_Risk'].map(cat_map)
    reg_df['Risk_Label']    = reg_df['Predicted_Risk'].apply(lambda c: get_risk_label(c))

    return reg_df

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
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌽 Consider intercropping maize with legumes to improve soil fertility.",
                "💧 Ensure consistent soil moisture especially during pollination.",
            ]
        else:
            crop_rec = [
                "🌽 Expand maize cultivation to underutilized areas in the region.",
                "🌱 Test improved hybrid maize varieties to boost yield potential.",
            ]
    else:
        if risk_code == 2:
            crop_rec = [
                f"🌱 Use improved {crop} varieties adapted to local conditions.",
                f"💧 Apply supplemental irrigation during critical growth stages of {crop}.",
            ]
        elif risk_code == 1:
            crop_rec = [
                f"🌱 Apply recommended fertilizers for {crop} production in {region}.",
                f"💧 Optimize water management practices for {crop} cultivation.",
            ]
        else:
            crop_rec = [
                f"🌱 Continue good agricultural practices for {crop} in {region}.",
                f"📈 Explore yield improvement opportunities for {crop}.",
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

### Key Features:
1. **🤖 Prediction**: Select Region, Crop Type, and Year to forecast food security risk with dynamic field stand simulation.
2. **🔍 Explainable AI**: Understand feature contributions through SHAP value waterfall & summary charts.
3. **📊 Risk Analysis**: Dynamic regional summaries and feature breakdowns related to chosen crops and years.
4. **🗺️ Ethiopia Risk Map**: High-resolution geospatial food security mapping.
""")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Total Records", len(df))
    col2.metric("🌍 Regions", df["Region"].nunique())
    col3.metric("🌾 Crop Types", df["crop type"].nunique())

# =====================================================
# PREDICTION PAGE (Without close-up crop image)
# =====================================================

elif page == "🤖 Prediction":

    st.title("🤖 Early Warning Prediction")
    st.write("Select Region, Crop Type, and Year to predict food security risk.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        default_reg = st.session_state.get('region', sorted(df["Region"].unique())[0])
        selected_region = st.selectbox("🌍 Region", options=sorted(df["Region"].unique()), index=sorted(df["Region"].unique()).index(default_reg))
    with col2:
        default_crp = st.session_state.get('crop', sorted(df["crop type"].unique())[0])
        selected_crop = st.selectbox("🌾 Crop Type", options=sorted(df["crop type"].unique()), index=sorted(df["crop type"].unique()).index(default_crp))
    with col3:
        year_list = sorted(df["Year"].unique(), reverse=True)
        default_yr = st.session_state.get('year', year_list[0])
        selected_year = st.selectbox("📅 Year", options=year_list, index=year_list.index(default_yr))

    st.session_state['region'] = selected_region
    st.session_state['crop']   = selected_crop
    st.session_state['year']   = selected_year

    st.markdown("---")
    land_col, info_col = st.columns([1.2, 1.8])
    with land_col:
        baseline_fig = render_crop_field_simulation(selected_crop, risk_code=0)
        st.pyplot(baseline_fig, use_container_width=True)
        st.caption("ℹ️ *Field stand simulation: plant spacing will dynamically update to reflect predicted risk.*")
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

        # Two-column layout: Left = Dynamic Field Simulation, Right = Model Risk Results
        res_land_col, res_results_col = st.columns([1.2, 1.8])
        with res_land_col:
            field_fig = render_crop_field_simulation(selected_crop, risk_code=xgb_pred)
            st.pyplot(field_fig, use_container_width=True)

            density_info = get_field_density_description(xgb_pred)
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
                    f"""<div style='background-color:{xgb_color}; padding:30px 15px;
                    border-radius:12px; text-align:center;'>
                    <h3 style='color:white; margin:0;'>🚀 XGBoost</h3>
                    <h2 style='color:white; margin:8px 0 0 0;'>{get_risk_label(xgb_pred)}</h2>
                    </div>""",
                    unsafe_allow_html=True
                )
            with col2:
                rf_color = get_risk_color(rf_pred)
                st.markdown(
                    f"""<div style='background-color:{rf_color}; padding:30px 15px;
                    border-radius:12px; text-align:center;'>
                    <h3 style='color:white; margin:0;'>🌳 Random Forest</h3>
                    <h2 style='color:white; margin:8px 0 0 0;'>{get_risk_label(rf_pred)}</h2>
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
# RISK ANALYSIS PAGE (Dynamic & Related to Crop & Year)
# =====================================================

elif page == "📊 Risk Analysis":

    st.title("📊 Food Security Risk Analysis")
    st.write("Examine risk forecasts and feature dynamics dynamically across Ethiopian regions.")
    st.markdown("---")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        crop_options = sorted(df["crop type"].unique())
        cur_crop = st.session_state.get('crop', crop_options[0])
        idx_crop = crop_options.index(cur_crop) if cur_crop in crop_options else 0
        analysis_crop = st.selectbox("🌾 Filter by Crop Type", crop_options, index=idx_crop, key="analysis_crop_select")
    with col_sel2:
        year_options = sorted(df["Year"].unique(), reverse=True)
        cur_year = st.session_state.get('year', year_options[0])
        idx_year = year_options.index(cur_year) if cur_year in year_options else 0
        analysis_year = st.selectbox("📅 Filter by Year", year_options, index=idx_year, key="analysis_year_select")

    st.session_state['crop'] = analysis_crop
    st.session_state['year'] = analysis_year

    dyn_risk_df = get_dynamic_regional_risk(analysis_crop, analysis_year)

    st.markdown("---")

    high_count = (dyn_risk_df['Predicted_Risk'] == 2).sum()
    med_count  = (dyn_risk_df['Predicted_Risk'] == 1).sum()
    low_count  = (dyn_risk_df['Predicted_Risk'] == 0).sum()
    avg_ews    = dyn_risk_df['Early_Warning_Score'].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔴 High Risk Regions", f"{high_count} / {len(dyn_risk_df)}")
    m2.metric("🟡 Medium Risk Regions", f"{med_count} / {len(dyn_risk_df)}")
    m3.metric("🟢 Low Risk Regions", f"{low_count} / {len(dyn_risk_df)}")
    m4.metric("🚨 Avg Early Warning Score", f"{avg_ews:.2f}")

    st.markdown("---")

    st.subheader(f"🌍 Food Security Risk by Region for {analysis_crop} ({analysis_year})")
    fig_dyn_bar = px.bar(
        dyn_risk_df.sort_values('Predicted_Risk', ascending=False),
        x='Region',
        y='Predicted_Risk',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
        },
        hover_data={'Predicted_Risk': True, 'Risk_Category': True, 'Early_Warning_Score': ':.2f', 'Yield_Growth_Rate': ':.2f'},
        labels={'Predicted_Risk': 'Risk Level (0=Low, 1=Med, 2=High)', 'Region': 'Ethiopian Region'},
        title=f"Predicted Food Security Risk Across Regions for {analysis_crop} in {analysis_year}"
    )
    fig_dyn_bar.update_layout(yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=['Low (0)', 'Medium (1)', 'High (2)']))
    st.plotly_chart(fig_dyn_bar, use_container_width=True)

    st.markdown("---")

    st.subheader("🔬 Feature Dynamics & Risk Correlation")
    st.caption("How Yield Growth Rate and Early Warning Score interact to influence regional risk.")

    fig_scatter = px.scatter(
        dyn_risk_df,
        x='Yield_Growth_Rate',
        y='Early_Warning_Score',
        color='Risk_Category',
        size='Area cultivated(Ha)',
        hover_name='Region',
        color_discrete_map={
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
        },
        labels={'Yield_Growth_Rate': 'Yield Growth Rate', 'Early_Warning_Score': 'Early Warning Score'},
        title=f"Regional Feature Dynamics: Early Warning Score vs. Yield Growth Rate ({analysis_crop}, {analysis_year})"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    selected_reg = st.session_state.get('region', dyn_risk_df['Region'].iloc[0])
    st.subheader(f"📈 Historical Risk Trend for Selected Region: {selected_reg} ({analysis_crop})")

    trend_df = df[
        (df["Region"] == selected_reg) &
        (df["crop type"] == analysis_crop)
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

        fig_trend = px.line(
            yearly, x='Year',
            y=['XGBoost_Risk', 'RF_Risk'],
            title=f"Multi-Year Risk Trajectory: {selected_reg} ({analysis_crop})",
            markers=True,
            labels={"value": "Risk Level (0=Low, 1=Med, 2=High)", "variable": "Model"}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info(f"No multi-year historical records found for {selected_reg} - {analysis_crop}.")

    st.markdown("---")

    st.subheader(f"📋 Regional Risk Summary & Key Indicators ({analysis_crop}, {analysis_year})")
    display_cols = [
        'Region', 'Risk_Label', 'Early_Warning_Score', 'Yield_Growth_Rate',
        'Area_Efficiency', 'Yield_Anomaly', 'Production(kg)', 'Area cultivated(Ha)'
    ]
    st.dataframe(
        dyn_risk_df[display_cols].sort_values('Early_Warning_Score', ascending=False).reset_index(drop=True),
        use_container_width=True
    )

# =====================================================
# ETHIOPIA RISK MAP PAGE (Dynamic with Crop & Year)
# =====================================================

elif page == "🗺️ Ethiopia Risk Map":

    st.title("🗺️ Ethiopia Food Security Risk Map")
    st.write("Geospatial risk mapping dynamically calculated for the selected crop and year.")
    st.markdown("---")

    map_c1, map_c2 = st.columns(2)
    with map_c1:
        crop_opts = sorted(df["crop type"].unique())
        map_crop = st.selectbox("🌾 Crop Type", crop_opts, index=crop_opts.index(st.session_state.get('crop', crop_opts[0])), key="map_crop_sel")
    with map_c2:
        year_opts = sorted(df["Year"].unique(), reverse=True)
        map_year = st.selectbox("📅 Year", year_opts, index=year_opts.index(st.session_state.get('year', year_opts[0])), key="map_year_sel")

    st.session_state['crop'] = map_crop
    st.session_state['year'] = map_year

    map_df = get_dynamic_regional_risk(map_crop, map_year)

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

    map_df['lat'] = map_df['Region'].apply(lambda r: get_region_coords(r)[0])
    map_df['lon'] = map_df['Region'].apply(lambda r: get_region_coords(r)[1])
    map_df['color'] = map_df['Risk_Category'].map({
        'Low Risk':    '#28a745',
        'Medium Risk': '#fd7e14',
        'High Risk':   '#dc3545'
    }).fillna('#6c757d')
    map_df['size'] = 18

    st.subheader(f"🗺️ Ethiopia — Geospatial Risk Map ({map_crop}, {map_year})")

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
        hover_data={'Risk_Category': True, 'Predicted_Risk': True, 'Early_Warning_Score': ':.2f',
                    'lat': False, 'lon': False, 'size': False},
        text='Region',
        labels={'Risk_Category': 'Risk Level', 'Predicted_Risk': 'Risk Code'},
        title=f"Ethiopia Food Security Risk Distribution — {map_crop} ({map_year})",
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
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=1),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig_main, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Regional Risk Table")
    st.dataframe(
        map_df[['Region', 'Risk_Label', 'Early_Warning_Score', 'Yield_Growth_Rate', 'Area_Efficiency', 'Production(kg)']].sort_values("Early_Warning_Score", ascending=False),
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
2. Machine learning models (**XGBoost** and **Random Forest**) predict food security risk
3. Dynamic aerial field canopy simulation depicts crop spacing directly linked to risk
4. SHAP provides explainability and actionable agronomic recommendations
5. Risk Analysis & Ethiopia Maps dynamically correlate with selected crop types and years

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
