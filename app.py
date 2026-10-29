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
# DYNAMIC CROP FIELD SIMULATOR (Scaled to Cultivated Land)
# =====================================================

def render_crop_field_simulation(crop_name, risk_code, cultivated_ha=None):
    """
    Dynamically generates an aerial crop field visualization.
    Reflects both the predicted risk level spacing and the cultivated land scale.
    """
    code = int(risk_code) if risk_code is not None else 0
    if code not in [0, 1, 2]:
        code = 0

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=110)

    if code == 2:  # High Risk — Plants are far apart, dry soil, severe drought
        grid_step = 2.4
        plant_size = 130
        plant_color = "#b2973b"
        plant_edge  = "#5d4037"
        soil_color  = "#d7ccc8"
        spacing_text = "FAR APART (>150 cm spacing)"
        density_pct = "18% Canopy Cover"
        status_label = "🔴 HIGH RISK — SPARSE / DROUGHT STRESSED"
        marker_shape = "v"
        noise_level = 0.35
        plants_per_ha = 15000

        np.random.seed(42)
        for _ in range(7):
            cx = np.random.uniform(1.0, 9.0)
            cy = np.random.uniform(1.0, 9.0)
            ax.plot([cx, cx + np.random.uniform(-1.8, 1.8)],
                    [cy, cy + np.random.uniform(-1.8, 1.8)],
                    color="#8d6e63", linewidth=1.2, alpha=0.55, linestyle=":")

    elif code == 1:  # Medium Risk — Moderate spacing
        grid_step = 1.35
        plant_size = 175
        plant_color = "#8bc34a"
        plant_edge  = "#33691e"
        soil_color  = "#a1887f"
        spacing_text = "MODERATE (75-90 cm spacing)"
        density_pct = "52% Canopy Cover"
        status_label = "🟡 MEDIUM RISK — MODERATE DENSITY"
        marker_shape = "^"
        noise_level = 0.16
        plants_per_ha = 42000

    else:  # Low Risk (code 0) — Dense, lush, thriving
        grid_step = 0.8
        plant_size = 220
        plant_color = "#2e7d32"
        plant_edge  = "#1b5e20"
        soil_color  = "#4e342e"
        spacing_text = "DENSE (30-40 cm close spacing)"
        density_pct = "94% Canopy Cover"
        status_label = "🟢 LOW RISK — OPTIMAL DENSE STAND"
        marker_shape = "^"
        noise_level = 0.08
        plants_per_ha = 78000

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

    # Cultivated land & population string
    if cultivated_ha is not None and cultivated_ha > 0:
        total_plants = plants_per_ha * cultivated_ha
        if total_plants >= 1e9:
            pop_str = f"{total_plants / 1e9:.2f} Billion"
        elif total_plants >= 1e6:
            pop_str = f"{total_plants / 1e6:.2f} Million"
        else:
            pop_str = f"{total_plants:,.0f}"
        land_header = f"🌱 {crop_name} Field Canopy ({cultivated_ha:,.1f} Ha Filled)\n{status_label}"
        banner_text = f"Cultivated Land: {cultivated_ha:,.1f} Ha | Stand Pop: ~{pop_str} plants | {density_pct}"
    else:
        land_header = f"🌱 {crop_name} Field Canopy Simulation\n{status_label}"
        banner_text = f"Spacing: {spacing_text} | {density_pct} | ~{plants_per_ha:,.0f} plants/Ha"

    ax.set_title(land_header, fontsize=10.5, fontweight="bold", color="#111111", pad=8)
    ax.text(
        5, 0.45, banner_text,
        fontsize=8, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#999")
    )

    plt.tight_layout()
    return fig

def get_field_density_description(risk_code, cultivated_ha=None):
    """Return human-readable agronomic description of crop spacing and land coverage."""
    code = int(risk_code) if risk_code is not None else 0
    ha_text = f" across {cultivated_ha:,.1f} Ha" if cultivated_ha else ""

    if code == 2:
        return {
            "title": "🔴 Sparse Stand (Wide Spacing & Severe Moisture Stress)",
            "desc": f"Crops{ha_text} are widely spaced with dry soil gaps. Indicates severe germination failure, drought stress, and critical food security risk.",
            "badge_color": "#dc3545",
            "plants_per_ha": 15000,
            "canopy_cover": "18%"
        }
    elif code == 1:
        return {
            "title": "🟡 Moderate Stand (Medium Spacing & Emerging Stress)",
            "desc": f"Crops{ha_text} display moderate canopy closure with intermittent thinned patches, reflecting moderate moisture deficit and medium vulnerability.",
            "badge_color": "#fd7e14",
            "plants_per_ha": 42000,
            "canopy_cover": "52%"
        }
    else:
        return {
            "title": "🟢 Dense Stand (Optimal Close Spacing & High Health)",
            "desc": f"Crops{ha_text} exhibit full vegetative canopy closure, robust vegetative density, and strong yield potential under low-risk conditions.",
            "badge_color": "#28a745",
            "plants_per_ha": 78000,
            "canopy_cover": "94%"
        }

# =====================================================
# DYNAMIC REGIONAL RISK COMPUTATION
# =====================================================

def get_dynamic_regional_risk(crop_type, year):
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

        area_val = float(base['Area_Filled'].median() if 'Area_Filled' in base.columns else base['Area cultivated(Ha)'].median())

        row_data = {
            'Region': reg,
            'crop type': crop_type,
            'Year': year,
            'Region_Code': reg_code,
            'Crop_Code': crop_code,
            'Area cultivated(Ha)': area_val,
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

def get_risk_reasons(region, crop, year, risk_code, input_data, cultivated_ha=None):
    reasons = []
    yield_growth = float(input_data['Yield_Growth_Rate'].iloc[0])
    prod_growth  = float(input_data['Production_Growth_Rate'].iloc[0])
    yield_anom   = float(input_data['Yield_Anomaly'].iloc[0])
    area_eff     = float(input_data['Area_Efficiency'].iloc[0])
    ews          = float(input_data['Early_Warning_Score'].iloc[0])
    stability    = float(input_data['Yield_Stability'].iloc[0])

    risk_code = int(risk_code)

    if cultivated_ha is not None and cultivated_ha > 0:
        reasons.append(f"🌾 **Cultivated Land Allocation:** A total of **{cultivated_ha:,.1f} Hectares** is filled with **{crop}** in {region} for {year}.")

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
            f"🆘 **Immediate action required** on the **{cultivated_ha:,.0f} Ha** of {crop} — alert regional agricultural bureaus.",
            "🏦 Activate strategic emergency grain reserves to buffer local deficits.",
            "📋 Deploy urgent agronomist teams to inspect crop stand density and pest incidence.",
            "🤝 Mobilize targeted humanitarian cash/food transfers for vulnerable farming households.",
        ]
    elif risk_code == 1:
        base_rec = [
            f"⚠️ **Close monitoring needed** for the **{cultivated_ha:,.0f} Ha** under {crop}.",
            "📊 Track rainfall and soil moisture indicators across key growing zones.",
            "📦 Build local community grain buffer stocks as a precautionary cushion.",
            "🧑🌾 Distribute farmer training on moisture-conserving tillage and mulching.",
        ]
    else:
        base_rec = [
            f"✅ **Conditions are stable** across the **{cultivated_ha:,.0f} Ha** planted with {crop}.",
            "📈 Maintain current agronomic practices and input distribution channels.",
            "📊 Continue regular seasonal surveillance to sustain favorable crop yields.",
        ]

    if "maize" in crop_lower or "corn" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌽 Distribute early-maturing, drought-tolerant maize varieties (e.g. DK8031, Melkassa).",
                "💧 Initiate emergency supplemental furrow irrigation during silking.",
                "🐛 Rigorously scout for Fall Armyworm (FAW) and apply biopesticides.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌽 Encourage intercropping with food legumes (haricot beans, cowpea) to safeguard soil moisture.",
                "💧 Monitor critical tassel-filling stages against localized dry spells.",
            ]
        else:
            crop_rec = [
                "🌽 Promote optimal nitrogen side-dressing and post-harvest hermetic storage.",
            ]
    elif "teff" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌾 Prioritize row-planting and lodging-resistant teff cultivars (e.g. Quncho, Boset).",
                "💧 Implement soil water retention bunds across sloped fields.",
            ]
        else:
            crop_rec = [
                "🌾 Expand certified teff seed usage and mechanize row planting.",
            ]
    else:
        if risk_code == 2:
            crop_rec = [
                f"🌱 Supply certified stress-tolerant seed varieties for {crop}.",
                f"💧 Optimize water conservation techniques across {crop} fields.",
            ]
        elif risk_code == 1:
            crop_rec = [
                f"🌱 Apply recommended balanced fertilization (NPS + Urea) for {crop}.",
            ]
        else:
            crop_rec = [
                f"🌱 Continue best agronomic management practices for {crop} in {region}.",
            ]

    rec = base_rec + crop_rec
    return reasons, rec

def get_cultivated_land_info(region, crop_type, year):
    """Retrieve historical cultivated land, production, and yield for given inputs."""
    exact_mask = (df["Region"] == region) & (df["crop type"] == crop_type) & (df["Year"] == year)
    exact_df = df[exact_mask]

    if not exact_df.empty:
        row = exact_df.iloc[0]
        area = float(row['Area_Filled']) if 'Area_Filled' in row and not pd.isna(row['Area_Filled']) else float(row['Area cultivated(Ha)'])
        prod = float(row['Production(kg)'])
        yield_val = float(row['Yeild (kg/ha)']) if 'Yeild (kg/ha)' in row and not pd.isna(row['Yeild (kg/ha)']) else (prod / area if area > 0 else 0)
        is_exact = True
    else:
        rc_df = df[(df["Region"] == region) & (df["crop type"] == crop_type)]
        if not rc_df.empty:
            area = float(rc_df['Area_Filled'].median() if 'Area_Filled' in rc_df.columns else rc_df['Area cultivated(Ha)'].median())
            prod = float(rc_df['Production(kg)'].median())
            yield_val = float(rc_df['Yeild (kg/ha)'].median() if 'Yeild (kg/ha)' in rc_df.columns else (prod / area if area > 0 else 0))
        else:
            area = float(df['Area cultivated(Ha)'].median())
            prod = float(df['Production(kg)'].median())
            yield_val = float(df['Yeild (kg/ha)'].median() if 'Yeild (kg/ha)' in df.columns else (prod / area if area > 0 else 0))
        is_exact = False

    # Regional total cultivated area for that year to compute land share
    reg_year_mask = (df["Region"] == region) & (df["Year"] == year)
    reg_year_df = df[reg_year_mask]
    if not reg_year_df.empty:
        total_reg_area = float(reg_year_df['Area cultivated(Ha)'].sum())
    else:
        reg_df = df[df["Region"] == region]
        total_reg_area = float(reg_df.groupby('Year')['Area cultivated(Ha)'].sum().median()) if not reg_df.empty else area * 3

    land_share_pct = (area / total_reg_area * 100) if total_reg_area > 0 else 0.0
    land_share_pct = min(100.0, land_share_pct)

    return {
        "cultivated_ha": area,
        "production_kg": prod,
        "yield_kgha": yield_val,
        "total_regional_ha": total_reg_area,
        "land_share_pct": land_share_pct,
        "is_exact_record": is_exact
    }

def predict_risk(region, crop_type, year, custom_area=None):
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

    area_val = custom_area if custom_area is not None else float(base['Area cultivated(Ha)'].median())

    input_data = pd.DataFrame([[
        region_code, crop_code, year,
        area_val,
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

Features:
- 🌾 Cultivated Land Analytics
- 🎯 Risk Classification (XGBoost / RF)
- 🌱 Dynamic Crop Canopy Simulator
- 🔍 SHAP Explainability

Country: 🇪🇹 Ethiopia
""")

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🌾 Agricultural Early Warning System")
    st.subheader("Explainable AI-Based Food Security Risk & Cultivated Land Mapping for Ethiopia")
    st.markdown("---")

    st.markdown("""
Welcome to the **Agricultural Early Warning System (AEWS)**.

This system forecasts food security risk and models land productivity across Ethiopian regions using machine learning.

👈 **Select '🤖 Prediction' in the sidebar to test predictions for any Region, Crop Type, and Year.**

### Core Capabilities:
1. **🤖 Prediction & Cultivated Land Modeling**: Predicts food security risk and visualizes the crop stand canopy across the cultivated land parcel.
2. **🌾 Land Allocation Insights**: Quantifies hectarage, regional land share, and standing crop population.
3. **🔍 Explainable AI (SHAP)**: Identifies the agronomic and climate drivers behind every risk forecast.
4. **📊 Dynamic Risk Analysis**: Explores multi-year yield trajectories and regional vulnerability matrices.
5. **🗺️ Geospatial Risk Map**: Visualizes food security risk distributions interactively across Ethiopia.
""")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Records", f"{len(df):,}")
    col2.metric("🌍 Administrative Regions", df["Region"].nunique())
    col3.metric("🌾 Tracked Crop Types", df["crop type"].nunique())
    col4.metric("📅 Temporal Range", f"{int(df['Year'].min())} – {int(df['Year'].max())}")

# =====================================================
# PREDICTION PAGE (With Cultivated Land & Risk Display)
# =====================================================

elif page == "🤖 Prediction":

    st.title("🤖 Early Warning Prediction & Cultivated Land Analysis")
    st.write("Enter the **Region**, **Crop Type**, and **Year** to display the predicted risk level and analyze the cultivated land filled with that crop.")
    st.markdown("---")

    # Input Selectors
    col1, col2, col3 = st.columns(3)
    with col1:
        reg_options = sorted(df["Region"].unique())
        default_reg = st.session_state.get('region', reg_options[0])
        selected_region = st.selectbox("🌍 Select Region", options=reg_options, index=reg_options.index(default_reg))
    with col2:
        crop_options = sorted(df["crop type"].unique())
        default_crp = st.session_state.get('crop', crop_options[0])
        selected_crop = st.selectbox("🌾 Select Crop Type", options=crop_options, index=crop_options.index(default_crp))
    with col3:
        year_list = sorted(df["Year"].unique(), reverse=True)
        default_yr = st.session_state.get('year', year_list[0])
        selected_year = st.selectbox("📅 Select Year", options=year_list, index=year_list.index(default_yr))

    st.session_state['region'] = selected_region
    st.session_state['crop']   = selected_crop
    st.session_state['year']   = selected_year

    # Retrieve Cultivated Land Data for Selected Query
    land_info = get_cultivated_land_info(selected_region, selected_crop, selected_year)

    # Historical Baseline Preview Card
    st.markdown("---")
    with st.container():
        st.markdown(f"#### 📍 Historical Agronomic Profile: **{selected_crop}** in **{selected_region} ({selected_year})**")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("🌾 Cultivated Land (Ha)", f"{land_info['cultivated_ha']:,.1f} Ha", help="Area of land cultivated and filled with this crop.")
        pcol2.metric("📦 Production Output", f"{land_info['production_kg']:,.0f} kg", help="Total historical production.")
        pcol3.metric("⚡ Yield Productivity", f"{land_info['yield_kgha']:,.1f} kg/Ha", help="Productivity per hectare.")
        pcol4.metric("🗺️ Regional Land Share", f"{land_info['land_share_pct']:.1f}%", help="Percentage of total regional cultivated farmland filled with this crop.")

    # Optional What-If Cultivated Area Adjustment
    with st.expander("⚙️ Optional: Simulate What-If Cultivated Area"):
        custom_ha_input = st.number_input(
            "Modify Cultivated Land (Ha) for Simulation:",
            min_value=1.0,
            max_value=2000000.0,
            value=float(round(land_info['cultivated_ha'], 1)),
            step=500.0,
            help="Adjust this value to test risk and population on a hypothetical land size."
        )

    st.markdown("---")

    # Predict Button
    if st.button("🔍 Predict Risk & Analyze Cultivated Land", use_container_width=True, type="primary"):

        active_ha = custom_ha_input if custom_ha_input else land_info['cultivated_ha']

        xgb_pred, rf_pred, input_data = predict_risk(
            selected_region, selected_crop, selected_year, custom_area=active_ha
        )

        st.session_state['region']        = selected_region
        st.session_state['crop']          = selected_crop
        st.session_state['year']          = selected_year
        st.session_state['xgb_pred']      = xgb_pred
        st.session_state['rf_pred']       = rf_pred
        st.session_state['input_data']    = input_data
        st.session_state['cultivated_ha'] = active_ha
        st.session_state['predicted']     = True

        st.subheader("🎯 Food Security Risk & Cultivated Land Assessment")
        st.caption(f"Results for **{selected_crop}** cultivated in **{selected_region}** ({selected_year})")

        # Top Metric Row: Risk Level + Cultivated Land KPIs
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        # 1. Cultivated Land Metric
        m_col1.metric(
            label="🌾 Cultivated Land Filled",
            value=f"{active_ha:,.1f} Ha",
            delta=f"{active_ha / 100:,.1f} km² ({land_info['land_share_pct']:.1f}% of regional land)",
            delta_color="normal"
        )

        # 2. Risk Level Metric
        risk_label_str = get_risk_label(xgb_pred)
        m_col2.metric(
            label="🎯 Predicted Risk Level",
            value=risk_label_str,
            delta="XGBoost Primary Model",
            delta_color="off"
        )

        # 3. Estimated Harvest Output
        est_prod_tons = (land_info['yield_kgha'] * active_ha) / 1000.0
        m_col3.metric(
            label="📦 Est. Harvest Output",
            value=f"{est_prod_tons:,.0f} Tons",
            delta=f"{land_info['yield_kgha']:,.1f} kg/Ha yield",
            delta_color="normal"
        )

        # 4. Total Plant Stand on Cultivated Land
        density_details = get_field_density_description(xgb_pred, active_ha)
        total_stand = density_details["plants_per_ha"] * active_ha
        if total_stand >= 1e9:
            stand_str = f"~{total_stand / 1e9:.2f} Billion"
        elif total_stand >= 1e6:
            stand_str = f"~{total_stand / 1e6:.2f} Million"
        else:
            stand_str = f"~{total_stand:,.0f}"

        m_col4.metric(
            label="🌱 Total Crop Stand Population",
            value=stand_str,
            delta=f"{density_details['canopy_cover']} Canopy Density",
            delta_color="normal"
        )

        st.markdown("---")

        # Detailed Section: Simulation on Left, Risk Badges & Land Impact on Right
        res_land_col, res_results_col = st.columns([1.2, 1.8])

        with res_land_col:
            st.markdown("##### 🛰️ Aerial Field Canopy Simulation")
            field_fig = render_crop_field_simulation(selected_crop, risk_code=xgb_pred, cultivated_ha=active_ha)
            st.pyplot(field_fig, use_container_width=True)

            st.markdown(
                f"""<div style='background-color:{density_details["badge_color"]}15; border-left: 4px solid {density_details["badge_color"]}; padding: 10px 14px; border-radius: 8px;'>
                <b style='color: {density_details["badge_color"]}; font-size: 13px;'>{density_details["title"]}</b><br>
                <span style='color: #333; font-size: 12px;'>{density_details["desc"]}</span>
                </div>""",
                unsafe_allow_html=True
            )

        with res_results_col:
            st.markdown("##### 🤖 Model Predictions & Consensus")
            col_xgb, col_rf = st.columns(2)
            with col_xgb:
                xgb_color = get_risk_color(xgb_pred)
                st.markdown(
                    f"""<div style='background-color:{xgb_color}; padding:22px 15px;
                    border-radius:10px; text-align:center;'>
                    <h4 style='color:white; margin:0;'>🚀 XGBoost</h4>
                    <h2 style='color:white; margin:6px 0 0 0; font-size: 22px;'>{get_risk_label(xgb_pred)}</h2>
                    </div>""",
                    unsafe_allow_html=True
                )
            with col_rf:
                rf_color = get_risk_color(rf_pred)
                st.markdown(
                    f"""<div style='background-color:{rf_color}; padding:22px 15px;
                    border-radius:10px; text-align:center;'>
                    <h4 style='color:white; margin:0;'>🌳 Random Forest</h4>
                    <h2 style='color:white; margin:6px 0 0 0; font-size: 22px;'>{get_risk_label(rf_pred)}</h2>
                    </div>""",
                    unsafe_allow_html=True
                )

            # Consensus statement
            if xgb_pred == rf_pred:
                st.success(f"✅ Both models strongly agree on **{get_risk_label(xgb_pred)}** for {selected_crop} in {selected_region}.")
            else:
                st.warning(f"⚠️ Model divergence: XGBoost predicts **{get_risk_label(xgb_pred)}**, while Random Forest predicts **{get_risk_label(rf_pred)}**.")

            # Cultivated Land Allocation Card
            st.markdown("##### 🌾 Cultivated Land Allocation Summary")
            st.markdown(
                f"""
                - **Total Hectares Filled:** `{active_ha:,.1f} Ha`
                - **Regional Farmland Share:** `{land_info['land_share_pct']:.2f}%` of all crop land in {selected_region}
                - **Plant Spacing & Geometry:** `{density_details['title'].split('(')[-1].replace(')', '')}`
                - **Expected Grain Output:** `{est_prod_tons:,.1f} Metric Tons`
                """
            )

            # Visual land coverage bar
            st.progress(float(min(1.0, max(0.01, land_info['land_share_pct'] / 100.0))))
            st.caption(f"*{selected_crop} represents {land_info['land_share_pct']:.1f}% of total cultivated land in {selected_region}.*")

        st.markdown("---")

        # Agronomic Action Plan for the Cultivated Land
        st.subheader("💡 Agronomic Early Warning Advisory for Cultivated Land")
        reasons, recommendations = get_risk_reasons(
            selected_region, selected_crop, selected_year, xgb_pred, input_data, cultivated_ha=active_ha
        )

        col_r, col_rec = st.columns(2)
        with col_r:
            st.markdown("##### 📌 Key Drivers Influencing Risk")
            for r in reasons:
                st.markdown(f"- {r}")

        with col_rec:
            st.markdown("##### 🚜 Actionable Recommendations for this Land")
            for r in recommendations:
                st.markdown(f"- {r}")

    elif 'predicted' not in st.session_state:
        st.info("👆 Confirm your **Region**, **Crop Type**, and **Year** above, then click **'🔍 Predict Risk & Analyze Cultivated Land'**.")

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
        st.info("💡 Go to **🤖 Prediction** page first and run an analysis to view filtered data here.")

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
        region        = st.session_state['region']
        crop          = st.session_state['crop']
        year          = st.session_state['year']
        input_data    = st.session_state['input_data']
        xgb_pred      = st.session_state['xgb_pred']
        cultivated_ha = st.session_state.get('cultivated_ha', None)

        st.markdown(f"Showing SHAP explanation for: **{region}** | **{crop}** | **{year}** | **Cultivated Area: {cultivated_ha:,.1f} Ha**")
        st.markdown(f"Model Prediction: **{get_risk_label(xgb_pred)}**")
        st.markdown("---")
    else:
        st.info("💡 Go to **🤖 Prediction** page first and run a prediction to view SHAP explanations.")
        st.stop()

    reasons, recommendations = get_risk_reasons(region, crop, year, xgb_pred, input_data, cultivated_ha=cultivated_ha)

    col_r, col_rec = st.columns(2)
    with col_r:
        st.subheader("📌 Identified Risk Drivers")
        for r in reasons:
            st.markdown(f"- {r}")

    with col_rec:
        st.subheader("💡 Agronomic Interventions")
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

    st.subheader("🔎 Local Feature Attributions for this Prediction")
    st.caption("Red bars increase risk | Green bars decrease risk")
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
    ax_bar.set_xlabel("SHAP Value (Positive = increases food insecurity risk, Negative = protects security)", fontsize=10)
    ax_bar.set_title(
        f"Feature Contributions — {get_risk_label(xgb_pred)} ({region} | {crop} | {year})",
        fontsize=12, fontweight="bold", pad=12
    )
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

# =====================================================
# RISK ANALYSIS PAGE
# =====================================================

elif page == "📊 Risk Analysis":

    st.title("📊 Food Security Risk Analysis")
    st.write("Examine risk forecasts and land utilization dynamics across Ethiopian regions.")
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
        hover_data={'Predicted_Risk': True, 'Risk_Category': True, 'Early_Warning_Score': ':.2f', 'Area cultivated(Ha)': ':,.1f'},
        labels={'Predicted_Risk': 'Risk Level (0=Low, 1=Med, 2=High)', 'Region': 'Ethiopian Region'},
        title=f"Predicted Risk Distribution Across Regions for {analysis_crop} in {analysis_year}"
    )
    fig_dyn_bar.update_layout(yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=['Low (0)', 'Medium (1)', 'High (2)']))
    st.plotly_chart(fig_dyn_bar, use_container_width=True)

    st.markdown("---")

    st.subheader("🔬 Cultivated Land vs. Early Warning Score")
    st.caption("Bubble size indicates total cultivated land (Ha) filled with the selected crop.")

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
        title=f"Cultivated Land Dynamics: Early Warning Score vs. Yield Growth Rate ({analysis_crop}, {analysis_year})"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    st.subheader(f"📋 Regional Risk & Cultivated Land Summary ({analysis_crop}, {analysis_year})")
    display_cols = [
        'Region', 'Risk_Label', 'Area cultivated(Ha)', 'Production(kg)',
        'Early_Warning_Score', 'Yield_Growth_Rate', 'Area_Efficiency'
    ]
    st.dataframe(
        dyn_risk_df[display_cols].sort_values('Area cultivated(Ha)', ascending=False).reset_index(drop=True),
        use_container_width=True
    )

# =====================================================
# ETHIOPIA RISK MAP PAGE
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
        size='Area cultivated(Ha)',
        size_max=32,
        hover_name='Region',
        hover_data={'Risk_Category': True, 'Predicted_Risk': True, 'Early_Warning_Score': ':.2f',
                    'Area cultivated(Ha)': ':,.1f', 'lat': False, 'lon': False},
        text='Region',
        labels={'Risk_Category': 'Risk Level', 'Predicted_Risk': 'Risk Code'},
        title=f"Ethiopia Food Security Risk & Cultivated Land Distribution — {map_crop} ({map_year})",
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

    st.subheader("📋 Regional Risk & Cultivated Land Table")
    st.dataframe(
        map_df[['Region', 'Risk_Label', 'Area cultivated(Ha)', 'Production(kg)', 'Early_Warning_Score', 'Yield_Growth_Rate', 'Area_Efficiency']].sort_values("Area cultivated(Ha)", ascending=False),
        use_container_width=True
    )

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Application")
    st.markdown("""
### Agricultural Early Warning System (AEWS)

This application predicts food security risk and analyzes cultivated land across Ethiopian regions using machine learning.

**How it works:**
1. User selects **Region**, **Crop Type**, and **Year**.
2. The system retrieves the exact **Cultivated Land (Ha)** filled with that crop and calculates land share.
3. Machine learning models (**XGBoost** and **Random Forest**) predict the food security risk category.
4. An aerial field simulator illustrates the crop parcel, showing spacing, canopy closure, and stand population.
5. SHAP values explain key agronomic and climate risk drivers, paired with customized recommendations.

**Models:**
- 🚀 XGBoost Classifier
- 🌳 Random Forest Classifier

**Risk Levels:**
- 🟢 **Low Risk** — Stable food security with dense, thriving canopy (~78,000 plants/Ha).
- 🟡 **Medium Risk** — Emerging stress requiring close monitoring (~42,000 plants/Ha).
- 🔴 **High Risk** — Severe deficit requiring emergency response (~15,000 plants/Ha).

**Version:** 2.0  
**Country:** 🇪🇹 Ethiopia
""")
