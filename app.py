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
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import json

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
    try:
        with open('ethiopia_regions.geojson', encoding='utf-8') as f:
            geojson_data = json.load(f)
        return geojson_data
    except Exception as e:
        return None

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

# Canonical 10 Ethiopian Administrative Regions & Coordinates
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

def normalize_region_name(name):
    """Normalize region spelling across GeoJSON and datasets."""
    if not isinstance(name, str):
        return ""
    n = name.strip().lower()
    if "tigray" in n: return "Tigray"
    if "afar" in n: return "Afar"
    if "amhara" in n: return "Amhara"
    if "oromi" in n: return "Oromia"
    if "somali" in n: return "Somali"
    if "benishangul" in n or "benshangul" in n: return "Benishangul-Gumuz"
    if "snnp" in n or "southern" in n: return "SNNPR"
    if "gambel" in n: return "Gambela"
    if "harar" in n: return "Harari"
    if "dire" in n: return "Dire Dawa"
    if "addis" in n: return "Addis Ababa"
    if "sidama" in n: return "Sidama"
    return name.strip()

# =====================================================
# DYNAMIC REGIONAL RISK COMPUTATION
# =====================================================

def get_dynamic_regional_risk(crop_type, year):
    regional_rows = []
    crop_code = crop_encoder.transform([crop_type])[0]

    # Ensure all regions in the dataset are always included
    all_regions = sorted(df['Region'].unique())

    for reg in all_regions:
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
        if np.isnan(area_val) or area_val < 0:
            area_val = 0.0

        row_data = {
            'Region': reg,
            'crop type': crop_type,
            'Year': year,
            'Region_Code': reg_code,
            'Crop_Code': crop_code,
            'Area cultivated(Ha)': area_val,
            'Production(kg)': float(base['Production(kg)'].median() if not pd.isna(base['Production(kg)'].median()) else 0.0),
            'Yield_Growth_Rate': float(base['Yield_Growth_Rate'].median() if not pd.isna(base['Yield_Growth_Rate'].median()) else 0.0),
            'Production_Growth_Rate': float(base['Production_Growth_Rate'].median() if not pd.isna(base['Production_Growth_Rate'].median()) else 0.0),
            'Area_Efficiency': float(base['Area_Efficiency'].median() if not pd.isna(base['Area_Efficiency'].median()) else 0.0),
            'Regional_Avg_Yield': float(base['Regional_Avg_Yield'].median() if not pd.isna(base['Regional_Avg_Yield'].median()) else 0.0),
            'Crop_Avg_Yield': float(base['Crop_Avg_Yield'].median() if not pd.isna(base['Crop_Avg_Yield'].median()) else 0.0),
            'Yield_Anomaly': float(base['Yield_Anomaly'].median() if not pd.isna(base['Yield_Anomaly'].median()) else 0.0),
            'Rolling_Yield_Trend': float(base['Rolling_Yield_Trend'].median() if not pd.isna(base['Rolling_Yield_Trend'].median()) else 0.0),
            'Yield_Stability': float(base['Yield_Stability'].median() if not pd.isna(base['Yield_Stability'].median()) else 0.0),
            'Production_Area_Ratio': float(base['Production_Area_Ratio'].median() if not pd.isna(base['Production_Area_Ratio'].median()) else 0.0),
            'Early_Warning_Score': float(base['Early_Warning_Score'].median() if not pd.isna(base['Early_Warning_Score'].median()) else 0.0),
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

    # Add display safe sizes so scatter plots NEVER drop zero-area regions
    min_bubble = 15
    max_bubble = 60
    max_area = reg_df['Area cultivated(Ha)'].max()
    if max_area > 0:
        reg_df['Plot_Bubble_Size'] = min_bubble + (reg_df['Area cultivated(Ha)'] / max_area) * (max_bubble - min_bubble)
    else:
        reg_df['Plot_Bubble_Size'] = min_bubble

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
        reasons.append(f"🌾 **Cultivated Land Allocation:** A total of **{cultivated_ha:,.1f} Hectares** is planted with **{crop}** in {region} for {year}.")

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
            "📋 Deploy urgent agronomist teams to inspect crop stand and pest incidence.",
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
- 🔍 SHAP Explainability
- 🗺️ Geospatial Risk Mapping

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
1. **🤖 Prediction & Cultivated Land Modeling**: Predicts food security risk and analyzes cultivated land parameters.
2. **🌾 Land Allocation Insights**: Quantifies hectarage, regional land share, and estimated harvest volume.
3. **🔍 Explainable AI (SHAP)**: Identifies the agronomic and climate drivers behind every risk forecast.
4. **📊 Dynamic Risk Analysis**: Explores multi-year yield trajectories and regional vulnerability matrices across all administrative regions.
5. **🗺️ Geospatial Risk Map**: Visualizes the complete 10 Ethiopian administrative regions on a unified map.
""")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Records", f"{len(df):,}")
    col2.metric("🌍 Administrative Regions", df["Region"].nunique())
    col3.metric("🌾 Tracked Crop Types", df["crop type"].nunique())
    col4.metric("📅 Temporal Range", f"{int(df['Year'].min())} – {int(df['Year'].max())}")

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🤖 Prediction":

    st.title("🤖 Early Warning Prediction & Cultivated Land Analysis")
    st.write("Enter the **Region**, **Crop Type**, and **Year** to display the predicted risk level and analyze the cultivated land.")
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

    land_info = get_cultivated_land_info(selected_region, selected_crop, selected_year)

    st.markdown("---")
    with st.container():
        st.markdown(f"#### 📍 Historical Agronomic Profile: **{selected_crop}** in **{selected_region} ({selected_year})**")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("🌾 Cultivated Land (Ha)", f"{land_info['cultivated_ha']:,.1f} Ha", help="Area of land cultivated and planted with this crop.")
        pcol2.metric("📦 Production Output", f"{land_info['production_kg']:,.0f} kg", help="Total historical production.")
        pcol3.metric("⚡ Yield Productivity", f"{land_info['yield_kgha']:,.1f} kg/Ha", help="Productivity per hectare.")
        pcol4.metric("🗺️ Regional Land Share", f"{land_info['land_share_pct']:.1f}%", help="Percentage of total regional cultivated farmland planted with this crop.")

    with st.expander("⚙️ Optional: Simulate What-If Cultivated Area"):
        custom_ha_input = st.number_input(
            "Modify Cultivated Land (Ha) for Simulation:",
            min_value=1.0,
            max_value=2000000.0,
            value=float(round(land_info['cultivated_ha'], 1)),
            step=500.0,
            help="Adjust this value to test risk on a hypothetical land size."
        )

    st.markdown("---")

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

        m_col1, m_col2, m_col3 = st.columns(3)

        m_col1.metric(
            label="🌾 Cultivated Land",
            value=f"{active_ha:,.1f} Ha",
            delta=f"{active_ha / 100:,.1f} km² ({land_info['land_share_pct']:.1f}% of regional land)",
            delta_color="normal"
        )

        risk_label_str = get_risk_label(xgb_pred)
        m_col2.metric(
            label="🎯 Predicted Risk Level",
            value=risk_label_str,
            delta="XGBoost Primary Model",
            delta_color="off"
        )

        est_prod_tons = (land_info['yield_kgha'] * active_ha) / 1000.0
        m_col3.metric(
            label="📦 Est. Harvest Output",
            value=f"{est_prod_tons:,.0f} Tons",
            delta=f"{land_info['yield_kgha']:,.1f} kg/Ha yield",
            delta_color="normal"
        )

        st.markdown("---")

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

        if xgb_pred == rf_pred:
            st.success(f"✅ Both models strongly agree on **{get_risk_label(xgb_pred)}** for {selected_crop} in {selected_region}.")
        else:
            st.warning(f"⚠️ Model divergence: XGBoost predicts **{get_risk_label(xgb_pred)}**, while Random Forest predicts **{get_risk_label(rf_pred)}**.")

        st.markdown("##### 🌾 Cultivated Land Allocation Summary")
        st.markdown(
            f"""
            - **Total Hectares Planted:** `{active_ha:,.1f} Ha`
            - **Regional Farmland Share:** `{land_info['land_share_pct']:.2f}%` of all crop land in {selected_region}
            - **Expected Grain Output:** `{est_prod_tons:,.1f} Metric Tons`
            """
        )

        st.progress(float(min(1.0, max(0.01, land_info['land_share_pct'] / 100.0))))
        st.caption(f"*{selected_crop} represents {land_info['land_share_pct']:.1f}% of total cultivated land in {selected_region}.*")

        st.markdown("---")

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
# RISK ANALYSIS PAGE (ALL REGIONS GUARANTEED VISIBLE)
# =====================================================

elif page == "📊 Risk Analysis":

    st.title("📊 Food Security Risk Analysis")
    st.write("Examine risk forecasts and land utilization dynamics across all Ethiopian administrative regions.")
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

    # 1. Bar Chart: Every single region displayed explicitly
    st.subheader(f"🌍 Predicted Risk Level by Region ({analysis_crop}, {analysis_year})")
    
    # Sort with all regions kept intact
    bar_df = dyn_risk_df.sort_values(by=['Predicted_Risk', 'Early_Warning_Score'], ascending=[False, False])
    
    fig_dyn_bar = px.bar(
        bar_df,
        x='Region',
        y='Predicted_Risk',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
        },
        hover_data={
            'Predicted_Risk': True,
            'Risk_Category': True,
            'Early_Warning_Score': ':.3f',
            'Area cultivated(Ha)': ':,.1f',
            'Production(kg)': ':,.0f'
        },
        labels={'Predicted_Risk': 'Risk Code (0=Low, 1=Med, 2=High)', 'Region': 'Administrative Region'},
        title=f"Complete Regional Distribution for {analysis_crop} ({analysis_year}) — {len(bar_df)} Regions Shown"
    )
    fig_dyn_bar.update_layout(
        yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=['Low (0)', 'Medium (1)', 'High (2)']),
        xaxis=dict(categoryorder='total descending'),
        height=480
    )
    st.plotly_chart(fig_dyn_bar, use_container_width=True)

    st.markdown("---")

    # 2. Scatter Plot: Fixed so 0-hectare or unplanted regions are NEVER omitted
    st.subheader("🔬 Regional Early Warning Score vs. Yield Growth Rate")
    st.caption("All administrative regions are guaranteed rendered. Marker size represents relative cultivated area (minimum visible size guaranteed).")

    fig_scatter = px.scatter(
        dyn_risk_df,
        x='Yield_Growth_Rate',
        y='Early_Warning_Score',
        color='Risk_Category',
        size='Plot_Bubble_Size',  # Non-zero safe size preventing dropped points
        text='Region',
        hover_name='Region',
        color_discrete_map={
            'Low Risk':    '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk':   '#dc3545'
        },
        hover_data={
            'Plot_Bubble_Size': False,
            'Area cultivated(Ha)': ':,.1f',
            'Production(kg)': ':,.0f',
            'Yield_Growth_Rate': ':.2f',
            'Early_Warning_Score': ':.3f'
        },
        labels={'Yield_Growth_Rate': 'Yield Growth Rate', 'Early_Warning_Score': 'Early Warning Score'},
        title=f"All {len(dyn_risk_df)} Regions: Early Warning Score vs. Yield Growth Rate ({analysis_crop}, {analysis_year})"
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(height=520)
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
# ETHIOPIA RISK MAP (TRUE 10 REGIONS GEOGRAPHICAL MAP)
# =====================================================

elif page == "🗺️ Ethiopia Risk Map":

    st.title("🗺️ Ethiopia Food Security Risk Map")
    st.write("Geospatial risk mapping dynamically calculated across all 10 Ethiopian administrative regions.")
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

    # Standardize region names for geo matching
    map_df['Norm_Region'] = map_df['Region'].apply(normalize_region_name)
    
    # Assign verified centroid coordinates for all regions
    def get_coords(name):
        norm = normalize_region_name(name)
        return REGION_COORDS.get(norm, (9.145, 40.489))

    map_df['lat'] = map_df['Region'].apply(lambda r: get_coords(r)[0])
    map_df['lon'] = map_df['Region'].apply(lambda r: get_coords(r)[1])

    # Check if geojson polygon matching is feasible
    geojson_feature_key = None
    if ethiopia_geojson and 'features' in ethiopia_geojson and len(ethiopia_geojson['features']) > 0:
        sample_props = ethiopia_geojson['features'][0].get('properties', {})
        # Detect the property holding region names (e.g., 'NAME_1', 'name', 'Region', 'ADM1_EN')
        for candidate in ['NAME_1', 'name', 'Region', 'ADM1_EN', 'reg_name', 'admin1Name']:
            if candidate in sample_props:
                geojson_feature_key = f"properties.{candidate}"
                break

    st.subheader(f"🗺️ Ethiopia Administrative Regions — Food Security Risk ({map_crop}, {map_year})")

    # Render Choropleth Map if GeoJSON has matching region boundaries
    rendered_choropleth = False
    if ethiopia_geojson and geojson_feature_key:
        try:
            # Map normalized names directly into geojson features for 100% boundary matching
            prop_name = geojson_feature_key.replace("properties.", "")
            for feature in ethiopia_geojson['features']:
                raw_val = feature.get('properties', {}).get(prop_name, '')
                feature['properties']['matched_region'] = normalize_region_name(raw_val)

            fig_map = px.choropleth(
                map_df,
                geojson=ethiopia_geojson,
                locations='Norm_Region',
                featureidkey="properties.matched_region",
                color='Risk_Category',
                color_discrete_map={
                    'Low Risk':    '#28a745',
                    'Medium Risk': '#fd7e14',
                    'High Risk':   '#dc3545'
                },
                hover_name='Region',
                hover_data={
                    'Norm_Region': False,
                    'Risk_Category': True,
                    'Early_Warning_Score': ':.3f',
                    'Area cultivated(Ha)': ':,.1f',
                    'Production(kg)': ':,.0f'
                },
                title=f"Ethiopia Regional Risk Map — {map_crop} ({map_year}) [Choropleth Boundary View]"
            )
            fig_map.update_geos(
                fitbounds="locations",
                visible=False
            )
            fig_map.update_layout(
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                height=650,
                legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=1)
            )
            st.plotly_chart(fig_map, use_container_width=True)
            rendered_choropleth = True
        except Exception:
            rendered_choropleth = False

    # Geospatial regional map overlay showing all 10 regions together
    if not rendered_choropleth:
        fig_scatter_map = px.scatter_geo(
            map_df,
            lat='lat',
            lon='lon',
            color='Risk_Category',
            color_discrete_map={
                'Low Risk':    '#28a745',
                'Medium Risk': '#fd7e14',
                'High Risk':   '#dc3545'
            },
            size='Plot_Bubble_Size',
            hover_name='Region',
            text='Region',
            hover_data={
                'Plot_Bubble_Size': False,
                'Risk_Category': True,
                'Early_Warning_Score': ':.3f',
                'Area cultivated(Ha)': ':,.1f',
                'Production(kg)': ':,.0f',
                'lat': False,
                'lon': False
            },
            title=f"Ethiopia 10 Administrative Regions Risk Map — {map_crop} ({map_year})"
        )

        fig_scatter_map.update_traces(
            textposition='top center',
            textfont=dict(size=12, color='#111111', family='Arial Black'),
            marker=dict(opacity=0.9, line=dict(width=1.5, color='white'))
        )

        fig_scatter_map.update_geos(
            visible=True,
            resolution=50,
            scope="africa",
            showcoastlines=True,  coastlinecolor="#777",
            showland=True,        landcolor="#f4f1ea",
            showocean=True,       oceancolor="#dbe9f4",
            showlakes=True,       lakecolor="#dbe9f4",
            showrivers=True,      rivercolor="#bed6ed",
            showsubunits=True,    subunitcolor="#b0b0b0",
            showcountries=True,   countrycolor="#444444",
            countrywidth=1.5,
            lonaxis_range=[32.5, 48.2],
            lataxis_range=[ 3.2, 15.2],
            center=dict(lat=9.145, lon=40.489)
        )

        fig_scatter_map.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter_map, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Administrative Regions Risk & Cultivated Land Table")
    st.dataframe(
        map_df[['Region', 'Risk_Label', 'Area cultivated(Ha)', 'Production(kg)', 'Early_Warning_Score', 'Yield_Growth_Rate', 'Area_Efficiency']]
        .sort_values("Area cultivated(Ha)", ascending=False).reset_index(drop=True),
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
2. The system retrieves historical and predicted agronomic indicators.
3. Machine learning models (**XGBoost** and **Random Forest**) predict the food security risk category.
4. Explanations (SHAP) and regional administrative maps identify vulnerable zones.

**Administrative Regions Covered:**
- Tigray, Afar, Amhara, Oromia, Somali, Benishangul-Gumuz, SNNPR, Gambela, Harari, Dire Dawa.

**Risk Levels:**
- 🟢 **Low Risk** — Stable food security with favorable yield potential.
- 🟡 **Medium Risk** — Emerging stress requiring close monitoring.
- 🔴 **High Risk** — Severe deficit requiring emergency interventions.

**Version:** 2.0  
**Country:** 🇪🇹 Ethiopia
""")
