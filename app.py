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
from pathlib import Path
from urllib.request import Request, urlopen

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
    "Maize":   "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_corn_on_the_cob.jpg/320px-Simple_corn_on_the_cob.jpg",
    "Wheat":   "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/WheatBread_transparency.png/320px-WheatBread_transparency.png",
    "Teff":    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eragrostis_tef_-_teff_%28aka%29.jpg/320px-Eragrostis_tef_-_teff_%28aka%29.jpg",
    "Barley":  "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Barleyleaf.jpg/320px-Barleyleaf.jpg",
    "Sorghum": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Sorghum_crop.jpg/320px-Sorghum_crop.jpg",
    "Millet":  "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Panicum_miliaceum_USDA.jpg/320px-Panicum_miliaceum_USDA.jpg",
    "Oats":    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg/320px-Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg",
}

DEFAULT_CROP_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Farmland_at_sunset.jpg/320px-Farmland_at_sunset.jpg"

# =====================================================
# CROP LAND/ACRES IMAGES — fields filled with each crop
# =====================================================

CROP_LAND_IMAGES = {
    # Maize — a field full of maize/corn plants
    "Maize":   "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Cornfield_banner.jpg/640px-Cornfield_banner.jpg",
    # Wheat — a golden wheat field
    "Wheat":   "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Wheat_field_in_Dorset%2C_England_-_July_2009.jpg/640px-Wheat_field_in_Dorset%2C_England_-_July_2009.jpg",
    # Teff — a teff crop field
    "Teff":    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eragrostis_tef_-_teff_%28aka%29.jpg/640px-Eragrostis_tef_-_teff_%28aka%29.jpg",
    # Barley — a barley field
    "Barley":  "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Barley_field.jpg/640px-Barley_field.jpg",
    # Sorghum — a sorghum crop field
    "Sorghum": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Sorghum_crop_Andhra_Pradesh_India.jpg/640px-Sorghum_crop_Andhra_Pradesh_India.jpg",
    # Millet — a millet field
    "Millet":  "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Panicum_miliaceum_USDA.jpg/640px-Panicum_miliaceum_USDA.jpg",
    # Oats — an oats field
    "Oats":    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg/640px-Avena_sativa_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-016.jpg",
}

DEFAULT_LAND_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Farmland_at_sunset.jpg/640px-Farmland_at_sunset.jpg"

def _normalize_crop_name(crop_name):
    """Normalize crop labels so names like 'maize ' and 'MAIZE' match."""
    return str(crop_name).strip().lower()

def get_crop_land_image(crop_name):
    """Return a field-image URL, matching crop names case-insensitively."""
    crop_lower = _normalize_crop_name(crop_name)
    for key, url in CROP_LAND_IMAGES.items():
        key_lower = _normalize_crop_name(key)
        if crop_lower == key_lower or key_lower in crop_lower or crop_lower in key_lower:
            return url
    return DEFAULT_LAND_IMAGE

@st.cache_data(show_spinner=False, ttl=3600)
def _download_image(url):
    """Download and cache image bytes; return None if the URL is unavailable."""
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=6) as response:
            content_type = response.headers.get("Content-Type", "")
            image_bytes = response.read()
            if response.status == 200 and "image" in content_type.lower() and image_bytes:
                return image_bytes
    except Exception:
        return None
    return None

def _local_crop_field_path(crop_name):
    """Find the bundled local field image for the selected crop."""
    crop = _normalize_crop_name(crop_name)
    image_map = {
        "maize": "maize_field.png",
        "wheat": "wheat_field.png",
        "teff": "teff_field.png",
        "barley": "barley_field.png",
        "sorghum": "sorghum_field.png",
        "millet": "millet_field.png",
        "oats": "oats_field.png",
    }
    base_dir = Path(__file__).resolve().parent
    for crop_key, filename in image_map.items():
        if crop_key in crop:
            path = base_dir / filename
            if path.is_file():
                return path
    fallback = base_dir / "maize_field.png"
    return fallback if fallback.is_file() else None


def _make_valid_fallback_png(crop_name):
    """Create a real PNG fallback using Pillow (never pass SVG bytes to st.image)."""
    from PIL import Image, ImageDraw
    crop = str(crop_name or "Crop").strip()[:35]
    img = Image.new("RGB", (900, 420), (188, 225, 242))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 175, 900, 420), fill=(133, 96, 49))
    draw.polygon([(0, 200), (150, 125), (280, 205), (430, 115), (610, 200), (760, 140), (900, 190), (900, 250), (0, 250)], fill=(120, 164, 100))
    # Draw repeated crop rows and green plants.
    for row in range(5):
        y = 235 + row * 34
        for col in range(18):
            x = 20 + col * 50 + (row % 2) * 20
            draw.line((x, y, x, y + 40), fill=(42, 100, 36), width=4)
            draw.ellipse((x-15, y+2, x+2, y+13), fill=(58, 139, 49))
            draw.ellipse((x, y+10, x+17, y+21), fill=(70, 151, 55))
    draw.rounded_rectangle((18, 15, 882, 75), radius=12, fill=(255, 255, 255))
    draw.text((450, 35), f"{crop} field illustration", fill=(31, 83, 39), anchor="mm")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def show_crop_land_image(crop_name, caption, use_container_width=True):
    """Display the selected crop's bundled PNG; use a valid generated PNG as fallback."""
    local_path = _local_crop_field_path(crop_name)
    if local_path is not None:
        try:
            # Verify the file is a readable image before asking Streamlit to display it.
            from PIL import Image
            with Image.open(local_path) as image:
                image.verify()
            st.image(str(local_path), caption=caption, use_container_width=use_container_width)
            return
        except Exception:
            pass

    # PNG bytes are generated locally, so this works even if network images are unavailable.
    fallback_png = _make_valid_fallback_png(crop_name)
    st.image(fallback_png, caption=f"{caption} (illustration fallback)",
             use_container_width=use_container_width)


def get_crop_image(crop_name):
    """Return close-up image of the crop."""
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

    # --- Reasons (same for all crops) ---
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

    # --- Crop-specific recommendations ---
    crop_lower = crop.lower()

    # Base recommendations per risk level
    if risk_code == 2:  # High Risk
        base_rec = [
            "🆘 **Immediate action required** — alert local agricultural authorities.",
            "🏦 Activate emergency food reserves and support programs.",
            "📋 Conduct urgent field assessments to verify crop conditions.",
            "🤝 Coordinate with NGOs and government agencies for humanitarian support.",
        ]
    elif risk_code == 1:  # Medium Risk
        base_rec = [
            "⚠️ **Monitor closely** — situation may worsen without intervention.",
            "📊 Strengthen early warning monitoring for the coming season.",
            "📦 Build buffer food stocks as a precautionary measure.",
            "🧑‍🌾 Provide farmers with training on improved agricultural practices.",
        ]
    else:  # Low Risk
        base_rec = [
            "✅ **Conditions are stable** — maintain current agricultural practices.",
            "📈 Continue investing in yield improvement programs.",
            "📊 Keep monitoring seasonal trends to detect early changes.",
        ]

    # Crop-specific additions
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

    elif "teff" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌿 Use improved teff varieties like QUNCHO or MAGNA with higher yields.",
                "💧 Avoid waterlogging — teff is highly sensitive to poor drainage.",
                "🌱 Apply DAP and urea fertilizers at recommended rates for the region.",
                "🚜 Use row planting instead of broadcasting to improve stand establishment.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌿 Practice row planting to improve weed management and yields.",
                "💧 Ensure proper land leveling to avoid waterlogging and runoff.",
                "🌱 Apply organic compost to improve soil structure for teff production.",
                "📊 Monitor rainfall patterns closely — teff needs consistent moisture.",
            ]
        else:
            crop_rec = [
                "🌿 Promote teff as a cash crop — high international demand exists.",
                "📦 Invest in improved threshing equipment to reduce post-harvest losses.",
                "🌱 Maintain soil fertility through compost and balanced fertilization.",
            ]

    elif "sorghum" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌾 Sorghum is drought-tolerant — prioritize it over other crops in dry areas.",
                "💧 Apply limited supplemental irrigation during grain-filling stage.",
                "🐛 Monitor for sorghum midge and stem borer — apply timely control.",
                "🌱 Use improved varieties like GAMBELLA-1 or MELKAM for better yield.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌾 Practice early planting to maximize use of early rains.",
                "🌿 Intercrop sorghum with cowpea to improve soil fertility.",
                "💧 Use tied ridges to conserve soil moisture in low-rainfall areas.",
                "📊 Track bird damage risks — deploy bird scarers during grain maturity.",
            ]
        else:
            crop_rec = [
                "🌾 Expand sorghum production as a food security buffer crop.",
                "📦 Promote sorghum-based food products to increase market demand.",
                "🌱 Use sorghum residues as animal feed and soil mulch.",
            ]

    elif "barley" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌾 Use improved barley varieties tolerant to cold and drought.",
                "💧 Supplement rainfall with irrigation during booting stage.",
                "🍄 Apply fungicides against barley net blotch and scald diseases.",
                "🌡️ Plant early to avoid frost damage at high altitudes.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌾 Apply balanced NPK fertilizers to maintain yield levels.",
                "🌿 Rotate barley with legumes to break pest and disease cycles.",
                "💧 Monitor soil moisture during grain filling — critical for yield.",
                "📊 Scout for aphids and other sucking pests regularly.",
            ]
        else:
            crop_rec = [
                "🌾 Promote malting barley varieties for brewery market linkages.",
                "📈 Expand barley in highland areas with stable rainfall.",
                "🌱 Use barley straw as livestock feed and soil organic matter.",
            ]

    elif "coffee" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "☕ Apply shade management to protect coffee from temperature extremes.",
                "🍄 Monitor for coffee berry disease and leaf rust — apply fungicides.",
                "💧 Provide supplemental irrigation during dry spells in flowering period.",
                "🌿 Rehabilitate old unproductive coffee trees by stumping.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "☕ Apply organic mulch under coffee trees to conserve soil moisture.",
                "🌿 Prune coffee trees to improve air circulation and reduce disease.",
                "💧 Ensure drainage in coffee fields to prevent root rot.",
                "📦 Improve post-harvest processing to maintain coffee quality.",
            ]
        else:
            crop_rec = [
                "☕ Invest in certification (organic, fair trade) for premium prices.",
                "📈 Expand coffee gardens in suitable agroforestry systems.",
                "🌱 Plant shade trees to improve coffee microclimate and biodiversity.",
            ]

    elif "sesame" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌻 Use drought-tolerant sesame varieties suited for lowland areas.",
                "💧 Avoid excessive irrigation — sesame is sensitive to waterlogging.",
                "🐛 Monitor for sesame webworm and apply control measures early.",
                "🌱 Apply phosphorus fertilizer to improve pod set in stressed conditions.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌻 Practice timely weeding — sesame is highly competitive with weeds early on.",
                "💧 Use furrow irrigation carefully to avoid stem rot.",
                "📦 Harvest sesame at the right maturity to avoid shattering losses.",
                "🌿 Rotate with cereals to break sesame disease cycles.",
            ]
        else:
            crop_rec = [
                "🌻 Expand sesame production — strong export market from Ethiopia.",
                "📦 Invest in cleaning and grading equipment for export quality.",
                "📈 Link farmers to export cooperatives for better prices.",
            ]

    elif "chickpea" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🫘 Use improved chickpea varieties resistant to botrytis gray mold.",
                "💧 Avoid waterlogged soils — chickpea is very susceptible to root rot.",
                "🌿 Apply seed treatment with rhizobium inoculant to fix nitrogen.",
                "🌡️ Plant early to avoid late-season heat stress during pod filling.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🫘 Rotate chickpea with cereals to reduce soil-borne diseases.",
                "💧 Apply irrigation carefully — only during critical flowering stage.",
                "🌱 Use raised beds to improve drainage and reduce waterlogging risk.",
                "📊 Scout for leaf miner and pod borer regularly.",
            ]
        else:
            crop_rec = [
                "🫘 Promote desi and kabuli chickpea for domestic and export markets.",
                "📦 Improve storage with hermetic bags to maintain seed quality.",
                "🌿 Use chickpea as a green manure crop to improve soil fertility.",
            ]

    elif "millet" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌾 Millet is highly drought tolerant — prioritize in arid zones.",
                "💧 Apply only minimal irrigation — millet thrives in low rainfall.",
                "🌱 Use improved pearl or finger millet varieties for higher yield.",
                "🐛 Monitor for downy mildew and apply seed treatment before planting.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌾 Plant millet early to make best use of onset rains.",
                "🌿 Intercrop with cowpea or groundnut for food security.",
                "💧 Use conservation tillage to preserve soil moisture.",
                "📦 Promote millet flour for local food products to boost demand.",
            ]
        else:
            crop_rec = [
                "🌾 Expand millet in marginal lands as food security crop.",
                "📈 Explore value addition — millet flour, porridge, and beer.",
                "🌱 Maintain soil fertility with compost and minimal tillage.",
            ]

    elif "potato" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🥔 Use certified disease-free seed potatoes to prevent blight.",
                "🍄 Apply fungicides against late blight — most destructive potato disease.",
                "💧 Ensure consistent irrigation — potato yields drop sharply under drought.",
                "🌡️ Avoid planting during extreme heat — cool temperatures favor potato.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🥔 Hill up potato plants to prevent greening and improve tuber set.",
                "💧 Irrigate regularly during tuber bulking stage for high yields.",
                "🍄 Scout weekly for early and late blight symptoms.",
                "📦 Harvest at proper maturity and store in cool, dark conditions.",
            ]
        else:
            crop_rec = [
                "🥔 Expand potato as a high-value food and income crop.",
                "📦 Invest in cold storage infrastructure for surplus potato.",
                "🌱 Promote improved potato varieties with high yield and disease resistance.",
            ]

    elif "bean" in crop_lower or "faba" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🫘 Use improved bean varieties tolerant to drought and rust.",
                "💧 Apply irrigation during flowering and pod filling stages.",
                "🍄 Apply fungicides against bean rust and anthracnose diseases.",
                "🌿 Apply rhizobium inoculant to improve nitrogen fixation.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🫘 Ensure proper spacing to improve air circulation and reduce disease.",
                "💧 Avoid overhead irrigation — use drip or furrow to reduce foliar disease.",
                "🌱 Rotate beans with cereals to break disease and pest cycles.",
                "📊 Monitor for bean fly — apply seed dressing before planting.",
            ]
        else:
            crop_rec = [
                "🫘 Promote bean production for both food security and soil health.",
                "📈 Link farmers to markets for export of dried beans.",
                "🌿 Use bean residues as organic matter to improve soil fertility.",
            ]

    elif "lentil" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🫘 Use drought-tolerant lentil varieties adapted to highland conditions.",
                "💧 Apply supplemental irrigation only during severe dry spells.",
                "🍄 Monitor for lentil wilt — use resistant varieties where available.",
                "🌡️ Plant early to avoid terminal drought during pod filling.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🫘 Apply rhizobium seed inoculant to enhance nitrogen fixation.",
                "🌿 Rotate lentils with cereals to reduce soil-borne diseases.",
                "💧 Ensure good drainage — lentils are sensitive to waterlogging.",
                "📊 Monitor for aphids and thrips during flowering.",
            ]
        else:
            crop_rec = [
                "🫘 Expand lentil as a high-protein food security crop.",
                "📦 Promote lentil export — Ethiopia is a major lentil producer.",
                "🌱 Use lentil as a rotation crop to improve soil fertility.",
            ]

    elif "rice" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌾 Use drought-tolerant upland rice varieties for rain-fed areas.",
                "💧 Maintain adequate water depth in paddy fields during tillering.",
                "🍄 Monitor for rice blast disease — apply fungicides at boot stage.",
                "🌿 Apply split nitrogen doses to reduce lodging and improve grain yield.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌾 Practice alternate wetting and drying to reduce water use.",
                "💧 Ensure irrigation channels are well maintained before planting season.",
                "🐛 Scout for stem borer — use pheromone traps for early detection.",
                "🌱 Apply zinc sulfate to address zinc deficiency common in rice soils.",
            ]
        else:
            crop_rec = [
                "🌾 Expand irrigated rice in lowland areas along rivers.",
                "📈 Invest in rice milling infrastructure to add value locally.",
                "🌿 Use rice straw as compost or animal feed to reduce burning.",
            ]

    elif "enset" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌿 Enset is drought tolerant — maintain existing plants and avoid uprooting.",
                "🍄 Monitor for enset xanthomonas wilt — remove and destroy infected plants.",
                "🌱 Plant disease-free suckers from certified sources only.",
                "👨‍🌾 Strengthen community-based enset management practices.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌿 Maintain adequate spacing to reduce disease spread in enset gardens.",
                "💧 Apply mulch around enset plants to conserve soil moisture.",
                "🌱 Propagate improved enset varieties for higher kocho yield.",
                "📊 Monitor plant health quarterly for early disease detection.",
            ]
        else:
            crop_rec = [
                "🌿 Expand enset cultivation — it is a key food security crop in southern Ethiopia.",
                "📦 Invest in enset processing (kocho, bulla) for nutrition and income.",
                "🌱 Promote agroforestry with enset as an understory crop.",
            ]

    elif "sunflower" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🌻 Use drought-tolerant sunflower hybrids for water-stressed areas.",
                "💧 Apply irrigation at flowering and seed-filling stages only.",
                "🐛 Monitor for sunflower head moth and apply control measures.",
                "🌡️ Avoid late planting — sunflower needs full season to mature.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🌻 Apply boron micronutrient to improve seed set in sunflower.",
                "💧 Avoid waterlogging during early vegetative growth.",
                "🌿 Rotate sunflower with cereals to manage Sclerotinia stem rot.",
                "📦 Harvest at proper maturity — test seeds for oil content.",
            ]
        else:
            crop_rec = [
                "🌻 Expand sunflower for edible oil production and income.",
                "📈 Link farmers to oil processing cooperatives for value addition.",
                "🌱 Use sunflower meal as livestock feed after oil extraction.",
            ]

    elif "sugar" in crop_lower or "cane" in crop_lower:
        if risk_code == 2:
            crop_rec = [
                "🎋 Maintain irrigation infrastructure — sugarcane is very water demanding.",
                "💧 Apply full irrigation during elongation and maturation stages.",
                "🐛 Monitor for sugarcane borer — apply biological control agents.",
                "🌱 Replant with high-yielding disease-resistant varieties.",
            ]
        elif risk_code == 1:
            crop_rec = [
                "🎋 Apply split nitrogen to reduce lodging and improve cane yield.",
                "💧 Monitor soil moisture — sugarcane needs consistent water supply.",
                "🌿 Practice trash mulching to conserve moisture and control weeds.",
                "📊 Monitor ratoon crop health for early detection of disease.",
            ]
        else:
            crop_rec = [
                "🎋 Expand sugarcane in irrigated lowland areas.",
                "📈 Invest in sugar processing to increase value-added production.",
                "🌱 Use press mud from sugar mills as organic fertilizer.",
            ]

    else:
        # Generic crop-specific recommendations
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


    # Show crop image and land image when crop is selected
    st.markdown("---")
    img_col, land_col, info_col = st.columns([1, 1, 2])
    with img_col:
        crop_img_url = get_crop_image(selected_crop)
        st.image(crop_img_url, caption=f"🌾 {selected_crop}", use_container_width=True)
    with land_col:
        show_crop_land_image(
            selected_crop,
            caption=(f"🌱 {selected_crop} Field (illustrative acreage visual; entered area: "
                     f"shown in the panel)"),
            use_container_width=True
        )
    with info_col:
        st.markdown(f"### Selected Input")
        st.markdown(f"- 🌍 **Region:** {selected_region}")
        st.markdown(f"- 🌾 **Crop Type:** {selected_crop}")
        st.markdown(f"- 📅 **Year:** {selected_year}")
        st.markdown("#### 🌱 Land Area (Acres)")
        selected_acres = st.number_input(
            "Enter the cultivated land area in acres",
            min_value=0.1,
            max_value=1000000.0,
            value=50.0 if str(selected_crop).strip().lower() == "maize" else 1.0,
            step=1.0,
            key=f"land_acres_{selected_crop}_{selected_region}_{selected_year}",
            help="Enter the actual acreage for the selected field. The image is illustrative and does not measure land area."
        )
        st.metric("Selected Field Area", f"{selected_acres:,.1f} acres")
        st.caption(f"Equivalent area: {selected_acres * 0.404686:,.2f} hectares")
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

    st.markdown("---")

    # Gallery of the 7 crop types on the Prediction page
    st.markdown("### 🌾 Crop Field Gallery")
    crop_names = ["Maize", "Wheat", "Teff", "Barley", "Sorghum", "Millet", "Oats"]
    gallery_cols = st.columns(4)
    for i, crop_name in enumerate(crop_names):
        with gallery_cols[i % 4]:
            show_crop_land_image(
                crop_name,
                caption=f"🌱 {crop_name} Field",
                use_container_width=True
            )

    # Show prediction results only after the button is clicked
    if st.button("🔍 Predict Risk", use_container_width=True):
        xgb_pred, rf_pred, input_data = predict_risk(
            selected_region, selected_crop, selected_year
        )
        st.session_state["region"] = selected_region
        st.session_state["crop"] = selected_crop
        st.session_state["year"] = selected_year
        st.session_state["xgb_pred"] = xgb_pred
        st.session_state["rf_pred"] = rf_pred
        st.session_state["input_data"] = input_data
        st.session_state["predicted"] = True

        st.subheader("🎯 Prediction Results")
        st.markdown(
            f"**Region:** {selected_region} | **Crop:** {selected_crop} | **Year:** {selected_year}"
        )
        st.markdown("---")
        res_img_col, res_results_col = st.columns([1, 2])
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
            st.success(
                f"✅ Both models agree: **{get_risk_label(xgb_pred)}** for "
                f"{selected_region} - {selected_crop} in {selected_year}."
            )
        else:
            st.warning(
                f"⚠️ Models disagree. XGBoost: **{get_risk_label(xgb_pred)}** | "
                f"Random Forest: **{get_risk_label(rf_pred)}**."
            )
    elif not st.session_state.get("predicted", False):
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
    st.caption("Red bars increase risk | Green bars decrease risk | Sorted by importance")
    shap_single = explainer.shap_values(input_data)

    # Extract SHAP values safely
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

    # Build dataframe sorted by absolute SHAP value
    shap_df = pd.DataFrame({
        "Feature":    input_data.columns.tolist(),
        "SHAP Value": sv
    }).sort_values("SHAP Value", key=abs, ascending=True)

    # Color: red = increases risk, green = decreases risk
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

    # Add value labels on each bar
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

    # Add vertical line at 0
    ax_bar.axvline(x=0, color="black", linewidth=0.8, linestyle="--")

    # Labels and styling
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

    # Custom legend
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

    # Summary table below the chart
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

    # ── Legend ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; gap:20px; margin-bottom:16px;'>
        <span style='background:#28a745;color:white;padding:6px 16px;border-radius:8px;font-weight:700;'>🟢 Low Risk</span>
        <span style='background:#fd7e14;color:white;padding:6px 16px;border-radius:8px;font-weight:700;'>🟡 Medium Risk</span>
        <span style='background:#dc3545;color:white;padding:6px 16px;border-radius:8px;font-weight:700;'>🔴 High Risk</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Ethiopian region centre coordinates ─────────────────────────────────
    # 10 regions + 2 city administrations
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

    # Build a combined dataframe with risk info + coordinates
    def get_region_coords(region_name):
        # exact match first
        if region_name in REGION_COORDS:
            return REGION_COORDS[region_name]
        # partial match
        for key, coords in REGION_COORDS.items():
            if key.lower() in region_name.lower() or region_name.lower() in key.lower():
                return coords
        return (9.145, 40.489)  # Ethiopia centre fallback

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
    map_df['size'] = 18  # marker size

    # ── Full Ethiopia scatter map ────────────────────────────────────────────
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

    # ── Individual region cards (3 per row) ──────────────────────────────────
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
                # Mini scatter map zoomed to this region
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

                # Risk badge
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

    # ── Highlight selected region ─────────────────────────────────────────────
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



    # ── Full Ethiopia choropleth map ─────────────────────────────────────────
    st.subheader("🗺️ Ethiopia — Full Country Risk Map")
    st.caption("Hover over a region to see its name and risk level.")

    fig_ethiopia = px.choropleth(
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
        hover_data={'Region': True, 'Risk_Category': True, 'Predicted_Risk': ':.2f'},
        title="Ethiopia Food Security Risk by Region",
        labels={'Risk_Category': 'Risk Level', 'Predicted_Risk': 'Risk Score'}
    )
    fig_ethiopia.update_geos(
        fitbounds="locations",
        visible=False,
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="#f5f5f0",
        showocean=True,
        oceancolor="#cce5ff",
        showlakes=True,
        lakecolor="#cce5ff",
        showrivers=True,
        rivercolor="#aad4f5",
    )
    fig_ethiopia.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.01,
            xanchor="right",  x=1
        )
    )
    st.plotly_chart(fig_ethiopia, use_container_width=True)

    st.markdown("---")

    # ── Per-region individual mini maps ─────────────────────────────────────
    st.subheader("📍 Individual Region Maps")
    st.caption("Each card shows the food security risk for one Ethiopian region.")

    # Build a colour lookup from risk_df
    region_color_map = {}
    region_risk_map  = {}
    for _, row in risk_df.iterrows():
        cat = row['Risk_Category']
        region_color_map[row['Region']] = (
            '#28a745' if 'Low'    in cat else
            '#fd7e14' if 'Medium' in cat else
            '#dc3545'
        )
        region_risk_map[row['Region']] = {
            'category': cat,
            'score':    row['Predicted_Risk']
        }

    all_regions = sorted(risk_df['Region'].unique())

    # Show 3 region cards per row
    cards_per_row = 3
    for row_start in range(0, len(all_regions), cards_per_row):
        cols = st.columns(cards_per_row)
        for col_idx, col in enumerate(cols):
            r_idx = row_start + col_idx
            if r_idx >= len(all_regions):
                break
            reg = all_regions[r_idx]
            info  = region_risk_map.get(reg, {'category': 'Unknown', 'score': 0})
            color = region_color_map.get(reg, '#6c757d')

            with col:
                # Mini choropleth for this single region
                region_row = risk_df[risk_df['Region'] == reg]
                fig_mini = px.choropleth(
                    region_row,
                    geojson=ethiopia_geojson,
                    featureidkey="properties.ADM1_EN",
                    locations='Region',
                    color='Risk_Category',
                    color_discrete_map={
                        'Low Risk':    '#28a745',
                        'Medium Risk': '#fd7e14',
                        'High Risk':   '#dc3545'
                    },
                    hover_data={'Region': True, 'Risk_Category': True},
                )
                fig_mini.update_geos(
                    fitbounds="locations",
                    visible=False,
                    showland=True,
                    landcolor="#f0f0e8",
                    showcoastlines=False,
                )
                fig_mini.update_layout(
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=200,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_mini, use_container_width=True,
                                key=f"mini_map_{reg}")

                # Risk badge below the mini map
                risk_code_r = (
                    0 if 'Low'    in info['category'] else
                    1 if 'Medium' in info['category'] else 2
                )
                st.markdown(
                    f"""<div style='background-color:{color}; padding:8px 12px;
                    border-radius:8px; text-align:center; margin-top:-10px;'>
                    <b style='color:white; font-size:14px;'>📍 {reg}</b><br>
                    <span style='color:white; font-size:13px;'>{get_risk_label(risk_code_r)}</span><br>
                    <span style='color:white; font-size:11px;'>Score: {info['score']:.2f}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
                st.markdown(" ")   # spacing

    st.markdown("---")

    # ── Highlight selected region if prediction was made ─────────────────────
    if 'predicted' in st.session_state:
        sel_region = st.session_state['region']
        st.subheader(f"📌 Your Selected Region: {sel_region}")

        region_risk = risk_df[risk_df['Region'] == sel_region]
        if not region_risk.empty:
            risk_cat  = region_risk['Risk_Category'].values[0]
            risk_val  = region_risk['Predicted_Risk'].values[0]
            risk_code = 0 if "Low" in risk_cat else (1 if "Medium" in risk_cat else 2)
            color     = get_risk_color(risk_code)

            hl_col1, hl_col2 = st.columns([1, 2])
            with hl_col1:
                # Larger map of selected region
                sel_row = risk_df[risk_df['Region'] == sel_region]
                fig_sel = px.choropleth(
                    sel_row,
                    geojson=ethiopia_geojson,
                    featureidkey="properties.ADM1_EN",
                    locations='Region',
                    color='Risk_Category',
                    color_discrete_map={
                        'Low Risk':    '#28a745',
                        'Medium Risk': '#fd7e14',
                        'High Risk':   '#dc3545'
                    },
                )
                fig_sel.update_geos(fitbounds="locations", visible=False,
                                    showland=True, landcolor="#f0f0e8")
                fig_sel.update_layout(
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=280, showlegend=False
                )
                st.plotly_chart(fig_sel, use_container_width=True,
                                key="selected_region_map")

            with hl_col2:
                st.markdown(
                    f"""<div style='background-color:{color}; padding:30px;
                    border-radius:12px; text-align:center;'>
                    <h2 style='color:white; margin:0;'>📍 {sel_region}</h2>
                    <h1 style='color:white; margin:10px 0 0 0;'>{get_risk_label(risk_code)}</h1>
                    <p style='color:white; margin:8px 0 0 0; font-size:18px;'>
                    Risk Score: {risk_val:.2f}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
        else:
            st.info(f"No risk map data available for {sel_region}.")

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
