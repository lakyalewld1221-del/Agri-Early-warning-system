# Agricultural Early Warning System

**Explainable AI-based food security risk mapping for Ethiopia**

An interactive Streamlit application that classifies the food security risk of Ethiopian regions and crops as **Low**, **Medium** or **High**, explains each prediction with SHAP, and shows the results on a regional risk map.

---

## Overview

Ethiopia's food security depends heavily on rain-fed agriculture, so yield shortfalls can quickly become food crises. Risk is often recognised only after harvests have already failed, national averages hide regional differences, and many machine-learning tools give no reasons for their predictions.

This project addresses these gaps with:

- a **risk classifier** (XGBoost and Random Forest) trained on regional crop area, production and yield data;
- **explainable predictions** using SHAP and plain-language reasons with crop-specific recommendations;
- a **choropleth risk map** of Ethiopian regions;
- an **interactive dashboard** for exploring predictions, trends and the dataset.

## Data

| Item | Description |
|---|---|
| Coverage | 10 regions, 7 cereal crops (teff, barley, wheat, maize, sorghum, millet, oats) |
| Period | 1996–2022 (2003 is missing), 26 years |
| Records | 1,820 region–crop–year observations (1,807 in the labelled dataset) |
| Variables | Cultivated area, production, yield, and engineered indicators |
| Target | `Risk_Level`: 0 = Low, 1 = Medium, 2 = High |

## Features

The models use 15 input features: region and crop codes, year, cultivated area, production, yield growth rate, production growth rate, area efficiency, regional and crop average yield, yield anomaly, rolling yield trend, yield stability, production-area ratio and an early warning score.

## Models and results

| Metric (test set, n = 362) | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 99.17% | 99.17% |
| Macro F1 | 0.990 | 0.990 |
| ROC-AUC (one-vs-rest) | 0.9995 | 0.9983 |

**Important note:** the risk label is derived mainly from yield anomaly and yield growth, which are also model inputs. The high accuracy therefore shows that the models learn the labelling rule, not that they forecast future risk. Using an independent outcome (for example IPC or FEWS NET classifications) is the main direction for future work.

## Dashboard pages

1. **Home**: overview and instructions
2. **Prediction**: choose a region, crop and year; compare both models
3. **Dataset**: browse the labelled data
4. **Explainable AI**: global and local SHAP explanations
5. **Risk Analysis**: risk trend over time for a region and crop
6. **Ethiopia Risk Map**: regional choropleth map and ranking table
7. **About**: method, models and risk levels

## Repository structure

```
├── app.py                          # Streamlit application
├── labeled_agri_risk_data.csv      # Labelled dataset with engineered features
├── X_test.csv                      # Test features (used for SHAP)
├── ethiopia_risk_map.csv           # Risk score and category per region
├── ethiopia_regions.geojson        # Region boundaries for the map
├── xgboost_model.pkl               # Trained XGBoost model
├── random_forest_model.pkl         # Trained Random Forest model
└── requirements.txt                # Python dependencies
```

## Installation and usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Suggested `requirements.txt` (pin the versions you trained with):

```
streamlit
pandas
numpy
scikit-learn==1.6.1
xgboost
shap
plotly
matplotlib
seaborn
joblib
```

> The saved models were created with scikit-learn 1.6.1. Using a different version may cause warnings or loading errors. Make sure the model filenames in `app.py` match the files in the repository.

## Technology

Python · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Plotly · Matplotlib · Seaborn · Streamlit · GeoJSON · joblib

## Limitations

- The model uses yield, production and area only; it does not include rainfall, temperature, vegetation indices, prices or conflict.
- Data are at regional level, so zone and woreda patterns are not visible.
- Predictions use historical values for the selected region, crop and year, not live data.
- The target is defined from yield, so it measures yield shortfall rather than direct food insecurity outcomes.

## Future work

- Add climate and satellite data (CHIRPS rainfall, NDVI, soil moisture) and market prices
- Use an independent target such as IPC or FEWS NET classifications
- Move to zone or woreda level and forecast future seasons
- Add live data updates, local-language support (Amharic, Afaan Oromo) and alerts

## Author

**[Your name]**, Data Science, Bahir Dar University
Contact: [your email]

## License

[Choose a license, e.g. MIT]
