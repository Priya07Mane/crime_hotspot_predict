import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pydeck as pdk
import json

# --- Configuration ---
MODEL_PATH = 'models/crime_hotspot_model_nogeo.pkl'
DATA_PATH = 'data/processed_crime_data.csv'
GEOJSON_PATH = 'karnataka.geojson'


def load_assets():
    """Load model, processed data, and geojson"""
    try:
        model_data = joblib.load(MODEL_PATH)
        processed_data = pd.read_csv(DATA_PATH)
        geojson_data = gpd.read_file(GEOJSON_PATH)
        return model_data, processed_data, geojson_data
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()


def classify_hotspots(df):
    """Classify districts as hotspots based on crime thresholds"""
    df['total_crimes'] = df['total_assault'] + df['total_harassment'] + df['total_rape']
    # Classify top 25% as hotspots (adjust threshold as needed)
    threshold = df['total_crimes'].quantile(0.75)
    df['is_hotspot'] = (df['total_crimes'] >= threshold).astype(int)
    # Calculate probability (normalized 0-1)
    df['probability'] = df['total_crimes'] / df['total_crimes'].max()
    return df


def safe_divide(numerator, denominator):
    """Handle division by zero"""
    return numerator / denominator if denominator > 0 else 0

# Debug check
def create_interactive_map(processed_data, geojson_data):
    geojson_dict = json.loads(geojson_data.to_json())
    name_mapping = {
    'Bagalkot': 'Bagalkot',
    'Bangalore Rural': 'Bengaluru District',
    'Bangalore Urban': 'Bengaluru City',
    'Belagavi': 'Belagavi District',
    'Bellary': 'Ballari',
    'Bidar': 'Bidar',
    'Vijayapura': 'Vijayapura',
    'Chamarajanagar': 'Chamarajnagar',
    'Chikballapura': 'Chikkaballapura',
    'Chikkamagaluru': 'Chikkamagaluru',
    'Chitradurga': 'Chitradurga',
    'Dakshina Kannada': 'Dakshina Kannada',
    'Davanagere': 'Davanagere',
    'Dharwad': 'Dharwad',
    'Gadag': 'Gadag',
    'Kalaburagi': 'Kalaburgi',
    'Hassan': 'Hassan',
    'Haveri': 'Haveri',
    'Kodagu': 'Kodagu',
    'Kolar': 'Kolar',
    'Koppal': 'Koppal',
    'Mandya': 'Mandya',
    'Mysuru': 'Mysuru District',
    'Raichur': 'Raichur',
    'Ramanagara': 'Ramanagara',
    'Shivamogga': 'Shimoga',
    'Tumakuru': 'Tumakuru',
    'Udupi': 'Udupi',
    'Uttara Kannada': 'Uttara Kannada',
    'Vijayanagara': 'Vijayanagara',
    'Yadgir': 'Yadgiri'
}

    processed_data = classify_hotspots(processed_data)
    for feature in geojson_dict["features"]:
        district_name = feature['properties']['NAME_2']
        match = processed_data[
            processed_data['district'].str.contains(district_name, case=False, regex=False)
        ]
        
        if not match.empty:
            feature['properties'].update({
                'is_hotspot': int(match.iloc[0]['is_hotspot']),
                'probability': float(match.iloc[0]['probability'])
            })
        # geo_district = feature['properties']['NAME_2']
        # district_name = name_mapping.get(geo_district, geo_district)
        # district_data = processed_data[processed_data['district'] == district_name]
        
        # if not district_data.empty:
        #     # Update GeoJSON properties with data from processed_data
        #     feature['properties'].update({
        #         'district': district_name,
        #         'total_assault': int(district_data['total_assault'].iloc[0]),
        #         'total_harassment': int(district_data['total_harassment'].iloc[0]),
        #         'is_hotspot': int(district_data['is_hotspot'].iloc[0]) if 'is_hotspot' in district_data else 0
        #     })
    
    m = folium.Map(location=[15.3173, 75.7139], zoom_start=6)
    
    folium.GeoJson(
        geojson_dict,
        style_function=lambda x: {
            'fillColor': '#ff0000' if  x['properties'].get('is_hotspot') else '#00ff00',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME_2', 'is_hotspot', 'probability'],
            aliases=['District:', 'Hotspot:', 'Risk:'],
            style=("background: white; font-family: sans-serif;")
        )
    ).add_to(m)
    return m

def make_prediction(input_data, model_data):
    """Make prediction using loaded model"""
    try:
        features = np.array([[
            input_data['total_assault'],
            input_data['total_harassment'],
            input_data['total_rape'],
            input_data['assault_victim_ratio'],
            input_data['harassment_victim_ratio'],
            input_data['public_transport_risk'],
            input_data['workplace_risk']
        ]])
        
        scaled = model_data['scaler'].transform(features)
        prediction = model_data['model'].predict(scaled)[0]
        probability = model_data['model'].predict_proba(scaled)[0][1]
        
        return {
            'is_hotspot': prediction,
            'probability': probability,
            'features': input_data
        }
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None
    
# --- Page Setup ---
st.set_page_config(
    page_title="Crime Hotspot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Karnataka Crime Hotspot Analysis")
st.markdown("""
    *Predicting high-risk districts using historical crime data*
    """)

# --- Data Loading ---
try:
    model_data, processed_data, geojson_data = load_assets()
    districts = sorted(processed_data['district'].unique())
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Prediction Controls")
    
    selected_district = st.selectbox(
        "Select District",
        districts,
        index=districts.index("Bengaluru City") if "Bengaluru City" in districts else 0
    )
    
    st.markdown("---")
    st.markdown("**Custom Scenario Analysis**")
    custom_mode = st.checkbox("Enable manual input")
    
    if custom_mode:
        total_assault = st.number_input("Total Assault Cases", min_value=0, value=120)
        total_harassment = st.number_input("Total Harassment Cases", min_value=0, value=45)
        total_rape = st.number_input("Rape Cases", min_value=0, value=8)
        assault_ratio = st.slider("Assault Victim Ratio", 0.0, 5.0, 1.2)
        harassment_ratio = st.slider("Harassment Victim Ratio", 0.0, 5.0, 1.0)
        transport_risk = st.slider("Public Transport Risk", 0.0, 1.0, 0.15)
        workplace_risk = st.slider("Workplace Risk", 0.0, 1.0, 0.05)

# --- Main Display ---
tab1, tab2, tab3 = st.tabs(["Interactive Map", "District Analysis", "Model Insights"])

with tab1:
    st.subheader("Karnataka Crime Hotspot Map")
    st_folium(create_interactive_map(processed_data, geojson_data), width=1000, height=600)

with tab2:
    if custom_mode:
        input_data = {
            'total_assault': total_assault,
            'total_harassment': total_harassment,
            'total_rape': total_rape,
            'assault_victim_ratio': assault_ratio,
            'harassment_victim_ratio': harassment_ratio,
            'public_transport_risk': transport_risk,
            'workplace_risk': workplace_risk
        }
    else:
        district_data = processed_data[processed_data['district'] == selected_district].iloc[0]
        input_data = {
            'total_assault': district_data['total_assault'],
            'total_harassment': district_data['total_harassment'],
            'total_rape': district_data['total_rape'],
            'assault_victim_ratio': district_data['assault_victim_ratio'],
            'harassment_victim_ratio': district_data['harassment_victim_ratio'],
            'public_transport_risk': district_data['public_transport_risk'],
            'workplace_risk': district_data['workplace_risk']
        }
    
    result = make_prediction(input_data, model_data)
    
    if result:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Result")
            status = "🟢 Not a Hotspot" if not result['is_hotspot'] else "🔴 Crime Hotspot"
            st.markdown(f"### {status}")
            st.metric("Risk Probability", f"{result['probability']:.1%}")
            if result.get('is_hotspot', False):
                st.warning("Requires attention")
            else:
                st.success("Normal crime levels")

        
        with col2:
            st.subheader("Key Metrics")
            st.metric("Total Assault Cases", int(result['features']['total_assault']))
            st.metric("Total Harassment Cases", int(result['features']['total_harassment']))
            st.metric("Public Transport Risk", f"{result['features']['public_transport_risk']:.1%}")

with tab3:
    st.subheader("Model Performance")
    try:
        importances = pd.DataFrame({
            'Feature': model_data['required_features'],
            'Importance': model_data['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importances.set_index('Feature'))
        st.dataframe(importances, hide_index=True)
    except Exception as e:
        st.error(f"Couldn't display feature importance: {str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown("""
    *Data Source: Karnataka Police Department*  
    *Last Updated: {}*
    """.format(datetime.now().strftime("%Y-%m-%d")))

