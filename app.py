import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
from urllib.parse import quote

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.markdown("""
    <style>
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Make widget labels white */
.stApp label,
.stSelectbox > label,
.stNumberInput > label,
.stSidebar label {
    color: #ffffff !important;
}

/* Sidebar option text fix */
.css-17eq0hr, .css-1kyxreq, .css-q8sbsg {
    color: #ffffff !important;
}

/* App background */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(6,16,24,0.45), rgba(6,16,24,0.45)),
                      url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=2000&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Limit main content width */
.block-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 1.2rem;
}

/* Hero area */
.hero {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 18px;
    animation: fadeIn 0.9s ease both;
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: url('https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=2000&q=80') center/cover no-repeat;
    opacity: 0.08;
    filter: grayscale(60%);
}
.hero .hero-glass {
    backdrop-filter: blur(6px) saturate(120%);
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    padding: 32px;
    border-radius: 12px;
    text-align:center;
}

.hero-title {
    font-size: 34px;
    color: #fff;
    font-weight: 800;
    text-shadow: 0 6px 18px rgba(0,0,0,0.6);
}
.hero-sub {
    color: rgba(255,255,255,0.9);
    font-size: 15px;
}

/* Car Preview */
.car-preview-wrap {
    display: flex;
    justify-content: center;
    width: 100%;
    margin-top: 14px;
    margin-bottom: 50px;  /* increased from 14 */
    animation: fadeIn 0.7s ease both;
}

.car-preview {
    text-align: center;
    background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,250,250,0.92));
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(2,12,27,0.18);
}
.car-preview img {
    border-radius: 10px;
    transition: transform 0.35s ease;
}
.car-preview img:hover {
    transform: scale(1.02);
}
.car-title { font-weight:700; margin-top:8px; }

/* Neumorphic section title */
.neu {
    background: #f0f3f7;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 6px 6px 14px rgba(163,177,198,0.35), -6px -6px 14px rgba(255,255,255,0.7);
    margin-bottom: 14px;
}

/* Buttons */
.stButton > button {
    color: black;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 16px;
    border: none;
    transition: 0.3s ease;
}

/* Hover effect */
.stButton > button:hover {
    color: white;
    background-color: #0A3D 62;
    cursor: pointer;
}

/* Center button container */
.center-button {
    display: flex;
    justify-content: center;
}

/* Result box */
.result {
    background: linear-gradient(180deg,#0A3D62,#07324b);
    color: #fff;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    box-shadow: 0 10px 30px rgba(2,12,27,0.3);
    margin-top: 14px;
}

/* Sidebar glass */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    backdrop-filter: blur(6px);
    border-right: 1px solid rgba(255,255,255,0.04);
}

.small-muted { color: rgba(255,255,255,0.8); font-size:13px; }

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)


# ---------------------------
# Load model artifacts (your existing pipeline)
# ---------------------------
with open("car_price_pipeline.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
label_encoders = bundle["encoder"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]
feature_order = bundle["feature_order"]


brand_models = {
    "Maruti": ["Alto","Swift","Swift Dzire", "Baleno hatchback","Celerio","Ertiga", "Dzire", "Ignis","Wagon R","Ritz","S-Cross","Vitara Brezza","Ciaz","Other"],
    "Hindustan": ["Ambassador"],
    "Audi": ["A4", "A6","A7","Q3", "Q5", "Q6", "Q7", "A3","Other"],
    "Hyundai": ["i20", "i10", "Creta","Eon", "Accent", "Venue","Other"],
    "Honda": ["City", "Civic", "Amaze", "Jazz", "WR-V","Other"],
    "Toyota": ["Innova", "Fortuner", "Glanza", "Etios", "Urban Cruiser","Camry","Other"],
    "Tata": ["Nexon", "Altroz", "Harrier", "Tiago", "Safari","Punch","Bolt","Vista", "Indica","Indigo","Other"],
    "Mahindra": ["XUV500", "XUV700", "Scorpio", "Thar", "Bolero","KUV100","Other"],
    "Ford": ["EcoSport", "Figo", "Endeavour","Ikon","Other"],
    "BMW": ["3 Series", "5 Series", "3 Series 320d Luxury Line","1 Series", "X5", "sDrive",
            "xDrive","6 Series","X1","Other"],
    "Volkswagen": ["Polo", "Vento", "Taigun","Other"],
    "Skoda": ["Rapid", "Slavia", "Kushaq","Laura","Superb","Other"],
    "Volvo": ["XC60", "XC90","Other"],
    "Force": ["Gurkha"],    
    "Bentley": ["Continental"],
    "Chevrolet": ["Cruze", "Aveo", "Beat","Enjoy","Other"],
    "Datsun": ["GO","Redi-Go","Other"],
    "Fiat": ["Punto", "Linea","Other"],
    "Jaguar": ["F-Pace","XF", "XJ","XE"],
    "Jeep": ["Compass", "Wrangler","Other"],
    "Land Rover": ["Discovery Sport", "Range Rover Evoque","Other"],
    "Isuzu": ["D-Max"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLA", "GLC", "GLE","M-Class","Other"],
    "Mini": ["Cooper"],
    "Mitsubishi": ["Outlander", "Pajero Sport","Other"],
    "Nissan": ["Terrano", "Micra", "Sunny","Other"],
    "Porsche": ["Cayenne", "Macan","Other"],
    "Renault": ["Duster", "Kwid", "Captur", "Triber","Other"],
    "Lamborghini": ["Gallardo"],
}

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.markdown("<div style='padding:12px;border-radius:10px;color:#fff;font-weight:700;font-size:16px;text-align:center;'>🔎 Select Car</div>", unsafe_allow_html=True)
st.sidebar.write("")  # spacing

st.sidebar.markdown("<div style='padding:10px;border-radius:10px;background:rgba(255,255,255,0.03);backdrop-filter:blur(5px)'>", unsafe_allow_html=True)

brand = st.sidebar.selectbox("🏷 Brand", options=label_encoders["Brand"].classes_)
models = brand_models.get(brand, ["Other"])
car_model = st.sidebar.selectbox("🚘 Model", options=models)
car_type = st.sidebar.selectbox("🚙 Car Type",
    ["Hatchback","Sedan", "SUV/Cross-over"]
)

st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='padding:8px;color:rgba(255,255,255,0.8);font-size:13px;text-align:center;'>Tip: choose brand first to populate models</div>", unsafe_allow_html=True)

# ---------------------------
# Image fetch function
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_car_image(brand: str, model: str):
    headers = {"User-Agent": "Mozilla/5.0"}

    if model and model != "Other":
        try:
            title = f"{brand} {model}"
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            r = requests.get(url, headers=headers, timeout=5)
            if r.ok:
                data = r.json()
                if "thumbnail" in data:
                    return data["thumbnail"]["source"]
        except:
            pass

    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(brand)}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.ok:
            data = r.json()
            if "thumbnail" in data:
                return data["thumbnail"]["source"]
    except:
        pass

    try:
        safe_query = quote(f"{brand} car")
        return f"https://source.unsplash.com/600x400/?{safe_query}"
    except:
        return None

image_url = fetch_car_image(brand, car_model)

# ---------------------------
# Hero (glass + title)
# ---------------------------
st.markdown("""
<div class="hero">
  <div class="hero-glass">
    <div class="hero-title">🚗 Car Price Prediction</div>
    <div class="hero-sub">Instant market valuation</div>
  </div>
</div>
""", unsafe_allow_html=True)


# Centered car preview
if image_url:
    st.markdown(f"""
    <div class="car-preview-wrap">
      <div class="car-preview">
        <img src="{image_url}" width="330" alt="car image">
        <div class="car-title">{brand} {car_model}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    year = st.number_input("📅 Year of Purchase", min_value=1990, max_value=datetime.datetime.now().year, value=2018)
    km_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, max_value=1000000, value=10000)
with col2:
    mileage = st.number_input("⛽ Mileage (km/l)", min_value=1, max_value=100, value=10)
    engine = st.number_input("🔧 Engine (CC)", min_value=400, max_value=7000, value=1197)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
col3, col4, col5 = st.columns([1,1,1])
with col3:
    seats = st.selectbox("🪑 Seats", [2,4,5,6,7,8], index = 2)
with col4:
    fuel_type = st.selectbox("⛽ Fuel Type", options=label_encoders["Fuel_Type"].classes_)
with col5:
    transmission = st.selectbox("⚙️ Transmission", options=label_encoders["Transmission"].classes_)

owner_type = st.selectbox("👤 Owner Type", options=label_encoders["Owner_Type"].classes_)

# small muted help text
st.markdown("<div class='small-muted'></div>", unsafe_allow_html=True)

# ---------------------------
# Feature engineering & prepare input
# ---------------------------
type_addition = 0
if car_type == "Sedan":
    type_addition = 1
elif car_type == "SUV / Crossover":
    type_addition = 2

car_age = datetime.datetime.now().year - year

input_df = pd.DataFrame([{
    "Kilometers_Driven": km_driven,
    "Mileage": mileage,
    "Engine": engine,
    "Seats": seats,
    "car_age": car_age,
    "Fuel_Type": fuel_type,
    "Transmission": transmission,
    "Owner_Type": owner_type,
    "Brand": brand,
}])

# Encode & scale
for col in cat_cols:
    input_df[col] = label_encoders[col].transform(input_df[col])

input_df[num_cols] = scaler.transform(input_df[num_cols])
input_df = input_df[feature_order]

# ---------------------------
# Predict button & output
# ---------------------------
st.markdown("""
<style>
div.stButton > button {
    display: block;
    margin-left: 321px ;
    margin-right: -170px ;
}
</style>
""", unsafe_allow_html=True)

if st.button("🔮 Predict Price"):
    prediction = model.predict(input_df)[0]
    final_price = prediction + type_addition
    st.markdown(
        f"<div class='result'>💰 Estimated Price: <strong>₹ {round(final_price,2)} Lakhs</strong></div>",
        unsafe_allow_html=True

    )





