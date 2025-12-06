# 🚗 Car Price Predictor

A sleek and interactive **Car Price Prediction** web app built with **Streamlit**. Instantly estimate the market price of a car based on brand, model, type, and key specifications. The app features a modern UI with car image previews, neumorphic inputs, and responsive design.

---

## 🔹 Features

- **Predict Car Prices**: Estimate prices in lakhs for various car brands and models.  
- **Interactive Inputs**: Select brand, model, car type, year of purchase, kilometers driven, mileage, engine, seats, fuel type, transmission, and owner type.  
- **Dynamic Car Image Preview**: Fetches car images from Wikipedia or Unsplash for a visual preview.  
- **Neumorphic UI**: Stylish inputs and buttons for a modern look.  
- **Responsive Design**: Works on desktop and mobile devices.  
- **Instant Results**: Shows estimated price immediately after prediction.

---

## 🛠️ Tech Stack

- **Python 3.x**  
- **Streamlit** – for interactive web app interface  
- **Pandas & NumPy** – for data manipulation  
- **Scikit-learn** – for machine learning model  
- **Requests** – for fetching car images  
- **Pickle** – for loading pre-trained model and encoders  

---

## ⚙️ How It Works

1. User selects **brand**, **model**, and **car type** from the sidebar.  
2. Inputs car specifications such as **year, mileage, engine, seats, fuel type, transmission, owner type**.  
3. App fetches a **car image** dynamically from Wikipedia or Unsplash.  
4. Predicts the **estimated price** using a pre-trained machine learning model.  
5. Displays the **result in a stylish card** below the Predict button.  

---

## 🌟 Customization

- **Add More Brands/Models**: Update the `brand_models` dictionary in `app.py`.  
- **Change Background or UI**: Modify the `<style>` section in `app.py`.  
- **Update Model**: Replace `car_price_pipeline.pkl` with a new trained pipeline.  

---

## 📌 Notes

- Currently supports major car brands in India.  
- Model predicts approximate market prices and may not reflect exact resale value.  
- Ensure internet connectivity for fetching car images.  

---

## ❤️ Acknowledgements

- Built with **Streamlit** for easy and interactive web app deployment.  
- Car images fetched from **Wikipedia API** and **Unsplash**.  
- Inspired by modern car price prediction tools and sleek UI designs.
