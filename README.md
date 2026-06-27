# 🚦 TrafficPulse: Los Angeles Traffic Speed & Congestion Prediction

TrafficPulse is an end-to-end machine learning project that predicts short-term traffic speed and congestion levels for **Los Angeles** using historical traffic sensor data, weather conditions, and road network features.

The system is built around the ****METR**-LA traffic sensor dataset**, so the current version is designed specifically for the Los Angeles area. Users can enter latitude and longitude coordinates near Los Angeles, and the system finds the nearest **METR**-LA sensor, predicts future traffic speed, and visualizes congestion on an interactive dashboard.

---

## 📌 Project Overview

Urban traffic congestion is affected by several factors such as time of day, historical traffic patterns, road structure, and weather. Traditional traffic prediction systems often rely only on historical averages, which may fail during dynamic conditions.

TrafficPulse solves this by combining:

- Historical traffic speed data from **METR**-LA sensors
- Weather features such as temperature, rain, precipitation, cloud cover, and wind speed
- Road network features from OpenStreetMap
- Machine learning models for 5-minute, 15-minute, and 30-minute traffic speed prediction
- An interactive Streamlit dashboard for visualization and prediction

---

## 🎯 Problem Statement

Given a latitude and longitude in the Los Angeles area, predict the traffic speed and congestion level for the nearest **METR**-LA traffic sensor.

The system supports three prediction horizons:

```text 5 minutes ahead 15 minutes ahead 30 minutes ahead ```

The output includes:

- Nearest sensor ID,
- Distance from user input to nearest sensor,
- Predicted traffic speed in mph,
- Congestion level: Low / Medium / High

---

## 🚀 Key Features

### Machine Learning

- Predicts short-term traffic speed using trained CatBoost models
- Supports multiple prediction horizons:

    * 5 minutes
    * 15 minutes
    * 30 minutes
- Uses sensor-level traffic history and contextual features
- Converts predicted speed into congestion level

### Geospatial Intelligence

- Finds the nearest **METR**-LA traffic sensor from user-entered coordinates
- Uses latitude and longitude based nearest-sensor matching
- Road features are extracted from OpenStreetMap data
- Dashboard map is centered around Los Angeles because the model is trained on Los Angeles sensors

### Weather-Aware Prediction

The model uses weather features such as:

```text
- temperature_2m
- relative_humidity_2m
- precipitation
- rain
- weather_code
- cloud_cover
- wind_speed_10m
- wind_gusts_10m
```

### Interactive Dashboard

Built using Streamlit with:

- Input panel for latitude, longitude, and prediction horizon
- Prediction cockpit
- Los Angeles congestion map
- Sensor-level speed trends
- Multi-horizon prediction comparison
- Road and weather context panel
- **JSON** export of prediction results
- Debug and data preview options

---

## 🧠 Congestion Level Logic

Predicted speed is converted into congestion level using the following rule:

```text
- Speed >= 50 mph       → Low congestion
- 30 mph <= Speed < 50  → Medium congestion
- Speed < 30 mph        → High congestion
```

---

## 🏗️ System Architecture

```text
### User Input
Latitude + Longitude + Horizon
    |
    v
### Nearest Sensor Matching
Find closest **METR**-LA traffic sensor
    |
    v
### Feature Preparation
Traffic + Weather + Road features
    |
    v
ML Model Prediction
CatBoost model for selected horizon
    |
    v
### Congestion Classification
Low / Medium / High
    |
    v
### Streamlit Dashboard
Map + Charts + Explanation + Export
```

---

## 📂 Project Structure

```text
TrafficPulse/
│
├── app.py
├── main.py
├── requirements.txt
├── **README**.md
│
├── src/
│   ├── predict.py
│   ├── load_artifacts.py
│   ├── nearest_sensor.py
│   └── preprocessing.py
│
├── models/
│   ├── model_5min.pkl
│   ├── model_15min.pkl
│   ├── model_30min.pkl
│   └── highway_encoder.pkl
│
├── data/
│   └── processed/
│       └── final_model_data.parquet
│
├── notebooks/
│   ├── 01_dataset_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
└── assets/
    └── screenshots/
```

---

## 📊 Datasets Used

### 1. METR-LA Traffic Dataset

The main traffic dataset contains 5-minute interval speed readings from Los Angeles traffic sensors.

Format:

```timestamp × sensor_id speed matrix ```

It was converted into long format:

```timestamp | sensor_id | speed ```

### 2. Sensor Location Dataset

Contains sensor metadata:

```sensor_id latitude longitude ```

This is used for nearest sensor detection and map visualization.

### 3. Weather Dataset

Historical weather data was fetched for the Los Angeles region and merged with traffic data using hourly timestamps.

### 4. OpenStreetMap Road Network Data

Road attributes were extracted using **OSM** data:

``` highway type road length number of lanes max speed oneway flag distance to road ```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Removing complete all-zero traffic timestamps
- Treating zero or negative speed values as invalid
- Interpolating short missing traffic gaps
- Converting traffic data from wide format to long format
- Merging traffic data with sensor coordinates
- Merging traffic data with weather features
- Matching sensors to nearest road segments
- Cleaning road attributes such as lanes and maxspeed
- Encoding categorical road type features

Important cleaning decision:

``` All-sensor zero rows were treated as system outage timestamps. Remaining zero/negative speeds were treated as missing values. Short missing gaps were interpolated. Long invalid gaps were avoided to prevent fake traffic patterns. ```

---

## 🤖 Model

The project uses CatBoost regression models for predicting future traffic speed.

Separate models are trained for:

``` 5-minute prediction 15-minute prediction 30-minute prediction ```

Model input features include:

``` speed latitude longitude weather features road features time-based features lag features rolling traffic features ```

Model output:

``` predicted_speed_mph ```

---

## 🖥️ Running the Project Locally

### 1. Clone the repository

```bash git clone [https://github.com/SambitSekhar12/Traffic-Pulse.git](https://github.com/SambitSekhar12/Traffic-Pulse.git) cd Traffic-Pulse ```

### 2. Create virtual environment

For Windows:

``` python -m venv venv venv\Scripts\activate ```

For macOS/Linux:

``` python -m venv venv source venv/bin/activate ```

### 3. Install dependencies

``` pip install -r requirements.txt ```

### 4. Run CLI prediction

``` python main.py ```

Example input:

``` Enter latitude: 34.**15497** Enter longitude: -**118**.**31829** Enter horizon (5min/15min/30min): 5min ```

### 5. Run Streamlit dashboard

``` streamlit run main.py ```

Then open the local Streamlit **URL** in your browser.

---

## 🧪 Example Prediction

Input:

``` Latitude: 34.**15497** Longitude: -**118**.**31829** Horizon: 5min ```

Example output:

``` Nearest sensor ID: **773869** Distance: 0.**000** km Predicted speed: 58.42 mph Congestion level: Low ```

Actual output may vary depending on the trained model and latest available dataset row.

---

## 📦 Requirements

Main libraries used:

``` streamlit plotly pydeck pandas numpy catboost scikit-learn joblib geopandas shapely osmnx pyarrow ```

Install all dependencies using:

``` pip install -r requirements.txt ```

---

## ⚠️ Important Note

This dashboard is currently built for **Los Angeles only**.

The model is trained using **METR**-LA traffic sensors. If a user enters coordinates outside Los Angeles, the system will still find the nearest LA sensor, but the prediction may not be meaningful.

Recommended input area:

```text Los Angeles, California, **USA** ```

---

## 📤 Output Export

The dashboard allows prediction results to be exported as **JSON**.

Example:

```json
{
    *input*:
    {
      *latitude*: 34.**15497**,
      *longitude*: -**118**.**31829**,
      *selected_horizon*: *5min*
    },
    *nearest_sensor*:
    {
      *sensor_id*: *773869*,
      *distance_km*: 0.0
    },
    *predictions*:
    {
      *5min*:
      {
        *predicted_speed_mph*: 58.42,
        *congestion_level*: *Low*
      }
    }
}
```

---

## 🔮 Future Improvements

Possible future extensions:

- Add live traffic **API** integration
- Add real-time weather updates
- Add **SHAP**-based model explanations
- Add sensor-click interaction on map
- Add high-congestion heatmap layer
- Add route-level congestion prediction
- Add support for other cities
- Deploy the Streamlit dashboard online
- Add MLflow experiment tracking
- Add Docker deployment

---

## 👨‍💻 Tech Stack

``` Python Pandas NumPy Scikit-learn CatBoost GeoPandas OSMnx OpenStreetMap Streamlit Plotly PyDeck Joblib ```

---

## 🙌 Acknowledgement

This project uses publicly available traffic, weather, and map-based data sources for educational and portfolio purposes.

TrafficPulse is designed as an applied machine learning project to demonstrate:

- End-to-end ML pipeline development
- Geospatial data processing
- Weather-aware traffic prediction
- Model deployment through Streamlit
- Interactive dashboard development

---

## 📄 License

This project is for educational and portfolio purposes.

If you use or modify this project, please provide appropriate credit.
