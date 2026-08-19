# 🚦 AI-Powered Traffic Accident Analytics Platform

An AI-powered Predictive Analytics Web App for **road accident severity prediction, risk analysis, and interactive data visualization** using Machine Learning.

## 📌 Project Overview

Road accident datasets contain valuable information about traffic conditions, road characteristics, time, lighting, vehicle count, and accident type. This project uses that information to build a machine-learning system that predicts accident severity and presents analytical insights through a web dashboard.

## 🎯 Objectives

- Predict road accident severity using Machine Learning.
- Analyze accident patterns across time, road conditions, lighting, and traffic factors.
- Provide interactive visual analytics through a web interface.
- Support data-driven road safety decision making.
- Compare ML models and select a suitable predictive model.

## 📊 Dataset

The prepared dataset contains approximately **200,352 accident records**.

### Features

- `SPEED_ZONE`
- `ACCIDENT_TYPE`
- `LIGHT_CONDITION`
- `ROAD_GEOMETRY`
- `DAY_OF_WEEK`
- `HOUR`
- `NO_OF_VEHICLES`

### Target

`SEVERITY`

Target categories include:

- Other injury accident
- Serious injury accident
- Fatal accident

> The dataset is imbalanced, so model evaluation should consider precision, recall, F1-score, and the confusion matrix in addition to accuracy.

## 🧠 Machine Learning Workflow

```text
Raw Accident Data
       ↓
Data Cleaning & Preprocessing
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Train / Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Prediction API / Web App
       ↓
Interactive Analytics Dashboard
```

## 🖥️ Planned Web App Features

### Dashboard
- Total accidents
- Fatal accidents
- Serious injury accidents
- Other injury accidents
- Key accident trends

### Accident Severity Prediction
Users can enter:

- Speed zone
- Accident type
- Light condition
- Road geometry
- Day of week
- Hour
- Number of vehicles

The application returns the predicted accident severity.

### Analytics
- Severity distribution
- Accidents by hour
- Accidents by day of week
- Accidents by speed zone
- Accident type analysis
- Lighting-condition analysis
- Vehicle-count analysis

## 🗂️ Recommended Project Structure

```text
AI-Powered-Traffic-Accident-Analytics-Platform/
│
├── app/
│   └── README.md
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── src/
│   └── README.md
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit or Flask
- HTML / CSS / JavaScript (if using a custom frontend)
- Git & GitHub

## 📈 Model Evaluation

The project should compare multiple classification models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Because the target classes are imbalanced, **macro F1-score and per-class recall** are especially important when selecting the final model.

## 🔐 Data & Repository Policy

Large raw datasets and generated model binaries should not be committed to GitHub unless their size and license permit it. Use Git LFS or an external data-storage solution when necessary.

## 🚀 Future Enhancements

- Real-time accident-risk scoring
- Interactive geographic accident maps
- Explainable AI using feature importance / SHAP
- Traffic-density integration
- Weather-data integration
- Emergency-response recommendations
- Cloud deployment
- Real-time monitoring dashboard

## 👩‍💻 Project Status

**Development in progress** — data preparation is complete and the Machine Learning training, evaluation, and web integration stages are being developed.

## 📄 License

This project is intended for educational, research, and demonstration purposes.
