# Project Documentation

## 1. Title

**AI-Powered Traffic Accident Analytics Platform**

## 2. Problem Statement

Road accidents are influenced by multiple factors such as speed zones, accident types, lighting conditions, road geometry, time, day, and the number of vehicles involved. Traditional reporting systems mainly describe historical accidents and may not provide an easy way to estimate severity for a new accident scenario.

## 3. Proposed Solution

This project develops a Machine Learning-powered analytical web application that learns patterns from historical accident records, predicts accident severity, and presents useful insights through interactive visualizations.

## 4. Input Features

| Feature | Description |
| --- | --- |
| SPEED_ZONE | Speed zone associated with the accident location |
| ACCIDENT_TYPE | Type/category of accident |
| LIGHT_CONDITION | Lighting condition during the accident |
| ROAD_GEOMETRY | Road layout or geometry |
| DAY_OF_WEEK | Day on which the accident occurred |
| HOUR | Hour of occurrence |
| NO_OF_VEHICLES | Number of vehicles involved |

## 5. Prediction Target

`SEVERITY` is treated as a multiclass classification target with categories such as fatal, serious injury, and other injury accidents.

## 6. Expected Output

The application provides:

1. Predicted accident severity.
2. Risk-oriented interpretation.
3. Interactive accident statistics.
4. Feature-based analytical insights.
5. Model performance metrics.

## 7. Development Phases

### Phase 1 — Data Preparation
- Load dataset.
- Remove invalid or duplicate records.
- Handle missing values.
- Encode categorical variables.
- Prepare training features and target.

### Phase 2 — Exploratory Data Analysis
- Study severity distribution.
- Analyze accidents by time and day.
- Analyze speed-zone patterns.
- Study lighting and road geometry.
- Identify relationships between vehicle count and severity.

### Phase 3 — Machine Learning
- Split data into training and testing sets.
- Train classification models.
- Handle class imbalance where appropriate.
- Compare evaluation metrics.
- Select the best-performing model.

### Phase 4 — Web Application
- Build dashboard.
- Add prediction form.
- Display charts.
- Integrate the trained model.
- Add user-friendly prediction results.

### Phase 5 — Deployment
- Test the application locally.
- Validate predictions and UI.
- Deploy the web application to a suitable hosting platform.

## 8. Evaluation Metrics

Accuracy alone should not determine model quality because the dataset contains imbalanced target classes. Precision, recall, F1-score, confusion matrix, and per-class performance should also be considered.

## 9. Educational Value

The project demonstrates the complete data-to-application workflow: data preprocessing, exploratory analysis, machine learning, model evaluation, visualization, and web deployment.
