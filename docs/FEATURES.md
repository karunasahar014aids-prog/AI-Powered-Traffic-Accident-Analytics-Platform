# Platform Features

## AI Accident Severity Prediction
Predicts accident severity from the selected speed zone, accident type, light condition, road geometry, day, hour, and number of vehicles.

## Risk Level
Maps the predicted severity to a simple presentation-friendly risk level: Low, Medium, or High. This is a derived communication layer, not a calibrated probability.

## Safety Recommendations
Provides practical caution guidance based on the predicted severity and selected conditions.

## Accident Analytics
The dashboard is designed to surface severity distribution and patterns by hour, day of week, speed zone, accident type, lighting condition, and vehicle count.

## Prediction Explanation
The prediction screen can display the input factors used by the model so users can understand which context was supplied for the prediction. Future versions can add model-derived feature importance or SHAP explanations.

## Hotspot / High-Risk Pattern Analysis
The platform can highlight high-frequency or high-severity patterns by available categorical/time features. Geographic hotspot mapping is planned when reliable location coordinates are incorporated.

## Trend Analysis
Time-based charts can be used to identify changes and peak periods in historical accident records.

## Deployment Note
The trained model artifact must be made reliably available in the production environment before live prediction can be claimed as fully operational. The large raw dataset should remain outside the normal Git repository when appropriate.
