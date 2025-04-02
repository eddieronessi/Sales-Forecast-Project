import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, request, jsonify

# Load dataset (replace with your actual dataset path)
data = pd.read_csv("sales_data.csv", parse_dates=['Date'], index_col='Date')

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x=data.index, y=data['Sales'])
plt.title("Sales Over Time")
plt.show()

# Feature Engineering
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['DayOfWeek'] = data.index.dayofweek
data['Lag_1'] = data['Sales'].shift(1)  # Previous day sales

data.dropna(inplace=True)  # Remove NaN values caused by shifting

# Split data
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# XGBoost Model
X_train, y_train = train.drop(columns=['Sales']), train['Sales']
X_test, y_test = test.drop(columns=['Sales']), test['Sales']

xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ARIMA Model
arima_model = ARIMA(train['Sales'], order=(5, 1, 0))
arima_fit = arima_model.fit()
y_pred_arima = arima_fit.forecast(steps=len(test))

# Evaluate Models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_arima, "ARIMA")

# Plot Predictions
plt.figure(figsize=(12, 6))
plt.plot(test.index, y_test, label="Actual Sales", marker='o')
plt.plot(test.index, y_pred_xgb, label="XGBoost Predictions", linestyle='dashed')
plt.plot(test.index, y_pred_arima, label="ARIMA Predictions", linestyle='dotted')
plt.legend()
plt.title("Sales Forecasting Comparison")
plt.show()

# Flask API Deployment
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Lag_1'] = data['Sales'] if 'Sales' in data else np.nan
    df.drop(columns=['Date'], inplace=True)
    df.fillna(method='bfill', inplace=True)
    prediction = xgb_model.predict(df)
    return jsonify({'Predicted Sales': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
