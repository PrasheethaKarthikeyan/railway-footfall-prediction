import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math

# Load dataset
df = pd.read_csv("data.csv")

# Remove bad repeated header rows if any
df = df[df["hour"] != "hour"]
df = df.dropna()

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# Features and target
X = df[['hour', 'day', 'weekend', 'trains_arrival', 'trains_departure', 'holiday', 'festival']]
y = df['footfall']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

accuracy_percent = r2 * 100

print("Model trained successfully!")
print(f'R2 Score: {r2:.4f}')
print(f'Accuracy: {accuracy_percent:.2f}%')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# Save model
pickle.dump(model, open("model.pkl", "wb"))