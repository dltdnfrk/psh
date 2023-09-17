from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/hyunjun/Downloads/SAMPLE/2023_smartFarm_AI_hackathon_dataset.csv")

# 1. Handle missing values
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# 2. Feature engineering
df['Temp_Humidity_Interaction'] = df['inTp'] * df['inHd']
df['Solar_Watering_Interaction'] = df['acSlrdQy'] * df['cunt']

# 3. Create target variable
df['unit_yield'] = df['outtrn_cumsum'] / df['frmAr']

# Function to calculate unit_yield
def calculate_unit_yield(row):
    if row['frmAr'] > 100:
        return row['outtrn_cumsum'] / row['frmAr']
    else:
        return row['outtrn_cumsum']

# 4. Feature selection
selected_features = [
    'cunt', 'frmAr', 'hvstGrupp', 'acSlrdQy', 'frtstGrupp', 'outTp', 
    'inTp', 'outWs', 'flanGrupp', 'frmYear', 'Temp_Humidity_Interaction',
    'Solar_Watering_Interaction'
]
target = ['unit_yield']

# 5. Standardization
scaler = StandardScaler()
X = df[selected_features]
y = df[target]
X_scaled = scaler.fit_transform(X)

# 6. Data split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape data for LSTM
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# 7. LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# Assuming the data is sorted in ascending order by time
# Take the last 16 weeks of feature data
last_16_weeks_features = X_scaled[-16:]

# Reshape to 3D array for LSTM
# This should match the input shape of your LSTM model
last_16_weeks_features = np.reshape(last_16_weeks_features, (last_16_weeks_features.shape[0], 1, last_16_weeks_features.shape[1]))

# Use the model to predict production for each of the next 16 weeks
future_16_week_predictions = model.predict(last_16_weeks_features)

# Sum up the 16 weeks of predictions to get the total future production
future_16_week_cumulative_production = np.sum(future_16_week_predictions)


# Model prediction
y_pred = model.predict(X_test)

# Reshape y_pred to 2D array
y_pred = y_pred.reshape(-1, 1)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)
print("Predicted cumulative production for the next 16 weeks:", future_16_week_cumulative_production)
