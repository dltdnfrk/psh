import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Read the dataset
df = pd.read_csv("/Users/hyunjun/Downloads/SAMPLE/2023_smartFarm_AI_hackathon_dataset.csv")

# Handle NaN values (fill with median for numerical columns)
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Function to calculate unit_yield
def calculate_unit_yield(row):
    if row['frmAr'] > 100:
        return row['outtrn_cumsum'] / row['frmAr']
    else:
        return row['outtrn_cumsum']

# Calculate 'unit_yield' for each row
df['unit_yield'] = df.apply(calculate_unit_yield, axis=1)

# Feature selection and target variable
selected_features = ['cunt', 'frmAr', 'hvstGrupp', 'acSlrdQy', 'frtstGrupp', 'outTp', 'inTp', 'outWs', 'flanGrupp', 'frmYear']
target = ['unit_yield']

X = df[selected_features]
y = df[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape data for LSTM
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(LSTM(50))
    model.add(Dense(1, activation='linear'))
    return model

# Model creation and compilation
input_shape = (1, X_train.shape[-1])
model = build_lstm_model(input_shape)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mse'])

# Model training
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Model prediction
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")
