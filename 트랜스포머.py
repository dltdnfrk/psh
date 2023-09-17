import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# 데이터 불러오기
data_path = "/Users/hyunjun/Downloads/SAMPLE/2023_smartFarm_AI_hackathon_dataset.csv"
df = pd.read_csv(data_path)

# 특성 선택 및 타겟 변수 설정
selected_features = ['cunt', 'frmAr', 'hvstGrupp', 'acSlrdQy', 'frtstGrupp', 'outTp', 'inTp', 'outWs', 'flanGrupp', 'frmYear']
target = ['outtrn_cumsum']

X = df[selected_features]
y = df[target]

# 농가구역(frmDist) 별로 데이터 스케일링
scaled_dfs = []
scaler = StandardScaler()

for frmDist in df['frmDist'].unique():
    sub_df = df[df['frmDist'] == frmDist]
    sub_df_features = sub_df[selected_features]
    sub_df_target = sub_df[target]

    # 특성 스케일링
    sub_df_features_scaled = scaler.fit_transform(sub_df_features)
    sub_df_features_scaled = pd.DataFrame(sub_df_features_scaled, columns=selected_features)

    # 스케일링된 특성과 타겟 변수 합치기
    sub_df_scaled = pd.concat([sub_df_features_scaled, sub_df_target.reset_index(drop=True)], axis=1)
    scaled_dfs.append(sub_df_scaled)

# 모든 농가구역의 스케일링된 데이터를 하나로 합침
final_scaled_df = pd.concat(scaled_dfs, axis=0).reset_index(drop=True)
final_scaled_df.head()

# 이상치 제거를 위한 Robust Scaler 사용
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 분할 후 형상을 변경
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 트랜스포머 모델 함수
def build_transformer_model(input_shape, num_heads, dim_feedforward, dropout_rate):
    inputs = Input(shape=input_shape)
    x = inputs
    
    # 멀티헤드 어텐션
    x = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward 네트워크
    x = Dense(dim_feedforward, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(input_shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # 평탄화 및 출력 레이어
    x = Flatten()(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# 하이퍼파라미터 설정
input_shape = (1, X_train.shape[-1]) # 입력 차원을 수정
num_heads = 4 #멀티헤드 어텐션의 헤드 수
dim_feedforward = 128 #피드포워드 네트워크의 차원
dropout_rate = 0.1 #드롭아웃 비율

# 모델 생성 및 컴파일
model = build_transformer_model(input_shape, num_heads, dim_feedforward, dropout_rate)  # <---- 여기서 변수를 사용
model.compile(optimizer=Adam(learning_rate=1e-4), loss=MeanSquaredError(), metrics=['mse'])

# 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")