import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import os
from datetime import datetime

# Cargar datos 
df_divisas = pd.read_csv("monedas.csv")

# Pivoteo para tener monedas como columnas
df_divisas = df_divisas.pivot_table(values="valor", columns="moneda", index="fecha").reset_index()
df_divisas["fecha"] = pd.to_datetime(df_divisas["fecha"])
df_divisas = df_divisas.sort_values('fecha')

# Escalado
data = df_divisas[["USD", "COP"]].values
train_size = int(len(data) * 0.8)

scaler_usd = MinMaxScaler()
scaler_cop = MinMaxScaler()

# Fit solo con datos de entrenamiento
scaler_usd.fit(data[:train_size, [0]])
scaler_cop.fit(data[:train_size, [1]])

# Transformar todo el dataset
usd_escalada = scaler_usd.transform(data[:, [0]])
cop_escalada = scaler_cop.transform(data[:, [1]])
data_escalada = np.hstack((usd_escalada, cop_escalada))

# Crear secuencias para LSTM
def crear_secuencias_con_fechas(dataset, fechas, pasos):
    X, y, fechas_y = [], [], []
    for i in range(pasos, len(dataset)):
        X.append(dataset[i - pasos:i, 0])   # USD últimos 'pasos' días
        y.append(dataset[i, 1])             # COP actual
        fechas_y.append(fechas[i])          # Fecha del valor de salida
    return np.array(X), np.array(y), np.array(fechas_y)

dias_previos = 30
X, y, fechas_y = crear_secuencias_con_fechas(data_escalada, df_divisas['fecha'].values, dias_previos)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Dividir en train y test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
fechas_train, fechas_test = fechas_y[:train_size], fechas_y[train_size:]

# Modelo LSTM 
modelo = Sequential([
    Input(shape=(X_train.shape[1], 1)),

    LSTM(256, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

modelo.compile(optimizer=AdamW(learning_rate=0.0007), loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-8)

history = modelo.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluación
pred = modelo.predict(X_test)

y_test_inv = scaler_cop.inverse_transform(y_test.reshape(-1, 1))
pred_inv = scaler_cop.inverse_transform(pred)

mae_real = mean_absolute_error(y_test_inv, pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
r2 = r2_score(y_test_inv, pred_inv)

# Guardar métricas y predicciones
os.makedirs("resultados", exist_ok=True)
fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Guardar métricas
archivo_metricas = f"resultados/metricas_{fecha_hora}.txt"
with open(archivo_metricas, "w", encoding="utf-8") as file:
    file.write("📊 Resultados del modelo LSTM\n")
    file.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    file.write(f"RMSE: {rmse:.2f} COP\n")
    file.write(f"R²: {r2:.4f}\n")
    file.write(f"MAE real: {mae_real:.2f} COP\n")

# Guardar predicciones en CSV
df_resultados = pd.DataFrame({
    "fecha": fechas_test,
    "real_COP": y_test_inv.flatten(),
    "pred_COP": pred_inv.flatten()
})
archivo_pred = f"resultados/predicciones_{fecha_hora}.csv"
df_resultados.to_csv(archivo_pred, index=False)

print(f"\n✅ Métricas guardadas en: {archivo_metricas}")
print(f"✅ Predicciones guardadas en: {archivo_pred}")
print(f"RMSE: {rmse:.2f} | R²: {r2:.4f} | MAE: {mae_real:.2f}")
