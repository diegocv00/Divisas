import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW,Adam
import os
from datetime import datetime

# Cargar datos 
df_divisas = pd.read_csv("monedas.csv")

# Pivoteo para tener monedas como columnas
df_divisas = df_divisas.pivot_table(values="valor", columns="moneda", index="fecha").reset_index()
df_divisas["fecha"] = pd.to_datetime(df_divisas["fecha"])
df_divisas = df_divisas.sort_values('fecha')

df_divisas["USD_change"] = df_divisas["USD"].pct_change()
df_divisas["COP_change"] = df_divisas["COP"].pct_change()
df_divisas.fillna(0, inplace=True)


scaler = MinMaxScaler()
cols = ["USD", "COP", "USD_change", "COP_change"]
data_escalada = scaler.fit_transform(df_divisas[cols])


# Crear secuencias para LSTM
def crear_secuencias_multivariadas(dataset, fechas, pasos, col_target=1):
    X, y, fechas_y = [], [], []
    for i in range(pasos, len(dataset)):
        X.append(dataset[i - pasos:i, :])      # todas las features
        y.append(dataset[i, col_target])       # COP actual (columna 1)
        fechas_y.append(fechas[i])             # Fecha del valor de salida
    return np.array(X), np.array(y), np.array(fechas_y)


dias_previos = 45
X, y, fechas_y = crear_secuencias_multivariadas(data_escalada, df_divisas['fecha'].values, dias_previos)

# Dividir en train y test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
fechas_train, fechas_test = fechas_y[:train_size], fechas_y[train_size:]

# Modelo LSTM 
modelo = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

optimizador_adam = Adam()
modelo.compile(optimizer=optimizador_adam, loss='mse', metrics=['mae'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=14,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=14,
    min_lr=1e-8
)


history = modelo.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

pred = modelo.predict(X_test)

# Evaluación
pred = modelo.predict(X_test)

# Índice de COP dentro de las columnas escaladas
col_cop = df_divisas.columns.get_loc("COP")  # 

# Convertir los valores predichos y reales a formato completo 
# para poder usar el scaler.inverse_transform correctamente
dummy_pred = np.zeros((len(pred), data_escalada.shape[1]))
dummy_true = np.zeros((len(y_test), data_escalada.shape[1]))

# Insertamos las predicciones en la posición correspondiente a COP
dummy_pred[:, col_cop] = pred.reshape(-1)
dummy_true[:, col_cop] = y_test.reshape(-1)

# Invertimos el escalado
pred_inv = scaler.inverse_transform(dummy_pred)[:, col_cop]
y_test_inv = scaler.inverse_transform(dummy_true)[:, col_cop]

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
