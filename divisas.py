
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import os
from datetime import datetime

df_divisas = pd.read_csv("Divisas/monedas.csv")

df_divisas

"""Pivoteando para hacer las monedas columnas"""

df_divisas = df_divisas.pivot_table(values="valor", columns="moneda",index="fecha").reset_index()
df_divisas

"""Se separa la fecha en dia,mes y año"""

df_divisas["fecha"] = pd.to_datetime(df_divisas["fecha"])
df_divisas["Dia"] = df_divisas["fecha"].dt.day
df_divisas["Mes"] = df_divisas["fecha"].dt.month
df_divisas["Año"] = df_divisas["fecha"].dt.year
df_divisas = df_divisas.sort_values('fecha')

df_divisas

"""Funcion que filtra por mes y grafica por dia del mes seleccionado para la moneda deseada


"""

def grafica_dias_del_mes(moneda: str):
    plt.figure(figsize=(14,8))
    sns.lineplot(data=df_divisas.tail(30), x="fecha", y=moneda)
    plt.title(f"Evolución de {moneda} los últimos 30 días")
    plt.xlabel("Día")
    plt.ylabel(moneda)
    plt.xticks(rotation=45)
    plt.show()

"""Grafica de COP en los últimos 30 días"""

grafica_dias_del_mes("COP")

"""Grafica de USD en los últimos 30 días"""

grafica_dias_del_mes("USD")

"""Ahora se usará una red LSTM para predecir los precios de COP según USD"""

scaler_usd = MinMaxScaler()
scaler_cop = MinMaxScaler()

usd_escalada = scaler_usd.fit_transform(df_divisas[["USD"]])
cop_escalada = scaler_cop.fit_transform(df_divisas[["COP"]])


data_escalada = np.hstack((usd_escalada, cop_escalada))

def crear_secuencias_con_fechas(dataset, fechas, pasos):
    X, y, fechas_y = [], [], []
    for i in range(pasos, len(dataset)):
        X.append(dataset[i - pasos:i, 0])   # USD últimos 'pasos' días
        y.append(dataset[i, 1])             # COP actual
        fechas_y.append(fechas[i])          # Fecha del valor de salida
    return np.array(X), np.array(y), np.array(fechas_y)

dias_previos = 30  #días previos de USD para predecir COP
X, y, fechas_y = crear_secuencias_con_fechas(data_escalada, df_divisas['fecha'].values, dias_previos)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
fechas_train, fechas_test = fechas_y[:train_size], fechas_y[train_size:]

modelo = Sequential([
    Input(shape=(X_train.shape[1], 1)),

    # Primera capa LSTM
    LSTM(256, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    # Segunda capa LSTM
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    # Tercera capa LSTM
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    # Capas densas intermedias
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    # Capa de salida
    Dense(1)
])

modelo.compile(optimizer=AdamW(learning_rate=0.0007), loss='mse', metrics=['mae'])

modelo.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-8
)


history = modelo.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

pred = modelo.predict(X_test)

y_test_inv = scaler_cop.inverse_transform(y_test.reshape(-1,1))
pred_inv = scaler_cop.inverse_transform(pred)

mae_real = mean_absolute_error(y_test_inv, pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
r2 = r2_score(y_test_inv, pred_inv)

# Crear carpeta "resultados" si no existe
os.makedirs("resultados", exist_ok=True)

# Generar nombre automático con fecha y hora actual
fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
archivo_metricas = f"resultados/metricas_{fecha_hora}.txt"

# Guardar las métricas en el archivo
with open(archivo_metricas, "w", encoding="utf-8") as file:
    file.write("📊 Resultados del modelo LSTM\n")
    file.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    file.write(f"RMSE: {rmse:.2f} COP\n")
    file.write(f"R²: {r2:.4f}\n")
    file.write(f"MAE real: {mae_real:.2f} COP\n")


print(f"\n Métricas guardadas automáticamente en: {archivo_metricas}")



