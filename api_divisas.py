import requests
from datetime import datetime, timedelta
import time
import pandas as pd
import os

API_KEY = 'e21af9c760fa509a88f68c238fc629a3'
ARCHIVO = "monedas.csv"

datos = []
fecha_actual = datetime.now()

for i in range(3):  # últimos 31 días
    fecha = (fecha_actual - timedelta(days=i)).strftime('%Y-%m-%d')
    url = f"https://data.fixer.io/api/{fecha}?access_key={API_KEY}"
    querystring = {"base": "EUR", "symbols": "COP,USD"}

    response = requests.get(url, params=querystring, timeout=10)
    data = response.json()

    time.sleep(5)  # evitar límite de API

    # Si hay datos válidos
    if "rates" in data:
        for moneda, valor in data["rates"].items():
            datos.append({
                "fecha": data["date"],
                "moneda": moneda,
                "valor": valor
            })
        print("finalizado día:", fecha)
    else:
        print("❌ Error en la fecha:", fecha, "-", data.get("error", "sin detalle"))

# Crear DataFrame nuevo
df_nuevo = pd.DataFrame(datos)

# Si el archivo ya existe, combinarlo
if os.path.exists(ARCHIVO):
    df_existente = pd.read_csv(ARCHIVO)
    df_total = pd.concat([df_existente, df_nuevo], ignore_index=True)
else:
    df_total = df_nuevo

# Quitar duplicados (por fecha y moneda)
df_total.drop_duplicates(subset=["fecha", "moneda"], inplace=True)

# Ordenar por fecha descendente
df_total = df_total.sort_values("fecha", ascending=False)

# Guardar CSV actualizado
df_total.to_csv(ARCHIVO, index=False)

print(f"\n✅ Archivo '{ARCHIVO}' actualizado correctamente con {len(df_total)} registros.")
