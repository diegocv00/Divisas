
import requests
from datetime import datetime, timedelta
import time
import pandas as pd

API_KEY = 'e21af9c760fa509a88f68c238fc629a3'

datos = []
fecha_actual = datetime.now()

for i in range(31): #ejecutar para los ultimos 31 dias
    fecha = (fecha_actual-timedelta(days=i)).strftime('%Y-%m-%d')
    url = f"https://data.fixer.io/api/{fecha}?access_key={API_KEY}" # asignar fecha a la url
    querystring = {"base":"EUR","symbols":"COP,USD"}

    response = requests.get(url, params=querystring,timeout=10)
    data=response.json()

    time.sleep(5) # Pausa de 5 segundos entre solicitudes

    for moneda, valor in data['rates'].items():
        datos.append({
            'fecha': data['date'],
            'moneda': moneda,
            'valor': valor
        })
    print("finalizado dia: ", fecha)

print(datos)
df = pd.DataFrame(datos)

print(df.head())
df.to_csv("monedas.csv", index=False)

#eliminar duplicados de csv
df_f= pd.read_csv("monedas.csv")
df_f.drop_duplicates(inplace=True)

df_f.to_csv("monedas.csv", index=False)
