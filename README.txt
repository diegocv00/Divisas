💱 Predicción Automática de Divisas (LSTM)

Este proyecto obtiene y predice tasas de cambio usando una red LSTM y se actualiza automáticamente cada mes con GitHub Actions.

⚙️ Funcionalidad

- Descarga datos desde Fixer.io (`api_divisas.py`).
- Entrena un modelo LSTM (`divisas.py`).
- Guarda resultados en:
  - `monedas.csv` → Datos actualizados  
  - `resultados/metricas_fecha.txt` → Métricas (RMSE, MAE, R²)
- Se ejecuta automáticamente el **día 1 de cada mes** a las **12:00 UTC** (~7:00 a.m. Colombia).
