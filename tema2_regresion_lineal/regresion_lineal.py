import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(42)
X_horas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y_desempenio = np.array([55, 60, 63, 68, 72, 78, 83, 87, 92, 96])
modelo_lr = LinearRegression()
modelo_lr.fit(X_horas, y_desempenio)
y_pred = modelo_lr.predict(X_horas)
print("=== REGRESIÓN LINEAL ===")
print(f"Pendiente (Impacto por hora): {modelo_lr.coef_[0]:.4f}")
print(f"Intersección (Puntaje base): {modelo_lr.intercept_:.4f}")
print(f"Coeficiente R^2: {r2_score(y_desempenio, y_pred):.4f}")
print(f"Error Promedio (RMSE): {np.sqrt(mean_squared_error(y_desempenio, y_pred)):.4f}")
