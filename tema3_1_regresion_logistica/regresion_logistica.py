import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Datos sintéticos: [Antigüedad en meses, Tickets de soporte] -> Churn (0=Se queda, 1=Se va)
X_churn = np.array([[10, 1], [20, 2], [30, 3], [70, 7], [80, 8], [90, 9]])
y_churn = np.array([0, 0, 0, 1, 1, 1])

modelo_log = LogisticRegression()
modelo_log.fit(X_churn, y_churn)

# Probar con un nuevo cliente: 50 meses de antigüedad y 5 tickets
nuevo_cliente = [[50, 5]]
pred_clase = modelo_log.predict(nuevo_cliente)
pred_prob = modelo_log.predict_proba(nuevo_cliente)

print("=== REGRESIÓN LOGÍSTICA ===")
print(f"Predicción final para nuevo cliente: Clase {pred_clase[0]} (0=Permanece, 1=Cancela)")
print(f"Probabilidades calculadas: P(Permanecer)={pred_prob[0][0]*100:.1f}%, P(Cancelar)={pred_prob[0][1]*100:.1f}%")
