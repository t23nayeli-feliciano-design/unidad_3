import os
import pandas as pd
import numpy as np
import joblib
import boto3
import io
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Simulación de dataset de transacciones bancarias (10% de Fraude)
X_bank, y_bank = make_classification(
    n_samples=1000, n_features=8, weights=[0.90, 0.10], random_state=42
)

# Dividir en Entrenamiento (80%) y Prueba (20%) manteniendo la proporción de fraudes
X_train, X_test, y_train, y_test = train_test_split(
    X_bank, y_bank, test_size=0.20, random_state=42, stratify=y_bank
)

# Escalado z-score de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Búsqueda automática de los mejores hiperparámetros (GridSearchCV)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 8],
    'criterion': ['gini', 'entropy']
}

rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(
    estimator=rf_base, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor modelo optimizado
best_rf = grid_search.best_estimator_

# 3. Evaluación final con el conjunto de prueba
y_pred_final = best_rf.predict(X_test_scaled)
y_proba_final = best_rf.predict_proba(X_test_scaled)[:, 1]

print("=== PIPELINE COMPLETO DE CLASIFICACIÓN EN UBUNTU EC2 ===")
print("Mejores Hiperparámetros Encontrados:", grid_search.best_params_)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_final))
print("\nReporte de Clasificación Completo:")
print(classification_report(y_test, y_pred_final, target_names=['Normal (0)', 'Fraude (1)']))
print(f"Métrica ROC-AUC Score: {roc_auc_score(y_test, y_proba_final):.4f}")

# 4. Guardar el modelo en la carpeta de la práctica en Ubuntu EC2
ruta_actual = os.path.dirname(os.path.abspath(__file__))
joblib.dump(best_rf, os.path.join(ruta_actual, 'modelo_fraude_opt.pkl'))
joblib.dump(scaler, os.path.join(ruta_actual, 'scaler_fraude.pkl'))
print("\n-> Modelo y escalador guardados exitosamente como archivos '.pkl'.")
