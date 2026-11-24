# Tema 4: Evaluación, Validación Cruzada y Optimización de Modelos

## 4.1 Matriz de Confusión y Métricas de Calidad
En problemas desbalanceados (como detección de fraudes donde solo el 1% o 10% son casos positivos), la exactitud (*Accuracy*) puede ser engañosa. Por ello se emplea la Matriz de Confusión para evaluar:
* **Precisión (Precision):** De todos los casos clasificados como positivos, cuántos realmente lo eran (evita falsas alarmas).
* **Exhaustividad (Recall):** De todos los casos positivos reales existentes, cuántos logró capturar el modelo (evita que se escapen fraudes).
* **F1-Score:** Media armónica entre Precisión y Recall.

## 4.2 Práctica Integradora: Detección de Fraudes con GridSearchCV en Ubuntu EC2
Pipeline completo que integra división estratificada de datos, escalado estándar (`StandardScaler`), validación cruzada de 5 pliegues (`5-Fold Cross Validation`), búsqueda automática de hiperparámetros (`GridSearchCV`) y serialización del modelo (`joblib`) en archivos `.pkl` listos para despliegue en AWS S3 o servicios web.

## Interpretación de Resultados del Pipeline
El reporte de clasificación evalúa la clase `Fraude (1)`, logrando capturar la mayoría de los casos fraudulentos reales (alto *Recall*) y alcanzando una métrica **ROC-AUC** elevada que confirma la capacidad del modelo para distinguir entre transacciones legítimas y fraudulentas. Los archivos `modelo_fraude_opt.pkl` y `scaler_fraude.pkl` quedan generados en la instancia EC2.
