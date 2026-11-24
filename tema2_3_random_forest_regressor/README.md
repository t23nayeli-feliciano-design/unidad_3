# Tema 2.3: Bosques Aleatorios para Regresión (Random Forest Regressor)

## Concepto Teórico
Un árbol de decisión toma decisiones tipo 'si / entonces'. Un Bosque Aleatorio combina cientos de estos árboles independientes; cada árbol predice una cifra y el bosque promedia todos los resultados bajo el principio de la 'sabiduría del grupo', logrando mayor precisión que un solo árbol individual.

## Interpretación de Resultados
La propiedad `feature_importances_` indica el porcentaje de relevancia de cada variable en las decisiones del modelo (por ejemplo, `0.65` representa el 65% del peso). Esto aporta explicabilidad para identificar exactamente qué factores influyen más en la predicción final.
