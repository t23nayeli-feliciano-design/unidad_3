# Tema 3.2: Árboles de Decisión y Bosques Aleatorios (Random Forest Classifier)

## Concepto Teórico
Un Árbol de Decisión funciona como un diagrama de flujo de preguntas ('¿El cliente gana más de $20,000? Sí -> ¿Tiene tarjetas de crédito? No -> Riesgo Alto'). Un Random Forest Classifier genera cientos de estos diagramas de flujo y decide la categoría final mediante 'votación de la mayoría' (por ejemplo, si 80 de 100 árboles votan 'Fraude', la respuesta del sistema será 'Fraude').

## Interpretación de Resultados
El Random Forest obtiene un porcentaje de precisión superior al árbol individual porque combina múltiples perspectivas descorrelacionadas, eliminando los sesgos y errores individuales de un solo árbol.
