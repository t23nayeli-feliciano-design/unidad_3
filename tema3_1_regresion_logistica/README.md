# Tema 3.1: Regresión Logística (Logistic Regression)

## Concepto Teórico
A pesar de llamarse 'regresión', este es un algoritmo de clasificación. En lugar de trazar una línea recta para predecir un número, utiliza una curva en forma de 'S' (función sigmoide) que convierte cualquier resultado en una probabilidad entre 0% y 100%. Si la probabilidad de que un cliente cancele su suscripción es mayor al 50%, lo clasifica como 'Cliente en Riesgo'.

## Interpretación de Resultados
Para el cliente analizado (`[50 meses, 5 tickets]`), el modelo calcula la probabilidad de permanencia frente a la de cancelación. Al ubicarse por debajo del umbral de riesgo del 50%, el modelo asigna la clasificación correspondiente según la frontera de decisión aprendida.
