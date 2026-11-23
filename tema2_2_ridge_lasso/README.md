# Tema 2.2: Regresión con Regularización (Ridge y Lasso)

## Concepto Teórico
Cuando tenemos docenas de variables de entrada, algunas pueden ser ruidosas o irrelevantes. La regularización actúa como un 'filtro de castigo':
* **Ridge (L2):** Reduce el impacto de variables poco importantes sin eliminarlas del todo.
* **Lasso (L1):** Tiene la capacidad especial de 'apagar' por completo (poner en cero) los coeficientes de variables inútiles, realizando una selección automática de las características valiosas.

## Interpretación de Resultados
En la salida de Lasso se observan coeficientes exactamente en `0.00` para las 3 variables que no aportaban información útil, descubriendo automáticamente que solo 2 características influyen en el resultado final. Ridge mantuvo las 5 variables pero suavizó sus valores para evitar inestabilidad en el modelo.
