import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression

# Crear dataset con 5 variables, pero solo 2 son realmente útiles
X_raw, y_raw = make_regression(n_samples=100, n_features=5, n_informative=2, noise=10.0, random_state=42)

# Modelo Ridge (L2) y Lasso (L1)
ridge = Ridge(alpha=1.0)
ridge.fit(X_raw, y_raw)

lasso = Lasso(alpha=2.0)
lasso.fit(X_raw, y_raw)

print("=== REGULARIZACIÓN RIDGE Y LASSO ===")
print("Coeficientes Ridge (Contrae valores):", np.round(ridge.coef_, 2))
print("Coeficientes Lasso (Elimina inútiles):", np.round(lasso.coef_, 2))
