from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generar dataset sintético para clasificación binaria
X_data, y_data = make_classification(n_samples=300, n_features=5, n_classes=2, random_state=42)

# 1. Árbol de decisión individual
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_data, y_data)

# 2. Bosque Aleatorio (Ensamble)
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_data, y_data)

print("=== ÁRBOLES VS RANDOM FOREST CLASSIFIER ===")
print(f"Precisión Árbol Individual: {accuracy_score(y_data, dt.predict(X_data))*100:.2f}%")
print(f"Precisión Random Forest: {accuracy_score(y_data, rf.predict(X_data))*100:.2f}%")
