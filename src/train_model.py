import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar dataset
try:
    df = pd.read_csv('dataset/light_environment_data.csv')
except FileNotFoundError:
    print("Primero debes ejecutar data_collector.py para generar datos.")
    exit()

print(f"Dataset cargado con {len(df)} muestras.")
print("Distribución de clases:")
print(df['label'].value_counts())

# Separar Features (X) y Target (y)
X = df.drop('label', axis=1)
y = df['label']

# Split 80% entrenamiento - 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar Modelo
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluar
score = clf.score(X_test, y_test)
print(f"\nExactitud del modelo: {score*100:.2f}%")

y_pred = clf.predict(X_test)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# (Opcional) Importancia de características
feature_importances = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("\nImportancia de las variables (¿Qué define más el ambiente?):")
print(feature_importances)