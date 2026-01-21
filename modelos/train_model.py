import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

def entrenar_modelos(ruta_train):
    df = pd.read_csv(ruta_train)
    X = df.drop(columns=["Etiqueta"])
    y = df["Etiqueta"]

    modelos = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svc": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    for nombre, modelo in modelos.items():
        modelo.fit(X, y)
        joblib.dump(modelo, f"./modelos/modelo_{nombre}.pkl")
        print(f"✅ Modelo {nombre} guardado.")

if __name__ == "__main__":
    entrenar_modelos("./datasets/train.csv")
