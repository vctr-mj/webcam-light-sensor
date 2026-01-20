import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def entrenar_modelos(ruta_train):
    df = pd.read_csv(ruta_train)
    X = df.drop(columns=["Etiqueta"])
    y = df["Etiqueta"]

    modelos = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svc": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "logistic": LogisticRegression(max_iter=1000, random_state=42)
    }

    for nombre, modelo in modelos.items():
        modelo.fit(X, y)
        joblib.dump(modelo, f"../modelos/modelo_{nombre}.pkl")
        print(f"✅ Modelo {nombre} guardado.")

if __name__ == "__main__":
    entrenar_modelos("../datasets/train.csv")
