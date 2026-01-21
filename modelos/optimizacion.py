import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def optimizar_rf(ruta_train):
    df = pd.read_csv(ruta_train)
    X = df.drop(columns=["Etiqueta"])
    y = df["Etiqueta"]
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    clf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print("Mejores parámetros RF:", grid.best_params_)
    return grid.best_estimator_

def optimizar_svc(ruta_train):
    df = pd.read_csv(ruta_train)
    X = df.drop(columns=["Etiqueta"])
    y = df["Etiqueta"]
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }
    svc = SVC(probability=True, random_state=42)
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print("Mejores parámetros SVC:", grid.best_params_)
    return grid.best_estimator_

def optimizar_knn(ruta_train):
    df = pd.read_csv(ruta_train)
    X = df.drop(columns=["Etiqueta"])
    y = df["Etiqueta"]
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print("Mejores parámetros KNN:", grid.best_params_)
    return grid.best_estimator_

def optimizar_gb(ruta_train):
    df = pd.read_csv(ruta_train)
    X = df.drop(columns=["Etiqueta"])
    y = df["Etiqueta"]
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10]
    }
    gb = GradientBoostingClassifier(random_state=42)
    grid = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print("Mejores parámetros GradientBoosting:", grid.best_params_)
    return grid.best_estimator_

if __name__ == "__main__":
    optimizar_rf("./datasets/train.csv")
    optimizar_svc("./datasets/train.csv")
    optimizar_knn("./datasets/train.csv")
    optimizar_gb("./datasets/train.csv")
