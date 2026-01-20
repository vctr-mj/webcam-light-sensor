import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(ruta_csv, test_size=0.2, random_state=42):
    df = pd.read_csv(ruta_csv)
    X = df.drop(columns=["Etiqueta", "Archivo_Imagen", "Fecha", "Hora"])
    y = df["Etiqueta"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train["Etiqueta"] = y_train
    X_test["Etiqueta"] = y_test
    X_train.to_csv("../datasets/train.csv", index=False)
    X_test.to_csv("../datasets/test.csv", index=False)
    print("✅ Split realizado: train.csv y test.csv")

if __name__ == "__main__":
    ruta = "../datasets/DATASET_FEATURES.csv"
    split_train_test(ruta)
