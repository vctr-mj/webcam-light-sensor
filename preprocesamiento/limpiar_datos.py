import pandas as pd

def cargar_y_limpiar_csv(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df = df.drop_duplicates()
    df = df.dropna()
    features = ["mean_h", "mean_s", "mean_v", "std_v", "mean_a", "mean_b", "std_l", "skew_v", "v_95"]
    for col in features:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df = df[(df[col] >= q1) & (df[col] <= q99)]
    return df

if __name__ == "__main__":
    ruta = "../datasets/DATASET_MAESTRO_COMPLETO.csv"
    df_limpio = cargar_y_limpiar_csv(ruta)
    df_limpio.to_csv("../datasets/DATASET_LIMPIO.csv", index=False)
    print("✅ Datos limpios guardados en DATASET_LIMPIO.csv")
