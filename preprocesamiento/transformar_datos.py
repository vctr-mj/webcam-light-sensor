import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def transformar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    le_etiqueta = LabelEncoder()
    le_usuario = LabelEncoder()
    df["Etiqueta_cod"] = le_etiqueta.fit_transform(df["Etiqueta"])
    df["Usuario_cod"] = le_usuario.fit_transform(df["Usuario"])
    features = ["mean_h", "mean_s", "mean_v", "std_v", "mean_a", "mean_b", "std_l", "skew_v", "v_95"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    # Normaliza también las nuevas features si existen
    nuevas = [col for col in df.columns if col not in features + ["Fecha", "Hora", "Usuario", "Archivo_Imagen", "Etiqueta", "Etiqueta_cod", "Usuario_cod"]]
    if nuevas:
        df[nuevas] = scaler.fit_transform(df[nuevas])
    return df

if __name__ == "__main__":
    ruta = "./datasets/DATASET_LIMPIO.csv"
    df_trans = transformar_datos(ruta)
    df_trans.to_csv("./datasets/DATASET_TRANSFORMADO.csv", index=False)
    print("✅ Datos transformados y normalizados guardados en DATASET_TRANSFORMADO.csv")
