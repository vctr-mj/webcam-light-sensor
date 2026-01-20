import pandas as pd

def agregar_features(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df["v_range"] = df["v_95"] - df["mean_v"]
    df["h_s_product"] = df["mean_h"] * df["mean_s"]
    df["v_skew_abs"] = df["skew_v"].abs()
    return df

if __name__ == "__main__":
    ruta = "../datasets/DATASET_TRANSFORMADO.csv"
    df_feat = agregar_features(ruta)
    df_feat.to_csv("../datasets/DATASET_FEATURES.csv", index=False)
    print("✅ Features adicionales guardadas en DATASET_FEATURES.csv")
