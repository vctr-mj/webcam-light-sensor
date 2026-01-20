import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURACIÓN ---
RUTA_DATASET = os.path.join("datasets", "DATASET_MAESTRO_COMPLETO.csv")
CARPETA_RESULTADOS = "resultados_graficos"

if not os.path.exists(CARPETA_RESULTADOS):
    os.makedirs(CARPETA_RESULTADOS)

# --- CARGA DE DATOS ---
if not os.path.exists(RUTA_DATASET):
    print(f"❌ Error: No se encuentra {RUTA_DATASET}")
    print("Ejecuta primero 'unir_datasets.py'")
    exit()

df = pd.read_csv(RUTA_DATASET)
print(f"✅ Datos cargados: {len(df)} registros.")

# Convertir etiqueta a numérico para análisis de correlación
df['Etiqueta_Code'] = df['Etiqueta'].astype('category').cat.codes

# --- 0. DISTRIBUCIÓN DE CLASES ---
plt.figure(figsize=(7,4))
sns.countplot(x='Etiqueta', hue='Etiqueta', data=df, palette='Set1', legend=False)
plt.title("Distribución de Clases (Etiquetas)")
plt.xlabel("Etiqueta")
plt.ylabel("Cantidad")
plt.savefig(os.path.join(CARPETA_RESULTADOS, "0_Distribucion_Clases.png"))
plt.close()

# --- 0.1. Estadísticas descriptivas por clase ---
desc = df.groupby('Etiqueta').describe().T
desc.to_csv(os.path.join(CARPETA_RESULTADOS, "0_Stats_Descriptivas_por_Clase.csv"))
print("Estadísticas descriptivas por clase guardadas.")

# --- 1. MATRIZ DE CORRELACIÓN (Para ver qué variables sirven más) ---
print("Generando Matriz de Correlación...")
plt.figure(figsize=(12, 10))
# Seleccionamos solo columnas numéricas relevantes
cols_corr = ['mean_h', 'mean_s', 'mean_v', 'std_v', 
             'mean_a', 'mean_b', 'std_l', 'skew_v', 'v_95', 'Etiqueta_Code'
            ]
corr = df[cols_corr].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Mapa de Calor: Correlación entre Variables y Etiqueta")
plt.savefig(os.path.join(CARPETA_RESULTADOS, "1_Matriz_Correlacion.png"))
plt.close()

# --- 2. BOXPLOT: LA PRUEBA DEL "PROBLEMA DE FONDO" (mean_b) ---
# Explicación: Este gráfico demuestra si el "mean_b" (Eje Azul-Amarillo)
# realmente separa la luz de pantalla (Azul) de la natural (Amarillo).
print("Generando Boxplot de Separación de Clases...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Etiqueta', y='mean_b', data=df, hue='Etiqueta', palette="Set2", legend=False)
plt.title("Distribución del Eje Azul-Amarillo (mean_b) por Fuente de Luz")
plt.ylabel("Valor Canal B (Negativo=Azul / Positivo=Amarillo)")
plt.xlabel("Tipo de Fuente")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(CARPETA_RESULTADOS, "2_Discriminacion_Luz_Azul.png"))
plt.close()

# --- 3. PAIRPLOT: VISTA GENERAL DE SEPARABILIDAD ---
# Explicación: Muestra cómo se agrupan las clases usando 3 variables clave.
print("Generando Gráfico de Dispersión (Pairplot)...")
vars_clave = ['mean_b', 'std_v', 'skew_v'] # Las 3 más importantes según teoría
sns.pairplot(df, vars=vars_clave, hue='Etiqueta', palette='bright', plot_kws={'alpha': 0.6})
plt.savefig(os.path.join(CARPETA_RESULTADOS, "3_Separacion_Clases_Scatter.png"))
plt.close()

# --- 4. HISTOGRAMAS DE VARIABLES ---
print("Generando histogramas de variables...")
vars_hist = ['mean_h', 'mean_s', 'mean_v', 'std_v', 'mean_a', 'mean_b', 'std_l', 'skew_v', 'v_95']
for var in vars_hist:
    plt.figure(figsize=(8,4))
    for etiqueta in df['Etiqueta'].unique():
        sns.histplot(df[df['Etiqueta'] == etiqueta][var], label=etiqueta, kde=True, stat="density", bins=30, alpha=0.5)
    plt.title(f"Histograma de {var} por Etiqueta")
    plt.xlabel(var)
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, f"4_Hist_{var}.png"))
    plt.close()

# --- 5. OUTLIERS Y DISTRIBUCIÓN ---
print("Analizando outliers y distribución...")
for var in vars_hist:
    plt.figure(figsize=(8,4))
    # Usar hue y legend=False para evitar el warning de seaborn
    sns.boxplot(x='Etiqueta', y=var, data=df, hue='Etiqueta', palette="Set3", legend=False)
    plt.title(f"Boxplot de {var} por Etiqueta")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, f"5_Boxplot_{var}.png"))
    plt.close()

# --- 6. ANÁLISIS DE UTILIDAD PARA ML ---
# Comentario: Variables con alta correlación con 'Etiqueta_Code' y buena separación visual en histogramas/boxplots
# son candidatas útiles para modelos de ML. Si hay solapamiento fuerte, el modelo tendrá más dificultad.
# Guardamos correlación con la etiqueta para referencia rápida.
corr_etiqueta = corr['Etiqueta_Code'].drop('Etiqueta_Code').sort_values(ascending=False)
corr_etiqueta.to_csv(os.path.join(CARPETA_RESULTADOS, "6_Correlacion_Variables_vs_Etiqueta.csv"))
print("Correlación de variables con la etiqueta guardada.")

print(f"\n✅ ¡Análisis completado! Revisa la carpeta '{CARPETA_RESULTADOS}' para ver las imágenes y archivos generados.")