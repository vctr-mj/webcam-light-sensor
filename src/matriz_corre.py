import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset generado
df = pd.read_csv('dataset/light_environment_data_advanced.csv')

# Verificar que hay datos
print("Clases encontradas:", df['label'].unique())

# Codificar las etiquetas (Label Encoding) para poder correlacionarlas
# Ej: Natural=0, LED=1, Pantalla=2
le = LabelEncoder()
df['label_code'] = le.fit_transform(df['label'])

# Calcular matriz de correlación
# Excluimos la columna 'label' de texto original
corr_matrix = df.drop(columns=['label']).corr()

# --- VISUALIZACIÓN ---
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación de Variables de Iluminación")
plt.tight_layout()
plt.show()

# --- INTERPRETACIÓN AUTOMÁTICA ---
print("\n--- ANÁLISIS RÁPIDO ---")
# Ver correlación directa con la etiqueta (objetivo)
target_corr = corr_matrix['label_code'].sort_values(ascending=False)
print("Correlación de variables con la etiqueta (Importancia lineal preliminar):")
print(target_corr)

print("\nNota: Si 'mean_v' y 'v_95' tienen correlación > 0.9, considera quitar una para simplificar el modelo.")