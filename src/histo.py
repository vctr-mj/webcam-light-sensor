import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar el dataset
# Asegúrate de que la ruta sea la correcta según tu estructura de carpetas
file_path = 'dataset/light_environment_data_advanced.csv'
df = pd.read_csv(file_path)

# 2. Configuración de estilo
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 10))

# 3. Variables seleccionadas (Las "Ganadoras" del análisis de correlación)
# Seleccionamos las que tienen menos redundancia entre sí pero alto valor predictivo
features_to_plot = [
    'mean_v',  # Brillo (Intensidad)
    'mean_s',  # Saturación (Pureza del color)
    'mean_b',  # Temperatura (Azul vs Amarillo - Clave para pantallas)
    'skew_v'   # Distribución (¿Luz pareja o punto focal?)
]

# 4. Generar los gráficos
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 2, i + 1) # Crear una cuadrícula de 2x2
    
    # Usamos histplot con 'kde=True' para ver la línea de tendencia suave
    # 'hue="label"' es la magia: pinta cada clase de un color diferente
    sns.histplot(
        data=df, 
        x=feature, 
        hue="label", 
        kde=True, 
        element="step", 
        stat="density", 
        common_norm=False,
        palette="bright"
    )
    
    plt.title(f'Distribución de: {feature}', fontsize=12, fontweight='bold')
    plt.xlabel(feature)
    plt.ylabel('Densidad')

# Ajustar diseño y mostrar
plt.suptitle('Análisis Visual de Separación de Clases', fontsize=16)
plt.tight_layout()
plt.show()