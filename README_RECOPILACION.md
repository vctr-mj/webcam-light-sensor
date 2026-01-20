## 🛠️ Uso

### 1. Recolectar datos con la webcam

Ejecuta el script de captura:
```bash
python scripts/recolector.py
```
Sigue las instrucciones en pantalla para capturar imágenes bajo diferentes fuentes de luz.

### 2. Unir sesiones de captura

Para consolidar varias sesiones en un solo dataset:
```bash
python scripts/unir_datasets.py
```

### 3. Análisis exploratorio

Para analizar correlaciones y visualizar datos:
```bash
python scripts/analisis_variables.py
```

# Recopilación de datos con webcam

Este módulo permite la captura automatizada de imágenes y extracción de características para construir datasets de entrenamiento.

## Uso

1. Ejecuta el recolector:
   ```bash
   python src/recolector.py
   ```
2. Ingresa el usuario y tipo de iluminación cuando se solicite.
3. Las imágenes y el archivo CSV se guardarán en la carpeta `datasets/`.

## Salida

- Carpeta con imágenes capturadas.
- CSV con las características extraídas de cada imagen.

## Siguiente paso

Continúa con el procesamiento y entrenamiento siguiendo las instrucciones de `README_PROCESAMIENTO.md`.