# Procesamiento, Entrenamiento y Análisis de Datos

Este README describe cómo ejecutar el pipeline completo de procesamiento, entrenamiento y análisis de modelos usando los datos generados.

## 1. Preprocesamiento de datos

```bash
# Limpiar datos
python preprocesamiento/limpiar_datos.py

# Transformar datos (normalización y codificación)
python preprocesamiento/transformar_datos.py

# Ingeniería de features
python preprocesamiento/features_engineering.py
```

## 2. División en train/test

```bash
python división/split_train_test.py
```

## 3. Entrenamiento de modelos

Entrena 4 modelos clásicos y los guarda en la carpeta `modelos/`:

```bash
python modelos/train_model.py
```

## 4. Optimización de hiperparámetros (opcional)

```bash
python modelos/optimizacion.py
```

## 5. Exploración y análisis

Abre los notebooks para explorar y analizar resultados:

- `notebooks/exploracion.ipynb`: Exploración y visualización inicial de los datos.
- `notebooks/analisis_resultados.ipynb`: Comparación de accuracy y métricas de los modelos entrenados.

---

**Nota:**  
Asegúrate de tener los archivos CSV generados en la carpeta `datasets/` antes de ejecutar el pipeline.