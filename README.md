# Clasificación de Iluminación Ambiental con Computer Vision

Este proyecto implementa un sistema de clasificación de fuentes de luz (Natural, LED, Pantalla) utilizando una webcam estándar como sensor de luz matricial. El objetivo es mejorar algoritmos de balance de blancos y detectar entornos de trabajo nocivos (exceso de luz azul).

## Estructura

- **Recopilación de datos:**  
  Captura imágenes y extrae características con la webcam.  
  Ver instrucciones en [`README_RECOPILACION.md`](./README_RECOPILACION.md).

- **Procesamiento y entrenamiento:**  
  Limpieza, transformación, ingeniería de features, entrenamiento y análisis de modelos.  
  Ver instrucciones en [`README_PROCESAMIENTO.md`](./README_PROCESAMIENTO.md).

## Requisitos

- Python 3.8+
- Instala dependencias:
  ```bash
  pip install -r requirements.txt
  ```

## Flujo recomendado

1. Recopila datos usando la webcam.
2. Procesa los datos y entrena modelos.
3. Analiza los resultados y compara modelos.

Consulta los README específicos para cada etapa.

## 🚀 Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/vctr-mj/webcam-light-sensor
   cd webcam-light-sensor
   ```

2. Crea un entorno virtual:
   ```bash
   python -m venv .venv
   ```

3. Activa el entorno virtual:

   - En Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - En Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso de la carpeta `datasets_compartido` y el script `unir_datasets.py`

Para combinar los archivos de datos (CSV) de las carpetas `datasets/` y `datasets_compartido/` en un solo archivo maestro, sigue estos pasos:

1. Coloca los archivos `.csv` que deseas unir dentro de las carpetas en `datasets_compartido/` (pueden estar en subcarpetas).
2. Ejecuta el script de unión:
   ```bash
   python scripts/unir_datasets.py
   ```
   Esto generará (o actualizará) el archivo `datasets/DATASET_MAESTRO_COMPLETO.csv` con la unión de todos los archivos CSV encontrados en ambas carpetas.

**Notas:**
- El script ignora automáticamente el archivo maestro si ya existe, para evitar duplicados.
- Solo se unirán archivos con extensión `.csv`.
- Puedes agregar nuevos archivos a cualquiera de las dos carpetas y volver a ejecutar el script para actualizar el dataset maestro.