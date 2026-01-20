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