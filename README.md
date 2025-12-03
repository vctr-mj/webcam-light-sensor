# Webcam Light Environment Classifier 📸💡

Este proyecto es parte del curso de Machine Learning de la Maestría. El objetivo es crear un **dataset supervisado propio** utilizando hardware común (webcam de laptop) actuando como un sensor fotométrico para clasificar entornos lumínicos.

## 🎯 Objetivo
Clasificar el entorno del usuario basándose en las propiedades de la luz ambiental sin utilizar reconocimiento de objetos (Computer Vision profunda), sino mediante **Ingeniería de Características (Feature Engineering)** estadística sobre los canales de color.

## 📂 Estructura del Dataset
El dataset se genera automáticamente mediante el script `data_collector.py`. No se guardan imágenes (respetando la privacidad y reduciendo el peso), sino un vector de características extraído de cada frame:

| Feature | Descripción | Racional Teórico |
| :--- | :--- | :--- |
| `mean_r` | Promedio Canal Rojo | Detecta luces cálidas (incandescentes) o atardeceres. |
| `mean_g` | Promedio Canal Verde | Ayuda a balancear la detección de luz fluorescente. |
| `mean_b` | Promedio Canal Azul | Detecta luz fría (pantallas, luz día nublado). |
| `brightness_mean` | Promedio Escala de Grises | Intensidad total de luz (Lux aproximado). |
| `brightness_std` | Desviación Estándar (Grises) | Mide el contraste. Una luz directa genera sombras duras (alto std), luz difusa genera sombras suaves (bajo std). |

## 🚀 Instalación y Uso

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt