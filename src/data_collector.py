import cv2
import numpy as np
import pandas as pd
import time
import os

# Configuración
DATASET_PATH = 'dataset/light_environment_data.csv'
SAMPLES_PER_CLASS = 200  # Cantidad de datos por etiqueta

def extract_features(frame):
    """
    Convierte un frame de video en un vector numérico de características.
    No guardamos la imagen, solo sus estadísticas matemáticas.
    """
    # 1. Separar canales de color (BGR en OpenCV)
    b, g, r = cv2.split(frame)
    
    # 2. Características de Color (Dominancia de espectro)
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    
    # 3. Características de Luminosidad (Convertir a escala de grises)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_mean = np.mean(gray)       # Nivel de luz general
    brightness_std = np.std(gray)         # Contraste (¿Luz dura o difusa?)
    
    return [mean_r, mean_g, mean_b, brightness_mean, brightness_std]

def collect_data():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se puede acceder a la webcam.")
        return

    print("--- RECOLECTOR DE DATOS DE AMBIENTE ---")
    label_name = input("Ingresa la ETIQUETA para esta sesión (ej. 'dia', 'noche', 'artificial'): ").strip()
    
    data_buffer = []
    print(f"Preparando recolección para la clase: '{label_name}'...")
    print("Presiona 'q' para cancelar antes de terminar.")
    time.sleep(2) # Tiempo para acomodar la laptop

    count = 0
    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extraer características
        features = extract_features(frame)
        
        # Visualización para el usuario (Feedback visual)
        # Mostramos los valores en la pantalla
        cv2.putText(frame, f"Label: {label_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Muestras: {count}/{SAMPLES_PER_CLASS}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Brillo: {features[3]:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow('Recolector de Datos (Sensor de Luz)', frame)
        
        # Guardar datos en memoria
        # Agregamos la etiqueta al final de la lista de características
        data_buffer.append(features + [label_name])
        count += 1
        
        # Pequeño delay para no tomar frames idénticos
        # time.sleep(0.05) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recolección cancelada.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Guardar en CSV
    if count == SAMPLES_PER_CLASS:
        columns = ['mean_r', 'mean_g', 'mean_b', 'brightness_mean', 'brightness_std', 'label']
        df = pd.DataFrame(data_buffer, columns=columns)
        
        # Si no existe la carpeta, crearla
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        
        # Si el archivo no existe, escribir con header. Si existe, agregar sin header.
        if not os.path.isfile(DATASET_PATH):
            df.to_csv(DATASET_PATH, index=False)
        else:
            df.to_csv(DATASET_PATH, mode='a', header=False, index=False)
            
        print(f"¡Éxito! {count} muestras guardadas en {DATASET_PATH}")
    else:
        print("No se guardaron datos (ciclo incompleto).")

if __name__ == "__main__":
    collect_data()