import cv2
import numpy as np
import pandas as pd
import time
import os
from scipy.stats import skew

# Configuración
DATASET_PATH = 'dataset/light_environment_data_advanced.csv' 
SAMPLES_PER_CLASS = 200  

def extract_features(frame):
    """
    Extrae características extendidas para clasificación de iluminantes.
    Incluye HSV, CIELab y Momentos Estadísticos (Skewness).
    """
    # --- PREPROCESAMIENTO ---
    # Reducir ruido levemente para que los cálculos estadísticos sean más estables
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 1. ESPACIO HSV (Matiz, Saturación, Valor)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    
    # 2. ESPACIO CIELab (L=Luminosidad, a=Verde-Rojo, b=Azul-Amarillo)
    # Fundamental para distinguir temperatura de color (LED vs Pantalla)
    lab_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_frame)

    # --- EXTRACCIÓN DE CARACTERÍSTICAS ---
    
    # A. Estadísticas HSV (Línea base)
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_v  = np.std(v)   # Contraste global

    # B. Estadísticas CIELab (Temperatura de color)
    # El canal 'b' es crítico: valores bajos indican azul (pantallas), altos amarillo (sol/incandescente)
    mean_a = np.mean(a)
    mean_b = np.mean(b) 
    std_l  = np.std(l)

    # C. Momentos de Orden Superior y "White Patch"
    # Skewness: ¿Hay un punto de luz muy fuerte o es difusa?
    # Usamos el canal V (Brillo) para ver la distribución de la luz
    skew_v = skew(v.flatten()) 
    
    # Percentil 95 (V95): Representa los "highlights" o la fuente de luz directa
    # Más robusto que el Max absoluto (que puede ser ruido)
    v_95 = np.percentile(v, 95)

    # Vector de características (9 variables)
    features = [mean_h, mean_s, mean_v, std_v, mean_a, mean_b, std_l, skew_v, v_95]
    return features

def collect_data():
    cap = cv2.VideoCapture(0) # Asegúrate que el índice 0 es tu webcam correcta
    
    if not cap.isOpened():
        print("Error: No se puede acceder a la webcam.")
        return

    print("--- RECOLECTOR DE DATOS AVANZADO (HSV + LAB + STATS) ---")
    label_name = input("Ingresa la ETIQUETA (ej. 'natural', 'led', 'pantalla', 'mix'): ").strip()
    
    data_buffer = []
    print(f"Recolectando para: '{label_name}'...")
    time.sleep(1) 

    count = 0
    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret: break
            
        features = extract_features(frame)
        
        # Desempaquetar para visualización (solo algunas claves)
        # [mean_h, mean_s, mean_v, std_v, mean_a, mean_b, std_l, skew_v, v_95]
        f_mean_v = features[2]
        f_mean_b = features[5] # Canal b de Lab (Azul-Amarillo)
        f_skew   = features[7]

        # Visualización en pantalla
        cv2.putText(frame, f"Label: {label_name} ({count}/{SAMPLES_PER_CLASS})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Brillo (V): {f_mean_v:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # Si 'b' es bajo (<128 en uint8 opencv o negativo en matemáticas puras) es azulado
        cv2.putText(frame, f"Temp (Lab-b): {f_mean_b:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1) 
        cv2.putText(frame, f"Asimetria: {f_skew:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

        cv2.imshow('Recolector Avanzado', frame)
        
        data_buffer.append(features + [label_name])
        count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cancelado.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if count == SAMPLES_PER_CLASS:
        columns = [
            'mean_h', 'mean_s', 'mean_v', 'std_v', 
            'mean_a', 'mean_b', 'std_l', 
            'skew_v', 'v_95', 
            'label'
        ]
        df = pd.DataFrame(data_buffer, columns=columns)
        
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        if not os.path.isfile(DATASET_PATH):
            df.to_csv(DATASET_PATH, index=False)
        else:
            df.to_csv(DATASET_PATH, mode='a', header=False, index=False)
            
        print(f"¡Guardado! Dataset actualizado en {DATASET_PATH}")

if __name__ == "__main__":
    collect_data()