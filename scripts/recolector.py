import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
from scipy.stats import skew
import sys

# =============================================================================
# --- INICIALIZACIÓN DE LA CÁMARA (Mejora de velocidad) ---
# =============================================================================
print("⏳ Inicializando cámara...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Error crítico: Cámara no detectada.")
    exit()

# Descarta los primeros frames para estabilizar la imagen
for _ in range(10):
    cap.read()
print("✅ Cámara lista.")

# =============================================================================
# --- CONFIGURACIÓN Y VALIDACIÓN DE ENTRADAS ---
# =============================================================================

# Listas permitidas (Normalizadas a minúsculas para evitar errores)
USUARIOS_VALIDOS = ["alex", "edwin", "marco", "orlando", "victor"]
ETIQUETAS_VALIDAS = ["natural", "artificial", "pantallas", "mix"]

def solicitar_input(mensaje, opciones_validas):
    """Solicita input al usuario hasta que coincida con las opciones válidas."""
    while True:
        print(f"\n{mensaje}")
        print(f"Opciones: {opciones_validas}")
        entrada = input(">>> ").strip().lower()
        if entrada in opciones_validas:
            return entrada
        print("❌ Error: Opción no válida. Intenta de nuevo.")

# 1. Solicitar datos al inicio
print("--- CONFIGURACIÓN DE LA SESIÓN ---")
USUARIO_ACTUAL = solicitar_input("¿Quién está realizando la captura?", USUARIOS_VALIDOS)
ETIQUETA_LUZ = solicitar_input("¿Qué tipo de iluminación es?", ETIQUETAS_VALIDAS)

# 2. Configurar Directorios
# Estructura: datasets / NOMBRE / ETIQUETA / archivo.csv
CARPETA_RAIZ = "datasets"
CARPETA_SESION = os.path.join(CARPETA_RAIZ, USUARIO_ACTUAL + "_" + ETIQUETA_LUZ + "_" + datetime.now().strftime('%H-%M'))

if not os.path.exists(CARPETA_SESION):
    os.makedirs(CARPETA_SESION)
    print(f"✅ Carpeta creada: {CARPETA_SESION}")

TOTAL_FOTOS = 1000
SEGUNDOS_ESPERA = 0.1 # 100 ms entre capturas

# =============================================================================
# --- SENSOR VIRTUAL (Extracción de Características) ---
# =============================================================================
def extraer_features(imagen_bgr):
    # 1. Espacio HSV
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_h = np.mean(h) * 2.0  
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_v  = np.std(v)         

    # 2. Espacio CIELAB
    lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)

    mean_a = np.mean(a) - 128.0
    mean_b = np.mean(b_chan) - 128.0 
    std_l  = np.std(l)

    # 3. Estadísticas de Forma
    skew_v = skew(v.flatten())   
    v_95 = np.percentile(v, 95)  

    # --- SUGERENCIAS DE NUEVAS FEATURES ---
    # mean_r = np.mean(imagen_bgr[:,:,2])
    # mean_g = np.mean(imagen_bgr[:,:,1])
    # mean_b_rgb = np.mean(imagen_bgr[:,:,0])
    # hist_v = np.histogram(v, bins=10, range=(0,255))[0]
    # entropia = -np.sum(np.histogram(v, bins=256)[0]/v.size * np.log2(np.histogram(v, bins=256)[0]/v.size + 1e-7))
    # contraste = np.max(v) - np.min(v)
    # Puedes agregar estos valores al return y al CSV si lo deseas.

    return [mean_h, mean_s, mean_v, std_v, mean_a, mean_b, std_l, skew_v, v_95]

# =============================================================================
# --- PREPARACIÓN DEL ARCHIVO ---
# =============================================================================
ahora = datetime.now()
t_stamp = ahora.strftime('%d-%m_%H-%M') # dia-mes_hora-minuto
# %d-%m-%Y_%H-%M-%S

# Nombre solicitado: sesion_usuario_etiqueta_fecha_hora.csv
nombre_csv = f"sesion_{USUARIO_ACTUAL}_{ETIQUETA_LUZ}_{t_stamp}.csv"
ruta_csv = os.path.join(CARPETA_SESION, nombre_csv)

headers = ["Fecha", "Hora", "Usuario", "Archivo_Imagen", "Etiqueta", 
           "mean_h", "mean_s", "mean_v", "std_v", 
           "mean_a", "mean_b", "std_l", "skew_v", "v_95"]

with open(ruta_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# =============================================================================
# --- BUCLE DE CAPTURA ---
# =============================================================================
try:
    spinner = ['|', '/', '-', '\\']
    for i in range(TOTAL_FOTOS):
        ret, frame = cap.read()
        if ret:
            features = extraer_features(frame)
            
            # Guardar Imagen en la misma subcarpeta
            t_str = datetime.now().strftime("%H-%M-%S-%f")[:-3] # Hora con milisegundos
            nom_foto = f"img_{USUARIO_ACTUAL}_{ETIQUETA_LUZ}_{t_str}.jpg"
            ruta_foto = os.path.join(CARPETA_SESION, nom_foto)
            cv2.imwrite(ruta_foto, frame)
            
            # Guardar CSV
            fila = [datetime.now().strftime("%d/%m/%Y"), t_str, USUARIO_ACTUAL, nom_foto, ETIQUETA_LUZ] + features
            
            with open(ruta_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fila)
            
            # Mostrar ventana en vivo con información
            frame_vista = frame.copy()
            cv2.putText(frame_vista, f"Captura {i+1}/{TOTAL_FOTOS}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame_vista, f"Imagen: {nom_foto}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Webcam - Capturando", frame_vista)
            
            # Consola: solo mostrar el último estado, con spinner animado
            sys.stdout.write('\r')
            sys.stdout.write(f"{spinner[i%4]} [{i+1}/{TOTAL_FOTOS}] Guardando...")
            sys.stdout.flush()
            
            # Salida anticipada con ESC
            if cv2.waitKey(1) & 0xFF == 27:
                print("\n⚠️ Grabación detenida por el usuario (ESC).")
                break
        
        time.sleep(SEGUNDOS_ESPERA)

    # Al finalizar, salto de línea y mensaje final
    print(f"\n[{i+1}/{TOTAL_FOTOS}] Última captura guardada")

except KeyboardInterrupt:
    print("\n⚠️ Grabación detenida por el usuario.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Sesión finalizada.")