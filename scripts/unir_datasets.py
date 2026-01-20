import pandas as pd
import glob
import os

# Configuración
CARPETAS_BUSQUEDA = ["datasets", "datasets_compartido"]
NOMBRE_MAESTRO = "DATASET_MAESTRO_COMPLETO.csv"
RUTA_SALIDA = os.path.join("datasets", NOMBRE_MAESTRO)

print(f"--- ESCANEANDO CARPETAS: {CARPETAS_BUSQUEDA} ---")

archivos = []
for carpeta in CARPETAS_BUSQUEDA:
    patron = os.path.join(carpeta, "**", "*.csv")
    encontrados = glob.glob(patron, recursive=True)
    archivos.extend(encontrados)

# Filtramos para no intentar unir el archivo maestro consigo mismo si ya existe
archivos = [f for f in archivos if NOMBRE_MAESTRO not in f]

if archivos:
    print(f"✅ Se encontraron {len(archivos)} archivos de sesión.")
    lista_dfs = []
    
    for f in archivos:
        try:
            df = pd.read_csv(f)
            lista_dfs.append(df)
            # Mostramos solo el nombre del archivo para no saturar consola
            print(f"  -> Integrado: {os.path.basename(f)}")
        except Exception as e:
            print(f"  ❌ Error leyendo {os.path.basename(f)}: {e}")
    
    if lista_dfs:
        maestro = pd.concat(lista_dfs, ignore_index=True)
        maestro.to_csv(RUTA_SALIDA, index=False)
        
        print("\n" + "="*50)
        print(f"¡ÉXITO! Dataset Maestro generado en:\n{RUTA_SALIDA}")
        print("="*50)
        print(f"Total de registros: {len(maestro)}")
        print("Desglose por Usuario y Etiqueta:")
        print(maestro.groupby(['Usuario', 'Etiqueta']).size())
else:
    print("❌ No se encontraron archivos .csv en las subcarpetas de 'datasets' ni 'datasets_compartido'.")