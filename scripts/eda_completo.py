#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANALISIS EXPLORATORIO DE DATOS (EDA) COMPLETO
================================================================================
Proyecto: webcam-light-sensor
Descripcion: Script completo de EDA con estadisticas detalladas por columna,
             analisis por categoria y visualizaciones comprehensivas.

Autor: Orlando
Fecha: Enero 2026
================================================================================
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suprimir warnings para output limpio
warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURACION
# ================================================================================

# Rutas del proyecto
PROJECT_ROOT = Path("/home/orlando/maestria/webcam-light-sensor")
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "resultados_graficos" / "eda"

# Crear directorio de salida si no existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columnas objetivo para analisis
FEATURE_COLUMNS = ['std_v', 'mean_b', 'skew_v', 'v_95']

# Columna de categoria
CATEGORY_COLUMN = 'Etiqueta'

# Mapeo de categorias para nombres cortos
CATEGORY_MAP = {
    'natural': 'NA',
    'artificial': 'AR',
    'pantallas': 'PAN',
    'mix': 'MIX'
}

# Configuracion de visualizacion
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")


# ================================================================================
# FUNCIONES AUXILIARES
# ================================================================================

def cargar_dataset():
    """
    Carga el dataset maestro o combina train/test si no existe.

    Returns:
        pd.DataFrame: Dataset completo cargado
    """
    print("\n" + "="*80)
    print("CARGANDO DATASET")
    print("="*80)

    maestro_path = DATASETS_DIR / "DATASET_MAESTRO_COMPLETO.csv"

    if maestro_path.exists():
        print(f"[OK] Cargando DATASET_MAESTRO_COMPLETO.csv...")
        df = pd.read_csv(maestro_path)
        print(f"    - Registros cargados: {len(df):,}")
        return df

    # Si no existe el maestro, combinar train y test
    train_path = DATASETS_DIR / "train.csv"
    test_path = DATASETS_DIR / "test.csv"

    if train_path.exists() and test_path.exists():
        print("[INFO] DATASET_MAESTRO_COMPLETO.csv no encontrado.")
        print("       Combinando train.csv + test.csv...")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df = pd.concat([df_train, df_test], ignore_index=True)

        print(f"    - Registros train: {len(df_train):,}")
        print(f"    - Registros test: {len(df_test):,}")
        print(f"    - Total combinado: {len(df):,}")
        return df

    raise FileNotFoundError(
        "No se encontro DATASET_MAESTRO_COMPLETO.csv ni train.csv/test.csv"
    )


def calcular_estadisticas_columna(serie: pd.Series) -> dict:
    """
    Calcula estadisticas descriptivas completas para una columna.

    Args:
        serie: Serie de pandas con los datos numericos

    Returns:
        dict: Diccionario con todas las estadisticas calculadas
    """
    # Eliminar valores nulos para calculos
    datos = serie.dropna()

    # Estadisticas basicas
    count = len(datos)
    missing = serie.isna().sum()
    missing_pct = (missing / len(serie)) * 100 if len(serie) > 0 else 0

    # Min, Max, Rango
    min_val = datos.min() if count > 0 else np.nan
    max_val = datos.max() if count > 0 else np.nan
    rango = max_val - min_val if count > 0 else np.nan

    # Tendencia central
    mean_val = datos.mean() if count > 0 else np.nan
    median_val = datos.median() if count > 0 else np.nan

    # Moda (puede haber multiples, tomamos la primera)
    try:
        mode_result = stats.mode(datos, keepdims=True)
        mode_val = mode_result.mode[0] if count > 0 else np.nan
    except:
        mode_val = datos.mode().iloc[0] if count > 0 and len(datos.mode()) > 0 else np.nan

    # Dispersion
    std_val = datos.std() if count > 1 else np.nan
    var_val = datos.var() if count > 1 else np.nan

    # Forma de distribucion
    skewness = stats.skew(datos) if count > 2 else np.nan
    kurtosis = stats.kurtosis(datos) if count > 3 else np.nan

    # Percentiles
    percentiles = {}
    if count > 0:
        for p in [1, 5, 25, 50, 75, 95, 99]:
            percentiles[f'p{p}'] = np.percentile(datos, p)
    else:
        for p in [1, 5, 25, 50, 75, 95, 99]:
            percentiles[f'p{p}'] = np.nan

    # IQR y limites de outliers
    q1 = percentiles['p25']
    q3 = percentiles['p75']
    iqr = q3 - q1 if count > 0 else np.nan

    # Limites para outliers (metodo IQR)
    limite_inferior = q1 - 1.5 * iqr if not np.isnan(iqr) else np.nan
    limite_superior = q3 + 1.5 * iqr if not np.isnan(iqr) else np.nan

    # Contar outliers
    if count > 0 and not np.isnan(limite_inferior):
        n_outliers = ((datos < limite_inferior) | (datos > limite_superior)).sum()
        pct_outliers = (n_outliers / count) * 100
    else:
        n_outliers = 0
        pct_outliers = 0

    return {
        'count': count,
        'missing': missing,
        'missing_pct': missing_pct,
        'min': min_val,
        'max': max_val,
        'range': rango,
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'std': std_val,
        'variance': var_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'p1': percentiles['p1'],
        'p5': percentiles['p5'],
        'p25': percentiles['p25'],
        'p50': percentiles['p50'],
        'p75': percentiles['p75'],
        'p95': percentiles['p95'],
        'p99': percentiles['p99'],
        'iqr': iqr,
        'outlier_lower': limite_inferior,
        'outlier_upper': limite_superior,
        'n_outliers': n_outliers,
        'pct_outliers': pct_outliers
    }


def calcular_estadisticas_por_categoria(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Calcula estadisticas para una columna agrupada por categoria.

    Args:
        df: DataFrame con los datos
        columna: Nombre de la columna a analizar

    Returns:
        pd.DataFrame: Estadisticas por categoria
    """
    resultados = []

    for categoria in df[CATEGORY_COLUMN].unique():
        datos_cat = df[df[CATEGORY_COLUMN] == categoria][columna].dropna()

        if len(datos_cat) > 0:
            cat_short = CATEGORY_MAP.get(categoria, categoria)

            resultados.append({
                'columna': columna,
                'categoria': categoria,
                'cat_short': cat_short,
                'count': len(datos_cat),
                'min': datos_cat.min(),
                'max': datos_cat.max(),
                'range': datos_cat.max() - datos_cat.min(),
                'mean': datos_cat.mean(),
                'median': datos_cat.median(),
                'std': datos_cat.std(),
                'cv': (datos_cat.std() / datos_cat.mean() * 100) if datos_cat.mean() != 0 else np.nan,
                'skewness': stats.skew(datos_cat),
                'kurtosis': stats.kurtosis(datos_cat),
                'p25': np.percentile(datos_cat, 25),
                'p75': np.percentile(datos_cat, 75)
            })

    return pd.DataFrame(resultados)


# ================================================================================
# FUNCIONES DE VISUALIZACION
# ================================================================================

def crear_histograma_por_categoria(df: pd.DataFrame, columna: str, ax=None):
    """
    Crea histograma con overlay por categoria.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Colores para cada categoria
    colores = {'natural': '#2ecc71', 'artificial': '#e74c3c',
               'pantallas': '#3498db', 'mix': '#9b59b6'}

    for categoria in sorted(df[CATEGORY_COLUMN].unique()):
        datos = df[df[CATEGORY_COLUMN] == categoria][columna].dropna()
        cat_short = CATEGORY_MAP.get(categoria, categoria)
        ax.hist(datos, bins=50, alpha=0.5, label=f'{cat_short} (n={len(datos)})',
                color=colores.get(categoria, '#95a5a6'), edgecolor='white', linewidth=0.5)

    ax.set_xlabel(columna, fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.set_title(f'Distribucion de {columna} por Categoria', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    return ax


def crear_boxplot_por_categoria(df: pd.DataFrame, columna: str, ax=None):
    """
    Crea boxplot agrupado por categoria.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Preparar datos con nombres cortos
    df_plot = df.copy()
    df_plot['categoria_short'] = df_plot[CATEGORY_COLUMN].map(CATEGORY_MAP)

    # Orden de categorias
    orden = ['NA', 'AR', 'PAN', 'MIX']
    orden_disponible = [c for c in orden if c in df_plot['categoria_short'].unique()]

    colores = {'NA': '#2ecc71', 'AR': '#e74c3c', 'PAN': '#3498db', 'MIX': '#9b59b6'}
    palette = {c: colores.get(c, '#95a5a6') for c in orden_disponible}

    sns.boxplot(data=df_plot, x='categoria_short', y=columna, order=orden_disponible,
                palette=palette, ax=ax, width=0.6)

    ax.set_xlabel('Categoria', fontsize=11)
    ax.set_ylabel(columna, fontsize=11)
    ax.set_title(f'Boxplot de {columna} por Categoria', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    return ax


def crear_violin_por_categoria(df: pd.DataFrame, columna: str, ax=None):
    """
    Crea violin plot agrupado por categoria.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Preparar datos
    df_plot = df.copy()
    df_plot['categoria_short'] = df_plot[CATEGORY_COLUMN].map(CATEGORY_MAP)

    orden = ['NA', 'AR', 'PAN', 'MIX']
    orden_disponible = [c for c in orden if c in df_plot['categoria_short'].unique()]

    colores = {'NA': '#2ecc71', 'AR': '#e74c3c', 'PAN': '#3498db', 'MIX': '#9b59b6'}
    palette = {c: colores.get(c, '#95a5a6') for c in orden_disponible}

    sns.violinplot(data=df_plot, x='categoria_short', y=columna, order=orden_disponible,
                   palette=palette, ax=ax, inner='box', cut=0)

    ax.set_xlabel('Categoria', fontsize=11)
    ax.set_ylabel(columna, fontsize=11)
    ax.set_title(f'Violin Plot de {columna} por Categoria', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    return ax


def crear_heatmap_estadisticas(df_stats: pd.DataFrame):
    """
    Crea un heatmap de estadisticas resumen.
    """
    # Preparar datos para heatmap
    metricas = ['mean', 'std', 'min', 'max', 'median', 'skewness']

    # Crear matriz pivoteada
    pivot_data = {}
    for metrica in metricas:
        if metrica in df_stats.columns:
            pivot_data[metrica] = df_stats.set_index('columna')[metrica]

    df_heatmap = pd.DataFrame(pivot_data)

    # Normalizar para mejor visualizacion
    df_normalized = (df_heatmap - df_heatmap.min()) / (df_heatmap.max() - df_heatmap.min())

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(df_normalized.T, annot=df_heatmap.T.round(2), fmt='', cmap='RdYlBu_r',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Valor Normalizado'})

    ax.set_title('Heatmap de Estadisticas por Columna\n(valores originales anotados, colores normalizados)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Columnas', fontsize=11)
    ax.set_ylabel('Metricas', fontsize=11)

    plt.tight_layout()
    return fig


# ================================================================================
# FUNCIONES PRINCIPALES DE ANALISIS
# ================================================================================

def analizar_estadisticas_por_columna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla completa de estadisticas para cada columna objetivo.
    """
    print("\n" + "="*80)
    print("ESTADISTICAS DESCRIPTIVAS POR COLUMNA")
    print("="*80)

    resultados = []

    for columna in FEATURE_COLUMNS:
        if columna in df.columns:
            print(f"\n[*] Analizando: {columna}")
            stats_dict = calcular_estadisticas_columna(df[columna])
            stats_dict['columna'] = columna
            resultados.append(stats_dict)

            # Mostrar resumen en consola
            print(f"    Count: {stats_dict['count']:,} | Missing: {stats_dict['missing']} ({stats_dict['missing_pct']:.2f}%)")
            print(f"    Min: {stats_dict['min']:.4f} | Max: {stats_dict['max']:.4f} | Range: {stats_dict['range']:.4f}")
            print(f"    Mean: {stats_dict['mean']:.4f} | Median: {stats_dict['median']:.4f} | Mode: {stats_dict['mode']:.4f}")
            print(f"    Std: {stats_dict['std']:.4f} | Var: {stats_dict['variance']:.4f}")
            print(f"    Skewness: {stats_dict['skewness']:.4f} | Kurtosis: {stats_dict['kurtosis']:.4f}")
            print(f"    IQR: {stats_dict['iqr']:.4f} | Outliers: {stats_dict['n_outliers']} ({stats_dict['pct_outliers']:.2f}%)")
        else:
            print(f"\n[!] ADVERTENCIA: Columna '{columna}' no encontrada en dataset")

    # Crear DataFrame y reordenar columnas
    df_stats = pd.DataFrame(resultados)
    cols_order = ['columna', 'count', 'missing', 'missing_pct', 'min', 'max', 'range',
                  'mean', 'median', 'mode', 'std', 'variance', 'skewness', 'kurtosis',
                  'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'iqr',
                  'outlier_lower', 'outlier_upper', 'n_outliers', 'pct_outliers']

    df_stats = df_stats[[c for c in cols_order if c in df_stats.columns]]

    return df_stats


def analizar_estadisticas_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla de estadisticas agrupadas por categoria.
    """
    print("\n" + "="*80)
    print("ESTADISTICAS POR CATEGORIA")
    print("="*80)

    resultados = []

    for columna in FEATURE_COLUMNS:
        if columna in df.columns:
            df_cat = calcular_estadisticas_por_categoria(df, columna)
            resultados.append(df_cat)

            print(f"\n[*] {columna}:")
            print("-" * 70)
            for _, row in df_cat.iterrows():
                print(f"    {row['cat_short']:>4}: n={row['count']:>5} | "
                      f"Min={row['min']:>8.2f} | Max={row['max']:>8.2f} | "
                      f"Mean={row['mean']:>8.2f} | Std={row['std']:>7.2f}")

    df_stats_cat = pd.concat(resultados, ignore_index=True)
    return df_stats_cat


def generar_resumen_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla resumen con min/max por columna y categoria.
    """
    print("\n" + "="*80)
    print("RESUMEN MIN/MAX POR COLUMNA Y CATEGORIA")
    print("="*80)

    resultados = []

    for columna in FEATURE_COLUMNS:
        if columna not in df.columns:
            continue

        # Global
        datos = df[columna].dropna()
        resultados.append({
            'columna': columna,
            'categoria': 'GLOBAL',
            'min': datos.min(),
            'max': datos.max(),
            'range': datos.max() - datos.min()
        })

        # Por categoria
        for categoria in sorted(df[CATEGORY_COLUMN].unique()):
            datos_cat = df[df[CATEGORY_COLUMN] == categoria][columna].dropna()
            cat_short = CATEGORY_MAP.get(categoria, categoria)
            resultados.append({
                'columna': columna,
                'categoria': cat_short,
                'min': datos_cat.min(),
                'max': datos_cat.max(),
                'range': datos_cat.max() - datos_cat.min()
            })

    df_minmax = pd.DataFrame(resultados)

    # Mostrar en consola
    for columna in FEATURE_COLUMNS:
        if columna in df.columns:
            print(f"\n{columna}:")
            df_col = df_minmax[df_minmax['columna'] == columna]
            for _, row in df_col.iterrows():
                print(f"    {row['categoria']:>6}: [{row['min']:>10.4f}, {row['max']:>10.4f}] (rango: {row['range']:.4f})")

    return df_minmax


def generar_visualizaciones(df: pd.DataFrame):
    """
    Genera todas las visualizaciones y las guarda como PNG.
    """
    print("\n" + "="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80)

    # 1. Panel de distribuciones (histogramas)
    print("\n[*] Generando histogramas por columna...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, columna in enumerate(FEATURE_COLUMNS):
        if columna in df.columns:
            crear_histograma_por_categoria(df, columna, ax=axes[i])

    plt.suptitle('Distribucion de Variables por Categoria', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'histogramas_por_categoria.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [OK] Guardado: histogramas_por_categoria.png")

    # 2. Panel de boxplots
    print("\n[*] Generando boxplots por columna...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, columna in enumerate(FEATURE_COLUMNS):
        if columna in df.columns:
            crear_boxplot_por_categoria(df, columna, ax=axes[i])

    plt.suptitle('Boxplots de Variables por Categoria', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'boxplots_por_categoria.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [OK] Guardado: boxplots_por_categoria.png")

    # 3. Panel de violin plots
    print("\n[*] Generando violin plots por columna...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, columna in enumerate(FEATURE_COLUMNS):
        if columna in df.columns:
            crear_violin_por_categoria(df, columna, ax=axes[i])

    plt.suptitle('Violin Plots de Variables por Categoria', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'violinplots_por_categoria.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [OK] Guardado: violinplots_por_categoria.png")

    # 4. Graficos individuales por columna
    print("\n[*] Generando graficos individuales...")
    for columna in FEATURE_COLUMNS:
        if columna not in df.columns:
            continue

        # Histograma individual
        fig, ax = plt.subplots(figsize=(12, 7))
        crear_histograma_por_categoria(df, columna, ax=ax)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f'hist_{columna}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    [OK] hist_{columna}.png")

        # Boxplot individual
        fig, ax = plt.subplots(figsize=(10, 7))
        crear_boxplot_por_categoria(df, columna, ax=ax)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f'boxplot_{columna}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    [OK] boxplot_{columna}.png")

        # Violin individual
        fig, ax = plt.subplots(figsize=(10, 7))
        crear_violin_por_categoria(df, columna, ax=ax)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f'violin_{columna}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    [OK] violin_{columna}.png")


def imprimir_resumen_final(df: pd.DataFrame, df_stats: pd.DataFrame, df_stats_cat: pd.DataFrame):
    """
    Imprime resumen comprehensivo en consola.
    """
    print("\n" + "="*80)
    print("RESUMEN FINAL DEL ANALISIS EXPLORATORIO")
    print("="*80)

    print(f"\n[DATASET]")
    print(f"  - Total de registros: {len(df):,}")
    print(f"  - Columnas analizadas: {', '.join(FEATURE_COLUMNS)}")

    print(f"\n[DISTRIBUCION POR CATEGORIA]")
    for categoria in sorted(df[CATEGORY_COLUMN].unique()):
        count = len(df[df[CATEGORY_COLUMN] == categoria])
        pct = count / len(df) * 100
        cat_short = CATEGORY_MAP.get(categoria, categoria)
        print(f"  - {cat_short:>4} ({categoria:>10}): {count:>5} registros ({pct:>5.1f}%)")

    print(f"\n[ESTADISTICAS CLAVE]")
    for _, row in df_stats.iterrows():
        print(f"\n  {row['columna'].upper()}:")
        print(f"    Rango: [{row['min']:.4f}, {row['max']:.4f}]")
        print(f"    Central: mean={row['mean']:.4f}, median={row['median']:.4f}")
        print(f"    Dispersion: std={row['std']:.4f}")
        print(f"    Forma: skew={row['skewness']:.4f}, kurt={row['kurtosis']:.4f}")
        print(f"    Outliers: {int(row['n_outliers'])} ({row['pct_outliers']:.1f}%)")

    print(f"\n[COMPARACION ENTRE CATEGORIAS]")
    for columna in FEATURE_COLUMNS:
        df_col = df_stats_cat[df_stats_cat['columna'] == columna]
        if len(df_col) > 0:
            print(f"\n  {columna}:")
            # Categoria con mayor media
            max_mean_cat = df_col.loc[df_col['mean'].idxmax()]
            min_mean_cat = df_col.loc[df_col['mean'].idxmin()]
            print(f"    Mayor media: {max_mean_cat['cat_short']} ({max_mean_cat['mean']:.4f})")
            print(f"    Menor media: {min_mean_cat['cat_short']} ({min_mean_cat['mean']:.4f})")
            print(f"    Diferencia: {max_mean_cat['mean'] - min_mean_cat['mean']:.4f}")

    print("\n" + "="*80)
    print("ARCHIVOS GENERADOS")
    print("="*80)
    print(f"\n[CSV]")
    print(f"  - {OUTPUT_DIR}/estadisticas_por_columna.csv")
    print(f"  - {OUTPUT_DIR}/estadisticas_por_categoria.csv")
    print(f"  - {OUTPUT_DIR}/resumen_minmax.csv")
    print(f"\n[PNG]")
    print(f"  - {OUTPUT_DIR}/histogramas_por_categoria.png")
    print(f"  - {OUTPUT_DIR}/boxplots_por_categoria.png")
    print(f"  - {OUTPUT_DIR}/violinplots_por_categoria.png")
    for col in FEATURE_COLUMNS:
        print(f"  - {OUTPUT_DIR}/hist_{col}.png")
        print(f"  - {OUTPUT_DIR}/boxplot_{col}.png")
        print(f"  - {OUTPUT_DIR}/violin_{col}.png")


# ================================================================================
# FUNCION PRINCIPAL
# ================================================================================

def main():
    """
    Funcion principal que ejecuta todo el analisis EDA.
    """
    print("\n" + "#"*80)
    print("#" + " "*30 + "EDA COMPLETO" + " "*35 + "#")
    print("#" + " "*20 + "webcam-light-sensor project" + " "*30 + "#")
    print("#"*80)

    try:
        # 1. Cargar datos
        df = cargar_dataset()

        # 2. Verificar columnas objetivo
        columnas_faltantes = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if columnas_faltantes:
            print(f"\n[ADVERTENCIA] Columnas no encontradas: {columnas_faltantes}")

        # 3. Estadisticas por columna
        df_stats = analizar_estadisticas_por_columna(df)
        df_stats.to_csv(OUTPUT_DIR / 'estadisticas_por_columna.csv', index=False)
        print(f"\n[OK] Guardado: estadisticas_por_columna.csv")

        # 4. Estadisticas por categoria
        df_stats_cat = analizar_estadisticas_por_categoria(df)
        df_stats_cat.to_csv(OUTPUT_DIR / 'estadisticas_por_categoria.csv', index=False)
        print(f"\n[OK] Guardado: estadisticas_por_categoria.csv")

        # 5. Resumen min/max
        df_minmax = generar_resumen_minmax(df)
        df_minmax.to_csv(OUTPUT_DIR / 'resumen_minmax.csv', index=False)
        print(f"\n[OK] Guardado: resumen_minmax.csv")

        # 6. Visualizaciones
        generar_visualizaciones(df)

        # 7. Resumen final
        imprimir_resumen_final(df, df_stats, df_stats_cat)

        print("\n" + "#"*80)
        print("#" + " "*25 + "ANALISIS COMPLETADO" + " "*33 + "#")
        print("#"*80 + "\n")

        return 0

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
