"""
Modulo de evaluacion y visualizacion de modelos de clasificacion por categoria.

Este script genera graficos de evaluacion para clasificadores multicategoria:
- Heatmap de ranking IG (Information Gain)
- Importancia de caracteristicas por categoria
- Matriz de confusion
- Reporte de clasificacion

Autor: Proyecto Webcam Light Sensor
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.feature_selection import mutual_info_classif

# Configuracion de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Configuracion de matplotlib
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


class IGRanker:
    """Calcula el ranking de Information Gain por categoria."""

    def __init__(self, X, y, feature_names, categories):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.categories = categories
        self.ig_scores = None

    def compute_scores(self):
        """Calcula IG scores para cada feature por categoria."""
        self.ig_scores = pd.DataFrame(
            index=self.feature_names,
            columns=self.categories
        )

        for cat in self.categories:
            # Convertir a problema binario: categoria vs resto
            y_binary = (self.y == cat).astype(int)
            scores = mutual_info_classif(self.X, y_binary, random_state=42)
            self.ig_scores[cat] = scores

        return self.ig_scores

    def get_scores_matrix(self):
        """Retorna la matriz de scores para heatmap."""
        if self.ig_scores is None:
            self.compute_scores()
        return self.ig_scores


class MultiCategoryClassifier:
    """Clasificador multicategoria basado en Random Forest."""

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.feature_names = None
        self.categories = None
        self.is_fitted = False

    def fit(self, X, y):
        """Entrena el modelo."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.categories = sorted(list(set(y)))
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Realiza predicciones."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        """Retorna probabilidades de prediccion."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def get_feature_importances(self):
        """Retorna las importancias de features."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")
        return dict(zip(self.feature_names, self.model.feature_importances_))


def plot_ig_ranking_heatmap(ranker, output_path):
    """
    Crea un heatmap de scores de Information Gain (features x categorias).

    Args:
        ranker: IGRanker con scores calculados
        output_path: Ruta donde guardar la imagen
    """
    scores_matrix = ranker.get_scores_matrix()

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        scores_matrix.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={'label': 'Information Gain Score'},
        linewidths=0.5
    )

    plt.title("Ranking de Information Gain por Categoria", fontsize=12, fontweight='bold')
    plt.xlabel("Categoria", fontsize=11)
    plt.ylabel("Caracteristica", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def plot_category_feature_importance(model, category, output_path, top_n=None):
    """
    Grafico de barras horizontal de importancia de features para una categoria.

    Args:
        model: MultiCategoryClassifier entrenado
        category: Nombre de la categoria (para el titulo)
        output_path: Ruta donde guardar la imagen
        top_n: Numero de features a mostrar (None = todas)
    """
    importances = model.get_feature_importances()

    # Ordenar por importancia
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_features = sorted_features[:top_n]

    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Crear grafico de barras horizontal
    plt.figure(figsize=(8, max(4, len(features) * 0.5)))

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(features)))[::-1]

    bars = plt.barh(range(len(features)), values, color=colors)
    plt.yticks(range(len(features)), features)

    # Agregar valores en las barras
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.xlabel("Importancia", fontsize=11)
    plt.ylabel("Caracteristica", fontsize=11)
    plt.title(f"Importancia de Caracteristicas\n(Modelo: {category})",
              fontsize=12, fontweight='bold')
    plt.xlim(0, max(values) * 1.15)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def plot_confusion_matrix_per_category(y_true, y_pred, categories, output_path):
    """
    Crea matriz de confusion con heatmap de seaborn.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        categories: Lista de nombres de categorias
        output_path: Ruta donde guardar la imagen
    """
    cm = confusion_matrix(y_true, y_pred, labels=categories)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Cantidad'},
        linewidths=0.5
    )

    plt.title("Matriz de Confusion", fontsize=12, fontweight='bold')
    plt.xlabel("Prediccion", fontsize=11)
    plt.ylabel("Valor Real", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def generate_classification_report(y_true, y_pred, output_path):
    """
    Genera y guarda el reporte de clasificacion en CSV.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        output_path: Ruta donde guardar el CSV
    """
    # Generar reporte como diccionario
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # Convertir a DataFrame
    df_report = pd.DataFrame(report_dict).T

    # Redondear valores
    df_report = df_report.round(4)

    # Guardar CSV
    df_report.to_csv(output_path, index=True)
    print(f"  Guardado: {output_path}")

    return df_report


def run_full_evaluation(classifier, X_test, y_test, output_dir, X_train=None, y_train=None):
    """
    Ejecuta la evaluacion completa y genera todos los graficos.

    Args:
        classifier: MultiCategoryClassifier entrenado
        X_test: Features de test
        y_test: Etiquetas de test
        output_dir: Directorio donde guardar resultados
        X_train: Features de entrenamiento (para IG ranking)
        y_train: Etiquetas de entrenamiento (para IG ranking)

    Returns:
        dict: Resumen de la evaluacion
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("EVALUACION COMPLETA DEL MODELO")
    print("="*60)

    # Obtener predicciones
    y_pred = classifier.predict(X_test)
    categories = classifier.categories
    feature_names = classifier.feature_names

    # Calcular metricas basicas
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Categorias: {categories}")
    print(f"Caracteristicas: {feature_names}")
    print(f"\nGenerando visualizaciones...")

    # 1. Matriz de confusion
    print("\n1. Matriz de Confusion")
    plot_confusion_matrix_per_category(
        y_test, y_pred, categories,
        os.path.join(output_dir, "confusion_matrix.png")
    )

    # 2. Reporte de clasificacion
    print("\n2. Reporte de Clasificacion")
    df_report = generate_classification_report(
        y_test, y_pred,
        os.path.join(output_dir, "classification_report.csv")
    )

    # 3. Importancia de caracteristicas
    print("\n3. Importancia de Caracteristicas")
    plot_category_feature_importance(
        classifier, "Random Forest",
        os.path.join(output_dir, "feature_importance.png")
    )

    # 4. IG Ranking Heatmap (si hay datos de entrenamiento)
    if X_train is not None and y_train is not None:
        print("\n4. Ranking de Information Gain")
        if isinstance(X_train, pd.DataFrame):
            X_train_values = X_train.values
        else:
            X_train_values = X_train

        ranker = IGRanker(X_train_values, y_train, feature_names, categories)
        ranker.compute_scores()
        plot_ig_ranking_heatmap(
            ranker,
            os.path.join(output_dir, "ig_ranking_heatmap.png")
        )

    # 5. Grafico de precision por categoria
    print("\n5. Precision por Categoria")
    _plot_metrics_per_category(df_report, categories, output_dir)

    # Resumen
    summary = {
        'accuracy': accuracy,
        'n_samples_test': len(y_test),
        'n_features': len(feature_names),
        'n_categories': len(categories),
        'categories': categories,
        'feature_names': feature_names,
        'report': df_report
    }

    # Guardar resumen
    _save_summary(summary, output_dir)

    print("\n" + "="*60)
    print("EVALUACION COMPLETADA")
    print(f"Resultados guardados en: {output_dir}")
    print("="*60)

    return summary


def _plot_metrics_per_category(df_report, categories, output_dir):
    """Grafico de barras con precision, recall y f1-score por categoria."""
    metrics = ['precision', 'recall', 'f1-score']

    # Filtrar solo categorias (excluir accuracy, macro avg, etc.)
    df_cats = df_report.loc[df_report.index.isin(categories), metrics]

    # Crear grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.25

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [df_cats.loc[cat, metric] if cat in df_cats.index else 0
                  for cat in categories]
        bars = ax.bar(x + i*width, values, width, label=metric.capitalize(), color=color)

        # Agregar valores
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Categoria', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metricas de Clasificacion por Categoria', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "metrics_per_category.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def _save_summary(summary, output_dir):
    """Guarda el resumen de la evaluacion en archivo de texto."""
    output_path = os.path.join(output_dir, "evaluation_summary.txt")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RESUMEN DE EVALUACION DEL MODELO\n")
        f.write("="*60 + "\n\n")

        f.write(f"Accuracy Global: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)\n")
        f.write(f"Muestras de Test: {summary['n_samples_test']}\n")
        f.write(f"Numero de Caracteristicas: {summary['n_features']}\n")
        f.write(f"Numero de Categorias: {summary['n_categories']}\n\n")

        f.write("Categorias:\n")
        for cat in summary['categories']:
            f.write(f"  - {cat}\n")

        f.write("\nCaracteristicas:\n")
        for feat in summary['feature_names']:
            f.write(f"  - {feat}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("METRICAS POR CATEGORIA:\n")
        f.write("-"*60 + "\n\n")

        report = summary['report']
        for cat in summary['categories']:
            if cat in report.index:
                f.write(f"{cat}:\n")
                f.write(f"  Precision: {report.loc[cat, 'precision']:.4f}\n")
                f.write(f"  Recall:    {report.loc[cat, 'recall']:.4f}\n")
                f.write(f"  F1-Score:  {report.loc[cat, 'f1-score']:.4f}\n")
                f.write(f"  Support:   {int(report.loc[cat, 'support'])}\n\n")

    print(f"  Guardado: {output_path}")


def load_data(train_path, test_path):
    """
    Carga los datos de entrenamiento y test.

    Args:
        train_path: Ruta al archivo train.csv
        test_path: Ruta al archivo test.csv

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["Etiqueta"])
    y_train = df_train["Etiqueta"]

    X_test = df_test.drop(columns=["Etiqueta"])
    y_test = df_test["Etiqueta"]

    return X_train, y_train, X_test, y_test


# --- MAIN ---
if __name__ == "__main__":
    # Rutas
    TRAIN_PATH = os.path.join(BASE_DIR, "datasets", "train.csv")
    TEST_PATH = os.path.join(BASE_DIR, "datasets", "test.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_graficos", "evaluacion_modelo")

    print("\n" + "="*60)
    print("MODULO DE EVALUACION DE MODELOS POR CATEGORIA")
    print("="*60)

    # Verificar que existen los archivos
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: No se encuentra {TRAIN_PATH}")
        sys.exit(1)
    if not os.path.exists(TEST_PATH):
        print(f"Error: No se encuentra {TEST_PATH}")
        sys.exit(1)

    # Cargar datos
    print("\nCargando datos...")
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)

    print(f"  Train: {len(X_train)} muestras")
    print(f"  Test:  {len(X_test)} muestras")
    print(f"  Caracteristicas: {list(X_train.columns)}")
    print(f"  Categorias: {sorted(y_train.unique())}")

    # Entrenar clasificador
    print("\nEntrenando MultiCategoryClassifier...")
    classifier = MultiCategoryClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    print("  Modelo entrenado correctamente.")

    # Ejecutar evaluacion completa
    summary = run_full_evaluation(
        classifier=classifier,
        X_test=X_test,
        y_test=y_test,
        output_dir=OUTPUT_DIR,
        X_train=X_train,
        y_train=y_train
    )

    # Imprimir resumen final
    print("\n" + "-"*60)
    print("RESUMEN FINAL")
    print("-"*60)
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print("\nMetricas por categoria:")
    report = summary['report']
    for cat in summary['categories']:
        if cat in report.index:
            f1 = report.loc[cat, 'f1-score']
            print(f"  {cat}: F1={f1:.4f}")

    print(f"\nArchivos generados en: {OUTPUT_DIR}")
