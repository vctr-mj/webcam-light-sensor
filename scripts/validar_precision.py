"""
Precision Validation Script for Multi-Category Classifier.

This script evaluates the trained model's performance at different
probability thresholds, comparing standard accuracy mode vs precision mode.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.multi_category_classifier import MultiCategoryClassifier


def plot_confusion_matrix(
    cm: np.ndarray,
    categories: list,
    title: str,
    output_path: Path
) -> None:
    """
    Plot and save a confusion matrix.

    Args:
        cm: Confusion matrix array.
        categories: List of category names.
        title: Plot title.
        output_path: Path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories
    )
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Run precision validation analysis."""
    # Paths
    model_path = PROJECT_ROOT / 'modelos' / 'multi_category_classifier.pkl'
    test_path = PROJECT_ROOT / 'datasets' / 'test.csv'
    output_dir = PROJECT_ROOT / 'resultados_graficos' / 'precision_analysis'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANALISIS DE PRECISION - Multi-Category Classifier")
    print("=" * 70)

    # Load model
    print("\n[1/5] Cargando modelo entrenado...")
    if not model_path.exists():
        print(f"  ERROR: Modelo no encontrado en {model_path}")
        print("  Ejecute primero: python modelos/multi_category_classifier.py")
        return

    classifier = MultiCategoryClassifier.load(model_path)
    print(f"  Modelo cargado: {classifier}")

    # Load test data
    print("\n[2/5] Cargando datos de prueba...")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=['Etiqueta'])
    y_test = test_df['Etiqueta']
    print(f"  Muestras de prueba: {len(X_test)}")
    print(f"  Distribucion de clases:")
    for cat, count in y_test.value_counts().items():
        print(f"    - {cat}: {count}")

    # Compare thresholds
    print("\n[3/5] Comparando umbrales de probabilidad...")
    thresholds_df = classifier.compare_thresholds(X_test, y_test)

    print("\n" + "-" * 70)
    print("COMPARACION DE UMBRALES")
    print("-" * 70)
    print(thresholds_df.to_string(index=False))
    print("-" * 70)

    # Save threshold comparison
    thresholds_csv_path = output_dir / 'threshold_comparison.csv'
    thresholds_df.to_csv(thresholds_csv_path, index=False)
    print(f"\n  Tabla guardada en: {thresholds_csv_path}")

    # Confusion matrix for threshold=0.5 (standard mode)
    print("\n[4/5] Generando matriz de confusion (threshold=0.5)...")
    eval_05 = classifier.evaluate_precision_mode(X_test, y_test, threshold=0.5)
    cm_05_path = output_dir / 'confusion_matrix_threshold_0.5.png'
    plot_confusion_matrix(
        eval_05['confusion_matrix'],
        classifier.categories,
        f"Matriz de Confusion (Threshold=0.5)\n"
        f"Accuracy: {eval_05['accuracy']:.3f} | Decididos: {eval_05['decided_samples']}/{eval_05['total_samples']}",
        cm_05_path
    )
    print(f"  Guardada: {cm_05_path}")

    # Confusion matrix for threshold=0.7 (precision mode)
    print("\n[5/5] Generando matriz de confusion (threshold=0.7)...")
    eval_07 = classifier.evaluate_precision_mode(X_test, y_test, threshold=0.7)
    cm_07_path = output_dir / 'confusion_matrix_threshold_0.7.png'
    plot_confusion_matrix(
        eval_07['confusion_matrix'],
        classifier.categories,
        f"Matriz de Confusion (Threshold=0.7)\n"
        f"Accuracy: {eval_07['accuracy']:.3f} | Decididos: {eval_07['decided_samples']}/{eval_07['total_samples']}",
        cm_07_path
    )
    print(f"  Guardada: {cm_07_path}")

    # Plot threshold comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Accuracy/Precision/Recall vs Threshold
    ax1 = axes[0]
    ax1.plot(thresholds_df['threshold'], thresholds_df['accuracy'], 'b-o', label='Accuracy', linewidth=2)
    ax1.plot(thresholds_df['threshold'], thresholds_df['precision'], 'g-s', label='Precision', linewidth=2)
    ax1.plot(thresholds_df['threshold'], thresholds_df['recall'], 'r-^', label='Recall', linewidth=2)
    ax1.set_xlabel('Umbral de Probabilidad')
    ax1.set_ylabel('Metrica')
    ax1.set_title('Metricas vs Umbral de Decision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right plot: Decided vs No Decidido
    ax2 = axes[1]
    ax2.bar(thresholds_df['threshold'] - 0.02, thresholds_df['decided'],
            width=0.04, label='Decididos', color='green', alpha=0.7)
    ax2.bar(thresholds_df['threshold'] + 0.02, thresholds_df['no_decidido'],
            width=0.04, label='No Decididos', color='red', alpha=0.7)
    ax2.set_xlabel('Umbral de Probabilidad')
    ax2.set_ylabel('Numero de Muestras')
    ax2.set_title('Muestras Decididas vs No Decididas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_plot_path = output_dir / 'threshold_comparison_plot.png'
    plt.savefig(comparison_plot_path, dpi=150)
    plt.close()
    print(f"\n  Grafico comparativo guardado: {comparison_plot_path}")

    # Print summary comparison
    print("\n" + "=" * 70)
    print("RESUMEN: MODO ESTANDAR vs MODO PRECISION")
    print("=" * 70)

    # Standard mode (threshold=0.5)
    print("\n[MODO ESTANDAR - Threshold 0.5]")
    print(f"  Total muestras:     {eval_05['total_samples']}")
    print(f"  Muestras decididas: {eval_05['decided_samples']} ({100 - eval_05['no_decidido_percentage']:.1f}%)")
    print(f"  Accuracy:           {eval_05['accuracy']:.4f}")
    print(f"  Precision:          {eval_05['precision']:.4f}")
    print(f"  Recall:             {eval_05['recall']:.4f}")

    # Precision mode (threshold=0.7)
    print("\n[MODO PRECISION - Threshold 0.7]")
    print(f"  Total muestras:     {eval_07['total_samples']}")
    print(f"  Muestras decididas: {eval_07['decided_samples']} ({100 - eval_07['no_decidido_percentage']:.1f}%)")
    print(f"  No decididas:       {eval_07['no_decidido_count']} ({eval_07['no_decidido_percentage']:.1f}%)")
    print(f"  Accuracy:           {eval_07['accuracy']:.4f}")
    print(f"  Precision:          {eval_07['precision']:.4f}")
    print(f"  Recall:             {eval_07['recall']:.4f}")

    # Improvement analysis
    if eval_05['accuracy'] > 0:
        acc_improvement = (eval_07['accuracy'] - eval_05['accuracy']) / eval_05['accuracy'] * 100
        prec_improvement = (eval_07['precision'] - eval_05['precision']) / eval_05['precision'] * 100
        print("\n[MEJORA CON MODO PRECISION]")
        print(f"  Mejora en Accuracy:  {acc_improvement:+.2f}%")
        print(f"  Mejora en Precision: {prec_improvement:+.2f}%")
        print(f"  Costo (no decididos): {eval_07['no_decidido_percentage']:.1f}% de muestras")

    # Per-category breakdown
    print("\n[DETALLE POR CATEGORIA - Threshold 0.7]")
    print("-" * 50)
    print(f"{'Categoria':<15} {'Total':>8} {'Correctos':>10} {'Accuracy':>10}")
    print("-" * 50)
    for cat in classifier.categories:
        cat_data = eval_07['per_category'][cat]
        print(f"{cat:<15} {cat_data['total']:>8} {cat_data['correct']:>10} {cat_data['accuracy']:>10.3f}")
    print("-" * 50)

    print("\n" + "=" * 70)
    print("Analisis completado. Resultados guardados en:")
    print(f"  {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
