#!/usr/bin/env python3
"""
Comprehensive Validation Script for MultiCategoryClassifier.

This script validates the MultiCategoryClassifier system with Information Gain rankings,
including model loading/training, evaluation metrics, confusion matrix visualization,
and comparison between standard and precision modes.

Author: Generated for webcam-light-sensor project
Date: 2025-01-27
"""

import sys
from pathlib import Path
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def load_or_train_model():
    """
    Load the existing trained model or train a new one if loading fails.

    Returns:
        tuple: (classifier, X_train, y_train, X_test, y_test, feature_cols)
    """
    from modelos.multi_category_classifier import MultiCategoryClassifier

    # Define paths
    model_path = PROJECT_ROOT / 'modelos' / 'multi_category_classifier.pkl'
    train_path = PROJECT_ROOT / 'datasets' / 'train.csv'
    test_path = PROJECT_ROOT / 'datasets' / 'test.csv'

    # Feature columns
    feature_cols = ['std_v', 'mean_b', 'skew_v', 'v_95']

    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[feature_cols]
    y_train = train_df['Etiqueta']
    X_test = test_df[feature_cols]
    y_test = test_df['Etiqueta']

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {feature_cols}")
    print(f"  Categories: {y_train.value_counts().to_dict()}")

    # Try to load existing model
    try:
        print(f"\nAttempting to load model from: {model_path}")
        clf = MultiCategoryClassifier.load(model_path)
        print("  Model loaded successfully!")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        print("\nTraining new model...")

        clf = MultiCategoryClassifier(top_k=4)
        clf.fit(X_train, y_train)

        # Save the newly trained model
        clf.save(model_path)
        print(f"  New model saved to: {model_path}")

    return clf, X_train, y_train, X_test, y_test, feature_cols


def print_ig_rankings(clf):
    """
    Print Information Gain rankings from the classifier.

    Args:
        clf: Fitted MultiCategoryClassifier instance
    """
    print("\n" + "=" * 70)
    print("INFORMATION GAIN RANKINGS")
    print("=" * 70)

    rankings = clf.get_rankings()

    print("\nComplete IG Matrix (Features x Categories):")
    print("-" * 70)
    print(rankings.to_string())

    print("\n\nTop Features by Category:")
    print("-" * 70)

    for category in clf.categories:
        category_scores = rankings[category].sort_values(ascending=False)
        print(f"\n{category.upper()}:")
        for i, (feature, score) in enumerate(category_scores.items(), 1):
            print(f"  {i}. {feature}: {score:.6f}")

    # Global ranking (average across categories)
    print("\n\nGlobal Feature Ranking (Average IG):")
    print("-" * 70)
    global_avg = rankings.mean(axis=1).sort_values(ascending=False)
    for i, (feature, score) in enumerate(global_avg.items(), 1):
        print(f"  {i}. {feature}: {score:.6f}")


def print_classification_report(clf, X_test, y_test):
    """
    Print classification report with detailed metrics.

    Args:
        clf: Fitted MultiCategoryClassifier instance
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Classification report dictionary
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = clf.evaluate(X_test, y_test)

    print(f"\n{'Category':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 65)

    for category in clf.categories:
        metrics = report[category]
        print(
            f"{category:<15} "
            f"{metrics['precision']:>12.4f} "
            f"{metrics['recall']:>12.4f} "
            f"{metrics['f1-score']:>12.4f} "
            f"{int(metrics['support']):>10}"
        )

    print("-" * 65)
    print(
        f"{'Accuracy':<15} "
        f"{'':>12} "
        f"{'':>12} "
        f"{report['accuracy']:>12.4f} "
        f"{int(report['weighted avg']['support']):>10}"
    )
    print(
        f"{'Macro Avg':<15} "
        f"{report['macro avg']['precision']:>12.4f} "
        f"{report['macro avg']['recall']:>12.4f} "
        f"{report['macro avg']['f1-score']:>12.4f} "
        f"{int(report['macro avg']['support']):>10}"
    )
    print(
        f"{'Weighted Avg':<15} "
        f"{report['weighted avg']['precision']:>12.4f} "
        f"{report['weighted avg']['recall']:>12.4f} "
        f"{report['weighted avg']['f1-score']:>12.4f} "
        f"{int(report['weighted avg']['support']):>10}"
    )

    return report


def generate_confusion_matrix(clf, X_test, y_test):
    """
    Generate and save confusion matrix plot.

    Args:
        clf: Fitted MultiCategoryClassifier instance
        X_test: Test features
        y_test: Test labels

    Returns:
        Path: Path to saved confusion matrix image
    """
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    y_pred = clf.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.categories)

    # Print numerical confusion matrix
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("-" * 70)
    cm_df = pd.DataFrame(cm, index=clf.categories, columns=clf.categories)
    print(cm_df.to_string())

    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, category in enumerate(clf.categories):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"  {category}: {accuracy:.2%} ({class_correct}/{class_total})")

    # Create and save visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.categories)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix - MultiCategoryClassifier', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the figure
    output_dir = PROJECT_ROOT / 'resultados_graficos' / 'validacion'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'confusion_matrix_multi_category.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")
    return output_path


def compare_prediction_modes(clf, X_test, y_test, threshold=0.7):
    """
    Compare standard prediction mode vs precision mode (with confidence threshold).

    In precision mode, predictions with max probability below threshold are marked
    as 'uncertain' and excluded from evaluation.

    Args:
        clf: Fitted MultiCategoryClassifier instance
        X_test: Test features
        y_test: Test labels
        threshold: Confidence threshold for precision mode
    """
    print("\n" + "=" * 70)
    print(f"COMPARISON: STANDARD MODE vs PRECISION MODE (threshold={threshold})")
    print("=" * 70)

    # Standard predictions
    y_pred_standard = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    max_proba = np.max(y_proba, axis=1)

    # Precision mode: filter low-confidence predictions
    high_confidence_mask = max_proba >= threshold
    n_confident = high_confidence_mask.sum()
    n_uncertain = len(y_test) - n_confident

    print(f"\n[STANDARD MODE] - All {len(y_test)} samples classified")
    print("-" * 50)

    # Standard mode metrics
    standard_report = classification_report(
        y_test, y_pred_standard,
        labels=clf.categories,
        target_names=clf.categories,
        output_dict=True
    )

    print(f"  Accuracy: {standard_report['accuracy']:.4f}")
    print(f"  Weighted F1: {standard_report['weighted avg']['f1-score']:.4f}")
    print(f"  Macro F1: {standard_report['macro avg']['f1-score']:.4f}")

    print(f"\n[PRECISION MODE] - Only {n_confident} high-confidence samples "
          f"({n_confident/len(y_test)*100:.1f}%)")
    print(f"  Excluded: {n_uncertain} uncertain samples ({n_uncertain/len(y_test)*100:.1f}%)")
    print("-" * 50)

    if n_confident > 0:
        y_test_confident = y_test.values[high_confidence_mask]
        y_pred_confident = y_pred_standard[high_confidence_mask]

        precision_report = classification_report(
            y_test_confident, y_pred_confident,
            labels=clf.categories,
            target_names=clf.categories,
            output_dict=True
        )

        print(f"  Accuracy: {precision_report['accuracy']:.4f}")
        print(f"  Weighted F1: {precision_report['weighted avg']['f1-score']:.4f}")
        print(f"  Macro F1: {precision_report['macro avg']['f1-score']:.4f}")

        # Show improvement
        acc_improvement = precision_report['accuracy'] - standard_report['accuracy']
        f1_improvement = precision_report['weighted avg']['f1-score'] - standard_report['weighted avg']['f1-score']

        print(f"\n[IMPROVEMENT]")
        print(f"  Accuracy: {'+' if acc_improvement >= 0 else ''}{acc_improvement:.4f}")
        print(f"  Weighted F1: {'+' if f1_improvement >= 0 else ''}{f1_improvement:.4f}")
    else:
        print("  No samples met the confidence threshold!")

    # Confidence distribution analysis
    print("\n[CONFIDENCE DISTRIBUTION]")
    print("-" * 50)

    confidence_bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

    for low, high in confidence_bins:
        in_bin = (max_proba >= low) & (max_proba < high) if high < 1.0 else (max_proba >= low)
        count = in_bin.sum()
        pct = count / len(max_proba) * 100

        if count > 0:
            correct_in_bin = (y_pred_standard[in_bin] == y_test.values[in_bin]).sum()
            accuracy_in_bin = correct_in_bin / count
            print(f"  [{low:.1f}-{high:.1f}): {count:4d} samples ({pct:5.1f}%) - Accuracy: {accuracy_in_bin:.2%}")
        else:
            print(f"  [{low:.1f}-{high:.1f}): {count:4d} samples ({pct:5.1f}%)")


def print_model_summary(clf):
    """
    Print a summary of the model configuration.

    Args:
        clf: Fitted MultiCategoryClassifier instance
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    print(clf.summary())


def main():
    """Main execution function."""
    print("=" * 70)
    print("MULTICATEGORYCLASSIFIER VALIDATION SCRIPT")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print()

    # Step 1: Load or train the model
    clf, X_train, y_train, X_test, y_test, feature_cols = load_or_train_model()

    # Step 2: Print model summary
    print_model_summary(clf)

    # Step 3: Print Information Gain rankings
    print_ig_rankings(clf)

    # Step 4: Print classification report
    report = print_classification_report(clf, X_test, y_test)

    # Step 5: Generate and save confusion matrix
    cm_path = generate_confusion_matrix(clf, X_test, y_test)

    # Step 6: Compare standard vs precision mode
    compare_prediction_modes(clf, X_test, y_test, threshold=0.7)

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  - Overall Accuracy: {report['accuracy']:.4f}")
    print(f"  - Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
    print(f"  - Confusion Matrix saved to: {cm_path}")
    print("\nCategories ranked by F1-Score:")

    category_f1 = [(cat, report[cat]['f1-score']) for cat in clf.categories]
    category_f1.sort(key=lambda x: x[1], reverse=True)

    for i, (cat, f1) in enumerate(category_f1, 1):
        print(f"  {i}. {cat}: {f1:.4f}")

    print("\n" + "=" * 70)
    return clf, report


if __name__ == "__main__":
    classifier, evaluation_report = main()
