"""
Multi-Category Light Source Classifier.

This module implements a hierarchical classification system that:
1. Uses CategoryIGRanker to rank features by Information Gain per category
2. Trains CategorySpecificModel binary classifiers for each light source type
3. Combines predictions using a meta-classifier (Logistic Regression)

The system provides category-specific feature selection for optimal classification
of natural, artificial, pantallas (screens), and mixed light sources.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.category_ranker import CategoryIGRanker
from modelos.category_models.base_category_model import CategorySpecificModel


class RankerAdapter:
    """
    Adapter to bridge CategoryIGRanker with CategorySpecificModel expected interface.

    The CategoryIGRanker uses get_top_k() returning a DataFrame, while
    CategorySpecificModel expects get_top_features() returning a list and
    get_feature_score() returning a float.
    """

    def __init__(self, ranker: CategoryIGRanker):
        """
        Initialize the adapter with a fitted CategoryIGRanker.

        Args:
            ranker: A fitted CategoryIGRanker instance.
        """
        self.ranker = ranker

    def get_top_features(self, category: str, k: int) -> List[str]:
        """
        Get top-K feature names for a category.

        Args:
            category: The target category.
            k: Number of top features to return.

        Returns:
            List of feature names ranked by Information Gain.
        """
        top_k_df = self.ranker.get_top_k(category, k)
        return list(top_k_df['feature'])

    def get_feature_score(self, category: str, feature: str) -> float:
        """
        Get the Information Gain score for a feature in a category.

        Args:
            category: The target category.
            feature: The feature name.

        Returns:
            The IG score for the feature-category pair.
        """
        if self.ranker.ig_matrix_ is None:
            raise RuntimeError("Ranker must be fitted before getting scores.")
        return float(self.ranker.ig_matrix_.loc[feature, category])


class MultiCategoryClassifier:
    """
    Multi-category classifier for light source classification.

    This classifier integrates:
    - CategoryIGRanker for feature ranking by Information Gain
    - CategorySpecificModel for per-category binary classification
    - LogisticRegression meta-classifier for final predictions

    The system uses a stacking approach where each binary classifier outputs
    a probability, and the meta-classifier combines these probabilities to
    produce the final multi-class prediction.

    Attributes:
        top_k (int): Number of top features to use per category.
        categories (List[str]): List of light source categories.
        ranker (CategoryIGRanker): Fitted feature ranker.
        category_models (Dict[str, CategorySpecificModel]): Trained binary models.
        meta_classifier (LogisticRegression): Stacking meta-classifier.

    Example:
        >>> classifier = MultiCategoryClassifier(top_k=5)
        >>> classifier.fit(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
        >>> report = classifier.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        top_k: int = 5,
        categories: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the MultiCategoryClassifier.

        Args:
            top_k: Number of top features to select for each category based
                   on Information Gain. Defaults to 5.
            categories: List of light source categories. Defaults to
                       ['natural', 'artificial', 'pantallas', 'mix'].
        """
        self.top_k = top_k
        if categories is None:
            self.categories = ['natural', 'artificial', 'pantallas', 'mix']
        else:
            self.categories = list(categories)

        # Components (initialized during fit)
        self.ranker: Optional[CategoryIGRanker] = None
        self.category_models: Dict[str, CategorySpecificModel] = {}
        self.meta_classifier: Optional[LogisticRegression] = None
        self._is_fitted: bool = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'MultiCategoryClassifier':
        """
        Fit the complete multi-category classification system.

        This method performs the following steps:
        1. Initialize and fit CategoryIGRanker to compute IG rankings
        2. For each category, create and train a CategorySpecificModel
           using the top-K features for that category
        3. Collect probability predictions from all category models
        4. Train LogisticRegression meta-classifier on the probabilities

        Args:
            X: Training feature matrix. Can be DataFrame or numpy array.
               If array, feature names will be auto-generated.
            y: Training labels with category names.

        Returns:
            self: The fitted classifier instance.

        Raises:
            ValueError: If any category is not found in training labels.
        """
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(
                X,
                columns=[f'feature_{i}' for i in range(X.shape[1])]
            )

        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Step 1: Initialize and fit the feature ranker
        self.ranker = CategoryIGRanker(categories=self.categories)
        self.ranker.fit(X, y)

        # Create adapter for CategorySpecificModel compatibility
        ranker_adapter = RankerAdapter(self.ranker)

        # Step 2: Train a CategorySpecificModel for each category
        self.category_models = {}
        for category in self.categories:
            model = CategorySpecificModel(
                category=category,
                top_k=self.top_k,
                min_ig_threshold=0.001  # Low threshold to ensure features are selected
            )
            model.fit(X, y, ranker_adapter)
            self.category_models[category] = model

        # Step 3: Collect probability predictions from all models
        proba_matrix = self._get_probability_matrix(X)

        # Step 4: Train meta-classifier on probability features
        self.meta_classifier = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        self.meta_classifier.fit(proba_matrix, y)

        self._is_fitted = True
        return self

    def _get_probability_matrix(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Get probability predictions from all category models.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, n_categories) with probabilities.
        """
        proba_columns = []
        for category in self.categories:
            model = self.category_models[category]
            proba = model.predict_proba(X)
            proba_columns.append(proba)

        return np.column_stack(proba_columns)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict category labels for samples.

        Uses the meta-classifier to combine probability outputs from
        all category-specific models.

        Args:
            X: Feature matrix to predict.

        Returns:
            Array of predicted category labels.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._check_is_fitted()

        # Get probability matrix from category models
        proba_matrix = self._get_probability_matrix(X)

        # Use meta-classifier for final prediction
        return self.meta_classifier.predict(proba_matrix)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict probability distribution over categories.

        Returns the meta-classifier's probability outputs, which combine
        information from all category-specific models.

        Args:
            X: Feature matrix to predict.

        Returns:
            Array of shape (n_samples, n_categories) with probability
            for each category. Columns are ordered by self.categories.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._check_is_fitted()

        # Get probability matrix from category models
        proba_matrix = self._get_probability_matrix(X)

        # Use meta-classifier for probability predictions
        return self.meta_classifier.predict_proba(proba_matrix)

    def get_rankings(self) -> pd.DataFrame:
        """
        Get the Information Gain ranking matrix.

        Returns:
            DataFrame with features as rows and categories as columns,
            containing IG scores.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._check_is_fitted()
        return self.ranker.export_matrix()

    def get_category_model(self, category: str) -> CategorySpecificModel:
        """
        Get the binary classifier for a specific category.

        Args:
            category: The category name.

        Returns:
            The CategorySpecificModel for the specified category.

        Raises:
            RuntimeError: If the classifier has not been fitted.
            ValueError: If the category is not valid.
        """
        self._check_is_fitted()

        if category not in self.categories:
            raise ValueError(
                f"Unknown category: '{category}'. "
                f"Available categories: {self.categories}"
            )

        return self.category_models[category]

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray]
    ) -> Dict:
        """
        Evaluate the classifier on test data.

        Generates a classification report with precision, recall, F1-score,
        and support for each category.

        Args:
            X_test: Test feature matrix.
            y_test: True labels for test samples.

        Returns:
            Dictionary containing the classification report metrics.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._check_is_fitted()

        y_pred = self.predict(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        return classification_report(
            y_test,
            y_pred,
            labels=self.categories,
            target_names=self.categories,
            output_dict=True
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the entire classification system to disk.

        Saves all components including the ranker, category models,
        and meta-classifier using joblib.

        Args:
            path: File path to save the model. Should end in .pkl or .joblib.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._check_is_fitted()

        model_data = {
            'top_k': self.top_k,
            'categories': self.categories,
            'ranker': self.ranker,
            'category_models': self.category_models,
            'meta_classifier': self.meta_classifier
        }

        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MultiCategoryClassifier':
        """
        Load a saved classification system from disk.

        Args:
            path: File path to the saved model.

        Returns:
            A fitted MultiCategoryClassifier instance.
        """
        model_data = joblib.load(path)

        instance = cls(
            top_k=model_data['top_k'],
            categories=model_data['categories']
        )

        instance.ranker = model_data['ranker']
        instance.category_models = model_data['category_models']
        instance.meta_classifier = model_data['meta_classifier']
        instance._is_fitted = True

        return instance

    def _check_is_fitted(self) -> None:
        """Check if the classifier has been fitted and raise if not."""
        if not self._is_fitted:
            raise RuntimeError(
                "This MultiCategoryClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

    def predict_with_threshold(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.7
    ) -> np.ndarray:
        """
        Predict category labels with confidence threshold.

        Only accepts predictions where the maximum probability is greater
        than or equal to the threshold. Uncertain predictions are labeled
        as 'no_decidido'.

        Args:
            X: Feature matrix to predict.
            threshold: Minimum probability threshold to accept a prediction.
                      Defaults to 0.7.

        Returns:
            Array of predicted category labels, with 'no_decidido' for
            uncertain cases where max probability < threshold.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._check_is_fitted()

        # Get probabilities
        probas = self.predict_proba(X)
        max_probas = np.max(probas, axis=1)

        # Get standard predictions
        predictions = self.predict(X)

        # Create result array with 'no_decidido' for uncertain cases
        result = np.where(
            max_probas >= threshold,
            predictions,
            'no_decidido'
        )

        return result

    def evaluate_precision_mode(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        threshold: float = 0.7
    ) -> Dict:
        """
        Evaluate the classifier in precision mode (with threshold).

        Calculates precision, recall, and accuracy only for decided cases
        (where the model made a confident prediction).

        Args:
            X_test: Test feature matrix.
            y_test: True labels for test samples.
            threshold: Minimum probability threshold to accept a prediction.
                      Defaults to 0.7.

        Returns:
            Dictionary containing:
                - precision: Precision on decided cases
                - recall: Recall on decided cases
                - accuracy: Accuracy on decided cases
                - total_samples: Total number of samples
                - decided_samples: Number of samples where prediction was made
                - no_decidido_count: Number of uncertain samples
                - no_decidido_percentage: Percentage of uncertain samples
                - confusion_matrix: Confusion matrix for decided cases
                - per_category: Per-category metrics

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        from sklearn.metrics import (
            precision_score, recall_score, accuracy_score, confusion_matrix
        )

        self._check_is_fitted()

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        # Get predictions with threshold
        y_pred = self.predict_with_threshold(X_test, threshold=threshold)

        # Separate decided and undecided cases
        decided_mask = y_pred != 'no_decidido'
        decided_count = np.sum(decided_mask)
        no_decidido_count = len(y_pred) - decided_count

        result = {
            'threshold': threshold,
            'total_samples': len(y_pred),
            'decided_samples': int(decided_count),
            'no_decidido_count': int(no_decidido_count),
            'no_decidido_percentage': float(no_decidido_count / len(y_pred) * 100)
        }

        if decided_count > 0:
            y_true_decided = y_test[decided_mask]
            y_pred_decided = y_pred[decided_mask]

            # Calculate metrics on decided cases only
            result['accuracy'] = float(accuracy_score(y_true_decided, y_pred_decided))
            result['precision'] = float(precision_score(
                y_true_decided, y_pred_decided,
                labels=self.categories,
                average='weighted',
                zero_division=0
            ))
            result['recall'] = float(recall_score(
                y_true_decided, y_pred_decided,
                labels=self.categories,
                average='weighted',
                zero_division=0
            ))

            # Confusion matrix
            result['confusion_matrix'] = confusion_matrix(
                y_true_decided, y_pred_decided,
                labels=self.categories
            )

            # Per-category metrics
            per_category = {}
            for category in self.categories:
                cat_mask = y_true_decided == category
                if np.sum(cat_mask) > 0:
                    cat_correct = np.sum(
                        (y_true_decided == category) & (y_pred_decided == category)
                    )
                    cat_total = np.sum(cat_mask)
                    per_category[category] = {
                        'total': int(cat_total),
                        'correct': int(cat_correct),
                        'accuracy': float(cat_correct / cat_total)
                    }
                else:
                    per_category[category] = {
                        'total': 0,
                        'correct': 0,
                        'accuracy': 0.0
                    }
            result['per_category'] = per_category
        else:
            result['accuracy'] = 0.0
            result['precision'] = 0.0
            result['recall'] = 0.0
            result['confusion_matrix'] = np.zeros(
                (len(self.categories), len(self.categories)), dtype=int
            )
            result['per_category'] = {
                cat: {'total': 0, 'correct': 0, 'accuracy': 0.0}
                for cat in self.categories
            }

        return result

    def compare_thresholds(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compare model performance at different probability thresholds.

        Evaluates the model at each threshold and returns a DataFrame
        comparing accuracy, precision, recall, and the number of
        undecided samples.

        Args:
            X_test: Test feature matrix.
            y_test: True labels for test samples.
            thresholds: List of probability thresholds to evaluate.
                       Defaults to [0.5, 0.6, 0.7, 0.8, 0.9].

        Returns:
            DataFrame with columns:
                - threshold: The probability threshold
                - accuracy: Accuracy on decided cases
                - precision: Weighted precision on decided cases
                - recall: Weighted recall on decided cases
                - decided: Number of decided samples
                - no_decidido: Number of undecided samples
                - no_decidido_pct: Percentage of undecided samples

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._check_is_fitted()

        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        results = []
        for threshold in thresholds:
            eval_result = self.evaluate_precision_mode(
                X_test, y_test, threshold=threshold
            )
            results.append({
                'threshold': threshold,
                'accuracy': eval_result['accuracy'],
                'precision': eval_result['precision'],
                'recall': eval_result['recall'],
                'decided': eval_result['decided_samples'],
                'no_decidido': eval_result['no_decidido_count'],
                'no_decidido_pct': eval_result['no_decidido_percentage']
            })

        return pd.DataFrame(results)

    def summary(self) -> str:
        """
        Generate a text summary of the classification system.

        Returns:
            Multi-line string with system configuration and
            top features per category.
        """
        self._check_is_fitted()

        lines = ["=" * 60]
        lines.append("Multi-Category Light Source Classifier")
        lines.append("=" * 60)
        lines.append(f"\nConfiguration:")
        lines.append(f"  - Top-K features per category: {self.top_k}")
        lines.append(f"  - Categories: {', '.join(self.categories)}")

        lines.append("\nSelected Features by Category:")
        for category in self.categories:
            model = self.category_models[category]
            features = model.get_selected_features()
            lines.append(f"\n  {category.upper()}:")
            for feat in features:
                lines.append(f"    - {feat}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation of the classifier."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"MultiCategoryClassifier(top_k={self.top_k}, "
            f"categories={self.categories}, status={status})"
        )


if __name__ == "__main__":
    """
    Example usage: Train and evaluate the multi-category classifier.
    """
    # Paths to data
    train_path = PROJECT_ROOT / 'datasets' / 'train.csv'
    test_path = PROJECT_ROOT / 'datasets' / 'test.csv'
    model_save_path = PROJECT_ROOT / 'modelos' / 'multi_category_classifier.pkl'

    print("=" * 60)
    print("Multi-Category Light Source Classifier - Training")
    print("=" * 60)

    # Load training data
    print("\n[1/4] Loading training data...")
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=['Etiqueta'])
    y_train = train_df['Etiqueta']
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {list(X_train.columns)}")
    print(f"  Categories: {y_train.value_counts().to_dict()}")

    # Load test data
    print("\n[2/4] Loading test data...")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=['Etiqueta'])
    y_test = test_df['Etiqueta']
    print(f"  Test samples: {len(X_test)}")

    # Train the classifier
    print("\n[3/4] Training multi-category classifier...")
    classifier = MultiCategoryClassifier(top_k=4)  # Using 4 since we have 4 features
    classifier.fit(X_train, y_train)
    print("  Training complete!")

    # Print system summary
    print("\n" + classifier.summary())

    # Evaluate on test data
    print("\n[4/4] Evaluating on test set...")
    report = classifier.evaluate(X_test, y_test)

    print("\nClassification Report:")
    print("-" * 60)
    print(f"{'Category':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)

    for category in classifier.categories:
        metrics = report[category]
        print(
            f"{category:<15} "
            f"{metrics['precision']:>10.3f} "
            f"{metrics['recall']:>10.3f} "
            f"{metrics['f1-score']:>10.3f} "
            f"{int(metrics['support']):>10}"
        )

    print("-" * 60)
    print(
        f"{'Accuracy':<15} "
        f"{'':>10} "
        f"{'':>10} "
        f"{report['accuracy']:>10.3f} "
        f"{int(report['weighted avg']['support']):>10}"
    )
    print(
        f"{'Weighted Avg':<15} "
        f"{report['weighted avg']['precision']:>10.3f} "
        f"{report['weighted avg']['recall']:>10.3f} "
        f"{report['weighted avg']['f1-score']:>10.3f} "
        f"{int(report['weighted avg']['support']):>10}"
    )

    # Save the model
    print(f"\nSaving model to: {model_save_path}")
    classifier.save(model_save_path)
    print("Model saved successfully!")

    # Demonstrate loading
    print("\nDemonstrating model loading...")
    loaded_classifier = MultiCategoryClassifier.load(model_save_path)
    test_pred = loaded_classifier.predict(X_test[:5])
    print(f"  Sample predictions: {list(test_pred)}")
    print(f"  Actual labels:      {list(y_test[:5])}")

    print("\n" + "=" * 60)
    print("Training and evaluation complete!")
    print("=" * 60)
