"""
Category-specific binary classifier for light source classification.

This module implements a binary classifier that specializes in detecting
whether a sample belongs to a specific light category (natural, artificial,
pantallas, or mix). It uses feature selection based on information gain
and probability calibration for reliable predictions.
"""

from typing import Dict, List, Optional, Protocol, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureRankerProtocol(Protocol):
    """Protocol for feature rankers that provide information gain rankings."""

    def get_top_features(self, category: str, k: int) -> List[str]:
        """Return top-K features for a specific category."""
        ...

    def get_feature_score(self, category: str, feature: str) -> float:
        """Return the information gain score for a feature in a category."""
        ...


class CategorySpecificModel:
    """
    Binary classifier specialized for a single light source category.

    This model uses top-K feature selection based on information gain,
    trains a GradientBoostingClassifier, and applies probability calibration
    using isotonic regression for well-calibrated probability estimates.

    Attributes:
        category: The target category this model classifies (e.g., 'natural').
        top_k: Number of top features to select based on information gain.
        min_ig_threshold: Minimum information gain threshold for feature selection.
        selected_features: List of features selected after fitting.
        feature_importances_: Dictionary mapping features to their importances.

    Example:
        >>> model = CategorySpecificModel(category='natural', top_k=5)
        >>> model.fit(X_train, y_train, feature_ranker)
        >>> probabilities = model.predict_proba(X_test)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        category: str,
        top_k: int = 5,
        min_ig_threshold: float = 0.01
    ) -> None:
        """
        Initialize the category-specific model.

        Args:
            category: The target category for binary classification.
                     One of: 'natural', 'artificial', 'pantallas', 'mix'.
            top_k: Number of top features to select based on information gain.
                   Defaults to 5.
            min_ig_threshold: Minimum information gain score required for a
                            feature to be considered. Defaults to 0.01.
        """
        self.category = category
        self.top_k = top_k
        self.min_ig_threshold = min_ig_threshold

        # Model components (initialized during fit)
        self._scaler: Optional[StandardScaler] = None
        self._base_classifier: Optional[GradientBoostingClassifier] = None
        self._calibrated_classifier: Optional[CalibratedClassifierCV] = None

        # Feature information (set during fit)
        self._selected_features: List[str] = []
        self._feature_importances: Dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_ranker: FeatureRankerProtocol
    ) -> 'CategorySpecificModel':
        """
        Fit the model using top-K features selected by the feature ranker.

        This method:
        1. Gets top-K features from the ranker based on information gain
        2. Filters features by minimum IG threshold
        3. Scales the selected features
        4. Trains a GradientBoostingClassifier
        5. Calibrates probabilities using isotonic regression

        Args:
            X: Training feature matrix. Can be a DataFrame or numpy array.
               If array, feature selection will use column indices.
            y: Training labels. Will be converted to binary (1 for target
               category, 0 for all others).
            feature_ranker: Object implementing FeatureRankerProtocol that
                          provides feature rankings for the category.

        Returns:
            self: The fitted model instance.

        Raises:
            ValueError: If no features pass the information gain threshold.
        """
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Get top-K features from the ranker
        top_features = feature_ranker.get_top_features(self.category, self.top_k)

        # Filter by minimum information gain threshold
        self._selected_features = []
        for feature in top_features:
            ig_score = feature_ranker.get_feature_score(self.category, feature)
            if ig_score >= self.min_ig_threshold:
                self._selected_features.append(feature)

        if len(self._selected_features) == 0:
            raise ValueError(
                f"No features pass the minimum IG threshold of {self.min_ig_threshold} "
                f"for category '{self.category}'"
            )

        # Select only the chosen features
        X_selected = X[self._selected_features].values

        # Create binary labels: 1 if sample is target category, 0 otherwise
        if isinstance(y, pd.Series):
            y_binary = (y == self.category).astype(int).values
        else:
            y_binary = (np.array(y) == self.category).astype(int)

        # Initialize and fit the scaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_selected)

        # Initialize base classifier
        self._base_classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        # Train the base classifier first (needed for calibration)
        self._base_classifier.fit(X_scaled, y_binary)

        # Store feature importances from the base classifier
        self._feature_importances = {
            feature: float(importance)
            for feature, importance in zip(
                self._selected_features,
                self._base_classifier.feature_importances_
            )
        }

        # Create calibrated classifier with isotonic regression
        # We need to re-fit for proper calibration using cross-validation
        self._calibrated_classifier = CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            cv=5,
            method='isotonic'
        )
        self._calibrated_classifier.fit(X_scaled, y_binary)

        self._is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict binary class labels for samples.

        Args:
            X: Feature matrix to predict. Must contain the selected features.

        Returns:
            Binary predictions (1 = belongs to category, 0 = does not).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        X_scaled = self._prepare_features(X)
        return self._calibrated_classifier.predict(X_scaled)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict calibrated probability of belonging to this category.

        Args:
            X: Feature matrix to predict. Must contain the selected features.

        Returns:
            Array of shape (n_samples,) with probability of being in
            this category (class 1 probability).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        X_scaled = self._prepare_features(X)
        # Return probability of positive class (belongs to category)
        return self._calibrated_classifier.predict_proba(X_scaled)[:, 1]

    def get_selected_features(self) -> List[str]:
        """
        Get the list of features selected for this model.

        Returns:
            List of feature names that were selected based on
            information gain ranking.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return self._selected_features.copy()

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances from the trained classifier.

        Returns:
            Dictionary mapping feature names to their importance scores
            from the GradientBoostingClassifier.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return self._feature_importances.copy()

    def save(self, filepath: str) -> None:
        """
        Save the fitted model to disk.

        Args:
            filepath: Path where the model will be saved. Should end in .pkl.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()

        model_data = {
            'category': self.category,
            'top_k': self.top_k,
            'min_ig_threshold': self.min_ig_threshold,
            'scaler': self._scaler,
            'calibrated_classifier': self._calibrated_classifier,
            'selected_features': self._selected_features,
            'feature_importances': self._feature_importances
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CategorySpecificModel':
        """
        Load a fitted model from disk.

        Args:
            filepath: Path to the saved model file.

        Returns:
            A fitted CategorySpecificModel instance.
        """
        model_data = joblib.load(filepath)

        instance = cls(
            category=model_data['category'],
            top_k=model_data['top_k'],
            min_ig_threshold=model_data['min_ig_threshold']
        )

        instance._scaler = model_data['scaler']
        instance._calibrated_classifier = model_data['calibrated_classifier']
        instance._selected_features = model_data['selected_features']
        instance._feature_importances = model_data['feature_importances']
        instance._is_fitted = True

        return instance

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted and raise if not."""
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Prepare features for prediction by selecting and scaling.

        Args:
            X: Input feature matrix.

        Returns:
            Scaled feature array with only selected features.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        X_selected = X[self._selected_features].values
        return self._scaler.transform(X_selected)

    def __repr__(self) -> str:
        """Return string representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"CategorySpecificModel(category='{self.category}', "
            f"top_k={self.top_k}, status={status})"
        )
