"""
Category-Specific Binary Classifier.

This module implements a binary classifier specialized for detecting
a specific light source category using its top-K most discriminative features.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class CategorySpecificModel(BaseEstimator, ClassifierMixin):
    """
    Binary classifier for detecting a specific light source category.

    This model uses only the top-K features most discriminative for the
    target category, based on Information Gain ranking. It outputs the
    probability of the sample belonging to the target category.

    Attributes:
        category (str): The target category this model detects.
        feature_names (List[str]): Names of the top-K features used.
        classifier: The underlying binary classifier.

    Example:
        >>> model = CategorySpecificModel(category='natural', feature_names=['std_v', 'mean_b'])
        >>> model.fit(X_train, y_train)
        >>> probs = model.predict_proba_category(X_test)
    """

    def __init__(self, category: str, feature_names: List[str],
                 n_estimators: int = 100, max_depth: int = 3,
                 learning_rate: float = 0.1, random_state: int = 42):
        """
        Initialize the CategorySpecificModel.

        Args:
            category: The target category name.
            feature_names: List of feature names to use (top-K from IG ranking).
            n_estimators: Number of boosting stages for GradientBoosting.
            max_depth: Maximum depth of individual trees.
            learning_rate: Learning rate for gradient boosting.
            random_state: Random state for reproducibility.
        """
        self.category = category
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self._fitted = False
        self._all_feature_names: List[str] = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]) -> 'CategorySpecificModel':
        """
        Fit the binary classifier for the target category.

        Creates binary labels (1 for target category, 0 for others)
        and trains on the selected features only.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).

        Returns:
            self: The fitted model instance.
        """
        # Extract selected features
        if isinstance(X, pd.DataFrame):
            self._all_feature_names = list(X.columns)
            X_selected = X[self.feature_names].values
        else:
            # Assume feature_names are indices if X is numpy array
            raise ValueError("X must be a DataFrame when using feature names")

        # Create binary labels
        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y
        y_binary = (y_arr == self.category).astype(int)

        # Train classifier
        self.classifier.fit(X_selected, y_binary)
        self._fitted = True

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict binary labels for the target category.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Binary predictions (1 for target category, 0 for others).
        """
        self._check_fitted()
        X_selected = self._select_features(X)
        return self.classifier.predict(X_selected)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict probability of belonging to the target category.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with probabilities [P(not-category), P(category)].
        """
        self._check_fitted()
        X_selected = self._select_features(X)
        return self.classifier.predict_proba(X_selected)

    def predict_proba_category(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get probability of belonging to the target category only.

        This is a convenience method that returns only P(category).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) with P(category) values.
        """
        proba = self.predict_proba(X)
        return proba[:, 1]

    def _select_features(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Select the features this model uses."""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].values
        else:
            raise ValueError("X must be a DataFrame when using feature names")

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self._fitted:
            raise ValueError(f"CategorySpecificModel for '{self.category}' "
                           "must be fitted before use. Call fit(X, y) first.")

    def get_feature_importances(self) -> pd.Series:
        """
        Get feature importances from the underlying classifier.

        Returns:
            Series of feature importances indexed by feature name.
        """
        self._check_fitted()
        return pd.Series(
            self.classifier.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (f"CategorySpecificModel(category='{self.category}', "
               f"features={self.feature_names}, {status})")
