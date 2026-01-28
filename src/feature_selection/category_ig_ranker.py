"""
Category-specific Information Gain Feature Ranker.

This module implements a feature ranker that computes Information Gain
for each feature with respect to each category in a multi-class classification
problem using a one-vs-rest approach.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.feature_selection import mutual_info_classif


class CategoryIGRanker:
    """
    Ranks features by Information Gain for each category using one-vs-rest approach.

    For each category, binary labels are created (category vs. all others),
    and Information Gain is computed for each feature. This allows identifying
    which features are most discriminative for each specific light source type.

    Attributes:
        categories (List[str]): List of category names.
        feature_names (List[str]): Names of features after fitting.
        ig_matrix (pd.DataFrame): Matrix of IG scores [features x categories].
        rankings (Dict[str, List[str]]): Feature rankings per category.

    Example:
        >>> ranker = CategoryIGRanker(categories=['natural', 'artificial', 'pantallas', 'mix'])
        >>> ranker.fit(X_train, y_train)
        >>> top_features = ranker.get_top_k_features('natural', k=5)
    """

    def __init__(self, categories: Optional[List[str]] = None, random_state: int = 42):
        """
        Initialize the CategoryIGRanker.

        Args:
            categories: List of category names. If None, will be inferred from data.
            random_state: Random state for reproducibility in MI estimation.
        """
        self.categories = categories
        self.random_state = random_state
        self.feature_names: List[str] = []
        self.ig_matrix: Optional[pd.DataFrame] = None
        self.rankings: Dict[str, List[str]] = {}
        self._fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]) -> 'CategoryIGRanker':
        """
        Compute Information Gain for each feature-category pair.

        For each category, creates binary labels (1 for category, 0 for others)
        and computes mutual information between each feature and the binary target.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).

        Returns:
            self: The fitted ranker instance.
        """
        # Convert to numpy/pandas as needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_arr = X

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y

        # Infer categories if not provided
        if self.categories is None:
            self.categories = list(np.unique(y_arr))

        # Compute IG for each category (one-vs-rest)
        ig_scores = {}
        for category in self.categories:
            # Create binary labels: 1 for this category, 0 for all others
            y_binary = (y_arr == category).astype(int)

            # Compute mutual information (Information Gain)
            mi_scores = mutual_info_classif(
                X_arr, y_binary,
                discrete_features=False,
                random_state=self.random_state
            )
            ig_scores[category] = mi_scores

        # Create IG matrix DataFrame
        self.ig_matrix = pd.DataFrame(
            ig_scores,
            index=self.feature_names
        )

        # Compute rankings for each category
        for category in self.categories:
            sorted_features = self.ig_matrix[category].sort_values(ascending=False)
            self.rankings[category] = list(sorted_features.index)

        self._fitted = True
        return self

    def get_top_k_features(self, category: str, k: int = 5) -> List[str]:
        """
        Get the top-K features for a specific category.

        Args:
            category: The category name.
            k: Number of top features to return.

        Returns:
            List of feature names ranked by IG for the category.

        Raises:
            ValueError: If ranker is not fitted or category is unknown.
        """
        self._check_fitted()
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}. "
                           f"Available: {self.categories}")
        return self.rankings[category][:k]

    def get_ig_scores(self, category: str) -> pd.Series:
        """
        Get all IG scores for a specific category.

        Args:
            category: The category name.

        Returns:
            Series of IG scores indexed by feature name.
        """
        self._check_fitted()
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}")
        return self.ig_matrix[category].sort_values(ascending=False)

    def get_ranking_matrix(self) -> pd.DataFrame:
        """
        Get the complete IG ranking matrix.

        Returns:
            DataFrame with features as rows and categories as columns,
            containing IG scores.
        """
        self._check_fitted()
        return self.ig_matrix.copy()

    def get_feature_indices(self, category: str, k: int = 5) -> List[int]:
        """
        Get indices of top-K features for a category.

        Useful for subsetting numpy arrays.

        Args:
            category: The category name.
            k: Number of top features.

        Returns:
            List of feature indices.
        """
        self._check_fitted()
        top_features = self.get_top_k_features(category, k)
        return [self.feature_names.index(f) for f in top_features]

    def _check_fitted(self):
        """Check if the ranker has been fitted."""
        if not self._fitted:
            raise ValueError("CategoryIGRanker must be fitted before use. "
                           "Call fit(X, y) first.")

    def __repr__(self) -> str:
        if self._fitted:
            return (f"CategoryIGRanker(categories={self.categories}, "
                   f"n_features={len(self.feature_names)}, fitted=True)")
        return f"CategoryIGRanker(categories={self.categories}, fitted=False)"
