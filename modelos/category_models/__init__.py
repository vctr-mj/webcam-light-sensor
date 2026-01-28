"""
Category-specific models for light source classification.

This package provides binary classifiers specialized for individual
light source categories (natural, artificial, pantallas, mix).

Main Classes:
    CategorySpecificModel: Binary classifier for a single category using
                          top-K feature selection and probability calibration.

Example:
    >>> from modelos.category_models import CategorySpecificModel
    >>>
    >>> # Create model for 'natural' light detection
    >>> model = CategorySpecificModel(category='natural', top_k=5)
    >>> model.fit(X_train, y_train, feature_ranker)
    >>>
    >>> # Get calibrated probabilities
    >>> probs = model.predict_proba(X_test)
    >>>
    >>> # Save/load model
    >>> model.save('model_natural.pkl')
    >>> loaded_model = CategorySpecificModel.load('model_natural.pkl')
"""

from modelos.category_models.base_category_model import (
    CategorySpecificModel,
    FeatureRankerProtocol,
)

__all__ = [
    'CategorySpecificModel',
    'FeatureRankerProtocol',
]
