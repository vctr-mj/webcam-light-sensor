"""
Feature Selection Module - Seleccion de features usando Information Gain.

Este modulo proporciona herramientas para analizar y rankear features
basandose en Information Gain (Mutual Information) respecto a las
categorias de tipo de luz.

Classes:
    InformationGainCalculator: Calcula MI entre features y target binario
    CategoryIGRanker: Rankea features por IG para cada categoria de luz

Example:
    >>> from src.feature_selection import CategoryIGRanker
    >>> import pandas as pd
    >>>
    >>> # Cargar datos
    >>> df = pd.read_csv('datasets/train.csv')
    >>> X = df.drop(columns=['Etiqueta'])
    >>> y = df['Etiqueta']
    >>>
    >>> # Calcular rankings
    >>> ranker = CategoryIGRanker()
    >>> ranker.fit(X, y)
    >>>
    >>> # Obtener top-5 features para luz natural
    >>> top_natural = ranker.get_top_k('natural', k=5)
    >>> print(top_natural)
    >>>
    >>> # Exportar resultados
    >>> ranker.to_csv('outputs/feature_selection/')
"""

from .information_gain import InformationGainCalculator
from .category_ranker import CategoryIGRanker

__all__ = [
    'InformationGainCalculator',
    'CategoryIGRanker'
]

__version__ = '1.0.0'
