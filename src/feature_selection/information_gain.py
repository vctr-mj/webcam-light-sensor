"""
Modulo para calcular Information Gain (Mutual Information) entre features y categorias.

Utiliza sklearn.feature_selection.mutual_info_classif para el calculo.
Incluye manejo de estabilidad numerica (jitter, NaN handling).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from typing import Union, Optional


class InformationGainCalculator:
    """
    Calculador de Information Gain basado en Mutual Information.

    Utiliza sklearn.feature_selection.mutual_info_classif con parametros
    optimizados para estabilidad numerica.

    Attributes:
        n_neighbors: Numero de vecinos para estimacion de MI (default=3)
        random_state: Semilla para reproducibilidad (default=42)
        jitter: Ruido gaussiano para estabilidad numerica
    """

    def __init__(
        self,
        n_neighbors: int = 3,
        random_state: int = 42,
        jitter: float = 1e-10
    ):
        """
        Inicializa el calculador de Information Gain.

        Args:
            n_neighbors: Numero de vecinos para estimacion KNN de MI
            random_state: Semilla para reproducibilidad
            jitter: Cantidad de ruido gaussiano para estabilidad numerica
        """
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.jitter = jitter
        self._imputer = SimpleImputer(strategy='median')

    def _preprocess_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocesa features: imputacion de NaN y jitter para estabilidad.

        Args:
            X: Matriz de features (n_samples, n_features)

        Returns:
            Matriz preprocesada como numpy array
        """
        # Convertir a numpy si es DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values.copy()
        else:
            X_array = np.array(X, dtype=np.float64).copy()

        # Imputar valores faltantes con mediana
        if np.any(np.isnan(X_array)):
            X_array = self._imputer.fit_transform(X_array)

        # Agregar jitter para estabilidad numerica (evita division por cero)
        np.random.seed(self.random_state)
        X_array = X_array + np.random.normal(0, self.jitter, X_array.shape)

        return X_array.astype(np.float64)

    def _validate_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Valida y prepara el vector objetivo binario.

        Args:
            y: Vector objetivo binario (0/1 o boolean)

        Returns:
            Vector objetivo como numpy array de enteros

        Raises:
            ValueError: Si y no es binario
        """
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Convertir booleanos a int
        if y_array.dtype == bool:
            y_array = y_array.astype(int)

        # Validar que es binario
        unique_vals = np.unique(y_array[~pd.isna(y_array)])
        if len(unique_vals) > 2:
            raise ValueError(
                f"El target debe ser binario. Valores encontrados: {unique_vals}"
            )

        return y_array.astype(int)

    def calculate_mutual_info(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_binary: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Calcula Mutual Information entre features y target binario.

        Utiliza sklearn.feature_selection.mutual_info_classif con manejo
        de estabilidad numerica (jitter, imputacion de NaN).

        Args:
            X: Matriz de features (n_samples, n_features)
            y_binary: Vector objetivo binario (0/1)

        Returns:
            Array con MI scores para cada feature

        Example:
            >>> calc = InformationGainCalculator()
            >>> X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
            >>> y = np.array([0, 1, 1])
            >>> mi_scores = calc.calculate_mutual_info(X, y)
        """
        # Preprocesar
        X_processed = self._preprocess_features(X)
        y_processed = self._validate_target(y_binary)

        # Calcular Mutual Information
        mi_scores = mutual_info_classif(
            X_processed,
            y_processed,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            discrete_features=False  # Features son continuas
        )

        # Manejar posibles NaN en resultados
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)

        return mi_scores

    def calculate_with_names(
        self,
        X: pd.DataFrame,
        y_binary: Union[pd.Series, np.ndarray]
    ) -> pd.Series:
        """
        Calcula MI y retorna Series con nombres de features.

        Args:
            X: DataFrame con features (columnas nombradas)
            y_binary: Vector objetivo binario

        Returns:
            Series con MI scores indexado por nombre de feature
        """
        mi_scores = self.calculate_mutual_info(X, y_binary)
        return pd.Series(mi_scores, index=X.columns, name='mutual_info')
