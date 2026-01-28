"""
Modulo para ranking de features por categoria usando Information Gain.

Implementa estrategia One-vs-Rest para calcular IG de cada feature
respecto a cada categoria de luz.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

from .information_gain import InformationGainCalculator


class CategoryIGRanker:
    """
    Rankeador de features por Information Gain para cada categoria.

    Calcula IG usando estrategia One-vs-Rest: para cada categoria,
    el problema se convierte en binario (categoria vs resto).

    Attributes:
        categories: Lista de categorias de luz
        ig_calculator: Instancia de InformationGainCalculator
        ig_matrix_: DataFrame con scores IG (features x categories)
        feature_names_: Nombres de features despues de fit
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        n_neighbors: int = 3,
        random_state: int = 42
    ):
        """
        Inicializa el rankeador por categoria.

        Args:
            categories: Lista de categorias. Default: ['natural', 'artificial',
                       'pantallas', 'mix']
            n_neighbors: Parametro para calculo de MI
            random_state: Semilla para reproducibilidad
        """
        if categories is None:
            self.categories = ['natural', 'artificial', 'pantallas', 'mix']
        else:
            self.categories = list(categories)

        self.ig_calculator = InformationGainCalculator(
            n_neighbors=n_neighbors,
            random_state=random_state
        )

        # Atributos que se llenan en fit()
        self.ig_matrix_: Optional[pd.DataFrame] = None
        self.feature_names_: Optional[List[str]] = None
        self._fitted = False

    def _create_binary_target(
        self,
        y: pd.Series,
        positive_category: str
    ) -> np.ndarray:
        """
        Crea target binario para One-vs-Rest.

        Args:
            y: Serie con etiquetas de categoria
            positive_category: Categoria positiva (1), resto es negativo (0)

        Returns:
            Array binario donde 1 = positive_category, 0 = resto
        """
        return (y == positive_category).astype(int).values

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> 'CategoryIGRanker':
        """
        Calcula Information Gain para cada feature respecto a cada categoria.

        Utiliza estrategia One-vs-Rest: para cada categoria, crea un problema
        binario y calcula MI de cada feature.

        Args:
            X: DataFrame con features (n_samples, n_features)
            y: Serie con etiquetas de categoria

        Returns:
            self (para method chaining)

        Raises:
            ValueError: Si alguna categoria no existe en y
        """
        self.feature_names_ = list(X.columns)

        # Validar que todas las categorias existen en y
        available_categories = set(y.unique())
        for cat in self.categories:
            if cat not in available_categories:
                raise ValueError(
                    f"Categoria '{cat}' no encontrada en datos. "
                    f"Categorias disponibles: {available_categories}"
                )

        # Calcular IG para cada categoria
        ig_scores: Dict[str, np.ndarray] = {}

        for category in self.categories:
            # Crear target binario: 1 si es la categoria, 0 si no
            y_binary = self._create_binary_target(y, category)

            # Calcular MI
            mi_scores = self.ig_calculator.calculate_mutual_info(X, y_binary)
            ig_scores[category] = mi_scores

        # Crear DataFrame de resultados
        self.ig_matrix_ = pd.DataFrame(
            ig_scores,
            index=self.feature_names_
        )

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        """Verifica que el modelo este ajustado."""
        if not self._fitted:
            raise RuntimeError(
                "CategoryIGRanker no ha sido ajustado. Llama fit() primero."
            )

    def get_top_k(
        self,
        category: str,
        k: int = 5
    ) -> pd.DataFrame:
        """
        Obtiene las top-K features para una categoria especifica.

        Args:
            category: Nombre de la categoria
            k: Numero de features a retornar

        Returns:
            DataFrame con columnas ['feature', 'ig_score'] ordenado descendente

        Raises:
            ValueError: Si la categoria no existe
            RuntimeError: Si no se ha llamado fit()
        """
        self._check_fitted()

        if category not in self.categories:
            raise ValueError(
                f"Categoria '{category}' no valida. "
                f"Categorias disponibles: {self.categories}"
            )

        # Obtener scores para la categoria y ordenar
        scores = self.ig_matrix_[category].sort_values(ascending=False)
        top_k = scores.head(k)

        return pd.DataFrame({
            'feature': top_k.index,
            'ig_score': top_k.values
        }).reset_index(drop=True)

    def export_matrix(self) -> pd.DataFrame:
        """
        Exporta la matriz completa de IG scores.

        Returns:
            DataFrame con features como filas y categorias como columnas

        Raises:
            RuntimeError: Si no se ha llamado fit()
        """
        self._check_fitted()
        return self.ig_matrix_.copy()

    def global_ranking(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calcula ranking global de features usando promedio ponderado de IG.

        Args:
            weights: Diccionario {categoria: peso}. Si None, usa pesos iguales.

        Returns:
            DataFrame con columnas ['feature', 'global_ig'] ordenado descendente

        Raises:
            RuntimeError: Si no se ha llamado fit()
        """
        self._check_fitted()

        if weights is None:
            # Pesos iguales para todas las categorias
            weights = {cat: 1.0 for cat in self.categories}

        # Normalizar pesos
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Calcular promedio ponderado
        global_scores = np.zeros(len(self.feature_names_))

        for category, weight in normalized_weights.items():
            if category in self.ig_matrix_.columns:
                global_scores += weight * self.ig_matrix_[category].values

        # Crear DataFrame ordenado
        result = pd.DataFrame({
            'feature': self.feature_names_,
            'global_ig': global_scores
        })

        return result.sort_values('global_ig', ascending=False).reset_index(drop=True)

    def to_csv(
        self,
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Exporta rankings a archivos CSV.

        Genera los siguientes archivos:
        - ig_matrix.csv: Matriz completa features x categories
        - ig_ranking_{category}.csv: Top-K para cada categoria
        - ig_global_ranking.csv: Ranking global

        Args:
            output_dir: Directorio donde guardar los archivos

        Returns:
            Diccionario con nombres de archivo y sus paths

        Raises:
            RuntimeError: Si no se ha llamado fit()
        """
        self._check_fitted()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, Path] = {}

        # 1. Matriz completa
        matrix_path = output_path / 'ig_matrix.csv'
        self.ig_matrix_.to_csv(matrix_path)
        saved_files['matrix'] = matrix_path

        # 2. Ranking por categoria
        for category in self.categories:
            top_k = self.get_top_k(category, k=len(self.feature_names_))
            cat_path = output_path / f'ig_ranking_{category}.csv'
            top_k.to_csv(cat_path, index=False)
            saved_files[f'ranking_{category}'] = cat_path

        # 3. Ranking global
        global_rank = self.global_ranking()
        global_path = output_path / 'ig_global_ranking.csv'
        global_rank.to_csv(global_path, index=False)
        saved_files['global_ranking'] = global_path

        return saved_files

    def summary(self) -> str:
        """
        Genera un resumen textual de los rankings.

        Returns:
            String con resumen de top-3 features por categoria
        """
        self._check_fitted()

        lines = ["=" * 50]
        lines.append("Resumen de Information Gain por Categoria")
        lines.append("=" * 50)

        for category in self.categories:
            top_3 = self.get_top_k(category, k=3)
            lines.append(f"\n{category.upper()}:")
            for _, row in top_3.iterrows():
                lines.append(f"  - {row['feature']}: {row['ig_score']:.4f}")

        lines.append("\n" + "=" * 50)
        lines.append("Ranking Global (promedio ponderado):")
        lines.append("=" * 50)

        global_rank = self.global_ranking()
        for _, row in global_rank.head(5).iterrows():
            lines.append(f"  - {row['feature']}: {row['global_ig']:.4f}")

        return "\n".join(lines)
