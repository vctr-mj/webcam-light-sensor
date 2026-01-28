# Propuesta de Mejora: Modelos por Categoría con Information Gain Ranking

## Resumen Ejecutivo

Este documento presenta una propuesta técnica para mejorar el sistema de clasificación de iluminación ambiental mediante la implementación de **modelos específicos por categoría** (NA, AR, PAN, MIX) con **ranking de features basado en Information Gain**.

| Aspecto | Estado Actual | Propuesta |
|---------|--------------|-----------|
| **Arquitectura** | Modelo único multi-clase | One-vs-Rest con Meta-Ensemble |
| **Selección de Features** | Todas las features para todos | Features específicos por categoría |
| **Interpretabilidad** | Limitada | Alta (IG ranking + SHAP) |
| **Precisión Base** | 98.97% (Random Forest) | Mantener/mejorar con interpretabilidad |

---

## 1. Justificación Técnica

### 1.1 Problema con el Enfoque Actual

El modelo actual utiliza **todas las features** para clasificar **todas las categorías**:

```python
# Enfoque actual (train_model.py)
X = df.drop(columns=["Etiqueta"])  # Todas las features
y = df["Etiqueta"]                  # Todas las categorías
modelo.fit(X, y)                    # Un solo modelo
```

**Limitaciones identificadas:**

1. **No discrimina importancia por categoría**: Un feature como `mean_b` es crítico para detectar pantallas (PAN) pero menos relevante para distinguir NA de AR.

2. **Falta de interpretabilidad**: No sabemos qué features son más importantes para cada tipo de iluminación específicamente.

3. **Potencial overfitting**: Features irrelevantes para ciertas categorías pueden introducir ruido.

4. **Dificultad para mejorar edge cases**: Sin saber qué features afectan cada categoría, es difícil mejorar casos límite.

### 1.2 Evidencia del Análisis de Variables

Del análisis existente (`analisis_variables.py`), sabemos que:

| Feature | Comportamiento Observado |
|---------|-------------------------|
| `mean_b` | Discrimina fuertemente PAN (valores negativos/azul) vs NA (valores positivos/amarillo) |
| `std_v` | Alta variabilidad en MIX, baja en PAN |
| `skew_v` | Distribución asimétrica diferente por categoría |
| `mean_h`, `mean_s` | Patrones específicos para AR (luz artificial LED) |

**Conclusión**: Cada categoría tiene un **subconjunto óptimo de features** que la discrimina mejor.

---

## 2. Propuesta: Arquitectura One-vs-Rest con Meta-Ensemble

### 2.1 Diseño Conceptual

```
┌──────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                         │
│  [mean_h, mean_s, mean_v, std_v, mean_a, mean_b,         │
│   std_l, skew_v, v_95]                                   │
└─────────────────────────┬────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Modelo    │   │   Modelo    │   │   Modelo    │
│     NA      │   │     AR      │   │    PAN      │
│ (Top-K IG)  │   │ (Top-K IG)  │   │ (Top-K IG)  │
│             │   │             │   │             │
│ Features:   │   │ Features:   │   │ Features:   │
│ - mean_b    │   │ - mean_h    │   │ - mean_b    │
│ - std_v     │   │ - mean_s    │   │ - std_v     │
│ - skew_v    │   │ - std_v     │   │ - v_95      │
│ - v_95      │   │ - mean_b    │   │ - skew_v    │
│ - mean_a    │   │ - std_l     │   │ - mean_s    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       │    ┌────────────┤                 │
       │    │   ┌────────────┐             │
       │    │   │   Modelo   │             │
       │    │   │    MIX     │             │
       │    │   │ (Top-K IG) │             │
       │    │   └──────┬─────┘             │
       │    │          │                   │
       └────┴──────────┴───────────────────┘
                       │
              ┌────────▼────────┐
              │ META-CLASSIFIER │
              │ (LogisticReg)   │
              │                 │
              │ Input: P(NA),   │
              │ P(AR), P(PAN),  │
              │ P(MIX)          │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ PREDICCIÓN FINAL│
              │ NA | AR | PAN | │
              │      MIX        │
              └─────────────────┘
```

### 2.2 Ventajas de Esta Arquitectura

| Ventaja | Descripción |
|---------|-------------|
| **Interpretabilidad** | Sabemos exactamente qué features importan para cada categoría |
| **Especialización** | Cada modelo se optimiza para su categoría específica |
| **Flexibilidad** | Podemos mejorar una categoría sin afectar las demás |
| **Manejo de conflictos** | El meta-clasificador resuelve predicciones contradictorias |
| **Debugging facilitado** | Si PAN falla, sabemos que revisar `mean_b`, `std_v`, etc. |

---

## 3. Information Gain como Métrica de Ranking

### 3.1 Fundamento Matemático

**Information Gain (IG)** mide cuánta información aporta un feature para clasificar una categoría:

```
IG(Categoría, Feature) = H(Categoría) - H(Categoría | Feature)
```

Donde:
- `H(Categoría)` = Entropía de la distribución de la categoría
- `H(Categoría | Feature)` = Entropía condicional dado el feature

**Interpretación**:
- IG alto → El feature reduce mucho la incertidumbre → **Muy útil para clasificar**
- IG bajo → El feature aporta poca información → **Poco útil**

### 3.2 Por qué Information Gain vs Otras Métricas

| Métrica | Ventaja | Limitación | Uso Recomendado |
|---------|---------|------------|-----------------|
| **Information Gain** | Detecta relaciones no lineales | Puede favorecer features con muchos valores | ✅ **Nuestro caso** |
| Correlación | Simple de interpretar | Solo detecta relaciones lineales | Exploración inicial |
| Chi-cuadrado | Bueno para categóricos | No funciona bien con continuos | Features discretos |
| ANOVA F-value | Robusto para gaussianos | Asume normalidad | Datos normalizados |

**Justificación**: Nuestros features (HSV, CIELAB) tienen relaciones **no lineales** con las categorías de iluminación. IG captura estas relaciones mejor que la correlación lineal.

### 3.3 Ejemplo Esperado de Ranking

Basado en el análisis exploratorio existente, esperamos rankings similares a:

**Categoría: PAN (Pantallas)**
| Rank | Feature | IG Score (estimado) | Justificación |
|------|---------|---------------------|---------------|
| 1 | `mean_b` | ~0.78 | Luz azul de pantallas |
| 2 | `std_v` | ~0.45 | Uniformidad típica de pantallas |
| 3 | `v_95` | ~0.38 | Brillo consistente |
| 4 | `skew_v` | ~0.32 | Distribución simétrica |
| 5 | `mean_s` | ~0.28 | Saturación de color |

**Categoría: NA (Natural)**
| Rank | Feature | IG Score (estimado) | Justificación |
|------|---------|---------------------|---------------|
| 1 | `mean_b` | ~0.65 | Shift amarillo (positivo) |
| 2 | `std_v` | ~0.52 | Variación natural de luz |
| 3 | `skew_v` | ~0.48 | Distribución asimétrica |
| 4 | `mean_a` | ~0.35 | Tonos cálidos |
| 5 | `v_95` | ~0.30 | Rangos de brillo |

---

## 4. Implementación Propuesta

### 4.1 Nuevos Módulos a Crear

```
webcam-light-sensor/
├── src/
│   └── feature_selection/
│       ├── __init__.py
│       ├── information_gain.py      # Cálculo de IG
│       ├── category_ranker.py       # Ranking por categoría
│       └── feature_selector.py      # Selección de features
├── modelos/
│   ├── category_models/
│   │   ├── __init__.py
│   │   ├── base_category_model.py   # Clase base
│   │   ├── na_model.py              # Modelo Natural
│   │   ├── ar_model.py              # Modelo Artificial
│   │   ├── pan_model.py             # Modelo Pantallas
│   │   ├── mix_model.py             # Modelo Mix
│   │   └── meta_classifier.py       # Meta-clasificador
│   └── multi_category_classifier.py # Clasificador integrado
└── docs/
    └── MEJORAS_MODELOS_POR_CATEGORIA.md
```

### 4.2 Flujo de Entrenamiento

```python
# Pseudocódigo del nuevo flujo
def entrenar_sistema_multicategoria(train_path, top_k=5):
    # 1. Cargar datos
    df = pd.read_csv(train_path)
    X = df[FEATURE_COLUMNS]
    y = df['Etiqueta']

    # 2. Para cada categoría, calcular Information Gain
    rankings = {}
    for categoria in ['natural', 'artificial', 'pantallas', 'mix']:
        y_binary = (y == categoria).astype(int)
        ig_scores = mutual_info_classif(X, y_binary)
        rankings[categoria] = rank_features(X.columns, ig_scores)

    # 3. Entrenar modelo binario por categoría con top-K features
    modelos = {}
    for categoria in rankings:
        top_features = rankings[categoria][:top_k]
        X_cat = X[top_features]
        y_cat = (y == categoria).astype(int)

        modelo = GradientBoostingClassifier()
        modelo.fit(X_cat, y_cat)
        modelos[categoria] = modelo

    # 4. Entrenar meta-clasificador
    meta_features = obtener_probabilidades(modelos, X)
    meta_clf = LogisticRegression(multi_class='multinomial')
    meta_clf.fit(meta_features, y)

    return modelos, meta_clf, rankings
```

### 4.3 Outputs Generados

1. **Matriz de Ranking IG** (CSV):
```csv
feature,IG_natural,IG_artificial,IG_pantallas,IG_mix,global_score
mean_b,0.65,0.32,0.78,0.28,0.51
std_v,0.52,0.41,0.45,0.55,0.48
...
```

2. **Modelos Serializados** (PKL):
```
modelos/category_models/
├── modelo_na.pkl
├── modelo_ar.pkl
├── modelo_pan.pkl
├── modelo_mix.pkl
└── meta_classifier.pkl
```

3. **Reportes de Interpretabilidad**:
- SHAP values por categoría
- Feature importance plots
- Confusion matrix por categoría

---

## 5. Métricas de Evaluación

### 5.1 Métricas por Categoría

Para cada categoría evaluaremos:

| Métrica | Descripción | Target |
|---------|-------------|--------|
| **Precision** | TP / (TP + FP) | > 0.95 |
| **Recall** | TP / (TP + FN) | > 0.95 |
| **F1-Score** | 2 × (P × R) / (P + R) | > 0.95 |
| **AUC-ROC** | Área bajo curva ROC | > 0.98 |

### 5.2 Métricas de Estabilidad

| Métrica | Descripción | Target |
|---------|-------------|--------|
| **Jaccard Stability** | Consistencia de features seleccionados entre folds | > 0.70 |
| **Ranking Correlation** | Correlación de rankings entre folds | > 0.80 |

### 5.3 Comparación con Baseline

| Modelo | Accuracy | Interpretabilidad | Mantenibilidad |
|--------|----------|-------------------|----------------|
| RF Actual | 98.97% | Baja | Media |
| Multi-Categoría Propuesto | ≥98.5% | Alta | Alta |

**Nota**: El objetivo NO es necesariamente mejorar accuracy (ya es excelente), sino:
1. **Entender** qué features importan para cada categoría
2. **Facilitar** mejoras futuras en categorías específicas
3. **Documentar** el conocimiento del dominio en forma algorítmica

---

## 6. Plan de Implementación

### Fase 1: Feature Selection Module
- Implementar `CategoryIGRanker`
- Generar matriz de rankings
- Validar con análisis existente

### Fase 2: Category Models
- Implementar modelos binarios por categoría
- Calibrar probabilidades
- Validar individualmente

### Fase 3: Meta-Classifier
- Entrenar LogisticRegression sobre probabilidades
- Resolver conflictos entre categorías
- Evaluar sistema completo

### Fase 4: Interpretabilidad
- Integrar SHAP analysis
- Generar visualizaciones
- Documentar insights

---

## 7. Conclusión

La implementación de **modelos por categoría con ranking de Information Gain** representa una mejora arquitectónica significativa que:

1. **Mantiene** la alta precisión actual (98.97%)
2. **Añade** interpretabilidad completa por categoría
3. **Facilita** el debugging y mejora de edge cases
4. **Documenta** el conocimiento del dominio de forma algorítmica
5. **Prepara** el sistema para futuras extensiones (nuevas categorías, nuevos features)

Esta propuesta sigue las mejores prácticas de machine learning interpretable y se alinea con los objetivos académicos de entender profundamente el problema de clasificación de iluminación ambiental.

---

## Referencias

- Scikit-learn Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html
- SHAP Values: https://shap.readthedocs.io/
- Information Gain: https://en.wikipedia.org/wiki/Information_gain_(decision_tree)
- One-vs-Rest Classification: https://scikit-learn.org/stable/modules/multiclass.html
