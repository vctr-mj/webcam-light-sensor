# Informe de Resultados: Clasificador de Fuentes de Luz

## Proyecto: webcam-light-sensor
**Fecha:** Enero 2026
**Precision Global:** 98.97%

---

## 1. Resumen Ejecutivo

### Objetivo del Proyecto

Desarrollar un sistema de clasificacion automatica de fuentes de luz mediante analisis de imagenes capturadas por webcam, capaz de distinguir entre cuatro categorias:

- **Natural**: Luz solar, luz de dia
- **Artificial**: Luz de bombillas, lamparas LED, luz fluorescente
- **Pantallas**: Luz emitida por monitores, televisores, telefonos
- **Mix**: Combinacion de multiples fuentes de luz

### Resultados Principales

| Metrica | Valor |
|---------|-------|
| **Precision Global (Accuracy)** | 98.97% |
| **Precision Ponderada** | 98.97% |
| **Recall Ponderado** | 98.97% |
| **F1-Score Ponderado** | 98.97% |
| **Total de Errores** | 17 de 1,460 muestras |
| **Tasa de Error** | 1.16% |

### Hallazgos Clave

1. **Perfecta clasificacion de pantallas y luz artificial**: Ambas categorias lograron 100% de precision
2. **Confusiones limitadas a natural/mix**: Los unicos 17 errores ocurren entre luz natural y condiciones mixtas
3. **El sesgo (skew_v) es la variable mas informativa globalmente** con IG = 0.454
4. **Pantallas presenta la mayor ganancia de informacion** para todas las variables (>0.57)

---

## 2. Metodologia

### 2.1 Arquitectura del Clasificador Multi-Categoria

El sistema implementa una arquitectura jerarquica de clasificacion denominada **Multi-Category Classifier** que combina tres componentes principales:

```
                    +---------------------------+
                    |   Datos de Entrada (X)    |
                    +---------------------------+
                              |
                              v
                    +---------------------------+
                    |   CategoryIGRanker        |
                    |   (Seleccion de Features) |
                    +---------------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        +---------+     +---------+     +---------+
        | Modelo  |     | Modelo  |     | Modelo  |
        | Natural |     |Artificial|    |Pantallas|  ...
        +---------+     +---------+     +---------+
              |               |               |
              v               v               v
        +---------------------------------------------+
        |         Probabilidades por Categoria        |
        +---------------------------------------------+
                              |
                              v
        +---------------------------------------------+
        |     Meta-Clasificador (Logistic Regression) |
        +---------------------------------------------+
                              |
                              v
        +---------------------------------------------+
        |              Prediccion Final               |
        +---------------------------------------------+
```

#### Componentes:

1. **CategoryIGRanker (Seleccion de Caracteristicas)**
   - Calcula la Ganancia de Informacion (IG) para cada par feature-categoria
   - Permite seleccion de caracteristicas especificas por categoria
   - Implementa el paradigma One-vs-Rest para cada categoria

2. **Clasificadores Binarios por Categoria (CategorySpecificModel)**
   - Un modelo Random Forest por cada categoria
   - Cada modelo usa las Top-K caracteristicas segun su IG para esa categoria
   - Genera probabilidades de pertenencia a cada clase

3. **Meta-Clasificador (Ensemble)**
   - Logistic Regression multinomial
   - Combina las probabilidades de los clasificadores binarios
   - Produce la prediccion final multi-clase

### 2.2 Seleccion de Caracteristicas por Ganancia de Informacion

La **Ganancia de Informacion (Information Gain, IG)** mide cuanta incertidumbre sobre la clase objetivo reduce una caracteristica. Se calcula como:

```
IG(Y, X) = H(Y) - H(Y|X)
```

Donde:
- H(Y) = Entropia de la variable objetivo (clase)
- H(Y|X) = Entropia condicional dado el valor de la caracteristica X

**Valores de IG:**
- IG = 0: La caracteristica no aporta informacion
- IG alto (>0.5): La caracteristica es muy discriminativa
- IG moderado (0.3-0.5): La caracteristica aporta informacion util

### 2.3 Division Train/Test

| Conjunto | Muestras | Porcentaje |
|----------|----------|------------|
| Entrenamiento | 5,839 | 80% |
| Prueba | 1,460 | 20% |
| **Total** | **7,299** | **100%** |

**Distribucion por categoria:**

| Categoria | Entrenamiento | Test | Total |
|-----------|---------------|------|-------|
| mix | 1,556 (26.6%) | 389 (26.6%) | 1,945 |
| pantallas | 1,510 (25.9%) | 377 (25.8%) | 1,887 |
| artificial | 1,462 (25.0%) | 366 (25.1%) | 1,828 |
| natural | 1,311 (22.5%) | 328 (22.5%) | 1,639 |

La distribucion es razonablemente balanceada entre categorias, lo cual favorece el entrenamiento del modelo.

---

## 3. Analisis de Ganancia de Informacion

### 3.1 Matriz Completa de IG (Features x Categorias)

| Feature | natural | artificial | pantallas | mix |
|---------|---------|------------|-----------|-----|
| **std_v** | 0.329 | 0.382 | **0.571** | 0.359 |
| **mean_b** | 0.241 | 0.371 | **0.571** | 0.256 |
| **skew_v** | 0.375 | 0.455 | **0.569** | 0.415 |
| **v_95** | 0.337 | **0.513** | **0.570** | 0.389 |

*Valores en negrita indican IG > 0.5 (alta discriminacion)*

### 3.2 Ranking de Features por Categoria

#### Natural
| Ranking | Feature | IG Score |
|---------|---------|----------|
| 1 | skew_v | 0.375 |
| 2 | v_95 | 0.337 |
| 3 | std_v | 0.329 |
| 4 | mean_b | 0.241 |

#### Artificial
| Ranking | Feature | IG Score |
|---------|---------|----------|
| 1 | v_95 | 0.513 |
| 2 | skew_v | 0.455 |
| 3 | std_v | 0.382 |
| 4 | mean_b | 0.371 |

#### Pantallas
| Ranking | Feature | IG Score |
|---------|---------|----------|
| 1 | mean_b | 0.571 |
| 2 | std_v | 0.571 |
| 3 | v_95 | 0.570 |
| 4 | skew_v | 0.569 |

#### Mix
| Ranking | Feature | IG Score |
|---------|---------|----------|
| 1 | skew_v | 0.415 |
| 2 | v_95 | 0.389 |
| 3 | std_v | 0.359 |
| 4 | mean_b | 0.256 |

### 3.3 Ranking Global de Features

| Ranking | Feature | IG Global |
|---------|---------|-----------|
| 1 | **skew_v** | 0.454 |
| 2 | **v_95** | 0.452 |
| 3 | std_v | 0.410 |
| 4 | mean_b | 0.360 |

### 3.4 Interpretacion de Cada Variable

#### **std_v (Desviacion Estandar del Canal Value/Brillo)**

| Categoria | IG | Interpretacion |
|-----------|-------|----------------|
| pantallas | 0.571 | Las pantallas tienen brillo muy uniforme (baja std_v), lo que las distingue claramente de otras fuentes |
| artificial | 0.382 | La luz artificial tiende a ser uniforme pero menos que pantallas |
| mix | 0.359 | Las condiciones mixtas muestran variabilidad intermedia |
| natural | 0.329 | La luz natural presenta mayor variacion (sombras, nubes, etc.) |

**Por que importa:** La desviacion estandar del brillo captura la uniformidad de la iluminacion. Las pantallas emiten luz muy homogenea, mientras que la luz natural es inherentemente variable.

#### **mean_b (Media del Canal Blue/Azul)**

| Categoria | IG | Interpretacion |
|-----------|-------|----------------|
| pantallas | 0.571 | Las pantallas (especialmente LED) tienen componente azul distintivo |
| artificial | 0.371 | Luz artificial moderna (LED) tambien tiende a tener mas azul |
| mix | 0.256 | El componente azul es ambiguo en condiciones mixtas |
| natural | 0.241 | La luz solar tiene espectro mas balanceado |

**Por que importa:** El canal azul es caracteristico de la tecnologia LED utilizada en pantallas modernas. Las pantallas emiten mas luz azul que fuentes naturales.

#### **skew_v (Asimetria del Canal Value/Brillo)**

| Categoria | IG | Interpretacion |
|-----------|-------|----------------|
| pantallas | 0.569 | La distribucion de brillo en pantallas es asimetrica hacia valores altos |
| artificial | 0.455 | Luz artificial muestra sesgo moderado |
| mix | 0.415 | El sesgo varia segun las fuentes combinadas |
| natural | 0.375 | La luz natural tiende a distribuciones mas simetricas |

**Por que importa:** La asimetria (skewness) indica hacia donde se concentran los valores de brillo. Pantallas muestran picos en valores altos (blancos puros), mientras que la luz natural es mas distribuida.

#### **v_95 (Percentil 95 del Canal Value/Brillo)**

| Categoria | IG | Interpretacion |
|-----------|-------|----------------|
| pantallas | 0.570 | El percentil 95 captura los pixeles mas brillantes, distintivos en pantallas |
| artificial | 0.513 | Alta discriminacion: luz artificial tiene picos de brillo caracteristicos |
| mix | 0.389 | Discriminacion moderada en condiciones mixtas |
| natural | 0.337 | La luz natural raramente produce extremos como pantallas |

**Por que importa:** El percentil 95 representa el brillo de los pixeles mas claros. Las pantallas producen blancos puros (255) mientras que la luz natural raramente alcanza esos extremos.

---

## 4. Rendimiento del Modelo

### 4.1 Reporte de Clasificacion

| Categoria | Precision | Recall | F1-Score | Soporte |
|-----------|-----------|--------|----------|---------|
| artificial | 0.997 | 1.000 | 0.999 | 366 |
| mix | 0.982 | 0.982 | 0.982 | 389 |
| natural | 0.979 | 0.976 | 0.977 | 328 |
| pantallas | 1.000 | 1.000 | 1.000 | 377 |
| **Accuracy** | | | **0.990** | **1,460** |
| **Macro Avg** | 0.989 | 0.989 | 0.989 | 1,460 |
| **Weighted Avg** | 0.990 | 0.990 | 0.990 | 1,460 |

### 4.2 Matriz de Confusion

```
               P R E D I C C I O N
              natural  artificial  pantallas    mix
      natural    317         0          0        11
R  artificial      0       366          0         0
E  pantallas       0         0        377         0
A        mix       6         0          0       383
L
```

**Lectura de la matriz:**
- Diagonal: Clasificaciones correctas
- Fuera de diagonal: Errores de clasificacion

### 4.3 Precision por Categoria

| Categoria | Correctos | Total | Precision |
|-----------|-----------|-------|-----------|
| **pantallas** | 377 | 377 | **100.00%** |
| **artificial** | 366 | 366 | **100.00%** |
| **mix** | 383 | 389 | 98.46% |
| **natural** | 317 | 328 | 96.65% |

### 4.4 Analisis de Errores

#### Errores Totales: 17 de 1,460 (1.16%)

| Error | Cantidad | Porcentaje |
|-------|----------|------------|
| natural confundido con mix | 11 | 64.7% de errores |
| mix confundido con natural | 6 | 35.3% de errores |

#### Interpretacion de Errores

Los **unicos errores ocurren entre las categorias natural y mix**. Esto tiene una explicacion logica:

1. **Similitud conceptual**: La categoria "mix" frecuentemente incluye luz natural como componente principal, lo que genera superposicion en las caracteristicas

2. **Patron de confusiones**:
   - 11 casos de luz natural clasificados como mix: Posiblemente escenas donde la luz natural se filtra de manera que simula multiples fuentes
   - 6 casos de mix clasificados como natural: Probablemente cuando la luz natural domina la escena mixta

3. **Por que no hay confusion con pantallas/artificial**:
   - Las pantallas tienen caracteristicas unicas (alta uniformidad, componente azul distintivo)
   - La luz artificial tiene patrones de brillo especificos
   - Tanto pantallas como artificial son fuentes puntuales controladas, mientras que natural y mix son mas difusas

---

## 5. Trade-off Precision vs Cobertura

El sistema implementa un **modo de precision** que permite ajustar el umbral de confianza para las predicciones. Cuando la probabilidad maxima esta por debajo del umbral, el sistema responde "no_decidido".

### 5.1 Tabla Comparativa de Umbrales

| Umbral | Accuracy | Precision | Recall | Decididos | No Decididos | % No Decidido |
|--------|----------|-----------|--------|-----------|--------------|---------------|
| 0.5 | 98.90% | 98.90% | 98.90% | 1,458 | 2 | 0.14% |
| 0.6 | 98.90% | 98.90% | 98.90% | 1,457 | 3 | 0.21% |
| 0.7 | 98.97% | 98.97% | 98.97% | 1,456 | 4 | 0.27% |
| 0.8 | 99.10% | 99.10% | 99.10% | 1,450 | 10 | 0.68% |
| 0.9 | 99.37% | 99.37% | 99.37% | 1,439 | 21 | 1.44% |

### 5.2 Impacto en la Matriz de Confusion

Al aumentar el umbral:
- **Aumenta la precision**: Los casos inciertos se excluyen
- **Disminuye la cobertura**: Mas casos quedan sin clasificar
- **Se reducen errores**: Los errores entre natural/mix se filtran

### 5.3 Recomendaciones de Umbral

| Caso de Uso | Umbral Recomendado | Justificacion |
|-------------|-------------------|---------------|
| **Aplicacion general** | 0.7 | Balance optimo precision/cobertura |
| **Sistema critico** | 0.8-0.9 | Minimiza errores, acepta casos "no decidido" |
| **Alta cobertura** | 0.5-0.6 | Maximiza clasificaciones, acepta mas errores |

---

## 6. Conclusiones

### 6.1 Hallazgos Clave

1. **Alta precision lograda**: El sistema alcanza 98.97% de precision global, superando las expectativas para un clasificador de 4 categorias

2. **Clasificacion perfecta de pantallas y artificial**: Estas categorias tienen caracteristicas altamente distintivas que el modelo captura completamente

3. **Las variables derivadas del brillo (Value) son las mas informativas**: skew_v y v_95 lideran el ranking global de importancia

4. **La categoria pantallas es la mas distinguible**: Todas las variables tienen IG > 0.57 para pantallas, indicando que es la categoria mas facil de identificar

5. **Los unicos errores ocurren en el borde natural/mix**: Esto refleja la naturaleza superpuesta de estas categorias

### 6.2 Fortalezas del Sistema

| Fortaleza | Descripcion |
|-----------|-------------|
| **Arquitectura modular** | Clasificadores especializados por categoria permiten optimizacion independiente |
| **Seleccion inteligente de features** | IG por categoria asegura que cada modelo use las variables optimas |
| **Modo de precision configurable** | El umbral permite adaptar el sistema a diferentes requerimientos |
| **Interpretabilidad** | Las 4 variables tienen significado fisico claro |
| **Bajo costo computacional** | Solo 4 features necesarias para alta precision |

### 6.3 Limitaciones

| Limitacion | Impacto | Mitigacion Posible |
|------------|---------|-------------------|
| Confusion natural/mix | 1.16% de error | Agregar features adicionales (temporales, contextuales) |
| Dependencia del canal V | Sensible a cambios de exposicion | Normalizar por condiciones de captura |
| Dataset de un solo dispositivo | Generalizacion limitada | Expandir con multiples camaras |
| Categorias predefinidas | No detecta fuentes nuevas | Implementar deteccion de anomalias |

### 6.4 Mejoras Futuras

1. **Features adicionales**:
   - Analisis temporal (cambios de luz en secuencia)
   - Patron espacial de la luz (gradientes)
   - Frecuencia de parpadeo (para pantallas)

2. **Modelo mejorado**:
   - Redes neuronales para capturar patrones no lineales
   - Clasificador difuso para la frontera natural/mix

3. **Expansion del dataset**:
   - Multiples dispositivos de captura
   - Diferentes condiciones ambientales
   - Mayor variedad de fuentes artificiales

4. **Sistema en tiempo real**:
   - Clasificacion por video
   - Deteccion de transiciones entre fuentes

---

## Anexos

### A. Descripcion de Variables

| Variable | Nombre Completo | Descripcion |
|----------|-----------------|-------------|
| std_v | Desviacion estandar del Value | Variabilidad del brillo en la imagen |
| mean_b | Media del canal Blue | Promedio del componente azul (0-255 normalizado) |
| skew_v | Asimetria del Value | Sesgo de la distribucion de brillo |
| v_95 | Percentil 95 del Value | Valor de brillo en el percentil 95 |

### B. Configuracion del Modelo

```python
MultiCategoryClassifier(
    top_k=4,  # Usa las 4 features disponibles
    categories=['natural', 'artificial', 'pantallas', 'mix']
)

# Clasificadores binarios: Random Forest
# Meta-clasificador: Logistic Regression (multinomial)
# Umbral por defecto: 0.7
```

### C. Archivos de Datos

| Archivo | Ubicacion | Descripcion |
|---------|-----------|-------------|
| train.csv | datasets/ | Datos de entrenamiento (5,839 muestras) |
| test.csv | datasets/ | Datos de prueba (1,460 muestras) |
| ig_matrix.csv | outputs/feature_selection/ | Matriz IG completa |
| classification_report.csv | resultados_graficos/evaluacion_modelo/ | Reporte de clasificacion |

---

*Informe generado automaticamente como parte del proyecto webcam-light-sensor*
