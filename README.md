# Agua Potable - Análisis de Datos para Clasificación Binaria

Este repositorio contiene el análisis exploratorio de una base de datos de agua potable utilizada para un Trabajo Práctico de Matemática III.

El objetivo principal del proyecto es analizar un conjunto de datos reales relacionado con la potabilidad del agua, detectar valores atípicos, estudiar las escalas de las variables numéricas y justificar la necesidad de normalización antes de utilizar los datos en un modelo de clasificación binaria, como una red neuronal.

---

## Objetivo del proyecto

El objetivo del trabajo es preparar y analizar una base de datos para un problema de clasificación binaria.

La variable objetivo es:

- `Potability`

Esta columna indica si una muestra de agua es potable o no:

- `0`: agua no potable
- `1`: agua potable

Las demás columnas representan características físico-químicas del agua y funcionan como variables de entrada para el modelo.

---

## Columnas del dataset

El dataset contiene las siguientes variables:

| Columna | Descripción | Tipo de variable |
|---|---|---|
| `ph` | Nivel de acidez o alcalinidad del agua | Numérica continua |
| `Hardness` | Dureza del agua | Numérica continua |
| `Solids` | Cantidad de sólidos disueltos | Numérica continua |
| `Chloramines` | Concentración de cloraminas | Numérica continua |
| `Sulfate` | Concentración de sulfatos | Numérica continua |
| `Conductivity` | Conductividad eléctrica del agua | Numérica continua |
| `Organic_carbon` | Cantidad de carbono orgánico | Numérica continua |
| `Trihalomethanes` | Concentración de trihalometanos | Numérica continua |
| `Turbidity` | Turbidez del agua | Numérica continua |
| `Potability` | Indica si el agua es potable o no | Categórica binaria |

---

## Archivos del repositorio

```text
Agua-Potable-TP/
│
├── water_potability.csv
├── AguaPotableBoxPlot.py
├── MaxMin.py
│
├── boxplots_agua_potable/
│   ├── boxplot_ph.png
│   ├── boxplot_Hardness.png
│   ├── boxplot_Solids.png
│   └── ...
│
└── analisis_normalizacion/
    ├── resumen_min_max_rango.csv
    ├── histograma_ph.png
    ├── histograma_Hardness.png
    ├── histograma_Solids.png
    └── ...
    
