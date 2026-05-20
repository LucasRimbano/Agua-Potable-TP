import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("water_potability.csv")
carpeta_salida = Path("boxplots_agua_potable")
carpeta_salida.mkdir(exist_ok=True)

columnas = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]


print("\n======================================")
print("ANÁLISIS DE VALORES FALTANTES NaN")
print("======================================")

print("Cantidad total de muestras:", len(df))

resumen_nan = []

for columna in columnas:
    cantidad_nan = df[columna].isna().sum()
    porcentaje_nan = cantidad_nan / len(df) * 100

    resumen_nan.append({
        "Columna": columna,
        "Cantidad NaN": cantidad_nan,
        "Porcentaje NaN": porcentaje_nan
    })

    print(f"{columna}: {cantidad_nan} NaN ({porcentaje_nan:.2f}%)")

muestras_sin_nan = len(df.dropna())
muestras_perdidas = len(df) - muestras_sin_nan

print("\nCantidad de muestras si eliminamos todas las filas con NaN:")
print(muestras_sin_nan)

print("\nCantidad de muestras que se perderían:")
print(muestras_perdidas)


resumen_nan_df = pd.DataFrame(resumen_nan)
resumen_nan_df.to_csv(
    carpeta_salida / "resumen_valores_nan.csv",
    index=False
)

print("\nResumen de NaN guardado en:")
print(carpeta_salida / "resumen_valores_nan.csv")

df_sin_atipicos = df.copy()

for columna in columnas:
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)

    RIC = Q3 - Q1

    limite_inferior = Q1 - 1.5 * RIC
    limite_superior = Q3 + 1.5 * RIC

    atipicos = df[
        (df[columna] < limite_inferior) |
        (df[columna] > limite_superior)
    ]

    print("\n==============================")
    print("Columna:", columna)
    print("==============================")
    print("Q1 =", Q1)
    print("Q3 =", Q3)
    print("RIC =", RIC)
    print("Límite inferior =", limite_inferior)
    print("Límite superior =", limite_superior)
    print("Cantidad de atípicos =", len(atipicos))
    
    df_sin_atipicos = df_sin_atipicos[
    (df_sin_atipicos[columna].isna()) |
    (
        (df_sin_atipicos[columna] >= limite_inferior) &
        (df_sin_atipicos[columna] <= limite_superior)
    )
    ]
    
    plt.figure(figsize=(6, 5))
    plt.boxplot(df[columna].dropna())
    plt.title(f"Boxplot de {columna}")
    plt.ylabel(columna)

   
    ruta_imagen = carpeta_salida / f"boxplot_{columna}.png"
    plt.savefig(ruta_imagen, dpi=300, bbox_inches="tight")

    plt.close()

df_sin_atipicos.to_csv(
        carpeta_salida / "water_potability_sin_atipicos.csv",
        index=False
)

df_sin_atipicos_sin_nan = df_sin_atipicos.dropna()

df_sin_atipicos_sin_nan.to_csv(
    carpeta_salida / "water_potability_sin_atipicos_sin_nan.csv",
    index=False
)


df_media_luego_sin_atipicos = df.copy()


for columna in columnas:
    media = df_media_luego_sin_atipicos[columna].mean()
    df_media_luego_sin_atipicos[columna] = df_media_luego_sin_atipicos[columna].fillna(media)

for columna in columnas:
    Q1 = df_media_luego_sin_atipicos[columna].quantile(0.25)
    Q3 = df_media_luego_sin_atipicos[columna].quantile(0.75)

    RIC = Q3 - Q1

    limite_inferior = Q1 - 1.5 * RIC
    limite_superior = Q3 + 1.5 * RIC

    df_media_luego_sin_atipicos = df_media_luego_sin_atipicos[
        (df_media_luego_sin_atipicos[columna] >= limite_inferior) &
        (df_media_luego_sin_atipicos[columna] <= limite_superior)
    ]


df_media_luego_sin_atipicos.to_csv(
    carpeta_salida / "water_potability_media_luego_sin_atipicos.csv",
    index=False
)

print("\nListo. Los boxplots fueron guardados en la carpeta:")
print(carpeta_salida)
print("\nCSV sin valores atípicos guardado en:")
print(carpeta_salida / "water_potability_sin_atipicos.csv")
print("\nCSV sin valores atípicos y sin NaN guardado en:")
print(carpeta_salida / "water_potability_sin_atipicos_sin_nan.csv")
print("\nCantidad de muestras originales:", len(df))
print("Cantidad de muestras sin atípicos:", len(df_sin_atipicos))
print("Cantidad de muestras sin atípicos y sin NaN:", len(df_sin_atipicos_sin_nan))
print("Cantidad de muestras eliminadas al sacar atipicos:", len(df) - len(df_sin_atipicos))
print("Cantidad de muestras eliminadas sin atípicos y sin NaN:", len(df) - len(df_sin_atipicos_sin_nan))
print("\nCSV reemplazando NaN por media y luego eliminando atípicos guardado en:")
print(carpeta_salida / "water_potability_media_luego_sin_atipicos.csv")
print("Cantidad de muestras reemplazando NaN por media y luego eliminando atípicos:", len(df_media_luego_sin_atipicos))