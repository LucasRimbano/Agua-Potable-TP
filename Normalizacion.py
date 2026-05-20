import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analizar_y_normalizar_datos(ruta_csv, carpeta_salida="analisis_normalizacion", columna_objetivo="Potability"):
    df = pd.read_csv(ruta_csv) 
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(exist_ok=True)


    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
    if columna_objetivo in columnas_numericas:
        columnas_numericas = columnas_numericas.drop(columna_objetivo)

    resumen_antes_de_normalizar = []

    for columna in columnas_numericas:
        minimo = df[columna].min()
        maximo = df[columna].max()
        rango = maximo - minimo
        media = df[columna].mean()
        desvio = df[columna].std()
        cantidad_nan = df[columna].isna().sum()

        resumen_antes_de_normalizar.append({
            "Columna": columna,
            "Minimo": minimo,
            "Maximo": maximo,
            "Rango": rango,
            "Media": media,
            "Desvio estandar": desvio,
            "Cantidad NaN": cantidad_nan
        })

    resumen_antes_de_normalizar_df = pd.DataFrame(resumen_antes_de_normalizar)

    resumen_antes_de_normalizar_df.to_csv(
        carpeta / "resumen_antes_normalizacion.csv",
        index=False
    )
  

    df_sin_nan_sin_atipicos = df.copy()
   
    ruta_csv_limpio = carpeta / "water_potability_sin_atipicos_sin_nan.csv"
    df_sin_nan_sin_atipicos.to_csv(ruta_csv_limpio, index=False)

    df_normalizado = df_sin_nan_sin_atipicos.copy()

    for columna in columnas_numericas:
         minimo = df_sin_nan_sin_atipicos[columna].min()
         maximo = df_sin_nan_sin_atipicos[columna].max()
         df_normalizado[columna] = (df_sin_nan_sin_atipicos[columna] - minimo) / (maximo - minimo)
         
    

    ruta_csv_normalizado = carpeta / "water_potability_sin_atipicos_sin_nan_normalizado.csv"
    
    df_normalizado.to_csv(
        ruta_csv_normalizado,
        index=False
    )

    resumen_despues_de_normalizar = []

    for columna in columnas_numericas:
        minimo = df_normalizado[columna].min()
        maximo = df_normalizado[columna].max()
        rango = maximo - minimo
        media = df_normalizado[columna].mean()
        desvio = df_normalizado[columna].std()
        cantidad_nan = df_normalizado[columna].isna().sum()

        resumen_despues_de_normalizar.append({
            "Columna": columna,
            "Minimo": minimo,
            "Maximo": maximo,
            "Rango": rango,
            "Media": media,
            "Desvio estandar": desvio,
            "Cantidad NaN": cantidad_nan
        })

        
        plt.figure(figsize=(8, 5))
        plt.hist(df_normalizado[columna].dropna(), bins=30, edgecolor="black")

        plt.title(f"Histograma de {columna} - Datos normalizados")
        plt.xlabel(f"{columna} normalizado")
        plt.ylabel("Frecuencia")

        nombre_archivo = f"histograma_normalizado_{columna}.png"
        plt.savefig(carpeta / nombre_archivo, bbox_inches="tight")
        plt.close()

    resumen_despues_de_normalizar_df = pd.DataFrame(resumen_despues_de_normalizar)

    resumen_despues_de_normalizar_df.to_csv(
        carpeta / "resumen_despues_normalizacion.csv",
        index=False
    )

    print("Análisis y normalización terminados.")
    print(f"Archivos guardados en la carpeta: {carpeta.resolve()}")
    print()
    print("Resumen ANTES de normalizar:")
    print(resumen_antes_de_normalizar_df)
    print()
    print("Resumen DESPUÉS de normalizar:")
    print(resumen_despues_de_normalizar_df)
    print()
    print(f"CSV sin NaN y sin atipicos: {ruta_csv_limpio.resolve()}")
    print(f"CSV normalizado guardado en: {ruta_csv_normalizado.resolve()}")

    return df_normalizado, resumen_antes_de_normalizar_df, resumen_despues_de_normalizar_df

df_normalizado, resumen_antes_de_normalizar, resumen_despues_de_normalizar = analizar_y_normalizar_datos(
    "boxplots_agua_potable/water_potability_sin_atipicos_sin_nan.csv"
)