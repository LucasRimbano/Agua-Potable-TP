import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analizar_min_max_rango_histogramas(
    ruta_csv,
    carpeta_salida="analisis_normalizacion",
    columna_objetivo="Potability"
):
    
    df = pd.read_csv(ruta_csv)

 
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(exist_ok=True)

    
    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns

   
    if columna_objetivo in columnas_numericas:
        columnas_numericas = columnas_numericas.drop(columna_objetivo)

    resumen = []

    for columna in columnas_numericas:
        minimo = df[columna].min()
        maximo = df[columna].max()
        rango = maximo - minimo
        media = df[columna].mean()
        desvio = df[columna].std()

        resumen.append({
            "Columna": columna,
            "Minimo": minimo,
            "Maximo": maximo,
            "Rango": rango,
            "Media": media,
            "Desvio estandar": desvio
        })

       
        plt.figure(figsize=(8, 5))
        plt.hist(df[columna].dropna(), bins=30, edgecolor="black")

        plt.title(f"Histograma de {columna}")
        plt.xlabel(columna)
        plt.ylabel("Frecuencia")

        
        nombre_archivo = f"histograma_{columna}.png"
        plt.savefig(carpeta / nombre_archivo, bbox_inches="tight")

    
        plt.close()

 
    resumen_df = pd.DataFrame(resumen)

    resumen_df.to_csv(carpeta / "resumen_min_max_rango.csv", index=False)

    print("Análisis terminado.")
    print(f"Archivos guardados en la carpeta: {carpeta.resolve()}")

    return resumen_df


resumen = analizar_min_max_rango_histogramas("water_potability.csv")

print(resumen)