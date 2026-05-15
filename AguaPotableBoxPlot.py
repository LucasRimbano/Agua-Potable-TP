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

    
    plt.figure(figsize=(6, 5))
    plt.boxplot(df[columna].dropna())
    plt.title(f"Boxplot de {columna}")
    plt.ylabel(columna)

   
    ruta_imagen = carpeta_salida / f"boxplot_{columna}.png"
    plt.savefig(ruta_imagen, dpi=300, bbox_inches="tight")

    plt.close()

print("\nListo. Los boxplots fueron guardados en la carpeta:")
print(carpeta_salida)