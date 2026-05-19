import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


df = pd.read_csv("water_potability.csv")

carpeta_salida = Path("AGUA_POTABLE_TP_correlacion_pearson")
carpeta_salida.mkdir(exist_ok=True)


df_nan_reemplazados_media = df.copy()


columnas_numericas = df_nan_reemplazados_media.select_dtypes(include=["int64", "float64"]).columns

# Reemplazar NaN por la media en las columnas numéricas
for columna in columnas_numericas:
    media = df_nan_reemplazados_media[columna].mean()
    df_nan_reemplazados_media[columna] = df_nan_reemplazados_media[columna].fillna(media)

# Calcular matriz de correlación de Pearson
matriz_correlacion = df_nan_reemplazados_media.corr(method="pearson")


matriz_correlacion.to_csv(
    carpeta_salida / "matriz_correlacion_pearson.csv"
)

# Crear heatmap de correlación
plt.figure(figsize=(12, 8))

sns.heatmap(
    matriz_correlacion,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Heatmap de correlación de Pearson")
plt.tight_layout()

plt.savefig(
    carpeta_salida / "heatmap_correlacion_pearson.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Mostrar correlación de cada columna con Potability
correlacion_potability = matriz_correlacion["Potability"].drop("Potability")

print("\n======================================")
print("CORRELACIÓN DE CADA VARIABLE CON POTABILITY")
print("======================================")

print(correlacion_potability.sort_values(ascending=False))


correlacion_potability.sort_values(ascending=False).to_csv(
    carpeta_salida / "correlacion_con_potability.csv",
    header=["Correlacion Pearson"]
)

print("\nArchivos guardados en la carpeta:")
print(carpeta_salida.resolve())

print("\nArchivos generados:")
print("- matriz_correlacion_pearson.csv")
print("- correlacion_con_potability.csv")
print("- heatmap_correlacion_pearson.png")