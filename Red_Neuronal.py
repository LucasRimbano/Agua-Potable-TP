import pandas as pd
df = pd.read_csv("analisis_normalizacion/water_potability_normalizado.csv")

df[:] = df[:].fillna(0)
print(df)