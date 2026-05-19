import pandas as pd
df = pd.read_csv("analisis_normalizacion/water_potability_normalizado.csv")

print(df.isnull().sum())
# df[:] = df[:].fillna(0)
# print(df)

# 1843 muestras totals para 9 datos de entrada 
# ver el tema de las capas de entrada y cuantas
# 


# ver la cantidad de overfitting sobre el tema de neuronas ocultas
print()