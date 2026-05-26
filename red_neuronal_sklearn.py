import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("analisis_normalizacion/water_potability_sin_atipicos_sin_nan_normalizado.csv")

columnas_entrada = [
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

columna_salida = "Potability"

X = df[columnas_entrada]
y = df[columna_salida]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


def entrenar_y_evaluar_sklearn(nombre_modelo, arquitectura, learning_rate=0.01, max_iter=50000):
    modelo = MLPClassifier(
        hidden_layer_sizes=arquitectura,
        activation="relu",
        solver="sgd",
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy_test = accuracy_score(y_test, pred_test)

    print("======================================")
    print(nombre_modelo)
    print("Arquitectura:", arquitectura)
    print("Accuracy entrenamiento:", accuracy_train)
    print("Porcentaje entrenamiento:", accuracy_train * 100, "%")
    print("Accuracy test:", accuracy_test)
    print("Porcentaje test:", accuracy_test * 100, "%")
    print("======================================")
    print()

    return {
        "Modelo": nombre_modelo,
        "Arquitectura": arquitectura,
        "Accuracy entrenamiento": accuracy_train,
        "Accuracy test": accuracy_test
    }



# PRUEBAS DE ARQUITECTURAS !=


resultados = []

resultados.append(
    entrenar_y_evaluar_sklearn(
        "Red equivalente a NumPy",
        (9,)
    )
)

resultados.append(
    entrenar_y_evaluar_sklearn(
        "Red con 1 capa oculta de 16 neuronas",
        (16,)
    )
)

resultados.append(
    entrenar_y_evaluar_sklearn(
        "Red con 1 capa oculta de 32 neuronas",
        (32,)
    )
)

resultados.append(
    entrenar_y_evaluar_sklearn(
        "Red con 2 capas ocultas de 9 neuronas",
        (9, 9)
    )
)

resultados.append(
    entrenar_y_evaluar_sklearn(
        "Red con 2 capas ocultas de 16 y 8 neuronas",
        (16, 8)
    )
)

resultados.append(
    entrenar_y_evaluar_sklearn(
        "Red con 2 capas ocultas de 32 y 16 neuronas",
        (32, 16)
    )
)




resultados_df = pd.DataFrame(resultados)

print("RESUMEN FINAL")
print(resultados_df)

resultados_df.to_csv("resultados_sklearn_arquitecturas.csv", index=False)