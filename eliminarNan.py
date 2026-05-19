import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score



df = pd.read_csv("water_potability.csv")

# Separar entradas y salida
X = df.drop("Potability", axis=1)
y = df["Potability"]


def entrenar_y_evaluar(X, y, nombre_experimento, estrategia_nan=None):
    
    print(nombre_experimento)
 

    print("Cantidad de muestras:", len(X))

    # Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pasos = []


    if estrategia_nan is not None:
        pasos.append(("imputador", SimpleImputer(strategy=estrategia_nan)))

    # Normalización Min-Max
    pasos.append(("normalizador", MinMaxScaler()))

    # Red neuronal
    pasos.append((
        "modelo",
        MLPClassifier(
            solver="sgd",
            hidden_layer_sizes=(9, 9),
            activation="relu",
            max_iter=100000,
            learning_rate_init=0.05,
            random_state=42
        )
    ))

    pipeline = Pipeline(pasos)

    # Entrenar
    pipeline.fit(X_train, y_train)

    # Predecir
    y_pred = pipeline.predict(X_test)

    # Evaluar
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    return accuracy

#Pruebas para medir la probablidad de acierto y ver que hacemos 
df_sin_nan = df.dropna()

X_sin_nan = df_sin_nan.drop("Potability", axis=1)
y_sin_nan = df_sin_nan["Potability"]

accuracy_sin_nan = entrenar_y_evaluar(
    X_sin_nan,
    y_sin_nan,
    "Modelo eliminando filas con NaN",
    estrategia_nan=None
)

accuracy_media = entrenar_y_evaluar(
    X,
    y,
    "Modelo reemplazando NaN por media",
    estrategia_nan="mean"
)



columnas_a_eliminar_1 = ["Turbidity"]

X_sin_turbidity = X.drop(columnas_a_eliminar_1, axis=1)

accuracy_sin_turbidity = entrenar_y_evaluar(
    X_sin_turbidity,
    y,
    "Modelo usando media y eliminando Turbidity",
    estrategia_nan="mean"
)



columnas_a_eliminar_2 = ["Turbidity", "ph"]

X_sin_turbidity_ph = X.drop(columnas_a_eliminar_2, axis=1)

accuracy_sin_turbidity_ph = entrenar_y_evaluar(
    X_sin_turbidity_ph,
    y,
    "Modelo usando media y eliminando Turbidity y ph",
    estrategia_nan="mean"
)




columnas_a_eliminar_3 = ["Turbidity", "ph", "Trihalomethanes"]

X_sin_turbidity_ph_trihalo = X.drop(columnas_a_eliminar_3, axis=1)

accuracy_sin_turbidity_ph_trihalo = entrenar_y_evaluar(
    X_sin_turbidity_ph_trihalo,
    y,
    "Modelo usando media y eliminando Turbidity, ph y Trihalomethanes",
    estrategia_nan="mean"
)


print()
print("RESUMEN FINAL")
print("Accuracy eliminando NaN:", accuracy_sin_nan)
print("Accuracy usando media:", accuracy_media)
print("Accuracy eliminando Turbidity:", accuracy_sin_turbidity)
print("Accuracy eliminando Turbidity y ph:", accuracy_sin_turbidity_ph)
print("Accuracy eliminando Turbidity, ph y Trihalomethanes:", accuracy_sin_turbidity_ph_trihalo)