import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("analisis_normalizacion/water_potability_sin_atipicos_sin_nan_normalizado.csv")

np.random.seed(42)
#1. división train/test
#2. inicialización de pesos
#3. forward
#4. costo
#5. backpropagation
#6. entrenamiento con epochs

print(df.columns.tolist())

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

X = df[columnas_entrada].values
y = df[columna_salida].values.reshape(-1, 1)

print("Tamaño de X:", X.shape)
print("Tamaño de y:", y.shape)


def relu(z):
    return np.maximum(0, z)

def derivada_relu(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def derivada_sigmoid(a):
    return a * (1 - a)


def entrenamiento_y_test(X, y, porcentaje_test=0.3):
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    
    cantidad_test = int(X.shape[0] * porcentaje_test)

    
    indices_test = indices[:cantidad_test]
    indices_train = indices[cantidad_test:]

    X_train = X[indices_train]
    y_train = y[indices_train]

    
    X_test = X[indices_test]
    y_test = y[indices_test]

    print("Cantidad total de datos:", X.shape[0])
    print("Cantidad de datos para entrenamiento:", X_train.shape[0])
    print("Cantidad de datos para prueba:", X_test.shape[0])
    print("Cantidad de columnas de entrada:", X_train.shape[1])

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = entrenamiento_y_test(X, y)


def inicializar_pesos(cantidad_entradas,cantidad_neuronas_ocultas,cantidad_salidas):
    
    W1 = np.random.randn(cantidad_entradas,cantidad_neuronas_ocultas) * 0.01

    b1 =np.zeros((1,cantidad_neuronas_ocultas))

    W2 = np.random.randn(cantidad_neuronas_ocultas,cantidad_salidas) * 0.01

    b2 = np.zeros((1,cantidad_salidas))

    print("Tamañano de W1:", W1.shape)
    print("Tamañano de b1:", b1.shape)
    print("Tamañano de W2:", W2.shape)
    print("Tamañano de b2:", b2.shape)

    return W1, b1, W2, b2

cantidad_entradas = X_train.shape[1]
cantidad_neuronas_ocultas = 9 
cantidad_salidas = 1

W1, b1, W2, b2 = inicializar_pesos(
    cantidad_entradas,
    cantidad_neuronas_ocultas,
    cantidad_salidas
)



def forward(X, W1, b1, W2, b2):
    
    z1 = np.dot(X, W1) + b1 
    a1 = relu(z1)

   
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    

    return z1, a1, z2, a2

z1, a1, z2, a2 = forward(X_train, W1, b1, W2, b2)

def calcular_costo(a2, y):
    error = a2 - y 
    costo = np.mean(error ** 2)


    return costo


costo = calcular_costo(a2, y_train)



def backpropagation(X, y, z1, a1, z2, a2, W2):
    cantidad_datos = X.shape[0]

   
    error_salida = a2 - y

    
    dz2 = (2 / cantidad_datos) * error_salida * derivada_sigmoid(a2)

   
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    
    error_oculto = np.dot(dz2, W2.T)

    
    dz1 = error_oculto * derivada_relu(z1)

    
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    

    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = backpropagation(
    X_train,
    y_train,
    z1,
    a1,
    z2,
    a2,
    W2
)


def entrenar_red(X_train, y_train,X_test, y_test, W1, b1, W2, b2, epochs=50000, learning_rate=0.01):

    cantidad_datos = X_train.shape[0]
    
    historial_epochs = []
    historial_costo_entrenamiento = []
    historial_costo_test = []
    historial_accuracy_entrenamiento = []
    historial_accuracy_test = []

    epochs_a_guardar = np.linspace(0, epochs - 1, 100, dtype=int)

    for epoch in range(epochs):
        indice = np.random.randint(0, cantidad_datos)

        x = X_train[indice].reshape(1, -1)
        y = y_train[indice].reshape(1, 1)

       
        z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
        
        dW1, db1, dW2, db2 = backpropagation(x, y, z1, a1, z2, a2, W2)

      
        W1 = W1 - learning_rate * dW1   
        b1 = b1 - learning_rate * db1

        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

       
        if epoch in epochs_a_guardar:
            _, _, _, a2_train = forward(X_train, W1, b1, W2, b2)
            _, _, _, a2_test = forward(X_test, W1, b1, W2, b2)
            costo_entrenamiento = calcular_costo(a2_train, y_train)
            costo_test = calcular_costo(a2_test, y_test)
            
            accuracy_entrenamiento = calcular_accuracy(a2_train, y_train)
            accuracy_test = calcular_accuracy(a2_test, y_test)

            historial_epochs.append(epoch)
            historial_costo_entrenamiento.append(costo_entrenamiento)
            historial_costo_test.append(costo_test)
            historial_accuracy_entrenamiento.append(accuracy_entrenamiento)
            historial_accuracy_test.append(accuracy_test)

            print(
                "Epoch:", epoch,
                "| Costo train:", round(costo_entrenamiento, 4),
                "| Costo test:", round(costo_test, 4),
                "| Acc train:", round(accuracy_entrenamiento * 100, 2), "%",
                "| Acc test:", round(accuracy_test * 100, 2), "%"
            )
  

    return W1, b1, W2, b2,historial_epochs, historial_costo_entrenamiento, historial_costo_test, historial_accuracy_entrenamiento, historial_accuracy_test




def calcular_accuracy(a2, y):
    predicciones = (a2 >= 0.5).astype(int)

    aciertos = np.sum(predicciones == y)
    total = y.shape[0]

    accuracy = aciertos / total

    return accuracy




W1, b1, W2, b2, historial_epochs, historial_costo_entrenamiento, historial_costo_test, historial_accuracy_entrenamiento, historial_accuracy_test = entrenar_red(
    X_train,
    y_train,
    X_test,
    y_test,
    W1,
    b1,
    W2,
    b2,
    epochs=50000,
    learning_rate=0.01
)



def graficar_curvas_un_solo_grafico(
    historial_epochs,
    historial_costo_entrenamiento,
    historial_costo_test,
    historial_accuracy_entrenamiento,
    historial_accuracy_test
):

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Eje izquierdo: pérdida
    ax1.plot(historial_epochs, historial_costo_entrenamiento, label="Pérdida entrenamiento")
    ax1.plot(historial_epochs, historial_costo_test, label="Pérdida test")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Pérdida")
    ax1.grid(True)

    # Eje derecho: accuracy
    ax2 = ax1.twinx()
    ax2.plot(
        historial_epochs,
        np.array(historial_accuracy_entrenamiento) * 100,
        label="Accuracy entrenamiento",
        linestyle="--"
    )
    ax2.plot(
        historial_epochs,
        np.array(historial_accuracy_test) * 100,
        label="Accuracy test",
        linestyle="--"
    )
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)

    plt.title("Curvas de pérdida y precisión")

    # Unimos las leyendas de los dos ejes
    lineas_1, etiquetas_1 = ax1.get_legend_handles_labels()
    lineas_2, etiquetas_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lineas_1 + lineas_2,
        etiquetas_1 + etiquetas_2,
        loc="best"
    )

    plt.savefig("curvas_entrenamiento_test.png", dpi=300, bbox_inches="tight")
    plt.show()

graficar_curvas_un_solo_grafico(
    historial_epochs,
    historial_costo_entrenamiento,
    historial_costo_test,
    historial_accuracy_entrenamiento,
    historial_accuracy_test
)

def mostrar_accuracy_final(a2, y):
    predicciones = (a2 >= 0.5).astype(int)

    aciertos = np.sum(predicciones == y)
    total = y.shape[0]

    accuracy = aciertos / total

    print("Aciertos:", aciertos)
    print("Total:", total)
    print("Accuracy:", accuracy)
    print("Porcentaje de acierto:", accuracy * 100, "%")

    return accuracy

print("\n======================================")
print("EVALUACIÓN FINAL CON DATOS DE ENTRENAMIENTO")
print("======================================")

_, _, _, a2_train = forward(X_train, W1, b1, W2, b2)

accuracy_train = mostrar_accuracy_final(a2_train, y_train)

print("\n======================================")
print("EVALUACIÓN CON DATOS DE PRUEBA")
print("======================================")

_, _, _, a2_test = forward(X_test, W1, b1, W2, b2)

accuracy_test = mostrar_accuracy_final(a2_test, y_test)
