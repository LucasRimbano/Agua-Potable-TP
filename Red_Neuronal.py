import pandas as pd
import numpy as np

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

    print("Tamaño de z1:", z1.shape)
    print("Tamaño de a1:", a1.shape)
    print("Tamaño de z2:", z2.shape)
    print("Tamaño de a2:", a2.shape)

    return z1, a1, z2, a2

z1, a1, z2, a2 = forward(X_train, W1, b1, W2, b2)



def calcular_costo(a2, y):
    error = a2 - y 
    costo = np.mean(error ** 2)

    print("Tamaño del error:", error.shape)
    print("Costo:", costo)

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

    print("Tamaño de dW1:", dW1.shape)
    print("Tamaño de db1:", db1.shape)
    print("Tamaño de dW2:", dW2.shape)
    print("Tamaño de db2:", db2.shape)

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


def entrenar_red(X_train, y_train, W1, b1, W2, b2, epochs=50000, learning_rate=0.01):

    cantidad_datos = X_train.shape[0]

    for epoch in range(epochs):

        
        indice = np.random.randint(0, cantidad_datos)

        x = X_train[indice].reshape(1, -1)
        y = y_train[indice].reshape(1, 1)

       
        z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)

        
        dW1, db1, dW2, db2 = backpropagation(x, y, z1, a1, z2, a2, W2)

      
        W1_viejo = W1.copy()
        b1_viejo = b1.copy()
        W2_viejo = W2.copy()
        b2_viejo = b2.copy()

      
        W1 = W1 - learning_rate * dW1   
        b1 = b1 - learning_rate * db1

        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

       
        if epoch % 500 == 0:
            _, _, _, a2_train = forward(X_train, W1, b1, W2, b2)
            costo = calcular_costo(a2_train, y_train)

            print("Epoch:", epoch)
            print("Costo:", costo)

            print("Ejemplo W1 viejo:", W1_viejo[0][0])
            print("Ejemplo W1 nuevo:", W1[0][0])

            print("Ejemplo b1 viejo:", b1_viejo[0][0])
            print("Ejemplo b1 nuevo:", b1[0][0])

            print("Ejemplo W2 viejo:", W2_viejo[0][0])
            print("Ejemplo W2 nuevo:", W2[0][0])

            print("Ejemplo b2 viejo:", b2_viejo[0][0])
            print("Ejemplo b2 nuevo:", b2[0][0])

            print("-----------------------------------")

    return W1, b1, W2, b2

def calcular_accuracy(a2, y):
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
print("EVALUACIÓN CON DATOS DE PRUEBA")
print("======================================")

_, _, _, a2_test = forward(X_test, W1, b1, W2, b2)

accuracy_test = calcular_accuracy(a2_test, y_test)