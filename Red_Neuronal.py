import pandas as pd
import numpy as np

df = pd.read_csv("analisis_normalizacion/water_potability_sin_atipicos_sin_nan_normalizado.csv")


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
