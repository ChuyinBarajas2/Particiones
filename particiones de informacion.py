import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


datos = pd.read_csv('irisbin.csv')

#train_test_split
X = datos.iloc[:, :4] 
y = datos.iloc[:, 4]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#odd-row
filas_impares = datos.iloc[::4, :]
filas_pares = datos.iloc[1::4, :]

# Validación Cruzada por Bloques
n_splits = 5  # Ajusta el número de bloques según tus necesidades
block_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

model = RandomForestClassifier()

aciertos_block_cv = 0
total_iteraciones_block_cv = block_cv.get_n_splits(X)


for train_index, test_index in block_cv.split(X):
    X_train_block_cv, X_test_block_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_block_cv, y_test_block_cv = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train_block_cv, y_train_block_cv)
    y_pred_block_cv = model.predict(X_test_block_cv)
    accuracy_block_cv = accuracy_score(y_test_block_cv, y_pred_block_cv)
    if accuracy_block_cv == 1.0:
        aciertos_block_cv += 1


# Validación Cruzada con Conjuntos Fijos
X_fixed_train, X_fixed_test, y_fixed_train, y_fixed_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_fixed_train, y_fixed_train)
y_pred_fixed = model.predict(X_fixed_test)
accuracy_fixed = accuracy_score(y_fixed_test, y_pred_fixed)

#LOO
total_samples = X.shape[0]
aciertos_loo = 0

for i in range(total_samples):
    X_train_loo = X.drop(i)  
    y_train_loo = y.drop(i)

    model.fit(X_train_loo, y_train_loo)
    y_pred_loo = model.predict(X.iloc[i:i + 1]) 
    if y_pred_loo == y.iloc[i]:
        aciertos_loo += 1

# Gráficos
plt.figure(figsize=(12, 8))

# Gráfico de Train-Test Split
plt.subplot(2, 3, 1)
plt.title('Train-Test Split')
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 3], color='blue', label='Entrenamiento', marker='o')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 3], color='red', label='Prueba', marker='o')
plt.legend()

# Gráfico de Odd-Row
plt.subplot(2, 3, 2)
plt.title('Odd-Row')
plt.scatter(filas_impares.iloc[:, 0], filas_impares.iloc[:, 3], color='blue', label='Entrenamiento', marker='x')
plt.scatter(filas_pares.iloc[:, 0], filas_pares.iloc[:, 3], color='red', label='Prueba', marker='x')
plt.legend()

# Gráfico de Validación Cruzada por Bloques
plt.subplot(2, 3, 3)
plt.title('Validación Cruzada por Bloques')
plt.scatter(X_train_block_cv.iloc[:, 0], X_train_block_cv.iloc[:, 3], color='blue', label='Entrenamiento', marker='v')
plt.scatter(X_test_block_cv.iloc[:, 0], X_test_block_cv.iloc[:, 3], color='red', label='Prueba', marker='v')
plt.legend()

#Grafico de validacion fixed 
plt.subplot(2, 3, 4)
plt.title('Conjuntos Fijos')
plt.scatter(X_fixed_train.iloc[:, 0], X_fixed_train.iloc[:, 3], color='blue', label='Entrenamiento', marker='^')
plt.scatter(X_fixed_test.iloc[:, 0], X_fixed_test.iloc[:, 3], color='red', label='Prueba', marker='^')
plt.legend()

#Grafica LOO
plt.subplot(2, 3, 5)
plt.title('Leave-One-Out (LOO)')
plt.scatter(X.iloc[:, 0], X.iloc[:, 3], color='blue', label='Entrenamiento', marker='o')
plt.scatter(X.iloc[aciertos_loo:aciertos_loo+1, 0], X.iloc[aciertos_loo:aciertos_loo+1, 3], color='green', label='Prueba (Acierto)', marker='o')
plt.scatter(X.iloc[:aciertos_loo, 0], X.iloc[:aciertos_loo, 3], color='red', label='Prueba (Error)', marker='x')
plt.scatter(X.iloc[aciertos_loo+1:, 0], X.iloc[aciertos_loo+1:, 3], color='red', marker='x')
plt.legend()


plt.show()
