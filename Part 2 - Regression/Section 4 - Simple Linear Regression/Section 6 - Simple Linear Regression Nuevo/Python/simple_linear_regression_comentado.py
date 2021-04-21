from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)

# nuestra libreria ya hace es feature scaling

#------------------------------#
# Fitting  Simple linear regresoin to the training set
#------------------------------#

# vamos a crear un objeto, llamado regresor, va a
# ajustar/fit al training set
regresor = LinearRegression()
# ahora lo ajustamos al train set
# con esto nuestro modelo ya predijo todo, osea el salario
# en funcion de la experiencia
# regresor es la machine, ya aprendio la correlacion entre
# el salario y la experiencia
regresor.fit(X_train, y_train)

#------------------------------#
# predicting the Test set results
#------------------------------#
# vamos a crear un vector que va a tener las predicciones
# de los test set salaries,  y los vamos a poner estas predicciones
# en un vector que se llama y_pred
# vector que tiene las predicciones
y_pred = regresor.predict(X_test)
# a esto lo comparamos con  los verdadero

#------------------------------#
# Visualising the Training set results
#------------------------------#
# vamos a graficar los verdaderos, el TRAIN primero
# la observacion en rojo, y la linia de regresion en azul
plt.scatter(X_train, y_train, color='red')

# ahora hacemos la LINEA DE REGRESION
# la variable independiente es la X_train
# y la dependiente de la linea de regresion es lo que
# predice el modelo, NO EL DEL TEST, queremos las predicciones
# del train set, y no del test set
plt.plot(X_train, regresor.predict(X_train), color="blue")
plt.title("salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary $")
plt.show()

# grafico del test
plt.scatter(X_test, y_test, color='red')
# esta linea es la misma, porque la linea de se crea
# en funcion de training set
# no cambiar por x_test porque nuestro Regresor ya se entreno con X_train
plt.plot(X_test, regresor.predict(X_test), color="blue")
plt.title("salary vs Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary $")
plt.show()
