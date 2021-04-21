#Polynomial Regresor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#No usamos la columna de POSITION porque, basicamente es lo mismo que NIVEL
dataset = pd.read_csv('Position_Salaries.csv')

#cuando hacemos ML model queremos que nuestra matris de features que sea
#siempre una matriz y no un vector para eso lo cortamos con los ":"
X = dataset.iloc[:,1:2].values
# y es un vector
y = dataset.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
# No lo hacemos porque tenemos pocos datos, no hace falta,
# tampoco porque queremos hacer una precision muy precisa,
# no queremos perder el objetivo
# para tener mas precision en la prediccion tenemo que tener la mayor precision
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


#el linear regresin library ya hace feature scaling

## Training the Linear Regression model on the whole dataset
#vamos a hacer 2 modelos para comparan el polimial con
# el linear 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Training the Polynomial Regression model on the whole dataset
#importamos para que nos deje incluir terminos polinomicos
# en la linear regresion
from sklearn.preprocessing import PolynomialFeatures
#esto va a transformar nuestra matriz X, en una nueva matriz
# que se va a llamar x poly, que va atener variables independientes
# de x a la 1 (las que ya tenemos), hasta x a la n
poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X) # le agregamos polinomio adicionales
#automaticamente se crea una columna de 1 para añadir el termino independiente
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
# basicamente pongo esto poly_reg.fit_transform(X)
# para hacerlo mas generico, por si cambia X, porque
# x_poly ya esta definido
#osea con esto, lo hacemos al codigo aplicable a cualquier nueva
# matriz X
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#ahora vamos a hacer que predizca en lugar de 1 a 1, hasta 10
# de 0.1 hasta 10, osea la aumentamos la resolucion
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

#esto va a hacer que aumente de 0.1 en 0.1 un array
#osea creamos una nueva matriz
#PERO estoo nos da un vector y queremos una matriz
X_grid = np.arange(min(X), max(X), 0.1)
#Asi nos da una matriz, le indicamos las filas y columnas
#las filas son la cantidad de elementos de X y le damos 1 columna
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
#vamos a predecir el resultado de 6.5
#usamos el mismo metodo, pero para predecir 1 valor nomas
#necesita 2 dimensiones por eso los dos "[[ ]]"
lin_reg.predict([[6.5]])


# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))





