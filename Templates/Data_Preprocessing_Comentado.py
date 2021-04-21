# Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------#
# Importing the dataset
#---------------------------------------#
dataset = pd.read_csv('Data.csv')
# ahora vamos a crear una matrix de todos los valores independientes

# X contiene todas las columnas de los valores independientes
# [las lineas, las columnas] = agarramos todas las columas, menos la ultima
# y .values significa tomar todos los valores
X = dataset.iloc[:, :-1].values
# print(X)
# ahora agarramos el vector de variables dependientes
y = dataset.iloc[:, 3].values


#---------------------------------------#
# Taking care of missing data 
#---------------------------------------#
from sklearn.impute import SimpleImputer
# para encargarnos de los valores que falta en el dataset
# usamos la libreria de sklearn

# usamos NaN porque son los valores del dataset .csv
# que no tiene valores y se los representa asi
# axis= 0 significa la media de los columnas y el 1 de las filas
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# ahora debemos meter esto objeto en el dataset
# debemos meter en las columnas don hay missing data
# que serian las segunda y tercera columan, osea 1 y 2, pero
# el [] si ponemos 2 va hasta el 1, tiene que ir 3, por no incluye
# el final
imputer = imputer.fit(X[:, 1:3])

# ahora debemos reemplaza los valores que no tiene X
# con lA MEDIA,
# SELECCIONAmos las columas con la falta de datos
X[:, 1:3] = imputer.transform(X[:, 1:3])

#---------------------------------------#
# Encoding categorical data
#---------------------------------------#
#Para lidear con los valores categoricos usamos
from sklearn.preprocessing import LabelEncoder

labelenconder_X = LabelEncoder()
# llenamos la matriz encodeada, osea sacas las categorias,
# y las transforma en numeros

#asi las asignamos a las primera columa
X[:,0] = labelenconder_X.fit_transform(X[:,0])



#---------------------------------------#
# Encoding the Independent Variable
#---------------------------------------#
# tenemos que evitar que el ML piense que francia>alemania y asi
# ahora vamos a amplicar algo que se llama Dummy encoding
# donde el numero de categoria es igual al numero de columnas que se 
# agregan
# ESTO es para el dummy encodel el OneHotEncoder
# este metodo es viejo, se actualizo
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# de esta forma reemplazamos los paises por
# 3 columas con numeros, donde ninguno es mas grande
# que otro, osea no le asigna un orden
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)


#ahora vamos a haces lo mismo pero con las variables
# dependientes, las compras
#vamos a usar label Enconder
# porque el ML algorith va a saber que es una categoria
# y que no va a haber orden entre los dos

#ahora con esto nuestra compras son 1 y 0
#---------------------------------------#
# Encoding the Dependent Variable
#---------------------------------------#
labelenconder_y = LabelEncoder()
y[:] = labelenconder_y.fit_transform(y[:])



#---------------------------------------#
# Splitting the dataset into the Training set and Test set
#---------------------------------------#

#####################Training Set y Test Set #####
#vamos a dividir en esos dos sets nuestro dataset
# vamos a importar librerias

from sklearn.model_selection import train_test_split



#random_state nos va a dar los mismos resultados que el chabon
# del video, por eso lo usamos
# no hace falta ponerlo, es como la semilla
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0 )


#---------------------------------------#
#Feature Scaling
#---------------------------------------#

#osea tenemos que pasar todo a la misma escala
# por ejemplo el salario va de 50000-10000 y la edad
# de 50 a 80 por ejemplo, por lo tanto la distancia
# entre los puntos para uno es mucha y para otra es mas
# y el ML se puede confundir
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#para el train set, los fit y lo transfonsmormamos
# para el test solo transformamos
X_train = sc_X.fit_transform(X_train)
# no le hacemos el fit al test porque ya esta fitted al train set
# ya esta ajustado, si ajustamos X_train primero, X_test despues no hay que falta
X_test = sc_X.transform(X_test)
#

#en este caso no hace falta hacer feature scaling a las
# variables dependientes, porque esto es un ejemplo de 
# clasificacion, si tiene muchos valores, si se hace,
# por ejemplo en regresion


