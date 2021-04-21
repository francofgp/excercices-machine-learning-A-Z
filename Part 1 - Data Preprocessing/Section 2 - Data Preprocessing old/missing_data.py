# Data Preprocessing

# Importing the libraries
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')
# ahora vamos a crear una matrix de todos los valores independientes

# X contiene todas las columnas de los valores independientes
# [las lineas, las columnas] = agarramos todas las columas, menos la ultima
# y .values significa tomar todos los valores
X = dataset.iloc[:, :-1].values
# print(X)
# ahora agarramos el vector de variables dependientes
y = dataset.iloc[:, 3].values


# Taking care of missing data
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


