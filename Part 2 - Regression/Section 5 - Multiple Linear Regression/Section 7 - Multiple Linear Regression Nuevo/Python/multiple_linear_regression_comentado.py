# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Importing the libraries
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# %%
# Encoding categorical data

labelenconder_X = LabelEncoder()
X[:, 3] = labelenconder_X.fit_transform(X[:, 3])


# %%

# Encoding the Independent Variable
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# las columnas que estaban a la derecha de los estados, se mueven a la izquierda


# %%
# Avoiding the Dummy variable Trap

X = X[:, 1:]  # REMUEVO la columna de index 0, para remover la dummy variable, HAY ALGUNAS LIBRERIAS QUE REMUEVEN ESTO, otras que no


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# %%
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
# CUANDO LE APLICO AL FIT A ESTE objeto, ajusto los multiples linear regressor a mi training set, osea lo entreno aca
regressor.fit(X_train, y_train)


# %%
# Predicting the Test set results
y_pred = regressor.predict(X_test)


# %%
# Building the Optimas model with Backward elimination

# agregamos una columna de unos, en nuestra matriz de independent variables, porque necesitamos agrega y = (b0 x0{esto necesita el x0}) +b1x1+b2x2+...bnxn
# en muchas librerias incluyen esto y no hace falta hacer nada, pero para startmodels si necesitamos, para evaluar el p-value
# vamos a agregar una columna de 1 a la izquierda de X
#np.ones(filas, columnas)
# axis = 1 = columna
# tenemos que agregar al comienzo
# X = np.append(arr=X, np.ones(1,50).astype(int), axis = 1 )
# si hacemos eso de arriva le agregamos a la X una columan de nos al final
# para haces al revez, a la columna de unos le agregamos X y listo
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)


# %%
# vamos a crear una nueva matriz de features que va a ser la optima
# que va a tener la cantidad joya de independientes variables que tengan
# mayor impacto en el profit

# X[tomamos todas las lineas, especificamos los indices de las columnas(tenemos 5 en total en nuestro dataset)]
# X_opt = X[:,[0,1,2,3,4,5]] # en el tutorial hace esto pero no funciona
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
# ahora ajustamos nuestro regresor nuevamente con todos los V.Independientes, con una nueva libreria
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()


# %%
# ahora debemos ver cual tiene el valor p-value mas alto para eliminarlo
regressor_OLS.summary()


# %%
# a partir de esta tabla vamos a elimnar X2 porque tiene mucho p-value
# la constante tiene indice 0, x1 1, x2 2, ver la Variable X y te das cuenta
# a partir de esto

X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# y continuamos con esto


# %%
X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# %%
X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# %%
# tenia 0.6 de p_value se podia elegir o no
X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# %%
# backwardElimination automatico


def backwardElimination(x, sl):
    numVars = len(x[0])  # cantidad de variables independientes
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
X_Modeled = backwardElimination(X_opt, SL)
print(X_Modeled)
