# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
'''

#requiere FEATURE SCALING SVR

# Feature Scaling
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
#tiene que tener dos dimensiones el array no 1, 
#por el reshape
y = sc_y.fit_transform(y.reshape(-1,1))

# Fitting The SVR to the dataset
from sklearn.svm import SVR
#Elegimos este kernel
regressor = SVR(kernel = "rbf")
regressor.fit(X,y)
#aca ya tenemos el SVR

# Predicting a new result with Polynomial Regression
#esto no va a andar porque esta con feature scaling,
#hay que desescalarlo
y_pred = regressor.predict([[6.5]])
#esto es lo mismo que lo de arrriba pero no anda
y_pred = regressor.predict(np.array([[6.5]]))

#asi es lo correcto, lo invierto a la X porque es el valor
#que yo quiero predicir no escalado
#y eso me da Y escalado, pero yo lo quiero original
#por eso lo hago sc_y.inverse
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

print(y_pred)
# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



#MAYOR RESOLUCION
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
