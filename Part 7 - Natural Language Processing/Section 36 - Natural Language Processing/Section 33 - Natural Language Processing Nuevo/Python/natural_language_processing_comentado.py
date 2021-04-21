# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###### Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = "\t", quoting = 3)

##### Cleaning the text
import re

# ahora removemos las cosas que no sirven como "the, on, this, etc"
import regex as res
import nltk
# estas son las lista de palabras que son irrelevantes
# para despues removerlas
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#list de reviews limpias
corpus = []

#loop para todas las reviews
for i in range(0,1000):
    
    # primero con la expresion regular ponemo lo que queremos dejar(se puede
    # hacer al revez, poner lo que queres sacar, despues lo que queda cuando reemplazamos
    # osea para que cuando quitas un PUNTO por ejemplo queda un espacio
    #luego le pasas el archivo)
    review = re.sub("[^a-zA-Z]", " " ,dataset['Review'][i])
    review = review.lower()

    # a review lo hacemos string para sacar las cosas
    review = review.split()
    
    
    #set no es necesario
    #como a esto le vamos a hacer stemming lo comento y lo hago abajo
    # todo junto
    #review = [word for word in review if not word in set(stopwords.words("english"))]
    
    #o asi
    #for word in review:
    #    if  word in set(stopwords.words("english")):
    #        review.remove(word)
    #ahora hacemos stemming,  osae quitamos cosas como matar,matarse,mataremos, para que nos
    # quede la raiz de la palabra osea matar, para achicar la matriz de cosas importantes
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    
    # ahora convertimos todo de nuevo a un string
    review = " ".join(review)
    
    corpus.append(review)
    
    
#Ahora creamos el bag of word model, esto es tomar todas las palabras
# diferentes del corpus, sin duplicarlas, las unicas nomas
# asi creamos una matriz con filas que son las review y las columnas
# que son las palabras que encontramos
# y en cada celda va a haber un numero que corresponde a la cantidad de veces que 
# aparece en la review, esto se llama tokenization
# una matriz con muchos ceros se llama sparse matrix

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
# para mantener las palabras mas frecuentes usamos max_features parametro
# para tener 1500 y no 1565, que si tuvieras mas reviews serian como
# no se un millon, reducimes la sparcity de la matrix
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# CON NAIVES BAYES
# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# CON RANDOM FOREST
# Training the Classifier model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion = "entropy", random_state=0)

classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
