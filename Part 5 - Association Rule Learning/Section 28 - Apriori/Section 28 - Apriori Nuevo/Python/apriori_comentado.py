# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
#  header = None porque no tenemos titulos en las columas
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# tenemos que transformar esto en una lista de lista,
# por que? porque asi lo pide la libreria apyori
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#min_length minimo de item en la bolsa, basket, si tenemos 1 no tiene sentido
#el min_support es 0.003 porque suponemos que un producto
# se compra 3 vecer por dia durante 7 dias
#entonces hace 3*7/7500=0.003=0.0028 mas o menos; porque 7500 es la cantidad 
# de productos en la semana que se venden
#min_confidence= queremos que tengan un acierto de que 20% de lo que se recomienda
#tengan razon, porque si elegimos algo muy alto
#nos va a dar que por ejemplo los huevos y el agua
# mineral estan relacionados, (con 0.8 si ponemos por ejemplo)
#pero eso no es asi, porque en realidad el algoritmo va a detectar que se relacionan,
#no porque por comprar agua compran muchos huevos, sino porque se compran muchos en general

# min_lift = 3 es normal, 6 es excelente 

# Visualising the results

#los resultados ya estan ordenados por prioridad de lift entre otras cosas
#combina varias cosas/criterios
results = list(rules)
#hay que imprimir porque el variable explorer no te lo muestra
print(results)