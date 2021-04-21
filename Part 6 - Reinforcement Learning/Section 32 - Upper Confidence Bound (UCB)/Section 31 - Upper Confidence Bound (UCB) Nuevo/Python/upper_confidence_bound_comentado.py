# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementacion de UCB
import math
N = 10000 # numeros de usuarios
d = 10 #cantidad de ads
#numero de version que la ad  fue seleccionada
numbers_of_selections =[0] * d 
#suma de recompensa de cada ronda hasta n
sums_of_rewards = [0] * d

#en cada elemento va a estar la ad que seleccionamos hasta 10000
ads_selected = []
total_reward = 0


for n in range(0,N):
    ad = 0
    max_upper_bound = 0 #para saber cual es el mayor UCB 
    for i in range(0,d):
        # condiciones iniciales para las primeras 10 rounds
        # donde seleccionamos las primeras 10, porque
        # no tenemos informacion
        if (numbers_of_selections[i]>0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            #log(n+1) porque empieza en 0 python
            delta_i = math.sqrt((3/2) * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
            #averiguar el max UCB
        else:
            # porque hacemos esto?
            # para que la primera vez seleccionar las primeras 10 ads
            # de esta forma nos aseguramos que no pasen el upper bound
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound 
            #necesitamos saber el index de cual ad tiene el max upper balance
            ad = i
        
    #debemos agregar al vector de ad seleccionada
    ads_selected.append(ad)
    #le agregamos 1 a la ad que fue seleccionada
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    #ahora debemos actualizar el vector de recompesans
    #segun el dataset de simulacion
    reward = dataset.values[n, ad] #obtenes el valor de la recompensa
    #osea 0 o 1
    #ahora debemos actualizar el vectod e las recompensas
    sums_of_rewards[ad] +=  reward
    
    total_reward = total_reward +reward
    
# Grafico
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()