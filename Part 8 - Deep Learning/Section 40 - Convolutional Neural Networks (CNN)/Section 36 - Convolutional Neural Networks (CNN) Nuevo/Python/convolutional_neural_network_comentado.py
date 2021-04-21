# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN

#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#32 features detector de 3x3 cada cuadrado
# input_shape 64x64 es el tamaño de la imagen y el 3 es para que mantegan los colres
# 2 es blanco y negro
# relu es para para tener no linearidad
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#reducimos el tamaño del feature map
##los argumentos es el tamaño del cuadrado
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#esto para evitar el overfitting y mas accuracy
# sacamos input_shape porque ya tenemos algo antes
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

#aca le aplicamos augmenttion, osea a las imaganes,
#las movemos, le hacemos zoom etc para hacerlas mas random
# y aumentar la precision
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), #dimensiones de las imagenes
                                                 batch_size = 32,
                                                 class_mode = 'binary') #porque tenes dos categorias gatos o perros

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, #cantidad de imaganes en cada epoch
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000) #cantidad en test set