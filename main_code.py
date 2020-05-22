import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense,Conv2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
import cv2
from keras.utils.np_utils import to_categorical

file=pd.read_csv('Data.csv')
#print(image.head())
path=r"C:\Users\prati\PycharmProjects\AIVEHICLEPOSE\AI my priject"


data=np.array(file)
data=data.transpose()
print(data.shape)
for i in range(0,955):
    data[0][i] = path+'/'+ data[0][i]

X_train,Y_train=data[0],data[1]
for i in range(0,955):
    X_train[i]=mpimg.imread(X_train[i])

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))


def img_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img=cv2.resize(img,(80,80))
    img = img / 255
    img=np.expand_dims(img, axis=0)
    img=img.reshape((80,80,1))
    return img

X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))

#X_train.reshape(764,250,376,1)
#X_valid.reshape(191,250,376,1)

Y_train = to_categorical(Y_train, 4)

Y_valid = to_categorical(Y_valid, 4)

def nvidia_model():
    # create model
    model = Sequential()
    model.add(Conv2D(24, (5, 5), input_shape=(80,80,1), activation='relu'))
    model.add(Conv2D(36, (5, 5), activation='relu'))
    #model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #   model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    #   model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu'))
    #   model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax'))
    #   model.add(Dropout(0.5))



    optimizer = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    return model
model = nvidia_model()
print(model.summary())
model.fit(X_train, Y_train,epochs=10,validation_split=0.2,batch_size=50,verbose=1, shuffle = 1)
model.save('model.h5')
#plt.imshow(X_train[80])
#plt.show()
#plt.axis("off")
#print(X_valid.shape)
#print(X_train[0])
#print(Y_train[500])
#print(Y_train)
#image=cv2.imread(X_train[0])
#cv2.imshow('image',image)
#cv2.waitKey(0)