import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd

file=pd.read_csv('Data.csv')
model=load_model('model.h5')
img=cv2.imread('4.png')
new=cv2.imread('4.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = cv2.resize(img, (80, 80))
img = img / 255
print(img.shape)
img = np.expand_dims(img, axis=0)
img = img.reshape((1,80, 80, 1))
index1=(model.predict_classes(img))
index2=index1[0]
if index2==0:
    print('REAR VIEW')
elif index2==1:
    print('right side view')
elif index2==2:
    print('front view')
else:
    print('left side view')

cv2.imshow('img',new)
cv2.waitKey(0)

