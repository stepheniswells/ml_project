import numpy as np
import pandas as pd
import tensorflow as tf
import os
from CNN import CNN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import string
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

from os.path import join
from sklearn.model_selection import train_test_split

cats_dir = r"C:\Users\steph\Desktop\Spring 2023\cs4375\Project\archive\dataset\training_set\cats"
all_cats_path = [join(cats_dir,filename) for filename in os.listdir(cats_dir)]

images_dir = r"C:\Users\steph\Desktop\Spring 2023\cs4375\Project\archive\dataset\training_set\dogs"
images_path = [join(images_dir,filename) for filename in os.listdir(images_dir)]

all_paths = all_cats_path + images_path

df = pd.DataFrame({
    'path': all_paths,
    'is_cat': [1 if path in all_cats_path else 0 for path in all_paths] })

X = df.path
Y = df.is_cat

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.2)
X_test_paths = X_test
print(type(X_test))


X_train = [tf.keras.utils.load_img(img_path,target_size=(32,32)) for img_path in X_train]
X_train = np.array([tf.keras.utils.img_to_array(img) for img in X_train])

X_test = [tf.keras.utils.load_img(img_path,target_size=(32,32)) for img_path in X_test]
X_test = np.array([tf.keras.utils.img_to_array(img) for img in X_test])

print(X_train.shape,X_test.shape)
X_train /= 255
X_test /= 255


# filter = np.array([[1,2],[3,4]])
# input = np.array([[1,2,3],[4,5,6],[7,8,9]])
# cnn = CNN(input, 1, 1)
# cnn.something(3, input, filter)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

res = model.predict(X_test)
print(res)
numCorrect=0
for i in range(len(X_test)):
    #print(res[i], X_test_paths.iloc[i])
    if res[i][0] > res[i][1]:
        if "dog" in X_test_paths.iloc[i]:
            numCorrect+=1
    else:
        if "cat" in X_test_paths.iloc[i]:
            numCorrect+=1
    
print(numCorrect / len(X_test))

print(model.evaluate(X_test, y_test))


#X_train = X_train.reshape(np.size(X_train, 0), np.size(X_train, 1) * np.size(X_train, 2)*np.size(X_train, 3))
#X_test = X_test.reshape(np.size(X_test, 0), np.size(X_test, 1) * np.size(X_test, 2)*np.size(X_test, 3))