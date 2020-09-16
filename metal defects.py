# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:08:16 2020

@author: Administrator
"""


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


data = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#Training & test Data Reading
trainData = data.flow_from_directory(directory=r'D:\Bhushan\NEU Metal Surface Defects Data\train',
                                           target_size=(256,256),
                                           class_mode = 'categorical',
                                           batch_size = 32,
                                           subset='training',
                                           shuffle=True)

trainData.class_indices

testData = data.flow_from_directory(directory=r'D:\Bhushan\NEU Metal Surface Defects Data\train',
                                           target_size=(256,256),
                                           class_mode = 'categorical',
                                           batch_size = 32,
                                           subset='validation',
                                           shuffle=True)

testData.class_indices

#Deep Learning model
model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(256,256,3)))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))


model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(6,activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Model Fitting
history=model.fit_generator(generator=trainData,
                            epochs=15,
                            validation_data=testData,
                            verbose=1, shuffle=True,
                            steps_per_epoch=100)
                            
                           

#plot for Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#test Image
test=r'D:\Bhushan\NEU Metal Surface Defects Data\test\Inclusion\In_107.bmp'
img=image.load_img(test,target_size=(256,256))
plt.imshow(img)
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])

val1=model.predict(images)

    
# Test Data
val=model.predict_classes(testData[0][0])
from sklearn.metrics import confusion_matrix , accuracy_score
p=confusion_matrix(testData[0][1] , val)
acc=accuracy_score(testData[0][1] , val)

