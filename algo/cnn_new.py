import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2 as cv

##loading dataset
x=[]
y=[]
image_count=0
t=0
import os
path=r'C:/Users/HP/Downloads/Doctors Prescriptions Final Edit/final/Final Edit/Algo/data'
for (root,dirs,files) in os.walk(path):
    if files !=[]:
        l=len(files)
        print(image_count)
        for i in range(0,l): 
            t=t+1
            path=os.path.join(root,files[i])
            print(path)
            y.append(image_count)
            full_size_image = cv.imread(path,0)
            x.append(cv.resize(full_size_image, (28,28), interpolation=cv.INTER_CUBIC))
        image_count=image_count+1
x=np.asarray(x)
x = x.reshape(t,28,28,1).astype('float32')

y=np.asarray(y).astype('uint8')


from sklearn.model_selection import train_test_split
print("x",x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
l=xtrain.shape
s=[]
for i in range(10):
    s.append(random.randrange(0,l[0]))



m=1
for images in(s):
##    print(i)

   
    image = np.asarray(xtrain[images]).squeeze()
##    print(l)
    plt.subplot(2,5,m)
    plt.imshow(image)
    plt.title(ytrain[images])
    
    m=m+1

plt.show()
plt.close() 

xtrain=xtrain / 255
xtest=xtest / 255

# one hot encode outputs
from keras.utils import np_utils
ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)
num_classes = ytest.shape[1]

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
def baseline_model():
	# create model

    model = Sequential()
    model.add(Conv2D(32, (5, 5),input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=20, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(xtest, ytest, verbose=2)


    
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


print('acc:',scores[1]*100)
