import numpy as np
#for arange, shuffling data and storing labels in array format

import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model
#for all the layers of the neural network ie., pooling, dense, flatten, dropout, Model for feature selection
#converts data into categorical form of n number of classes (two classes in our case being uninfected and infected) 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#importing KNN and SVM models 

import cv2
from PIL import Image
import os
#for loading the data and data augmentation 

import matplotlib.pyplot as plt
import seaborn as sns
#for plotting the dataset and graphs

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
#accuracy score, splittig data into train and test 

print(os.listdir("cell_images/cell_images"))
#print the directory names

infected = os.listdir('cell_images/cell_images/Parasitized/') 
uninfected = os.listdir('cell_images/cell_images/Uninfected/')
#storing the path in these respective variables

data = []
labels = []

for i in infected:
    try:
    
        image = cv2.imread("cell_images/cell_images/Parasitized/"+i)
        #reading the images iteratively
        image_array = Image.fromarray(image , 'RGB')
        #converting the images to pixel values in the RGB scale
        resize_img = image_array.resize((50 , 50))
        #resizing the image into 50 x 50 pixels evenly 
        rotated45 = resize_img.rotate(45)
        #augmenting the data by rotating by 45 degrees
        rotated75 = resize_img.rotate(75)
        #augmenting the data by rotating by 75 degrees
        blur = cv2.blur(np.array(resize_img) ,(10,10))
        #agumenting the data by blurring the infected images to avoid overfitting and also cases where small blemishes might be anomalies
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        data.append(np.array(blur))
        #append all the augmented data to the data array 
        labels.append(1)
        labels.append(1)
        labels.append(1)
        labels.append(1)
        #append the labels as 1 for all infected images

    except AttributeError:
        print('')
    #exception handeling
    
for u in uninfected:
    try:
        
        image = cv2.imread("cell_images/cell_images/Uninfected/"+u)
        #reading the images iteratively
        image_array = Image.fromarray(image , 'RGB')
        #converting the images to pixel values in the RGB scale
        resize_img = image_array.resize((50 , 50))
        #resizing the image into 50 x 50 pixels evenly 
        rotated45 = resize_img.rotate(45)
        #augmenting the data by rotating by 45 degrees
        rotated75 = resize_img.rotate(75)
        #augmenting the data by rotating by 75 degrees
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        #append all the augmented data to the data array 
        #blurring is not done for uninfected images as they do not have a spot in the cell images
        labels.append(0)
        labels.append(0)
        labels.append(0)
        #append the labels as 0 for all infected images
        
    except AttributeError:
        print('')
    #exception handeling

cells = np.array(data)
labels = np.array(labels)
#saving the data array in the following variables as corresponding image and label

np.save('Cells' , cells)
np.save('Labels' , labels)
#locally saves the Cells and Labels as a numpy file

print('Cells : {} | labels : {}'.format(cells.shape , labels.shape))
#shape of cells and labels

plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(28):
    n += 1 
    r = np.random.randint(0 , cells.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(cells[r[0]])
    plt.title('{} : {}'.format('Infected' if labels[r[0]] == 1 else 'Unifected' ,
                               labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()
#plotting the infected and uninfected cell images data 

plt.figure(1, figsize = (10 , 4))
plt.subplot(1 , 2 , 1)
plt.imshow(cells[0])
plt.title('Infected Cell')
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(cells[60000])
plt.title('Uninfected Cell')
plt.xticks([]) , plt.yticks([])

plt.show()
#plots a single infected and uninfected cell image for our understanding

cells = cells.astype('float32') / 255
#min-max normalisation of our dataset

(train_x, test_x, train_y, test_y) = train_test_split(cells, labels, test_size=0.2, stratify=labels)
#split the data into training and testing

train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)
#converts data into categorical form of n number of classes (two classes in our case being uninfected and infected)

model_svm = Sequential()
#sequential model is a plain stack of layers where each layer has one input and output tensor (tensor = vector/matrix ie., a form of data)
model_svm.add(Conv2D(16,(5,5),padding='valid',activation="relu",input_shape = (50, 50, 3)))
#we state the input as 50,50,3 as 50,50 are the pixel size of each image and 3 being the RGB value associated with each image. 
#convolutional layer has 16 filters each of size 5,5
#padding is valid means zero padding
#activation function is rectified linear unit 
model_svm.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
#maxpooling chooses the largest of the 4 values obtained post convolutional layer
model_svm.add(Dropout(0.4))
#at random, 40% of the neurons are dropped to avoid overfitting
model_svm.add(Conv2D(32,(5,5),padding='valid',activation="relu"))
#convolutional layer has 32 filters each of size 5,5. as the dimensionality reduces, the relevance of the features increase. in order to accomodate the relevant features, we increase the number of filters
#padding is valid means zero padding
#activation function is rectified linear unit 
model_svm.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
#maxpooling chooses the largest of the 4 values obtained post convolutional layer
model_svm.add(Dropout(0.6))
#60% neurons dropoped
model_svm.add(Conv2D(64,(5,5),padding='valid',activation="relu"))
model_svm.add(Dropout(0.8))
model_svm.add(Flatten())
#flatten all the different inputs into a single dimension
model_svm.add(Dense(2, activation="softmax"))
#dense is a layer filled with neurons parallely
#softmax function ensures that the data recieved is onverted to probabilities that add to 1. 
#example: 0.1, 0.3 = ((0.1)/(0.1+0.3)), ((0.3)/(0.1+0.3)) 

model_feat = Model(inputs=model_svm.input,outputs=model_svm.get_layer(index=-1).output)
#specifynig input and output layer for model to perform feature selection
feat_train = model_feat.predict(train_x)
#stores the predicted values

svm = SVC(kernel='rbf')
#SVM has linear, polynomial, gaussian and rbf kernels where rbf performs the best as we have a complicated dataset
svm.fit(feat_train,np.argmax(train_y,axis=1))
#fit the model with train_x and train_y 

feat_test = model_feat.predict(test_x)
y_pred = feat_test
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]
#predicted values are obtained and stored in the suitable variable

from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
print('{} \n{} \n{}'.format(confusion_matrix(true , predictions) , 
                           classification_report(true , predictions) , 
                           accuracy_score(true , predictions)))
#gives the confusion matrix, classification report and accuracy score

model_knn = Sequential()
model_knn.add(Conv2D(16,(5,5),padding='valid',activation="relu",input_shape = (50, 50, 3)))
model_knn.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model_knn.add(Dropout(0.4))
model_knn.add(Conv2D(32,(5,5),padding='valid',activation="relu"))
model_knn.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model_knn.add(Dropout(0.6))
model_knn.add(Conv2D(64,(5,5),padding='valid',activation="relu"))
model_knn.add(Dropout(0.8))
model_knn.add(Flatten())
model_knn.add(Dense(2, activation="softmax"))

model_feat = Model(inputs=model_knn.input,outputs=model_knn.get_layer(index=-1).output)
feat_train = model_feat.predict(train_x)

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(feat_train,np.argmax(train_y,axis=-1))
#we use knn for image processing to test if knn model can learn and predict infected and uninfected cells correctly

feat_test = model_feat.predict(test_x)
y_pred = feat_test
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]

from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
print('{} \n{} \n{}'.format(confusion_matrix(true , predictions) , 
                           classification_report(true , predictions) , 
                           accuracy_score(true , predictions)))

model_ann = Sequential()
model_ann.add(Dense(16, input_shape=(50, 50, 3), activation='relu'))
model_ann.add(Dropout(0.4))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dropout(0.6))
model_ann.add(Dense(64, activation='relu'))
model_ann.add(Dropout(0.6))
model_ann.add(Flatten())
model_ann.add(Dense(2, activation='softmax'))

model_ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model = model_ann.fit(train_x, train_y, validation_split=0.2, batch_size=128, epochs=4, verbose=0)
#ANN is a sequence of dense layers packed with neurons that run for 4 epochs which can give us promising results

y_pred = model_ann.predict(test_x)
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]

from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
print('{} \n{} \n{}'.format(confusion_matrix(true , predictions) , 
                           classification_report(true , predictions) , 
                           accuracy_score(true , predictions)))

model_cnn = Sequential()
model_cnn.add(Conv2D(filters=50, kernel_size=3, padding="same", activation="relu", input_shape=(50, 50, 3)))
model_cnn.add(MaxPooling2D(pool_size=2))
model_cnn.add(Conv2D(filters=100, kernel_size=3, padding="same", activation="relu"))
model_cnn.add(MaxPooling2D(pool_size=2))
model_cnn.add(Conv2D(filters=250, kernel_size=3, padding="same", activation="relu"))
model_cnn.add(MaxPooling2D(pool_size=2))
model_cnn.add(Dropout(0.2))
model_cnn.add(Flatten())
model_cnn.add(Dense(250, activation="relu"))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(2, activation="softmax"))


model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model_cnn = model_cnn.fit(train_x, train_y, validation_split=0.2, batch_size=128, epochs=3, verbose=0)
#convolutional layers are used to treat the input features not just as independant results but also consider their context

y_pred = model_cnn.predict(test_x)
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]

from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
print('{} \n{} \n{}'.format(confusion_matrix(true , predictions) , 
                           classification_report(true , predictions) , 
                           accuracy_score(true , predictions)))

model_cnn1 = Sequential()
model_cnn1.add(Conv2D(filters=50, kernel_size=5, padding="same", activation="relu", input_shape=(50, 50, 3)))
model_cnn1.add(MaxPooling2D(pool_size=2))
model_cnn1.add(Conv2D(filters=100, kernel_size=5, padding="same", activation="relu"))
model_cnn1.add(MaxPooling2D(pool_size=2))
model_cnn1.add(Conv2D(filters=250, kernel_size=5, padding="same", activation="relu"))
model_cnn1.add(MaxPooling2D(pool_size=2))
model_cnn1.add(Dropout(0.2))
model_cnn1.add(Flatten())
model_cnn1.add(Dense(500, activation="relu"))
model_cnn1.add(Dropout(0.2))
model_cnn1.add(Dense(2, activation="softmax")) 

model_cnn1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model_cnn1 = model_cnn1.fit(train_x, train_y, validation_split=0.2, batch_size=128, epochs=3, verbose=0)

y_pred = model_cnn1.predict(test_x)
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
print('{} \n{} \n{}'.format(confusion_matrix(true , predictions) , 
                           classification_report(true , predictions) , 
                           accuracy_score(true , predictions)))

model_cnn2 = Sequential()
model_cnn2.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=(50, 50, 3)))
model_cnn2.add(MaxPooling2D(pool_size=2))
model_cnn2.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model_cnn2.add(MaxPooling2D(pool_size=2))
model_cnn2.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
model_cnn2.add(MaxPooling2D(pool_size=2))
model_cnn2.add(Dropout(0.2))
model_cnn2.add(Flatten())
model_cnn2.add(Dense(500, activation="relu"))
model_cnn2.add(Dropout(0.2))
model_cnn2.add(Dense(2, activation="softmax"))  

model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model_cnn = model_cnn.fit(train_x, train_y, validation_split=0.2, batch_size=128, epochs=3, verbose=0)

# Initially we took the filter size as 3 X 3 and the number of filters in each of the layers as 50, 100 and 250. The padding was initialized as 'same', following each convolutional layer we had a max pooling layer of size 2 x 2. After a set of three convolutional and pooling layers, we had a dropout of 0.2 and a dense layer with 250 neurons. Then we had a flattened input to our final dense layer of 2 neurons corresponding to our two classification labels.

# In an attempt to potentially increase the accuracy, we increased the filter size to 5 X 5 to capture the essence of the pixel values on a moderately larger scale. We also increased the number of neurons in our first dense layer to 500 to increase the combinational computations. 

# We find that increasing the kernel size ultimately only greatly increases the computational time and therefore we revert to 3 x 3.

# Having reverted to 3 x 3, we also increase the number of filters in our convolutional layers to 64, 128 and 256. As we sequentially progress through the convolutional layers, the intermediate outputs and subsequent inputs will drop in dimensionality but increase in relevance. This is why we increase the number of filters to capture the importance of these features which evidently gives us improved results.

y_pred = model_cnn.predict(test_x)
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
print('{} \n{} \n{}'.format(confusion_matrix(true , predictions) , 
                           classification_report(true , predictions) , 
                           accuracy_score(true , predictions)))

plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(28):
    n += 1 
    r = np.random.randint( 0  , test_x.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(test_x[r[0]])
    plt.title('true {} : pred {}'.format(true[r[0]] , predictions[r[0]]) )
    plt.xticks([]) , plt.yticks([])

plt.show()
#the predicted versus actual values are listed out along with cell images based on the predictions of the optimised cnn model