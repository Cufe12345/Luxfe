import matplotlib.pyplot as plt
# import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np
import sys


directory= os.path.dirname(__file__)

#To draw: we need more / images, couldnt recognise dots
#roots and factorials wont work due to the way they are drawn ie not a continous line maybe with factorial do something like i did with the = sign

#Labels for the letters
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#stores the labels for all other symbols ie not letters
labels2 = dict()

#path to the Letter data
path = directory+'/TrainData/Letters/images/images'
#path to the Math data
path2 = directory+'/TrainData/MathSymbols'
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))

def fetchLetter():
    '''Fetches the letter images train and test data from the folders and does some preprocessing on them.\n
    if you want more detail on the preprocessing look at test.py'''
    trainData = []
    testData = []
    for label in labels:
        folderPath = path+'/'+label
        count = 0
        total = 0
        index = 0
        # convert label to number
        labelFinal = labels.index(label)
        for img in os.listdir(folderPath):
            try:
                #Prints out the progress of the data fetching every 100 images
                total += 1
                index += 1
                if(index >= 100):
                    print(str(total)+" : "+str(len(os.listdir(folderPath))))
                    index = 0
                imgArr = cv2.imread(os.path.join(
                    folderPath, img), cv2.IMREAD_GRAYSCALE)
                shape = imgArr.shape
                rowStart = shape[0]
                rowEnd = 0
                colStart = shape[1]
                colEnd = 0
                for i in range(0, shape[0]):
                    for j in range(0, shape[1]):
                        pixel = imgArr[i][j]
                        if pixel != 0:
                            if (colStart > j):
                                colStart = j
                            if (colEnd < j):
                                colEnd = j
                            if (rowStart > i):
                                rowStart = i
                            if (rowEnd < i):
                                rowEnd = i
                #print(rowStart, rowEnd, colStart, colEnd)
                if (rowStart == rowEnd):
                    rowEnd = rowEnd + 1
                if (colStart == colEnd):
                    colEnd = colEnd + 1
                
                imgArr = imgArr[rowStart:rowEnd, colStart:colEnd]
                # cv2.imshow('image', imgArr)
                # cv2.waitKey(0)
                resizedImg = cv2.resize(imgArr, (50, 50))
                if count > 7:
                    testData.append([resizedImg, labelFinal])
                    count = 0
                else:
                    trainData.append([resizedImg, labelFinal])
                    count += 1
            except Exception as e:
                print(e)
    return np.array([trainData, testData], dtype=object)

def fetchMath():
    '''Fetches the Math images train and test data from the folders and does some preprocessing on them\n
    if you want more detail on the preprocessing look at test.py'''
    trainData = []
    testData = []
    newPath = path2+'/newSymbols' 
    folders = ['-','+','decimal','div','(',')','!','theta']
    index = 0
    for k in range(0, len(folders)):
        indexTrain = 0
        for img in os.listdir(newPath+'/'+folders[k]):
            try:
                labelName = folders[k]
                if(str(labelName) not in labels2):
                    labels2[str(labelName)] = index
                    index += 1
                labelFinal = labels2[str(labelName)]
                imgArr = cv2.imread(os.path.join(
                    newPath+'/'+folders[k], img), cv2.IMREAD_GRAYSCALE)
                imgArr = 255 - imgArr

                shape = imgArr.shape
                rowStart = shape[0]
                rowEnd = 0
                colStart = shape[1]
                colEnd = 0
                for i in range(0, shape[0]):
                    for j in range(0, shape[1]):
                        pixel = imgArr[i][j]
                        if pixel != 0:
                            if (colStart > j):
                                colStart = j
                            if (colEnd < j):
                                colEnd = j
                            if (rowStart > i):
                                rowStart = i
                            if (rowEnd < i):
                                rowEnd = i
                #print(rowStart, rowEnd, colStart, colEnd)
                if (rowStart == rowEnd):
                    rowEnd = rowEnd + 1
                if (colStart == colEnd):
                    colEnd = colEnd + 1
                
                imgArr = imgArr[rowStart:rowEnd, colStart:colEnd]
                resizedImg = cv2.resize(imgArr, (50, 50))
                if indexTrain > 7:
                    testData.append([resizedImg, labelFinal])
                    indexTrain = 0
                else:
                    indexTrain += 1
                    trainData.append([resizedImg, labelFinal])
            except Exception as e:
                print("Error2: "+str(e))

    
    print(str(len(trainData))+"_"+str(len(testData)))
    return np.array([trainData, testData], dtype=object)


def fetchDigits():
    '''Fetches the digit images train and test data from the folders and does some preprocessing on them\n
    if you want more detail on the preprocessing look at test.py'''

    path = directory+"/TrainData/Numbers/trainingSet/"
    #the folders of the images that we want to fetch
    folders = ['0','1','2','3','4','5','6','7','8','9','1N','2N','3N','4N','5N','8N','9N','0N']
    test_data = []
    train_data = []
    index = 0
    indexLabel = 0
    for k in range(0, len(folders)):   
        for img in os.listdir(path+'/'+folders[k]):
            # print("i: "+str(k)+" foldersLen: "+str(len(folders)))
            label = folders[k]

            #remove the N from the label if it is there
            if(label[len(label)-1] == 'N'):
                label = str(label[0:len(label)-1])
            if(str(label) not in labels2):
                    labels2[str(label)] = indexLabel
                    indexLabel += 1
            label = labels2[str(label)]
            imgArr = cv2.imread(os.path.join(path+'/'+folders[k], img), cv2.IMREAD_GRAYSCALE)

            #invert the image if it above 9 as the images are inverted
            if(k >9):
                imgArr = 255 - imgArr
            shape = imgArr.shape
            rowStart = shape[0]
            rowEnd = 0
            colStart = shape[1]
            colEnd = 0
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    pixel = imgArr[i][j]
                    if pixel != 0:
                        if (colStart > j):
                            colStart = j
                        if (colEnd < j):
                            colEnd = j
                        if (rowStart > i):
                            rowStart = i
                        if (rowEnd < i):
                            rowEnd = i
                #print(rowStart, rowEnd, colStart, colEnd)
            if (rowStart == rowEnd):
                rowEnd = rowEnd + 1
            if (colStart == colEnd):
                colEnd = colEnd + 1
            imgArr = imgArr[rowStart:rowEnd, colStart:colEnd]
            resizedImg = cv2.resize(imgArr, (50, 50))

            #add the image to the test or train data depending on the index
            if(index > 9):        
                test_data.append([resizedImg, label])
                index = 0
            else:
                train_data.append([resizedImg, label])
                index += 1
    return np.array([train_data, test_data], dtype=object)


def fetchExtra():
    '''Fetches the extra images train and test data you want from the folders and does some preprocessing on them\n
    if you want more detail on the preprocessing look at test.py'''

    trainData = []
    testData = []
    newPath = directory
    trainPath = newPath+'/ExtraData'
    testPath = newPath+'/ExtraDataTest' 
    folders = ['_alpha','_beta','_delta','_Delta1','_epsilon','_exists','_gamma','_infty','_int','_lambda','_mu','_omega','_Omega1','_pi','_sim','_sqrt','_theta','_sigma','_Sigma1','_neq','_1','_','+']
    index = 0
    
    for k in range(0, len(folders)):
        for img in os.listdir(trainPath+'/'+folders[k]):
            try:
                labelName = folders[k]
                if(str(labelName) not in labels2):
                    labels2[str(labelName)] = index
                    index += 1
                labelFinal = labels2[str(labelName)]
                imgArr = cv2.imread(os.path.join(
                    trainPath+'/' +folders[k], img), cv2.IMREAD_GRAYSCALE)

                shape = imgArr.shape
                rowStart = shape[0]
                rowEnd = 0
                colStart = shape[1]
                colEnd = 0
                for i in range(0, shape[0]):
                    for j in range(0, shape[1]):
                        pixel = imgArr[i][j]
                        if pixel != 0:
                            if (colStart > j):
                                colStart = j
                            if (colEnd < j):
                                colEnd = j
                            if (rowStart > i):
                                rowStart = i
                            if (rowEnd < i):
                                rowEnd = i
                #print(rowStart, rowEnd, colStart, colEnd)
                if (rowStart == rowEnd):
                    rowEnd = rowEnd + 1
                if (colStart == colEnd):
                    colEnd = colEnd + 1
                
                imgArr = imgArr[rowStart:rowEnd, colStart:colEnd]
                resizedImg = cv2.resize(imgArr, (50, 50))
                
                trainData.append([resizedImg, labelFinal])
            except Exception as e:
                print("Error2: "+str(e))
        for img in os.listdir(testPath+'/'+folders[k]):
            try:
                labelName = folders[k]
                if(str(labelName) not in labels2):
                    labels2[str(labelName)] = index
                    index += 1
                labelFinal = labels2[str(labelName)]
                imgArr = cv2.imread(os.path.join(
                    testPath+'/'+folders[k], img), cv2.IMREAD_GRAYSCALE)
                #imgArr = 255 - imgArr

                shape = imgArr.shape
                rowStart = shape[0]
                rowEnd = 0
                colStart = shape[1]
                colEnd = 0
                for i in range(0, shape[0]):
                    for j in range(0, shape[1]):
                        pixel = imgArr[i][j]
                        if pixel != 0:
                            if (colStart > j):
                                colStart = j
                            if (colEnd < j):
                                colEnd = j
                            if (rowStart > i):
                                rowStart = i
                            if (rowEnd < i):
                                rowEnd = i
                #print(rowStart, rowEnd, colStart, colEnd)
                if (rowStart == rowEnd):
                    rowEnd = rowEnd + 1
                if (colStart == colEnd):
                    colEnd = colEnd + 1
                
                imgArr = imgArr[rowStart:rowEnd, colStart:colEnd]
                resizedImg = cv2.resize(imgArr, (50, 50))
                
                testData.append([resizedImg, labelFinal])
            except Exception as e:
                print("Error2: "+str(e))

    
    print(str(len(trainData))+"_"+str(len(testData)))
    return np.array([trainData, testData], dtype=object)

#take arguments from command line
train_data = []
test_data = []
letter = False
modelName = "default"
epsilons = 30
if(len(sys.argv) > 1):
    if(sys.argv[1] == '-L'):
        print("Fetching letter data")
        letter = True
        data = fetchLetter()
        train_data = data[0]
        test_data = data[1]
    elif(sys.argv[1] == '-M'):
        print("Fetching math data")
        data2 = fetchMath()
        for i in range(0, len(data2[0])):
            train_data.append(data2[0][i])
        for i in range(0, len(data2[1])):
            test_data.append(data2[1][i])
    elif(sys.argv[1] == '-D'):
        print("Fetching digit data")
        data3 = fetchDigits()
        for i in range(0, len(data3[0])):
            train_data.append(data3[0][i])
        for i in range(0, len(data3[1])):
            test_data.append(data3[1][i])
    elif(sys.argv[1] == '-E'):
        print("Fetching extra data")
        data4 = fetchExtra()
        for i in range(0, len(data4[0])):
            train_data.append(data4[0][i])
        for i in range(0, len(data4[1])):
            test_data.append(data4[1][i])
    else:
        print("Invalid argument")
        exit()
    if(len(sys.argv) > 2):
        modelName = sys.argv[2]
    if(len(sys.argv) > 3):
        epsilons = int(sys.argv[3])
else:
    print("No arguments given")
    exit()

x_train = []
y_train = []
x_val = []
y_val = []
for key in labels:
    print(str(key)+" : "+str(labels.index(key)))
for key in labels2:
    print(str(key)+" : "+str(labels2[key]))
for feature, label in train_data:
    #print("Feature: "+str(feature.shape))
    x_train.append(feature)
    y_train.append(label)

for feature, label in test_data:
    # print("Feature: "+str(feature))
    #print("Label: "+str(type(label)))
    x_val.append(feature)
    y_val.append(label)

# normalise the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

# reshape the data
x_train = x_train.reshape(-1, 50, 50, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, 50, 50, 1)
y_val = np.array(y_val)

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=30,
    zoom_range=0.2,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# does some calculation like mean etc so its ready to augment data
datagen.fit(x_train)

# build the model
model = Sequential()

model.add(Conv2D(64, 3, padding='same',
          activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

if(letter):
    model.add(Dense((len(labels)), activation='softmax'))
else:
    model.add(Dense((len(labels2)), activation='softmax'))

model.summary()

# compile the model 0.00001
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epsilons,
                    validation_data=(x_val, y_val))

# save the model
model_json = model.to_json()
with open(path+"/"+modelName+".json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(path+"/"+modelName+".h5")
print("Saved model to disk")

if(letter):
    print("Num of output: "+str(len(labels)))
else:
    print("Num of output: "+str(len(labels2))) 

# plot the training and validation accuracy and loss at each epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epsilons)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
