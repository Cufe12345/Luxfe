import cv2
import numpy as np
import tensorflow as tf
import imutils
from imutils.contours import sort_contours
import math
import os

directory= os.path.dirname(__file__)

#stores the labels for the letter ai model ie if model outputs 0 then it is A etc
labelsL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#stores the labels for the number ai model ie if model outputs 0 then it is 0 etc
labelsN = ['0','1','2','3','4','5','6','7','8','9']

#labelsM = ['div','decimal','(',')','+','-','=','alpha','beta','cos','delta','gamma','geq','gt','infty','int','lambda','log','lt','mu','neq','pi','pm','rightarrow','sigma','sqrt','tan','theta','times','[',']','nothing','!','sin']

#stores the labels for the maths ai model ie if model outputs 0 then it is - etc
labelsM = ['-','+','.','/','(',')','!','theta']

#stores the labels for the extra ai model ie if model outputs 0 then it is _alpha
labelsE = ['_alpha','_beta','_delta','_Delta1','_epsilon','_exists','_gamma','_infty','_int','_lambda','_mu','_omega','_Omega1','_pi','_sim','_sqrt{ }','_theta','_sigma','_Sigma1','_neq','_1','_','+']


#stores manual probabilities meaning if the model outputs a percentage for that specific symbol above the probability in the dict then it is that symbol no matter how high the other models outputs are
probabilities = dict()
probabilities['_sqrt{ }'] = 0.85
probabilities['_exists'] = 0.972
probabilities['_theta'] = 0.85
probabilities['_infty'] = 0.95
probabilities['_pi'] = 0.90
probabilities['_int'] = 0.96
probabilities['_epsilon'] = 0.97
probabilities['+'] = [0.99,0.999,'T']





class Image:
    ''' class to store the extracted letters/symbols from the image'''
    # the image itself ie the 2d array of pixels with the same dimensions as the original image
    image = None

    # the start and end of the row and column of the image part of the image which is the actual letter
    # ie if you extract from image[rowStart:rowEnd, colStart:colEnd] you get the letter itself
    rowStart = 0
    rowEnd = 0
    colStart = 0
    colEnd = 0

    # the row and column the letter is in relative to the other letters
    rowNumber = None
    columnNumber = None

    # the number of pixels in the image
    pixelCount = None

    # the index of the image in the array of all images
    index = None

    # the character the image is predicted to be
    character = None


def checkSimilarity(image1, image2):
    '''Checks if the two images are similar based on the number of pixels that are different'''
    if image2 is None:
        return 1
    count = 0
    difference = 0
    for i in range(0, image1.shape[0]):
        for j in range(0, image1.shape[1]):
            count += 1
            if (image1[i][j] != image2[i][j]):
                difference += 1

    return difference/count



def sort(imgArray):
    '''imgArray is the array of each individual symbol extracted from the image \n
    uses the ai model to work out what all the symbols are then sorts them into rows and columns'''

    #stores each symbol extracted from the image and its properties
    allImages = []
    for img in imgArray:
        shape = img.shape
        rowStart = shape[0]
        rowEnd = 0
        colStart = shape[1]
        colEnd = 0
        pixelCount = 0

        # find the start and end of the row and column of the image ie remove all the black space around the image (background is black white is the writing)
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                pixel = img[i][j]
                if pixel != 0:
                    pixelCount += 1
                    if (colStart > j):
                        colStart = j
                    if (colEnd < j):
                        colEnd = j
                    if (rowStart > i):
                        rowStart = i
                    if (rowEnd < i):
                        rowEnd = i
        
        #include some of the black space as improves accuracy in ai model
        if (rowStart == rowEnd):
            rowEnd = rowEnd + 1
        if (colStart == colEnd):
            colEnd = colEnd + 1
        if (rowStart >= 5):
            rowStart -= 5
        if (colStart >= 5):
            colStart -= 5
        if (rowEnd + 5 < shape[0]):
            rowEnd += 5
        if (colEnd + 5 < shape[1]):
            colEnd += 5
        tempImg = img[rowStart:rowEnd, colStart:colEnd]
        before = tempImg

        # resize img to 50x50 as that is what the ai model was trained on
        tempImg = cv2.resize(tempImg, (50, 50))
        before2 = tempImg

        #normalise the image
        tempImg = np.array(tempImg) / 255

        #reshape the image to be 4d array as that is what the ai model was trained on
        tempImg = tempImg.reshape(-1, 50, 50, 1)

        #pass the image into the ai model's
        predictionLetter = loadedLetter_model.predict(tempImg)
        predictionNumber = loadedNumber_model.predict(tempImg)
        predictionMaths = loadedMaths_model.predict(tempImg)
        predictionExtra = loadedExtra_model.predict(tempImg)
        highestIndex = [-1,-1,-1,-1]
        highestValue = [0,0,0,0]
        
        #find the highest value and index of the highest value for each model
        for i in range(0, len(predictionLetter[0])):
            if (predictionLetter[0][i] > highestValue[0]):
                highestValue[0] = predictionLetter[0][i]
                highestIndex[0] = i
        for i in range(0, len(predictionNumber[0])):
            if (predictionNumber[0][i] > highestValue[1]):
                highestValue[1] = predictionNumber[0][i]
                highestIndex[1] = i
        for i in range(0, len(predictionMaths[0])):
            if (predictionMaths[0][i] > highestValue[2]):
                highestValue[2] = predictionMaths[0][i]
                highestIndex[2] = i
        for i in range(0, len(predictionExtra[0])):
            if (predictionExtra[0][i] > highestValue[3]):
                highestValue[3] = predictionExtra[0][i]
                highestIndex[3] = i
        highestTotal = -1
        finalPrediction = -1
        finalLabel = -1
        print("Potential Letter: "+ labelsL[highestIndex[0]] + " with a confidence of " + str(predictionLetter[0][highestIndex[0]]))
        print("Potential Number: "+ labelsN[highestIndex[1]] + " with a confidence of " + str(predictionNumber[0][highestIndex[1]]))
        print("Potential Maths: "+ labelsM[highestIndex[2]] + " with a confidence of " + str(predictionMaths[0][highestIndex[2]]))
        print("Potential Extra: "+ labelsE[highestIndex[3]] + " with a confidence of " + str(predictionExtra[0][highestIndex[3]]))
        # if(highestValue[1] > 0.99):
        #     highestTotal = 1
        #     finalPrediction = predictionNumber
        #     finalLabel = labelsN[highestIndex[highestTotal]]

        #run through the probabilities dict and if the highest value is above the probability then it is that symbol no matter what the other models output
        if(labelsM[highestIndex[2]] in probabilities):
            print("Highest Value: "+str(highestValue[2]))
            print("Highest Label: "+str(probabilities[labelsM[highestIndex[2]]][0]))
            print("Highest Label: "+str(probabilities[labelsM[highestIndex[2]]][1]))
            print("Highest Label: "+str(probabilities[labelsM[highestIndex[2]]][2]))
            print("1. ", highestValue[0])
            print("3.",labelsL[highestIndex[0]])


            if(highestValue[2] >= probabilities[labelsM[highestIndex[2]]][0] and highestValue[0] >= probabilities[labelsM[highestIndex[2]]][1] and labelsL[highestIndex[0]] == probabilities[labelsM[highestIndex[2]]][2] ):
                highestValue[2] = 1

        #the weighting of the maths model is significantly higher ie for nearly all cases it will be 0.99 or higher it only goes about 0.999 if it is a maths symbol
        if(highestValue[2] < 0.999):
            highestValue[2] = 0

         #run through the probabilities dict and if the highest value is above the probability then it is that symbol no matter what the other models output
        if(labelsE[highestIndex[3]] in probabilities):
            if(highestValue[3] >= probabilities[labelsE[highestIndex[3]]]):
                highestValue[3] = 1
        
        #works out which model has the highest likeliness and sets the final prediction and label to that
        if(highestValue[0] > highestValue[1] and highestValue[0] > highestValue[2] and highestValue[0] > highestValue[3]):
            highestTotal = 0
            finalPrediction = predictionLetter
            finalLabel = labelsL[highestIndex[highestTotal]]
        elif(highestValue[1] >= highestValue[0] and highestValue[1] >= highestValue[2] and highestValue[1] >= highestValue[3]):
            highestTotal = 1
            finalPrediction = predictionNumber
            finalLabel = labelsN[highestIndex[highestTotal]]
        elif(highestValue[2] >= highestValue[0] and highestValue[2] >= highestValue[1] and highestValue[2] >= highestValue[3]):
            highestTotal = 2
            finalPrediction = predictionMaths
            finalLabel = labelsM[highestIndex[highestTotal]]
        elif(highestValue[3] >= highestValue[0] and highestValue[3] >= highestValue[1] and highestValue[3] >= highestValue[2]):
            highestTotal = 3
            finalPrediction = predictionExtra
            finalLabel = labelsE[highestIndex[highestTotal]]
        # cv2.imshow(str(finalLabel), before)
        # cv2.waitKey(0)

        #if the highest value is above 0.5 then it is that symbol otherwise ignore it
        if(highestValue[highestTotal] > 0.5):
            print("IT is a " + finalLabel +
                      " with a confidence of " + str(finalPrediction[0][highestIndex[highestTotal]]))
            
            #create a new image object and store all the properties of the image in it then add it to the array of all images
            newImage = Image()
            newImage.image = img
            newImage.rowStart = rowStart
            newImage.rowEnd = rowEnd
            newImage.colStart = colStart
            newImage.colEnd = colEnd
            newImage.pixelCount = pixelCount
            newImage.character = finalLabel
            newImage.index = len(allImages)
            allImages.append(newImage)
        else:
            print("IT is NOT a " + str(highestIndex[highestTotal]) +
                      " with a confidence of " + str(finalPrediction[0][highestIndex[highestTotal]]))
            #cv2.imshow(str(labels[highestIndex]), before)
            #cv2.waitKey(0)
        
    #sort the images into rows
    class rows:
        rowStart = None
        rowEnd = None
        rowNumber = None
    allRows = []
    initialRow = rows()
    initialRow.rowStart = allImages[0].rowStart
    initialRow.rowEnd = allImages[0].rowEnd
    initialRow.rowNumber = 0
    allRows.append(initialRow)
    index = 0
    for image in allImages:
        for row in allRows:

            # the image's start is less than the row's start and the image's end is greater than the row's end meaning
            # the images contains the rows olds start and end meaning it is part of the row and the row's new start and end
            if (image.rowStart < row.rowStart and image.rowEnd > row.rowEnd):
                image.rowNumber = row.rowNumber
                allRows[row.rowNumber].rowStart = image.rowStart
                allRows[row.rowNumber].rowEnd = image.rowEnd
                break

            # the image's start is less than the row's start and the image's end is less than the row's end meaning
            # the image is inside the rows dimensions so is a part of it
            if (image.rowStart > row.rowStart and image.rowEnd < row.rowEnd):
                image.rowNumber = row.rowNumber
                break

            # count how many pixels are within the rows range
            count = 0
            for i in range(row.rowStart, row.rowEnd):
                for j in range(image.colStart, image.colEnd):
                    if (image.image[i][j] != 0):
                        count += 1

            # if more than half the pixels are within the row's range then the image is part of the row
            if (count/image.pixelCount > 0.5):
                image.rowNumber = row.rowNumber
        
        # if the image doesnt fit into any current row's then create a new row and add it to the array of all rows
        if (image.rowNumber is None):

            newRow = rows()
            newRow.rowStart = image.rowStart
            newRow.rowEnd = image.rowEnd
            newRow.rowNumber = len(allRows)
            allRows.append(newRow)
            image.rowNumber = newRow.rowNumber
        allImages[index] = image
        index += 1
    print(len(allRows))

    #sort the images into columns
    for row in allRows:
        allImagesInRow = []

        #assign each image a index value for later use
        index = 0
        for img in allImages:
            allImages[index].index = index
            index += 1
            #add all images in the row to the array
            if (img.rowNumber == row.rowNumber):
                allImagesInRow.append(img)

        #sort the images in the row by their start column
        allImagesInRow.sort(key=lambda x: x.colStart)
        print("Length Before: "+str(len(allImagesInRow)))
        removed = []
        #remove duplicates based on similarity
        for i in range(0, len(allImagesInRow)-1):
            if (i >= len(allImagesInRow)-1):
                break
            sim = checkSimilarity(
                allImagesInRow[i].image, allImagesInRow[i+1].image)
            print("sim: "+str(sim))
            if (sim < 0.00009):
                removed.append(allImagesInRow[i+1])
                allImagesInRow.pop(i+1)
                print("removed")
        print("Length After: "+str(len(allImagesInRow)))

        #assign each image a column number
        for i in range(0, len(allImagesInRow)):
            allImagesInRow[i].columnNumber = i
            allImages[allImagesInRow[i].index] = allImagesInRow[i]
        for i in range(0, len(removed)):
            allImages.remove(removed[i])

    return allImages

#Load Letter Model
json_file = open(
    directory+'/Model/Model/LetterModel.json', 'r')
letterModel = json_file.read()
json_file.close()
loadedLetter_model = tf.keras.models.model_from_json(letterModel)
loadedLetter_model.load_weights(
    directory+"/Model/Model/LetterModel.h5")

#Load Number Model
json_file = open(
    directory+'/Model/Model/NumberModel.json', 'r')
numberModel = json_file.read()
json_file.close()
loadedNumber_model = tf.keras.models.model_from_json(numberModel)
loadedNumber_model.load_weights(
    directory+"/Model/Model/NumberModel.h5")

#Load Maths symbol Model
json_file = open(
    directory+'/Model/Model/MathsModel.json', 'r')
mathsModel = json_file.read()
json_file.close()
loadedMaths_model = tf.keras.models.model_from_json(mathsModel)
loadedMaths_model.load_weights(
    directory+"/Model/Model/MathsModel.h5")


#Load Extra Model
json_file = open(
    directory+'/Model/Model/ExtraModel.json', 'r')
extraModel = json_file.read()
json_file.close()
loadedExtra_model = tf.keras.models.model_from_json(extraModel)
loadedExtra_model.load_weights(
    directory+"/Model/Model/ExtraModel.h5")

#read in the image as greyscale
img = cv2.imread(directory+'/Testing_Images/Formula6.jpg', 0)


#invert the image as the writing is white and the background is black
img = (255-img)
#img = cv2.GaussianBlur(img, (5, 5), 0)
#img = cv2.bitwise_not(img)
vis = img.copy()
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(img)
# print(regions)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', vis)
cv2.waitKey(0)
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)


imgArray = []
high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lowThresh = 0.5*high_thresh

#detect edges in the image
edged = cv2.Canny(img, lowThresh, high_thresh)

#find the contours in the image from the edges
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort the contours from left to right
cnts = sort_contours(cnts, method="left-to-right")[0]

#loop through each contour and draw it onto the mask then add the image with the mask applied to the array of images
for contour in cnts:
    tempImg = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    cv2.drawContours(tempImg, [contour], -1, (255, 255, 255), -1)
    imgArray.append(cv2.bitwise_and(img, img, mask=tempImg))
    # cv2.imshow('img', cv2.bitwise_and(img, img, mask=tempImg))
    # cv2.waitKey(0)

previous = None
tempImgArray = []
# does removes some duplicates based on similarity if they are next to each other in the array (not all) are removed
# as some are duplicates but not next to each other
for img2 in imgArray:
    if (checkSimilarity(img2, previous) < 0.0009):
        previous = img2
        continue
    tempImgArray.append(img2)
    previous = img2
imgArray = tempImgArray


finalImgArray = sort(imgArray)

#create a dict of all the rows and their images
arrayRows = dict()
for img2 in finalImgArray:
    if (not (img2.rowNumber in arrayRows)):
        arrayRows[img2.rowNumber] = [img2]
    else:
        arrayRows[img2.rowNumber].append(img2)

#sort the rows into order and add spaces between symbols where they should be ie how they are written
arrayRows2 = []
for row in arrayRows:
    arrayRows[row].sort(key=lambda x: x.colStart)
    total = 0
    previous = arrayRows[row][0].colEnd
    first = True

    #calculate all the spaces between the symbols in the row and add each space value to an array
    values = []
    for img2 in arrayRows[row]:
        if (first):
            first = False
            continue

        space = img2.colStart - previous
        total += space
        previous = img2.colEnd
        values.append(space)
    
    #if no spaces then continue
    if(len(values) == 0):
        continue

    #calculate the average space between the symbols
    average = total/(len(values))
    originalValues = values.copy()

    #calculate the standard deviation of the spaces
    for i in range(0, len(values)):
        values[i] = values[i] - average
    final = 0
    for i in range(0, len(values)):
        final += values[i]*values[i]

    standardDeviation = math.sqrt(final/len(values))

    #the threshold for a space to be considered a space ie anything above this value is a space
    value = average + (1*standardDeviation)
    print("mean: "+str(average))
    print("standard deviation: "+str(standardDeviation))
    print("value: "+str(value))

    #add spaces between the symbols where they should be
    for i in range(0, len(arrayRows[row])):

        if (i == 0):
            continue
        print("value["+str(i-1)+"]: "+str(originalValues[i-1]))
        if (originalValues[i-1] > value):
            print("space")
            spaceImage = Image()
            spaceImage.image = None
            spaceImage.character = ' '
            arrayRows[row].insert(i, spaceImage)

    arrayRows2.append(arrayRows[row])
arrayRows2.sort(key=lambda x: x[0].rowStart)
characters = []

#this turns symbols into what they should be ie as = has space between them the edge detection will see them as two symbols so this turns them into one symbol ie -- will become = only if it should be
for row in arrayRows2:
    characters.append([])
    previous = None
    for img2 in row:
        #cv2.imshow("img", img2.image)
        # cv2.waitKey(0)
        current = img2.character
        if(previous != None):
            if(current == '-' and previous.character == '-'):
                count = 0
                #Do a check to see if both - are in same row
                for i in range(previous.rowStart,previous.rowEnd):
                    for j in range(img2.colStart,img2.colEnd):
                        if(previous.image[i][j] != 0):
                            count += 1
                print("count: "+str(count))
                print("previous pixel count: "+str(previous.pixelCount*0.5))
                if(count > 0.5*previous.pixelCount):
                    characters[len(characters)-1].pop()
                    characters[len(characters)-1].append('=')
                    previous = None
                    continue
        previous = img2
        characters[len(characters)-1].append(img2.character)
print(characters)

#turn the 2d array of symbols into a string
finalString = ""
for row in characters:
    for character in row:
        finalString += character
    finalString += '\n'
print(finalString)

#replace some of the symbols with their latex equivalent
finalString = finalString.replace('div', '\div')
finalString = finalString.replace('_alpha', "\\alpha")
finalString = finalString.replace('_beta', '\\beta')
finalString = finalString.replace('_gamma', '\gamma')
finalString = finalString.replace('_delta', '\delta')
finalString = finalString.replace('_theta', '\\theta')
finalString = finalString.replace('_lambda', '\lambda')
finalString = finalString.replace('_mu', '\mu')
finalString = finalString.replace('_pi', '\pi')
finalString = finalString.replace('_sigma', '\sigma')
finalString = finalString.replace('_sqrt', '\sqrt')
finalString = finalString.replace('_tan', '\\tan')
finalString = finalString.replace('_times', '\\times')
finalString = finalString.replace('_infty', '\infty')
finalString = finalString.replace('_int', '\int')
finalString = finalString.replace('_log', '\log')
finalString = finalString.replace('pm', '\pm')
finalString = finalString.replace('rightarrow', '\\rightarrow')
finalString = finalString.replace('geq', '\geq')
finalString = finalString.replace('gt', '\gt')
finalString = finalString.replace('lt', '\lt')
finalString = finalString.replace('neq', '\\neq')
finalString = finalString.replace('nothing', '')
finalString = finalString.replace('sin', '\sin')
finalString = finalString.replace('cos', '\cos')

print(finalString)


cv2.destroyAllWindows()
