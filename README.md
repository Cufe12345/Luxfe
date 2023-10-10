

## Website

The website is still in development will update the read me when its complete

## Backend
TrainData folder contains all the images which were used in the training of the ai models
Testing Images contains images to test the accuracy of the models after training
# Scripts
training.py contains the code used to train the ai models. training.py [1st argument the model you want to train] [2nd argument the name of output file] [3rd argument number of epsilons] 
eg: training.py -M TestModel 30

[options]
-M = Maths model
-D = Digits model
-L = Letters model
-E = Extra's model

[Extra info]
The training will try to use the GPU if its avaliable ie you have the right drivers installed and GPU is supported by tensorflow

test.py contains the code to take in an image and return the latex using the provided ai model




