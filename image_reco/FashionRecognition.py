import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras import layers
from keras.layers import Dropout
from matplotlib import pyplot
import cv2
import time
#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
def summarizeLearningCurvesPerformances(histories, accuracyScores):

    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='green', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='red', label='test')

        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='green', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='red', label='test')

        #print accuracy for each split
        print("Accuracy for set {} = {}".format(i, accuracyScores[i]))

    pyplot.show()

    print('Accuracy: mean = {:.3f} std = {:.3f}, n = {}'.format(np.mean(accuracyScores) * 100, np.std(accuracyScores) * 100, len(accuracyScores)))
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def prepareData(trainX, trainY, testX, testY):

    #reshape the data to be of size [samples][width][height][channels]
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))


    #normalize the input values
    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255


    #Transform the classes labels into a binary matrix
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)



    return trainX, trainY, testX, testY
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineModel(input_shape, num_classes):

    #Initialize 
    model = keras.models.Sequential()

    #first hidden layer as a convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))


    #pooling layer
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))


    #flatten layer
    model.add(layers.Flatten())


    #dense layer of size 16 
    model.add(layers.Dense(16, activation='relu', kernel_initializer='he_uniform'))

    #output layer
    model.add(layers.Dense(num_classes, activation='softmax'))


    #Compile 
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY):

    model = defineModel(trainX.shape[1:], trainY.shape[1])


    # start time
    start_time = time.time()

    #Train 
    history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=1)

    end_time = time.time()
    print("Execution time: ", end_time - start_time)

    #Evaluate the model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('acc : %.3f' % (acc * 100.0)) #Accuracy in percentage format



    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY):

    k_folds = 5

    accuracyScores = []
    histories = []

    #cross validation datasets
    kfold = KFold(k_folds, shuffle=True, random_state=1)


    for train_idx, val_idx in kfold.split(trainX):

        #data for train and validation
        trainX_i, trainY_i, valX_i, valY_i = trainX[train_idx], trainY[train_idx], trainX[val_idx], trainY[val_idx]


        model = defineModel((28, 28, 1), 10)


        history = model.fit(trainX_i, trainY_i, epochs=5, batch_size=32, validation_data=(valX_i, valY_i), verbose=1)


        #Save the training related information in the histories list
        histories.append(history)


        #Evaluate the model on the test dataset
        _, acc = model.evaluate(testX, testY, verbose=1)


        #Save the accuracy in the list
        accuracyScores.append(acc)


    return histories, accuracyScores
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def main():

    #Load the Fashion MNIST dataset in Keras
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()


    #Print the size of the train/test dataset
    print(f"Training data size: {trainX.shape} samples")
    print(f"Test data size: {testX.shape} samples")

    pyplot.figure(figsize=(6,6))
    for i in range(9):
        pyplot.subplot(3, 3, i + 1)
        pyplot.imshow(trainX[i], cmap='gray')
        pyplot.axis('off')

    pyplot.show()

    

    trainX, trainY, testX, testY = prepareData(trainX, trainY, testX, testY)


    #Define, train and evaluate the model in the classical way
    defineTrainAndEvaluateClassic(trainX, trainY, testX, testY)



    #Define, train and evaluate the model using K-Folds strategy
    histories, accuracyScores = defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY)


    #System performance presentation
    summarizeLearningCurvesPerformances(histories, accuracyScores) 



    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
