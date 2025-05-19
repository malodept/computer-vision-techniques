import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras import layers
import matplotlib.pyplot as plt

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def baseline_model(num_pixels, num_classes):

    #Initialize
    model = keras.models.Sequential() 

    #hidden dense layer with 8 neurons
    model.add(layers.Dense(8, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    


    #output dense layer
    model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    


    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictMLP(X_train, Y_train, X_test, Y_test):

    #Reshape the MNIST dataset - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = X_train.shape[1] * X_train.shape[2] 
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32') 
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32') 


    #Normalize the input values
    X_train = X_train / 255
    X_test = X_test / 255


    #Transform the classes labels into a binary matrix
    Y_train = to_categorical(Y_train) 
    Y_test = to_categorical(Y_test) 
    num_classes = Y_test.shape[1]


    #model architecture 
    model = baseline_model(num_pixels, num_classes)


    #Train
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

    #System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0) 
    print("Baseline Error: {:.2f}".format(100-scores[1]*100))

    # Save the model
    model.save_weights("D:\malo\Documents\cours_tsp\cv\TP1_MNIST_Digit_Recognition_Moodle\TP1_MNIST_Digit_Recognition_Moodle\model.weights.h5")


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def CNN_model(input_shape, num_classes):

    #Initialize
    model = keras.models.Sequential()


    #first hidden layer as a convolutional layer
    model.add(layers.Conv2D(30, kernel_size=(5, 5), activation='relu', input_shape=input_shape))


    #pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(15, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))



    #Dropout layer
    model.add(layers.Dropout(0.2))


    # flatten layer
    model.add(layers.Flatten())


    #dense layer of size 128
    model.add(layers.Dense(128, activation='relu'))


    model.add(layers.Dense(50, activation='relu'))


    #Define the output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    


    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    #reshape the data to be of size [samples][width][height][channels]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    


    #normalize the input values from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255


    #One hot encoding - Transform the classes labels into a binary matrix
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    


    model = CNN_model((28, 28, 1), Y_test.shape[1])


    #Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=200, verbose=2)

    #plt.plot(history.history['val_accuracy'])      this is to plot the validation accuracy over epochs to see if the model is overfitting
    #plt.title('Validation Accuracy over Epochs')
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation Accuracy')
    #plt.show()

    #Final evaluation of the model - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("CNN Error: {:.2f}".format(100-scores[1]*100))


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #Load the MNIST dataset in Keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()


    #Train and predict on a MLP 
    #model = trainAndPredictMLP(X_train, Y_train, X_test, Y_test)


    #Train and predict on a CNN 
    model = trainAndPredictCNN(X_train, Y_train, X_test, Y_test)


    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
