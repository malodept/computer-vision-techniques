import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import tensorflow as tf

from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input

from operator import truediv


def display_activation(activation, col_size, row_size):
    print(activation.shape,'the shape')
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='viridis')
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            activation_index += 1
    plt.show()
        

def plotImage(img_tensor):
    fig = plt.figure(figsize=(10,5))
    plt.imshow(img_tensor,cmap="gray")
    plt.axis('off')
    plt.show()
    
def plotImages(x,y):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot the images in a 5x5 grid
    for i in range(5):
        for j in range(5):
            # Select the image and label
            index = i * 5 + j
            axes[i, j].imshow(x[index], cmap='gray')
            axes[i, j].set_title(str(y[index]))
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def plot_filter(filters, col_size, row_size):
    
    print(filters.shape,'the shape')
    f_ind=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(filters[:, :, 0, f_ind], cmap='gray')
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            f_ind += 1
    plt.show()

def plot_loss(history):
    # Plotting Loss and Accuracy
    plt.figure(figsize=(12, 8))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def CNN(input_shape, num_classes):
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamW(learning_rate=0.001),  
                  metrics=['accuracy'])

    return model

def process_data():
    
    (x_train, y_train), (x_test, y_test)  = mnist.load_data()
    
    
    #plotImages(x_train,y_train)

    # Lets store the number of rows and columns
    img_rows = x_train[0].shape[0]
    img_cols = x_train[0].shape[1]
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # store the shape of a single image 
    input_shape = (img_rows, img_cols, 1)
    
    # change our image type to float32 data type
    x_train = x_train.astype('float32') #uint8 originally
    x_test = x_test.astype('float32')
    
    # Normalize our data by changing the range from (0 to 255) to (0 to 1)
    x_train /= 255.0
    x_test /= 255.0
    
    # Now we one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    num_classes = y_test.shape[1]
    
    return  x_train, y_train, x_test, y_test, input_shape, num_classes


def train():
    

    #call the process data to get necessary processed data
    x_train,y_train, x_test, y_test,input_shape, num_classes = process_data()
    #define the model
    model = CNN(input_shape,num_classes)
    #show the summary of the models.
    print(model.summary())

    #fit the model
    history = model.fit(x_train,y_train,batch_size = 32,epochs = 10,verbose = 1,validation_data = (x_test, y_test))

    #plot the history
    plot_loss(history)
    #Evaluate the model loss and accuracy
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #save the weights
    model.save_weights("D:/malo/Documents/cours_tsp/cv/TP-3-Kernel-Vis/F-MNISTFilter.weights.h5")


def inspect():
    #load the dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = process_data()

    #load the model and its weights
    model = CNN(input_shape, num_classes)

    #Force model to initialize before loading weights
    model(tf.keras.Input(shape=input_shape))  

    # Load weights
    model.load_weights('D:/malo/Documents/cours_tsp/cv/TP-3-Kernel-Vis/F-MNISTFilter.weights.h5')

    # Process a dummy input to ensure the model is fully built
    dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
    _ = model.predict(dummy_input)

    #perform filter explorations by iterating the each of model layer and print the name. Also, print the filter weights and biases for convolution layer
    for layer in model.layers:
        print(layer.name)
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            print(filters.shape, biases.shape)

    # directly access the first layer using indexing (use variable indexing to hold the indexer)
    indexing = 0
    filters, biases = model.layers[indexing].get_weights()
    print(f"Filter Shape: {filters.shape}")
    print(f"Filters: {filters}")
    print(f"Bias Shape: {biases.shape}")
    print(f"Bias: {biases}")

    # normalize the filter between 0 to 1 for visualization and print the filters
    f_min, f_max = filters.min(), filters.max()
    print(f'Before Normalization, Min = {f_min}, Max = {f_max}')
    filters = (filters - f_min) / (f_max - f_min + 1e-8)  # ✅ Prevent division by zero
    print(f'After Normalization, Min = {filters.min()}, Max = {filters.max()}')

    # plot the filters and save manually
    plot_filter(filters, 4, int(truediv(filters.shape[-1], 4)))

    # get the layers to a list for successive features extractions. Use indexing on the model.layers
    layer_outputs = [layer.output for layer in model.layers]

    # Rebuild the activation model after forcing model initialization
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # get an example of test data from x_test (through indexing) and put them to a variable
    example = 30
    img_tensor = x_test[example]

    # plot the example image using plotImage() function
    plotImage(img_tensor)

    # Expand a first dimension to enable batching
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)

    # Process the image once to register it in the model graph
    _ = activation_model.predict(dummy_input)

    # perform prediction using the defined activation_model (through predict function)
    activations = activation_model.predict(img_tensor)

    # print the number of extracted activations
    print("Number of layer activations:", len(activations))

    # get selected activations through array indexer by using indexing variable (step 5) and print the shape
    sel_activation = activations[indexing]
    print(f"First Layer Activation Shape: {sel_activation.shape}")

    print(model.summary())

    # plot the last features (32) from the selected activation layer
    plt.matshow(sel_activation[0, :, :, -1], cmap='summer')
    plt.legend()

    # display activation function to plot the selected activation
    display_activation(sel_activation, 4, int(truediv(sel_activation.shape[-1], 4)))




def main():
    #Application 1 Step 1, call function train
    #train()


    #Application 2 Step 1, call function inspect
    inspect()

    #Applicaiton 3, repeat application 1 and 2 for Fashion MNIST (change the loading data to load fashionmnist instead). Be careful to save the trained model with different name

if __name__ == '__main__':
    main()

