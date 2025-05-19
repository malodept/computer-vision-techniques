import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def baseline_model(num_pixels, num_classes):
    model = keras.models.Sequential()
    model.add(layers.Dense(8, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_model_and_predict():
    (_,_), (X_test, Y_test) = mnist.load_data()
    num_pixels = X_test.shape[1] * X_test.shape[2]
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32') / 255
    Y_test = to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    model = baseline_model(num_pixels, num_classes)
    model.load_weights("D:\\malo\\Documents\\cours_tsp\\cv\\TP1_MNIST_Digit_Recognition_Moodle\\TP1_MNIST_Digit_Recognition_Moodle\\model.weights.h5")

    # Prédire les 5 premières images
    predictions = model.predict(X_test[:5])
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f'Pred: {predicted_labels[i]}')
    
    plt.show()

if __name__ == '__main__':
    load_model_and_predict()