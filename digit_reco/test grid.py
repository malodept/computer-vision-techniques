import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, models
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier  
import keras

# crée le modèle CNN
def create_model(optimizer='adam', dropout_rate=0.2):
    model = models.Sequential()
    model.add(layers.Conv2D(30, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(15, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# données MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Prétraitement 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalisation 
X_train /= 255
X_test /= 255

# One-hot encoding
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Enveloppe du modèle pour utiliser avec GridSearchCV
model = KerasClassifier(model=create_model, verbose=0, dropout_rate=0.2)

# hyperparamètres à tester
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'batch_size': [32, 64, 128],
    'epochs': [5, 10]
}


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train, Y_train)

print(f"Meilleurs paramètres : {grid_result.best_params_}")
print(f"Meilleure précision : {grid_result.best_score_}")

# Évaluation du modèle final sur les données de test
best_model = grid_result.best_estimator_
score = best_model.score(X_test, Y_test)
print(f"Précision sur le jeu de test : {score * 100:.2f}%")
