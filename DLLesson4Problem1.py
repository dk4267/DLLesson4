# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.backend import common as K
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
ytest = y_test
xtrain = X_train
xtest = X_test
# normalize inputs from 0-255 to 0.0-1.0
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain / 255.0
xtest = xtest / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
ytest = np_utils.to_categorical(ytest)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
history = model.fit(xtrain, y_train, validation_data=(xtest, ytest), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#before changes:  loss: 0.2211 - accuracy: 0.9228 - val_loss: 1.0722 - val_accuracy: 0.7035
#validation accuracy much lower than training accuracy - means it probably memorized some of the training data

#after problem 1: loss: 0.3568 - accuracy: 0.8723 - val_loss: 0.6188 - val_accuracy: 0.7954
#accuracy went down, but validation accuracy went up - did better with new data than the previous one

print("1. Predicted :")
print(model.predict_classes(xtest[[0], :]))
print("1. Actual: ")
print(y_test[0])
print("2. Predicted: ")
print(+ model.predict_classes(xtest[[1], :]))
print("2. Actual: ")
print(y_test[1])
print("3. Predicted: ")
print(model.predict_classes(xtest[[2], :]))
print("3. Actual: ")
print(y_test[2])
print("4. Predicted: ")
print(model.predict_classes(xtest[[3], :]))
print("4. Actual: ")
print(y_test[3])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plot history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
