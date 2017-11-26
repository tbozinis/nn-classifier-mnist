import keras
from keras import optimizers
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from keras.models import Model

EPOCHS = 20
BATCH_SIZE = 1000
DIM = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], DIM, DIM, 1)
x_test = x_test.reshape(x_test.shape[0], DIM, DIM, 1)
input_shape = (DIM, DIM, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

input_layer = Input(shape=input_shape)
hidden = Conv2D(28, 3, activation='relu', input_shape=input_shape)(input_layer)
hidden = Conv2D(28, 3, activation='relu')(hidden)
hidden = Dropout(0.25)(hidden)
hidden = Flatten()(hidden)
output = Dense(10, activation='softmax')(hidden)

mlp = Model(inputs=input_layer, outputs=output)
mlp.compile(optimizer=optimizers.Adam(lr=0.005), loss='mse', metrics=['accuracy'])

mlp.summary()

mlp.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2)

score = mlp.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])