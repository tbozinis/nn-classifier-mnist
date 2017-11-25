from keras import Model, optimizers
from keras.datasets import mnist
from keras.layers import Input, Dense

EPOCHS = 20
BATCH_SIZE = 2000
DIM = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255


input_layer = Input(shape=(DIM * DIM,))
first_layer = Dense(512, activation='sigmoid')(input_layer)
hidden = Dense(512, activation='sigmoid')(first_layer)
hidden = Dense(28, activation='sigmoid')(hidden)
output = Dense(1, activation='sigmoid')(hidden)

mlp = Model(inputs=input_layer, outputs=output)
mlp.compile(optimizer=optimizers.Adam(lr=0.005), loss='mse', metrics=['accuracy'])

mlp.summary()

mlp.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2)