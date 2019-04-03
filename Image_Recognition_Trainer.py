from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
import h5py

# from PIL import Image

# cat_image_pathname = "C:\Users\Mehmet\Pictures\AnimalImages\CAT1.jpg"
# cat_image = Image.open(cat_image_pathname)
# cat_image.show()

# display_image_pathname = input('Enter image pathname')
# display_image = Image.open(display_image_pathname)
# display_image.show()

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# index = int(input('Enter an image index'))
# display_image = X_train[index]
# display_label = y_train[index][0]

# from matplotlib import pyplot as plt

# red_image = Image.fromarray(display_image)
# red, green, blue = red_image.split()

# plt.imshow(red, cmap="Reds")
# plt.show()

new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_train /= 255
new_X_test /= 255
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(new_X_train, new_Y_train, epochs=10, batch_size=32)

model.save('Trained_Model.h5')
