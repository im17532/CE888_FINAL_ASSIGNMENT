# IMPORT NEEDED PACKAGES

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import random
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import Input, Dense, Dropout
from keras.models import Model


# FUNCTION TO DISPLAY THE MNIST AND CIFAR10 DATASETS
def display_digit(num, x_train):
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d' % num)
    plt.imshow(image, cmap = plt.get_cmap('gray_r'))
    plt.show()

# WARNING: BECAUSE OF HUGE AMOUNT OF DATA THE EXECUTION OF THE CODE MAY BE A LITTLE BIT SLOW

random.seed(777)
# READ THE MNIST DATASET AND MANIPULATE THE PIXELS OF x_train and x_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# STARTING WITH THE X_TRAIN
mnist_train_shuffled1 = np.empty([x_train.shape[0], x_train.shape[1], x_train.shape[2]])
mnist_train_shuffled2 = np.empty([x_train.shape[0], x_train.shape[1], x_train.shape[2]])
mnist_train_shuffled3 = np.empty([x_train.shape[0], x_train.shape[1], x_train.shape[2]])

# OBTAIN THE FIRST TASK OF THE TRAIN SET BY RESHUFFLING
for i in range(0, x_train.shape[0]):
    temp_shuffled = np.empty([x_train.shape[1], x_train.shape[2]])
    for j in range(0, x_train.shape[1]):
        np.random.shuffle(x_train[i][j])
        temp_shuffled[j] = x_train[i][j]
    mnist_train_shuffled1[i] = temp_shuffled

# OBTAIN THE SECOND TASK OF THE TRAIN SET BY RESHUFFLING
for i in range(0, x_train.shape[0]):
    temp_shuffled = np.empty([x_train.shape[1], x_train.shape[2]])
    for j in range(0, x_train.shape[1]):
        np.random.shuffle(x_train[i][j])
        temp_shuffled[j] = x_train[i][j]
    mnist_train_shuffled2[i] = temp_shuffled

# OBTAIN THE THIRD TASK OF THE TRAIN SET BY RESHUFFLING
for i in range(0, x_train.shape[0]):
    temp_shuffled = np.empty([x_train.shape[1], x_train.shape[2]])
    for j in range(0, x_train.shape[1]):
        np.random.shuffle(x_train[i][j])
        temp_shuffled[j] = x_train[i][j]
    mnist_train_shuffled3[i] = temp_shuffled


# CONTINUE WITH THE X_TEST

mnist_test_shuffled1 = np.empty([x_test.shape[0], x_test.shape[1], x_test.shape[2]])
mnist_test_shuffled2 = np.empty([x_test.shape[0], x_test.shape[1], x_test.shape[2]])
mnist_test_shuffled3 = np.empty([x_test.shape[0], x_test.shape[1], x_test.shape[2]])

# OBTAIN THE FIRST TASK OF THE TEST SET BY RESHUFFLING
for i in range(0, x_test.shape[0]):
    temp_shuffled = np.empty([x_test.shape[1], x_test.shape[2]])
    for j in range(0, x_test.shape[1]):
        np.random.shuffle(x_test[i][j])
        temp_shuffled[j] = x_test[i][j]
    mnist_test_shuffled1[i] = temp_shuffled

# OBTAIN THE SECOND TASK OF THE TEST SET BY RESHUFFLING
for i in range(0, x_test.shape[0]):
    temp_shuffled = np.empty([x_test.shape[1], x_test.shape[2]])
    for j in range(0, x_test.shape[1]):
        np.random.shuffle(x_test[i][j])
        temp_shuffled[j] = x_test[i][j]
    mnist_test_shuffled2[i] = temp_shuffled

# OBTAIN THE THIRD TASK OF THE TEST SET BY RESHUFFLING
for i in range(0, x_test.shape[0]):
    temp_shuffled = np.empty([x_test.shape[1], x_test.shape[2]])
    for j in range(0, x_test.shape[1]):
        np.random.shuffle(x_test[i][j])
        temp_shuffled[j] = x_test[i][j]
    mnist_test_shuffled3[i] = temp_shuffled

# # CODE IF NEEDED TO VISUALIZE MNIST IMAGES
# display_digit(0, x_train)
# display_digit(0, mnist_test_shuffled1)

# SPECIFY THE NN ARGUMENTS
num_train = 60000
num_test = 10000

batch_size = 128
num_classes = 10
epochs = 12

# MNIST IMAGES HAVE A SHAPE OF 28 X 28 AND 10 CLASSES IN TOTAL (0-9
height, width, depth = 28, 28, 1
num_classes = 10

# TASK1 MODIFICATIONS
# RESHAPE FOR TASK1
x_train1 = mnist_train_shuffled1.reshape(num_train, height * width)
x_test1 = mnist_test_shuffled1.reshape(num_test, height * width)
X_train1 = x_train1.astype('float32')
X_test1 = x_test1.astype('float32')

# CHANGE THE PIXELS RANGE FROM [0,255] --> [0,1]
x_train1 /= 255
x_test1 /= 255
# print('x_train1 shape:', x_train1.shape)
# print(x_train1.shape[0], 'train samples')
# print(x_test1.shape[0], 'test samples')

# TASK 2 MODIFICATION
# RESHAPE FOR TASK1
x_train2 = mnist_train_shuffled2.reshape(num_train, height * width)
x_test2 = mnist_test_shuffled2.reshape(num_test, height * width)
X_train2 = x_train2.astype('float32')
X_test2 = x_test2.astype('float32')

# CHANGE THE PIXELS RANGE FROM [0,255] --> [0,1]
x_train2 /= 255
x_test2 /= 255
# print('x_train2 shape:', x_train2.shape)
# print(x_train2.shape[0], 'train samples')
# print(x_test2.shape[0], 'test samples')

# TASK 3 MODIFICATION
x_train3 = mnist_train_shuffled3.reshape(num_train, height * width)
x_test3 = mnist_test_shuffled3.reshape(num_test, height * width)
X_train3 = x_train3.astype('float32')
X_test3 = x_test3.astype('float32')

# CHANGE THE PIXELS RANGE FROM [0,255] --> [0,1]
x_train3 /= 255
x_test3 /= 255
# print('x_train3 shape:', x_train3.shape)
# print(x_train3.shape[0], 'train samples')
# print(x_test3.shape[0], 'test samples')

# ONE - HOT ENCODE OF THE LABELS
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


## IMPORTANT: FOR THE TRAINING OF THE AUTOENCODERS AND NN USE ONE PART OF CODE AT A TIME
## COMMENTING THE OTHER LINES THAT HAVE TO DO WITH TRAINING OF OTHER AUTOENCODERS AND NEURAL NETWORK
# CREATE n = 3 AUTOENCODERS
input_img = Input(shape=(height * width,))

# CREATE AUTOENCODER NUMBER 1
x = Dense(height * width, activation='relu')(input_img)

encoded1 = Dense(height * width//2, activation='relu')(x)
encoded2 = Dense(height * width//8, activation='relu')(encoded1)

y = Dense(height * width//256, activation='relu')(encoded2)

decoded2 = Dense(height * width//8, activation='relu')(y)
decoded1 = Dense(height * width//2, activation='relu')(decoded2)

z = Dense(height * width, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)

encoder = Model(input_img, y)

autoencoder.compile(optimizer='adadelta', loss='mse')

# TRAIN THE AUTOENCODER WITH TASK1
autoencoder.fit(x_train1, x_train1,
      epochs = 3,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_test1, x_test1))
## loss: 0.0861

# CREATE AUTOENCODER NUMBER 2
x = Dense(height * width, activation='relu')(input_img)

encoded1 = Dense(height * width//2, activation='relu')(x)
dropout1 = Dropout(0.5)(encoded1)
encoded2 = Dense(height * width//8, activation='relu')(dropout1)

y = Dense(height * width//256, activation='relu')(encoded2)

decoded2 = Dense(height * width//8, activation='relu')(y)
dropout2 = Dropout(0.2)(decoded2)
decoded1 = Dense(height * width//2, activation='relu')(dropout2)

z = Dense(height * width, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)

encoder = Model(input_img, y)

autoencoder.compile(optimizer='adadelta', loss='mse')

# TRAIN THE AUTOENCODER AND THE NN WITH TASK2
autoencoder.fit(x_train2, x_train2,
      epochs = 3,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_test2, x_test2))
# loss: 0.0870


# CREATE AUTOENCODER NUMBER 3
x = Dense(height * width, activation='relu')(input_img)

encoded1 = Dense(height * width//2, activation='relu')(x)
dropout1 = Dropout(0.2)(encoded1)
encoded2 = Dense(height * width//8, activation='relu')(dropout1)

y = Dense(height * width//256, activation='relu')(encoded2)

decoded2 = Dense(height * width//8, activation='relu')(y)
dropout2 = Dropout(0.5)(decoded2)
decoded1 = Dense(height * width//2, activation='relu')(dropout2)

z = Dense(height * width, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)

#encoder is the model of the autoencoder slice in the middle
encoder = Model(input_img, y)

autoencoder.compile(optimizer='adadelta', loss='mse')

# TRAIN THE AUTOENCODER AND THE NN WITH TASK3
autoencoder.fit(x_train3, x_train3,
      epochs = 3,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_test3, x_test3))
# loss: 0.0876

# THE FIRST AUTO-ENCODER IS THE BEST ONE WITH LOSS: 0.0861
# WE ASSOCIATE A CLASSIFIER WITH THAT AND TRAIN THE RESULTED NEURAL NETWORK WITH EVERY SINGLE BATCH

# BEST AUTO-ENCODER
x = Dense(height * width, activation='relu')(input_img)

encoded1 = Dense(height * width//2, activation='relu')(x)
encoded2 = Dense(height * width//8, activation='relu')(encoded1)

y = Dense(height * width//256, activation='relu')(encoded2)

decoded2 = Dense(height * width//8, activation='relu')(y)
decoded1 = Dense(height * width//2, activation='relu')(decoded2)

z = Dense(height * width, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)

encoder = Model(input_img, y)
autoencoder.compile(optimizer='adadelta', loss='mse')

# TRAIN THE NN WITH TASK1
autoencoder.fit(x_train1, x_train1,
      epochs = 3,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_test1, x_test1))

# ASSOCIATE 'SOFTMAX' CLASSIFIER
out2 = Dense(num_classes, activation='softmax')(encoder.output)
newmodel = Model(encoder.input, out2)
newmodel.compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = ['accuracy'])
newmodel.fit(x_train1, y_train,
      epochs = 10,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_test1, y_test))

scores = newmodel.evaluate(x_test1, y_test, verbose=1)
print("Accuracy: ", scores[1])
# score: 0.7348

# TRAIN THE SAME NN WITH TASK2
autoencoder.fit(x_train2, x_train2,
      epochs = 3,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_test2, x_test2))

# USE soft-max classifier
out2 = Dense(num_classes, activation='softmax')(encoder.output)
newmodel = Model(encoder.input, out2)
newmodel.compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = ['accuracy'])
newmodel.fit(x_train2, y_train,
      epochs = 10,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_test2, y_test))
scores = newmodel.evaluate(x_test2, y_test, verbose=1)
print("Accuracy: ", scores[1])
# Score:0.7426

# TRAIN THE SAME NN WITH TASK2 BUT EVALUATE ON TASK1
autoencoder.fit(x_train2, x_train2,
      epochs=3,
      batch_size=128,
      shuffle=True,
      validation_data=(x_test1, x_test1))

# USE soft-max classifier
out2 = Dense(num_classes, activation='softmax')(encoder.output)
newmodel = Model(encoder.input, out2)
newmodel.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
newmodel.fit(x_train2, y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(x_test1, y_test))
scores = newmodel.evaluate(x_test1, y_test, verbose=1)
print("Accuracy: ", scores[1])
# Score:0.7685

# TRAIN THE SAME NN WITH TASK3 BUT EVALUATE ON TASK1
autoencoder.fit(x_train3, x_train3,
      epochs=3,
      batch_size=128,
      shuffle=True,
      validation_data=(x_test1, x_test1))

# USE soft-max classifier
out2 = Dense(num_classes, activation='softmax')(encoder.output)
newmodel = Model(encoder.input, out2)
newmodel.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
newmodel.fit(x_train3, y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(x_test1, y_test))
scores = newmodel.evaluate(x_test1, y_test, verbose=1)
print("Accuracy: ", scores[1])
# Score: 0.7698

# TRAIN THE SAME NN WITH TASK3 BUT EVALUATE ON TASK2
autoencoder.fit(x_train3, x_train3,
      epochs=3,
      batch_size=128,
      shuffle=True,
      validation_data=(x_test2, x_test2))

# USE 'SOFTMAX' CLASSIFIER
out2 = Dense(num_classes, activation='softmax')(encoder.output)
newmodel = Model(encoder.input, out2)
newmodel.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
newmodel.fit(x_train3, y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(x_test2, y_test))
scores = newmodel.evaluate(x_test2, y_test, verbose=1)
print("Accuracy: ", scores[1])
# Score:0.7644

# NOW THAT WE HAVE TRAINED WITH ALL THE BATCHES WE WILL
# SEE IF THE ACCURACY HAS DECREASED; IF SO, THAT MEANS THAT OUR NN FORGOT HOW TO PERFORM ON TASK1
# HOWEVER THE ACCURACY ALWAYS GOES UP PROVING THAT THE CONTINUAL LEARNING SEEMS TO WORK

# READ THE CIFAR10 DATASET AND MANIPULATE THE PIXELS OF THE x_trainCifar and x_testCifar
(x_trainCifar, y_trainCifar), (x_testCifar, y_testCifar) = cifar10.load_data()

x_trainCifar_shuffled1 = np.empty([int(x_trainCifar.shape[0]/100), x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
x_trainCifar_shuffled2 = np.empty([int(x_trainCifar.shape[0]/100), x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
x_trainCifar_shuffled3 = np.empty([int(x_trainCifar.shape[0]/100), x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])

# OBTAIN THE FIRST TASK OF THE CIFAR10 TRAIN SET BY RESHUFFLING
for i in range(0, int(x_trainCifar.shape[0]/100)):
    temp1 = np.empty([x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
    for j in range(0, x_trainCifar.shape[1]):
        temp2 = np.empty([x_trainCifar.shape[2], x_trainCifar.shape[3]])
        for z in range(0, x_trainCifar.shape[2]):
            np.random.shuffle(x_trainCifar[i][j][z])
            temp2[z] = x_trainCifar[i][j][z]
        temp1[j] = temp2
    x_trainCifar_shuffled1[i] = temp1

# OBTAIN THE SECOND TASK OF THE CIFAR10 TRAIN SET BY RESHUFFLING
for i in range(0, int(x_trainCifar.shape[0]/100)):
    temp1 = np.empty([x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
    for j in range(0, x_trainCifar.shape[1]):
        temp2 = np.empty([x_trainCifar.shape[2], x_trainCifar.shape[3]])
        for z in range(0, x_trainCifar.shape[2]):
            np.random.shuffle(x_trainCifar[i][j][z])
            temp2[z] = x_trainCifar[i][j][z]
        temp1[j] = temp2
    x_trainCifar_shuffled2[i] = temp1

# OBTAIN THE THIRD TASK OF THE CIFAR10 TRAIN SET BY RESHUFFLING
for i in range(0, int(x_trainCifar.shape[0]/100)):
    temp1 = np.empty([x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
    for j in range(0, x_trainCifar.shape[1]):
        temp2 = np.empty([x_trainCifar.shape[2], x_trainCifar.shape[3]])
        for z in range(0, x_trainCifar.shape[2]):
            np.random.shuffle(x_trainCifar[i][j][z])
            temp2[z] = x_trainCifar[i][j][z]
        temp1[j] = temp2
    x_trainCifar_shuffled3[i] = temp1

# CONTINUE WITH THE X_TEST

x_testCifar_shuffled1 = np.empty([int(x_testCifar.shape[0]/100), x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]])
x_testCifar_shuffled2 = np.empty([int(x_testCifar.shape[0]/100), x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]])
x_testCifar_shuffled3 = np.empty([int(x_testCifar.shape[0]/100), x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]])

# OBTAIN THE FIRST TASK OF THE CIFAR10 TEST SET BY RESHUFFLING
for i in range(0, int(x_testCifar.shape[0]/100)):
    temp1 = np.empty(([x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]]))
    for j in range(0, x_testCifar.shape[1]):
        temp2 = np.empty([x_testCifar.shape[2], x_testCifar.shape[3]])
        for z in range(0, x_testCifar.shape[2]):
            np.random.shuffle(x_testCifar[i][j][z])
            temp2[z] = x_testCifar[i][j][z]
        temp1[j] = temp2
    x_testCifar_shuffled1[i] = temp1

# OBTAIN THE SECOND TASK OF THE CIFAR10 TEST SET BY RESHUFFLING
for i in range(0, int(x_testCifar.shape[0]/100)):
    temp1 = np.empty(([x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]]))
    for j in range(0, x_testCifar.shape[1]):
        temp2 = np.empty([x_testCifar.shape[2], x_testCifar.shape[3]])
        for z in range(0, x_testCifar.shape[2]):
            np.random.shuffle(x_testCifar[i][j][z])
            temp2[z] = x_testCifar[i][j][z]
        temp1[j] = temp2
    x_testCifar_shuffled2[i] = temp1

# OBTAIN THE THIRD TASK OF THE CIFAR10 TEST SET BY RESHUFFLING
for i in range(0, int(x_testCifar.shape[0]/100)):
    temp1 = np.empty(([x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]]))
    for j in range(0, x_testCifar.shape[1]):
        temp2 = np.empty([x_testCifar.shape[2], x_testCifar.shape[3]])
        for z in range(0, x_testCifar.shape[2]):
            np.random.shuffle(x_testCifar[i][j][z])
            temp2[z] = x_testCifar[i][j][z]
        temp1[j] = temp2
    x_testCifar_shuffled3[i] = temp1

# SINCE WE ARE USING A SUBSET OF THE CIFAR10 DATASET WE HAVE TO TAKE THE RESPECTIVE DATASET ON THE Y_TRAINCIFAR AND Y_TESTCIFAR
y_trainCifarsub = np.empty([500, 1])
y_testCifarsub = np.empty([100, 1])

for i in range(0, int(y_trainCifar.shape[0]/100)):
    y_trainCifarsub[i] = y_trainCifar[i]

for i in range(0, int(y_testCifar.shape[0]/100)):
    y_testCifarsub[i] = y_testCifar[i]

# SPECIFY THE NN ARGUMENTS
num_train = 500
num_test = 100

height = 32
width = 32
depth = 3

num_classes = 10
# TASK1 MODIFICATIONS (RESHAPE AND NORMALIZATION)
# RESHAPE FOR TASK1
X_train1 = x_trainCifar_shuffled1.reshape(num_train, height * width * depth)
X_test1 = x_testCifar_shuffled1.reshape(num_test, height * width*depth)
X_train1 = X_train1.astype('float32')
X_test1 = X_test1.astype('float32')

X_train1 /= 255
X_test1 /= 255

# TASK2 MODIFICATIONS (RESHAPE AND NORMALIZATION)
# RESHAPE FOR TASK2
X_train2 = x_trainCifar_shuffled2.reshape(num_train, height * width * depth)
X_test2 = x_testCifar_shuffled2.reshape(num_test, height * width*depth)
X_train2 = X_train2.astype('float32')
X_test2 = X_test2.astype('float32')

X_train2 /= 255
X_test2 /= 255

# TASK3 MODIFICATIONS (RESHAPE AND NORMALIZATION)
# RESHAPE FOR TASK3
X_train3 = x_trainCifar_shuffled3.reshape(num_train, height * width * depth)
X_test3 = x_testCifar_shuffled3.reshape(num_test, height * width*depth)
X_train3 = X_train3.astype('float32')
X_test3 = X_test3.astype('float32')

X_train3 /= 255
X_test3 /= 255

# ONE - HOT ENCODE OF THE LABELS
Y_train = np_utils.to_categorical(y_trainCifarsub, num_classes)
Y_test = np_utils.to_categorical(y_testCifarsub, num_classes)

input_img = Input(shape=(height * width * depth,))
s = height * width * depth

## IMPORTANT: FOR THE TRAINING OF THE AUTOENCODERS AND NN USE ONE PART OF CODE AT A TIME
## COMMENTING THE OTHER LINES THAT HAVE TO DO WITH TRAINING OF OTHER AUTOENCODERS AND NEURAL NETWORK
# CREATE n = 3 AUTO-ENCODERS
# CREATE AUTOENCODER NUMBER 1
x = Dense(s, activation='relu')(input_img)

encoded = Dense(s//2, activation='relu')(x)
encoded = Dense(s//8, activation='relu')(encoded)

y = Dense(s//256, activation='relu')(x)

decoded = Dense(s//8, activation='relu')(y)
decoded = Dense(s//2, activation='relu')(decoded)

z = Dense(s, activation='sigmoid')(decoded)
model = Model(input_img, z)

model.compile(optimizer='adadelta', loss='mse')

# TRAIN THE AUTOENCODER WITH TASK1 CIFAR
model.fit(X_train1, X_train1,
      nb_epoch=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test1, X_test1))

mid = Model(input_img, y)
reduced_representation = mid.predict(X_test1)
# loss: 0.063

# CREATE AUTOENCODER NUMBER 2
x = Dense(s, activation='relu')(input_img)

encoded = Dense(s//2, activation='relu')(x)
dropout = Dropout(0.25)(encoded)
encoded = Dense(s//8, activation='relu')(dropout)

y = Dense(s//256, activation='relu')(x)

decoded = Dense(s//8, activation='relu')(y)
decoded = Dense(s//2, activation='relu')(decoded)

z = Dense(s, activation='sigmoid')(decoded)
model = Model(input_img, z)

model.compile(optimizer='adadelta', loss='mse') # reporting the accuracy

# TRAIN THE AUTOENCODER WITH TASK2 CIFAR
model.fit(X_train2, X_train2,
      nb_epoch=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test2, X_test2))

mid = Model(input_img, y)
reduced_representation = mid.predict(X_test2)
# Loss: 0.063

# CREATE AUTOENCODER NUMBER 3
x = Dense(s, activation='relu')(input_img)

encoded = Dense(s//2, activation='relu')(x)
dropout = Dropout(0.5)(encoded)
encoded = Dense(s//8, activation='relu')(dropout)

y = Dense(s//256, activation='relu')(x)

decoded = Dense(s//8, activation='relu')(y)
decoded = Dense(s//2, activation='relu')(decoded)

z = Dense(s, activation='sigmoid')(decoded)
model = Model(input_img, z)

model.compile(optimizer='adadelta', loss='mse') # reporting the accuracy

# TRAIN THE AUTOENCODER WITH TASK3 CIFAR
model.fit(X_train3, X_train3,
      nb_epoch=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test3, X_test3))

mid = Model(input_img, y)
reduced_representation = mid.predict(X_test3)
# loss: 0.631

# WE ASSUME THAT THE FIRST AUTO-ENCODER IS THE BEST ONE WITH LOSS: 0.0630
# WE ASSOCIATE A CLASSIFIER WITH THAT AND TRAIN THE RESULTED NEURAL NETWORK WITH EVERY SINGLE BATCH

# BEST AUTO-ENCODER
x = Dense(s, activation='relu')(input_img)

encoded = Dense(s//2, activation='relu')(x)
encoded = Dense(s//8, activation='relu')(encoded)

y = Dense(s//256, activation='relu')(x)

decoded = Dense(s//8, activation='relu')(y)
decoded = Dense(s//2, activation='relu')(decoded)

z = Dense(s, activation='sigmoid')(decoded)
model = Model(input_img, z)

model.compile(optimizer='adadelta', loss='mse')

# ASSOCIATE 'SOFTMAX' CLASSIFIER
out = Dense(num_classes, activation='softmax')(y)
reduced = Model(input_img, out)
reduced.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])


# TRAIN THE NN WITH TASK1 CIFAR
model.fit(X_train1, X_train1,
      nb_epoch=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test1, X_test1))

mid = Model(input_img, y)
reduced_representation = mid.predict(X_test1)

reduced.fit(X_train1, Y_train,
      nb_epoch=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test1, Y_test))
scores = reduced.evaluate(X_test1, Y_test, verbose=0)
print("Accuracy: ", scores[1])
# Score: 0.06

# TRAIN THE NN WITH TASK2 CIFAR BUT EVALUATE ON TASK1
model.fit(X_train2, X_train2,
      nb_epoch=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test1, X_test1))

mid = Model(input_img, y)
reduced_representation = mid.predict(X_test1)

reduced.fit(X_train2, Y_train,
      nb_epoch=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test1, Y_test))
scores = reduced.evaluate(X_test1, Y_test, verbose=0)
print("Accuracy: ", scores[1])
# Score: 0.06

# TRAIN THE NN WITH TASK3 CIFAR BUT EVALUATE ON TASK1
model.fit(X_train3, X_train3,
      nb_epoch=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test1, X_test1))

mid = Model(input_img, y)
reduced_representation = mid.predict(X_test1)

reduced.fit(X_train3, Y_train,
      nb_epoch=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test1, Y_test))
scores = reduced.evaluate(X_test1, Y_test, verbose=0)
print("Accuracy: ", scores[1])
# Score: 0.06

# TRAIN THE NN WITH TASK 2 AND OBTAIN THE ACCURACY
reduced.fit(X_train2, Y_train,
      nb_epoch=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test2, Y_test))
scores = reduced.evaluate(X_test2, Y_test, verbose=0)
print("Accuracy: ", scores[1])
# Score: 0.11

# TRAIN THE NN WITH TASK3 CIFAR BUT EVALUATE ON TASK2
model.fit(X_train3, X_train3,
      nb_epoch=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test2, X_test1))

mid = Model(input_img, y)
reduced_representation = mid.predict(X_test2)

reduced.fit(X_train3, Y_train,
      nb_epoch=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test2, Y_test))
scores = reduced.evaluate(X_test2, Y_test, verbose=0)
print("Accuracy: ", scores[1])
# Score: 0.11

# THE ACCURACY REMAINS THE SAME (BUT RATHER LOW). THIS MEANS THAT THE NN DOESN'T FORGET BUT IT'S LEARNING PROCEDURE IS LIMITED
