import csv

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import image


def read_samples(file_name):
    samples = []
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            row[1] = int(row[1])
            samples.append(row)

    for sample in samples:
        label = np.zeros(2, dtype=np.int8)
        label[sample[1]] = 1
        sample[1] = label

    return samples


def split_data():
    good_samples_file = "samples/good_samples.csv"
    good_samples = read_samples(good_samples_file)

    train_good, test_good = train_test_split(good_samples, test_size=0.2)

    bad_samples_file = "samples/bad_samples.csv"
    bad_samples = read_samples(bad_samples_file)

    train_bad, test_bad = train_test_split(bad_samples, test_size=0.2)

    train = train_good + train_bad
    test = test_good + test_bad

    return train, test


def generate_train_data(samples):
    while 1:
        batch_size = int(len(samples) / 11)
        for sample_index in range(0, len(samples), batch_size):
            x = np.zeros((batch_size, 100, 100, 1), dtype=np.float32)
            y = np.zeros((batch_size, 2))
            for index in range(sample_index, sample_index + batch_size):
                file_name = samples[index][0]
                square = image.load_square_from_file(file_name)
                expanded = np.expand_dims(square, axis=2)
                x[index - sample_index] = expanded
                y[index - sample_index] = samples[index][1]
            yield (x, y)


def generate_test_data(samples):
    while 1:
        batch_size = int(len(samples) / 10)
        for sample_index in range(0, len(samples), batch_size):
            x = np.zeros((batch_size, 100, 100, 1), dtype=np.float32)
            y = np.zeros((batch_size, 2))
            for index in range(sample_index, sample_index + batch_size):
                file_name = samples[index][0]
                square = image.load_square_from_file(file_name)
                expanded = np.expand_dims(square, axis=2)
                x[index - sample_index] = expanded
                y[index - sample_index] = samples[index][1]
            yield (x, y)


train, test = split_data()
input_shape = (100, 100, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


print(len(train))
print(len(test))
history = AccuracyHistory()
train_batch_size = int(len(train) / 11)
test_batch_size = int(len(test) / 5)
epochs = 10

model.fit_generator(generate_train_data(train),
                    samples_per_epoch=train_batch_size,
                    nb_epoch=epochs,
                    callbacks=[history],
                    validation_data=generate_test_data(test),
                    validation_steps=int(len(test) / test_batch_size))

model.save("model.h5")

score = model.evaluate_generator(generate_test_data(test), workers=4)
plt.plot(range(1, 10), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
