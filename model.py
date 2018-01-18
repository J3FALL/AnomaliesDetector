import csv

import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model

import image


def read_samples(file_name):
    samples = []
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            row[1] = int(row[1])

            if row[2] == 0.0 and row[3] == 0.0:
                next(reader)

            samples.append(row)

    for sample in samples:
        label = np.zeros(2, dtype=np.int8)
        label[sample[1]] = 1
        sample[1] = label

    a, b = divmod(len(samples), 50)

    return samples[0:len(samples) - b]


def split_data():
    good_samples_file = "samples/good_samples.csv"
    good_samples = read_samples(good_samples_file)

    # train_good, test_good = train_test_split(good_samples, test_size=0.2)

    bad_samples_file = "samples/bad_samples.csv"
    bad_samples = read_samples(bad_samples_file)

    # train_bad, test_bad = train_test_split(bad_samples, test_size=0.2)

    train = bad_samples
    test = bad_samples

    return train


def generate_train_data(samples):
    while 1:
        # print(len(samples))
        batch_size = int(len(samples) / 50)
        for sample_index in range(0, len(samples), batch_size):
            x = np.zeros((batch_size, 100, 100, 1), dtype=np.float32)
            y = np.zeros((batch_size, 2))
            for index in range(sample_index, sample_index + batch_size):
                try:
                    file_name = samples[index][0]
                except IndexError:
                    print(index)
                square = image.load_square_from_file(file_name)
                expanded = np.expand_dims(square, axis=2)
                x[index - sample_index] = expanded
                y[index - sample_index] = samples[index][1]
            yield (x, y)


def generate_test_data(samples):
    # while 1:
    batch_size = int(len(samples) / 25)
    for sample_index in range(0, len(samples), batch_size):
        x = np.zeros((batch_size, 100, 100, 1), dtype=np.float32)
        y = np.zeros((batch_size, 2))
        for index in range(sample_index, sample_index + batch_size):
            file_name = samples[index][0]
            square = image.load_square_from_file(file_name)
            # rows_sum = square.sum(axis=1)
            # new_matrix = square / rows_sum[:, np.newaxis]
            expanded = np.expand_dims(square, axis=2)
            x[index - sample_index] = expanded
            y[index - sample_index] = samples[index][1]
        yield (x, y)


def generate_test_outliers(samples):
    # while 1:
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


def get_validation_data():
    valid_samples_file = "samples/valid_samples.csv"
    samples = read_samples(valid_samples_file)

    return samples


def get_outliers_validation_data():
    file = "samples/valid_samples.csv"
    samples = read_samples(file)
    result = []
    for sample in samples:
        if sample[1][1] == 1:
            result.append(sample)

    return result[0:240]


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
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


train = split_data()
test = get_validation_data()
history = AccuracyHistory()
train_batch_size = int(len(train) / 50)
test_batch_size = int(len(test) / 25)
epochs = 10

'''
model.fit_generator(generate_train_data(train),
                    steps_per_epoch=27,
                    callbacks=[history],
                    epochs=10)

model.save("model.h5")
'''

model = load_model("model.h5")
out = get_outliers_validation_data()
print(model.metrics_names)
score = model.evaluate_generator(generate_test_outliers(out), steps=10)
print(score)

score = model.evaluate_generator(generate_test_data(test), steps=25)
print(score)
