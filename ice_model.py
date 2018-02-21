import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from netCDF4 import Dataset as NCFile
from sklearn.model_selection import train_test_split

from ice_data import Dataset


def init_model():
    input_shape = (100, 100, 2)
    num_squares = 44

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
    model.add(Dense(num_squares, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


def read_samples(file_name):
    dataset = Dataset.from_csv(file_name)

    # reduce amount of samples to be a factor of 50
    a, b = divmod(len(dataset.samples), 50)
    dataset.samples = dataset.samples[0: len(dataset.samples) - b]

    return dataset


def split_data(samples, test_size):
    train, test = train_test_split(samples, test_size=test_size)

    return train, test


class VarsContainer:
    def __init__(self):
        self.conc_dic = {}
        self.thic_dic = {}

        self.files_by_counts = {}

        self.limit = 43 * 23

    def values(self, file_name):

        if file_name in self.files_by_counts:
            self.files_by_counts[file_name] += 1
        else:
            print(file_name + " was loaded")
            self.files_by_counts[file_name] = 1
            nc = NCFile(file_name)
            conc = nc.variables['iceconc'][:]
            thic = nc.variables['icethic_cea'][:]

            self.conc_dic[file_name] = conc
            self.thic_dic[file_name] = thic

        if self.files_by_counts[file_name] == self.limit:
            conc_tmp = self.conc_dic[file_name]
            thic_tmp = self.thic_dic[file_name]

            del self.files_by_counts[file_name]
            del self.conc_dic[file_name]
            del self.thic_dic[file_name]

            return conc_tmp, thic_tmp

        else:
            return self.conc_dic[file_name], self.thic_dic[file_name]


def data_generator(samples, batch_size, vars_container):
    while 1:
        for sample_index in range(0, len(samples), batch_size):
            x = np.zeros((batch_size, 100, 100, 2), dtype=np.float32)
            y = np.zeros((batch_size, 44))
            for index in range(sample_index, sample_index + batch_size):
                nc_file = samples[index][0].nc_file
                conc, thic = vars_container.values(nc_file)
                ice_square = samples[index][0].ice_conc(conc)
                thic_square = samples[index][0].ice_thic(thic)
                # expanded = np.expand_dims(ice_square, axis=2)
                combined = np.stack(arrays=[ice_square, thic_square], axis=2)
                x[index - sample_index] = combined
                y[index - sample_index] = samples[index][1]
            yield (x, y)


def calc_batch_size(batch_len, min_size, max_size):
    for size in range(min_size, max_size):
        a, b = divmod(batch_len, size)
        if b == 0:
            return size


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


data = read_samples("samples/ice_samples.csv")

# d_conc = {}
# d_thic = {}
#
# idx = 0
# for sample in data.samples:
#     if sample.nc_file not in d_conc:
#         idx += 1
#         print(sample.nc_file)
#         nc = NCFile(sample.nc_file)
#         ice = nc.variables['iceconc'][:]
#         thic = nc.variables['icethic_cea'][:]
#         d_conc[sample.nc_file] = ice
#         d_thic[sample.nc_file] = thic
# print("NetCDFs were loaded")

train, test = split_data(data.samples, 0.2)

# train_middle = int(len(train) / 250)
# test_middle = int(len(test) / 100)
# train_batch_size = calc_batch_size(len(train), int(train_middle - 0.5 * train_middle),
#                                    int(train_middle + 0.5 * train_middle))
# test_batch_size = calc_batch_size(len(test), int(test_middle - 0.5 * test_middle),
#                                   int(test_middle + 0.5 * test_middle))

train_batch_size = 80
test_batch_size = 80
print(len(train))
print(len(test))

print(train_batch_size)
print(test_batch_size)

train_idx = []
for sample in train:
    train_idx.append(sample.index - 1)
train_idx = keras.utils.to_categorical(train_idx, 44)

test_idx = []
for sample in test:
    test_idx.append(sample.index - 1)
test_idx = keras.utils.to_categorical(test_idx, 44)

tr_samples = []
for idx in range(len(train)):
    tr_samples.append([train[idx], train_idx[idx]])

tt_samples = []
for idx in range(len(test)):
    tt_samples.append([test[idx], test_idx[idx]])
epochs = 20

container = VarsContainer()

model = init_model()
history = AccuracyHistory()
model.fit_generator(data_generator(tr_samples, train_batch_size, container),
                    steps_per_epoch=train_batch_size,
                    callbacks=[history],
                    epochs=epochs)
model.save("samples/model.h5")
# model = load_model("samples/model.h5")
# scores = model.predict_generator(data_generator(tt_samples, test_batch_size, d),
#                                  steps=int(len(test) / test_batch_size))

t = model.evaluate_generator(data_generator(tt_samples, test_batch_size, container),
                             steps=int(len(test) / test_batch_size))
print(t)
# print(scores)
