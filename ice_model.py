import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from netCDF4 import Dataset as NCFile
from sklearn.model_selection import train_test_split

from ice_data import Dataset
from ice_data import SQUARE_SIZE
from ice_data import count_predictions

# squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
#            *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
#            *list(range(139, 143))
#            ]

squares = [*list(range(1, 8)), *list(range(12, 19)), *list(range(24, 30))]
zones = [[5, 11, 0, 6], [1, 2, 3, 4, 7, 8, 9, 10, 12]]


def init_model():
    input_shape = (SQUARE_SIZE, SQUARE_SIZE, 3)
    num_squares = 176

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
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])

    return model


def init_basic_ocean_model(num_squares):
    input_shape = (50, 50, 2)

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


def init_advanced_ocean_model(num_squares):
    input_shape = (SQUARE_SIZE, SQUARE_SIZE, 3)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (10, 10), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_squares, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


def VGG(num_squares):
    input_shape = (SQUARE_SIZE, SQUARE_SIZE, 1)
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_squares, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
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

        self.limit = 90 * 23

    def values(self, file_name):

        if file_name in self.files_by_counts:
            self.files_by_counts[file_name] += 1
        else:
            print(file_name + " was loaded")
            self.files_by_counts[file_name] = 1
            nc = NCFile(file_name)
            # TODO: this should be params list-like
            if "satellite" in file_name:
                # TODO: check .filled and fix this anywhere
                conc = nc.variables['ice_conc'][:].filled(0) / 100.0
                thic = np.empty((1, 400, 100), np.float32)
            else:
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

    def normalize(self, data):
        return data / np.max(data)


def data_generator(samples, batch_size, vars_container, full_mask):
    while 1:
        for sample_index in range(0, len(samples), batch_size):
            x = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)
            y = np.zeros((batch_size, 176))
            for index in range(sample_index, sample_index + batch_size):
                nc_file = samples[index][0].nc_file
                conc, thic = vars_container.values(nc_file)
                ice_square = samples[index][0].ice_conc(conc)
                thic_square = samples[index][0].ice_thic(thic)
                square_size = samples[index][0].size
                mask_x, mask_y = divmod(samples[index][0].index - 1, 22)
                mask = full_mask[mask_x * square_size:mask_x * square_size + square_size,
                       mask_y * square_size:mask_y * square_size + square_size]

                # expanded = np.expand_dims(ice_square, axis=2)
                combined = np.stack(arrays=[ice_square, thic_square, mask], axis=2)
                x[index - sample_index] = combined
                y[index - sample_index] = samples[index][1]
            yield (x, y)


def ocean_data_generator(samples, batch_size, vars_container, mode):
    while 1:
        for sample_index in range(0, len(samples), batch_size):
            x = np.zeros((batch_size, SQUARE_SIZE, SQUARE_SIZE, 1), dtype=np.float32)
            y = np.zeros((batch_size, 20))

            if sample_index + batch_size > len(samples):
                offset = len(samples) - sample_index
            else:
                offset = batch_size
            for index in range(sample_index, sample_index + offset):
                nc_file = samples[index][0].nc_file
                conc, thic = vars_container.values(nc_file)
                conc_square = samples[index][0].ice_conc(conc)
                thic_square = samples[index][0].ice_thic(thic)
                # print(conc_square.shape)
                if mode == "conc":
                    combined = np.stack(arrays=[conc_square], axis=2)
                elif mode == "thic":
                    combined = np.stack(arrays=[thic_square], axis=2)
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


def big_grid():
    data = read_samples("samples/ice_samples.csv")

    train, test = split_data(data.samples, 0.2)

    # train_middle = int(len(train) / 250)
    # test_middle = int(len(test) / 100)
    # train_batch_size = calc_batch_size(len(train), int(train_middle - 0.5 * train_middle),
    #                                    int(train_middle + 0.5 * train_middle))
    # test_batch_size = calc_batch_size(len(test), int(test_middle - 0.5 * test_middle),
    #                                   int(test_middle + 0.5 * test_middle))

    train_batch_size = 50
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

    mask_file = NCFile("samples/bathy_meter_mask.nc")
    coastline_mask = mask_file.variables['Bathymetry'][:]
    mask_file.close()

    epochs = 60

    container = VarsContainer()

    model = init_model()
    history = AccuracyHistory()
    model.fit_generator(data_generator(tr_samples, train_batch_size, container, coastline_mask),
                        steps_per_epoch=train_batch_size,
                        callbacks=[history],
                        epochs=epochs)
    model.save("samples/model.h5")
    # model = load_model("samples/model.h5")
    # scores = model.predict_generator(data_generator(tt_samples, test_batch_size, d),
    #                                  steps=int(len(test) / test_batch_size))

    t = model.evaluate_generator(data_generator(tt_samples, test_batch_size, container, coastline_mask),
                                 steps=int(len(test) / test_batch_size))
    print(t)
    # print(scores)


def small_grid():
    data = read_samples("samples/ice_samples_small_grid.csv")
    train, test = split_data(data.samples, 0.2)

    print(len(train))
    print(len(test))
    train_batch_size = 80
    test_batch_size = 80

    train_idx = []
    for sample in train:
        train_idx.append(sample.index - 1)
    train_idx = keras.utils.to_categorical(train_idx, 176)

    test_idx = []
    for sample in test:
        test_idx.append(sample.index - 1)
    test_idx = keras.utils.to_categorical(test_idx, 176)

    tr_samples = []
    for idx in range(len(train)):
        tr_samples.append([train[idx], train_idx[idx]])

    tt_samples = []
    for idx in range(len(test)):
        tt_samples.append([test[idx], test_idx[idx]])

    mask_file = NCFile("samples/bathy_meter_mask.nc")
    coastline_mask = mask_file.variables['Bathymetry'][:]
    mask_file.close()

    epochs = 20

    container = VarsContainer()

    model = init_model()
    history = AccuracyHistory()
    model.fit_generator(data_generator(tr_samples, train_batch_size, container, coastline_mask),
                        steps_per_epoch=train_batch_size,
                        callbacks=[history],
                        epochs=epochs)
    model.save("samples/small_grid_model.h5")
    # model = load_model("samples/model.h5")
    # scores = model.predict_generator(data_generator(tt_samples, test_batch_size, d),
    #                                  steps=int(len(test) / test_batch_size))

    t = model.evaluate_generator(data_generator(tt_samples, test_batch_size, container, coastline_mask),
                                 steps=int(len(test) / test_batch_size))
    print(t)
    # print(scores)


def ocean_only():
    month = "12"
    data = read_samples("samples/sat_csvs/sat_" + month + ".csv")
    train, test = split_data(data.samples, 0.0)
    print(len(train))
    print(len(test))
    train_batch_size = 50
    train_idx = []
    for sample in train:
        train_idx.append(squares.index(sample.index - 1))
    train_idx = keras.utils.to_categorical(train_idx, len(squares))

    tr_samples = []
    for idx in range(len(train)):
        tr_samples.append([train[idx], train_idx[idx]])

    epochs = 40
    mode = "conc"
    container = VarsContainer()

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    set_session(tf.Session(config=config))

    # model = init_advanced_ocean_model(11)
    model = VGG(20)
    history = AccuracyHistory()
    model.fit_generator(ocean_data_generator(tr_samples, train_batch_size, container, mode),
                        steps_per_epoch=train_batch_size,
                        callbacks=[history],
                        epochs=epochs)
    model.save("samples/sat_csvs/" + mode + month + "_model.h5")
    count_predictions(month)


ocean_only()
