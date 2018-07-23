import csv
import glob
import os

import matplotlib

matplotlib.use('agg')

import operator
import keras
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.patches import Polygon, Patch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset as NCFile
from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
import tensorflow as tf

SQUARE_SIZE = 150
IMAGE_SIZE = {
    'x': 1100,
    'y': 400
}

# x: [from, to], y: [from, to]
BORDERS = [
    {'x': [100, 800],
     'y': [0, 200]},
    {'x': [200, 800],
     'y': [200, 300]}
]


def is_inside(x, y):
    for border in BORDERS:
        x_border = border['x']
        y_border = border['y']

        if x_border[0] <= x < x_border[1] and y_border[0] <= y < y_border[1]:
            return True

    return False


class Dataset:
    def __init__(self, file_name):
        self.file_name = file_name

        self.samples = []

    def dump_to_csv(self):
        with open(self.file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for sample in self.samples:
                writer.writerow(sample.raw_data())

    @staticmethod
    def from_csv(file_name):
        samples = []
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                samples.append(IceSample.from_raw_data(row))

        dataset = Dataset(file_name)
        dataset.samples = samples

        return dataset


class IceSample:
    def __init__(self, nc_file, index, size, time, label, x, y):
        self.nc_file = nc_file
        self.index = index
        self.size = size
        self.time = time

        self.x = x
        self.y = y

        # 0 - not-outlier
        self.label = label

    def get_borders(self):
        x, y = divmod(self.index - 1, int(IMAGE_SIZE['x'] / SQUARE_SIZE))
        return x, y

    def ice_conc(self, var):
        # nc = NCFile(self.nc_file)
        ice = var[self.time][self.y:self.y + self.size, self.x:self.x + self.size]
        # nc.close()

        return ice

    def ice_thic(self, var):
        # nc = NCFile(self.nc_file)
        thic = var[self.time][self.x * self.size:self.x * self.size + self.size,
               self.y * self.size:self.y * self.size + self.size]
        # nc.close()

        return thic

    def one_hot_index(self, num_classes=44):
        return keras.utils.to_categorical(self.index - 1, num_classes)

    def raw_data(self):
        return [str(self.nc_file), str(self.index), str(self.size), str(self.time), str(self.label), str(self.x),
                str(self.y)]

    @staticmethod
    def from_raw_data(raw):
        return IceSample(raw[0], int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4]), int(raw[5]), int(raw[6]))


class IceDetector:
    def __init__(self, alpha, month):
        self.alpha = alpha

        self.model = VGG(10)
        self.model.load_weights("samples/sat_with_square_sizes/150/conc" + month + "_model.h5")

        # ocean squares
        self.squares = [*list(range(1, 8)), *list(range(12, 19)), *list(range(24, 30))]
        # similar squares for september
        d = load_zones(month)
        self.similar = d
        print(self.similar)

    def detect(self, file_name):
        nc = NCFile(file_name)
        if "sat" in file_name or "SAT" in file_name:
            conc = nc.variables['ice_conc'][:].filled(0) / 100.0
            conc = conc[0]
        elif "ease2" in file_name:
            conc = nc.variables['ice_conc'][:][0]
        else:
            conc = nc.variables['iceconc'][:][0]
            # conc = ndimage.uniform_filter(conc, 10)

        nc.close()

        square = 0
        good_amount = 0

        # TODO: add list as params [conc, thic, etc]
        for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
            for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
                if is_inside(x, y):
                    sample = np.zeros((1, SQUARE_SIZE, SQUARE_SIZE, 1))
                    conc_square = conc[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]
                    combined = np.stack(
                        arrays=[conc_square],
                        axis=2)
                    sample[0] = combined
                    result = self.model.predict(sample)
                    predicted_index = np.argmax(result[0])

                    if predicted_index in self.similar[square]:
                        good_amount += 1

                    square += 1

        prediction = 1 if good_amount / square > self.alpha else 0

        return prediction, good_amount / square

    def is_outlier(self, predicted_idx, real_idx):
        out = True
        if predicted_idx != real_idx:
            for level in self.levels:
                if predicted_idx in level and real_idx in level:
                    out = False
        else:
            out = False

        return out


def construct_ice_dataset_ocean_only():
    dataset = Dataset("samples/ice_samples_ocean_only.csv")

    data_dir = "samples/ice_data/"

    size = 50
    times_amount = 24
    squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
               *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
               *list(range(139, 143))
               ]

    for nc_file in glob.iglob(data_dir + "**/*.nc", recursive=True):
        # open NetCDF, slice it to samples with size = (100, 100)
        # each square contains data for [0..24] hours
        for square_index in squares:
            for time in range(times_amount):
                dataset.samples.append(IceSample(nc_file, square_index + 1, size, time, 0))
    dataset.dump_to_csv()


def construct_ice_dataset():
    dataset = Dataset("samples/sat_only.csv")
    data_dir = "samples/ice_data/"

    squares = [*list(range(1, 7)), *list(range(12, 18)), *list(range(24, 29))]
    print(squares)
    times = [0]
    for nc_file in glob.iglob(data_dir + "**/*.nc", recursive=True):
        for square_index in squares:
            for time in times:
                if "satellite" in nc_file:
                    if time == 0:
                        dataset.samples.append(IceSample(nc_file, square_index + 1, SQUARE_SIZE, 0, 0))
                # else:
                #     dataset.samples.append(IceSample(nc_file, square_index + 1, SQUARE_SIZE, time, 0))
    dataset.dump_to_csv()


def construct_ice_sat_dataset(month, dir):
    dataset = Dataset(dir + "sat_" + month + ".csv")
    data_dir = "samples/conc_satellite/"

    squares = []
    for nc_file in glob.iglob(data_dir + "*/" + month + "/*.nc", recursive=True):
        square = 0
        for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
            for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
                if is_inside(x, y):
                    dataset.samples.append(IceSample(nc_file, square, SQUARE_SIZE, 0, 0, x, y))
                    if square not in squares:
                        squares.append(square)
                    square += 1

    # if month == "09":
    #     nemo_dir = "samples/NEMO_good_for_September/"
    #     for nc_file in glob.iglob(nemo_dir + "**/*.nc", recursive=True):
    #         square = 0
    #         for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
    #             for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
    #                 if is_inside(x, y):
    #                     dataset.samples.append(IceSample(nc_file, square, SQUARE_SIZE, 0, 0, x, y))
    #                     square += 1
    dataset.dump_to_csv()


def sat_dataset_full_year(dir="samples/sat_csvs/"):
    months = [str(idx) for idx in range(1, 13)]

    for month in months:
        if len(month) < 2:
            month = "0" + month
        construct_ice_sat_dataset(month, dir)


def draw_ice_data(file_name):
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]
    conc = nc.variables['iceconc'][:][0]
    thic = nc.variables['icethic_cea'][:][0]
    mask_file = NCFile("samples/bathy_meter_mask.nc")
    coastline_mask = mask_file.variables['Bathymetry'][:]
    mask_file.close()

    nc.close()

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, thic, latlon=True, cmap='jet')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    model = load_model("samples/model.h5")

    real_idx = 0
    for y in range(0, 400, 100):
        for x in range(0, 1100, 100):
            sample = np.zeros((1, 100, 100, 3))
            combined = np.stack(
                arrays=[conc[y:y + 100, x:x + 100], thic[y:y + 100, x:x + 100], coastline_mask[y:y + 100, x:x + 100]],
                axis=2)
            sample[0] = combined
            result = model.predict(sample)
            predicted_index = np.argmax(result[0])
            result_x, result_y = m(lon[y + 50][x + 50], lat[y + 50][x + 50])
            plt.text(result_x, result_y, str(predicted_index), ha='center', size=10, color="yellow")
            result_x, result_y = m(lon[y + 70][x + 50], lat[y + 70][x + 50])
            plt.text(result_x, result_y, str(real_idx), ha='center', size=10, color="yellow")
            result_x, result_y = m(lon[y + 90][x + 50], lat[y + 90][x + 50])
            plt.text(result_x, result_y, str(result[0][predicted_index]), ha='center', size=10, color="yellow")
            lat_poly = np.array([lat[y][x], lat[y][x + 99], lat[y + 99][x + 99], lat[y + 99][x]])
            lon_poly = np.array([lon[y][x], lon[y][x + 99], lon[y + 99][x + 99], lon[y + 99][x]])
            mapx, mapy = m(lon_poly, lat_poly)
            points = np.zeros((4, 2), dtype=np.float32)
            for j in range(0, 4):
                points[j][0] = mapx[j]
                points[j][1] = mapy[j]
            poly = Polygon(points, edgecolor='black', alpha=0.5)
            plt.gca().add_patch(poly)

            real_idx += 1

    plt.colorbar()
    plt.title(file_name)

    plt.show()


def draw_ice_small_grid(file_name):
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]
    conc = nc.variables['iceconc'][:][0]
    thic = nc.variables['icethic_cea'][:][0]
    mask_file = NCFile("samples/bathy_meter_mask.nc")
    coastline_mask = mask_file.variables['Bathymetry'][:]
    mask_file.close()

    nc.close()

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, thic, latlon=True, cmap='jet')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    model = load_model("samples/small_grid_model.h5")

    real_idx = 0
    for y in range(0, 400, 50):
        for x in range(0, 1100, 50):
            sample = np.zeros((1, 50, 50, 3))
            combined = np.stack(
                arrays=[conc[y:y + 50, x:x + 50], thic[y:y + 50, x:x + 50], coastline_mask[y:y + 50, x:x + 50]],
                axis=2)
            sample[0] = combined
            result = model.predict(sample)
            predicted_index = np.argmax(result[0])
            result_x, result_y = m(lon[y + 15][x + 25], lat[y + 15][x + 25])
            plt.text(result_x, result_y, str(predicted_index), ha='center', size=7, color="yellow")
            result_x, result_y = m(lon[y + 30][x + 25], lat[y + 30][x + 25])
            plt.text(result_x, result_y, str(real_idx), ha='center', size=7, color="yellow")
            result_x, result_y = m(lon[y + 45][x + 25], lat[y + 45][x + 25])
            plt.text(result_x, result_y, str(result[0][predicted_index]), ha='center', size=7, color="yellow")
            lat_poly = np.array([lat[y][x], lat[y][x + 49], lat[y + 49][x + 49], lat[y + 49][x]])
            lon_poly = np.array([lon[y][x], lon[y][x + 49], lon[y + 49][x + 49], lon[y + 49][x]])
            mapx, mapy = m(lon_poly, lat_poly)
            points = np.zeros((4, 2), dtype=np.float32)
            for j in range(0, 4):
                points[j][0] = mapx[j]
                points[j][1] = mapy[j]
            poly = Polygon(points, edgecolor='black', alpha=0.5)
            plt.gca().add_patch(poly)

            real_idx += 1

    plt.colorbar()
    plt.title(file_name)

    plt.show()


def MLP(num_squares):
    input_shape = (SQUARE_SIZE, SQUARE_SIZE, 1)
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
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


def draw_ice_ocean_only(file_name):
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]
    # month = "02"
    month = select_month(file_name)

    print("month: ", month)
    if "sat" in file_name:
        conc = nc.variables['ice_conc'][:].filled(0) / 100.0
        conc = conc[0]
        thic = np.empty((1, 400, 100), np.float32)
    elif "ease2" in file_name:
        conc = nc.variables['ice_conc'][:][0]
    else:
        conc = nc.variables['iceconc'][:][0]
        thic = nc.variables['icethic_cea'][:][0]

    nc.close()
    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, conc, latlon=True, cmap='RdYlBu_r', vmax=1)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    K.set_session(tf.Session(config=config))

    model = VGG(20)
    model.load_weights("samples/sat_csvs/conc" + month + "_model.h5")

    # model.load_weights("samples/sat_with_square_sizes/50/conc08_model.h5")
    # squares = [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26, 27, 28, 29]

    d = load_zones(month)
    similar = d
    print(d)
    square = 0
    for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
        for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
            if is_inside(x, y):
                sample = np.zeros((1, SQUARE_SIZE, SQUARE_SIZE, 1))

                combined = np.stack(
                    arrays=[conc[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]],
                    axis=2)
                sample[0] = combined
                result = model.predict(sample)
                predicted_index = np.argmax(result[0])

                y_offset = int(SQUARE_SIZE / 4)
                x_offset = int(SQUARE_SIZE / 2)
                result_x, result_y = m(lon[y + y_offset][x + x_offset], lat[y + y_offset][x + x_offset])
                plt.text(result_x, result_y, str(square), ha='center', size=7, color="yellow",
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
                result_x, result_y = m(lon[y + 2 * y_offset][x + x_offset], lat[y + 2 * y_offset][x + x_offset])
                plt.text(result_x, result_y, str(predicted_index), ha='center', size=7, color="yellow",
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
                result_x, result_y = m(lon[y + 3 * y_offset][x + x_offset], lat[y + 3 * y_offset][x + x_offset])
                plt.text(result_x, result_y, str(round(result[0][predicted_index], 3)), ha='center', size=7,
                         color="yellow", path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

                lat_poly = np.array(
                    [lat[y][x], lat[y][x + SQUARE_SIZE - 1], lat[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lat[y + SQUARE_SIZE - 1][x]])
                lon_poly = np.array(
                    [lon[y][x], lon[y][x + SQUARE_SIZE - 1], lon[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lon[y + SQUARE_SIZE - 1][x]])
                mapx, mapy = m(lon_poly, lat_poly)
                points = np.zeros((4, 2), dtype=np.float32)
                for j in range(0, 4):
                    points[j][0] = mapx[j]
                    points[j][1] = mapy[j]

                # check zones
                if predicted_index not in similar[square]:
                    # poly = Polygon(points, color='red', fill=False, linewidth=3)
                    # plt.gca().add_patch(poly)
                    if y >= 200:
                        poly = Polygon(points, color='yellow', fill=False, linewidth=3)
                        plt.gca().add_patch(poly)
                    else:
                        poly = Polygon(points, color='red', fill=False, linewidth=3)
                        plt.gca().add_patch(poly)
                else:
                    if predicted_index == square:
                        poly = Polygon(points, color='green', fill=False, linewidth=3)
                        plt.gca().add_patch(poly)
                    else:
                        poly = Polygon(points, color='yellow', fill=False, linewidth=3)
                        plt.gca().add_patch(poly)

                square += 1
    # plt.rcParams["figure.figsize"] = [5, 5]
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(cax=cax, label="Ice concentration")
    # plt.title("Outlier detection results for : ARCTIC_1h_ice_grid_TUV_20130912-20130912.nc_1.nc", fontsize=10, loc='left')

    red = Patch(color='red', label='Anomaly')
    yellow = Patch(color='yellow', label='Rather correct')
    green = Patch(color='green', label='Correct')
    plt.legend(loc='lower right', fontsize='medium', bbox_to_anchor=(1, 1), handles=[green, yellow, red])
    # plt.legend(loc='lower right', fontsize='medium', handles=[green, yellow, red])
    plt.savefig(file_name + "_bad_result.png", dpi=500)

    K.clear_session()


def draw_ice_zones(file_name):
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]

    if "SAT" in file_name:
        conc = nc.variables['ice_conc'][:].filled(0) / 100.0
        conc = conc[0]
        thic = np.empty((1, 400, 100), np.float32)
    else:
        conc = nc.variables['iceconc'][:][0]
        thic = nc.variables['icethic_cea'][:][0]

    nc.close()

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, conc, latlon=True, cmap='RdYlBu_r', vmax=1)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    squares = [*list(range(1, 8)), *list(range(12, 19)), *list(range(24, 30))]

    real_idx = 0
    for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
        for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
            if real_idx in squares:
                y_offset = int(SQUARE_SIZE / 4)
                x_offset = int(SQUARE_SIZE / 2)
                result_x, result_y = m(lon[y + y_offset][x + x_offset], lat[y + y_offset][x + x_offset])
                plt.text(result_x, result_y, str(squares.index(real_idx)), ha='center', size=10, color="yellow",
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

                lat_poly = np.array(
                    [lat[y][x], lat[y][x + SQUARE_SIZE - 1], lat[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lat[y + SQUARE_SIZE - 1][x]])
                lon_poly = np.array(
                    [lon[y][x], lon[y][x + SQUARE_SIZE - 1], lon[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lon[y + SQUARE_SIZE - 1][x]])
                mapx, mapy = m(lon_poly, lat_poly)
                points = np.zeros((4, 2), dtype=np.float32)
                for j in range(0, 4):
                    points[j][0] = mapx[j]
                    points[j][1] = mapy[j]
                poly = Polygon(points, color='green', fill=False, linewidth=3)
                plt.gca().add_patch(poly)

            real_idx += 1

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(cax=cax, label="Ice conc")
    plt.show()


def show_detection_results(file_name):
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]

    conc = nc.variables['iceconc'][:][0]
    # conc = nc.variables['ice_conc'][:].filled(0) / 100.0
    # conc = conc[0]
    nc.close()

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, conc, latlon=True, cmap='RdYlBu_r', vmax=1)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    squares = [*list(range(1, 8)), *list(range(12, 19)), *list(range(24, 30))]

    model = load_model("samples/sat_csvs/conc_model.h5")

    real_idx = 0
    for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
        for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
            if real_idx in squares:
                sample = np.zeros((1, SQUARE_SIZE, SQUARE_SIZE, 1))

                combined = np.stack(
                    arrays=[conc[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]],
                    axis=2)
                sample[0] = combined
                result = model.predict(sample)
                predicted_index = np.argmax(result[0])
                print(predicted_index)
                y_offset = int(SQUARE_SIZE / 4)
                x_offset = int(SQUARE_SIZE / 2)

                result_x, result_y = m(lon[y + y_offset][x + x_offset], lat[y + y_offset][x + x_offset])
                plt.text(result_x, result_y, str(predicted_index), ha='center', size=5, color="yellow",
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
                result_x, result_y = m(lon[y + 2 * y_offset][x + x_offset], lat[y + 2 * y_offset][x + x_offset])
                plt.text(result_x, result_y, str(squares.index(real_idx)), ha='center', size=5, color="yellow",
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
                result_x, result_y = m(lon[y + 3 * y_offset][x + x_offset], lat[y + 3 * y_offset][x + x_offset])
                plt.text(result_x, result_y, str(round(result[0][predicted_index], 3)), ha='center', size=5,
                         color="yellow", path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

                lat_poly = np.array(
                    [lat[y][x], lat[y][x + SQUARE_SIZE - 1], lat[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lat[y + SQUARE_SIZE - 1][x]])
                lon_poly = np.array(
                    [lon[y][x], lon[y][x + SQUARE_SIZE - 1], lon[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lon[y + SQUARE_SIZE - 1][x]])
                mapx, mapy = m(lon_poly, lat_poly)
                points = np.zeros((4, 2), dtype=np.float32)
                for j in range(0, 4):
                    points[j][0] = mapx[j]
                    points[j][1] = mapy[j]

                if predicted_index == squares.index(real_idx):
                    poly = Polygon(points, color='green', fill=False, linewidth=3)
                    plt.gca().add_patch(poly)
                else:
                    poly = Polygon(points, color='red', fill=False, linewidth=3)
                    plt.gca().add_patch(poly)

            real_idx += 1

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(cax=cax, label="Ice concentration ")
    plt.savefig("samples/sat_csvs/on_train.png", dpi=500)


def draw_ice_levels(file_name):
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]
    conc = nc.variables['iceconc'][:][0]
    thic = nc.variables['icethic_cea'][:][0]
    nc.close()

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, thic, latlon=True, cmap='jet')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
               *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
               *list(range(139, 143))
               ]
    levels = [list(range(4, 7)), [3, 7, *list(range(20, 25))], [2, 8, 19, 25, *list(range(37, 44))],
              [1, 9, 18, 26, 36, 44, *list(range(53, 62))],
              [0, *list(range(10, 13)), 17, 27, 28, 29, 35, 45, 46, 62, *list(range(69, 76)), *list(range(80, 86))],
              [*list(range(14, 17)), *list(range(31, 34)), *list(range(49, 52))]]
    print(levels)
    real_idx = 0
    for y in range(0, 400, 50):
        for x in range(0, 1100, 50):
            if real_idx in squares:
                sample = np.zeros((1, 50, 50, 2))
                combined = np.stack(
                    arrays=[conc[y:y + 50, x:x + 50], thic[y:y + 50, x:x + 50]],
                    axis=2)
                sample[0] = combined
                lat_poly = np.array([lat[y][x], lat[y][x + 49], lat[y + 49][x + 49], lat[y + 49][x]])
                lon_poly = np.array([lon[y][x], lon[y][x + 49], lon[y + 49][x + 49], lon[y + 49][x]])
                mapx, mapy = m(lon_poly, lat_poly)
                points = np.zeros((4, 2), dtype=np.float32)
                for j in range(0, 4):
                    points[j][0] = mapx[j]
                    points[j][1] = mapy[j]

                if squares.index(real_idx) in levels[0]:
                    poly = Polygon(points, edgecolor='green', alpha=1)
                    plt.gca().add_patch(poly)
                if squares.index(real_idx) in levels[1]:
                    poly = Polygon(points, edgecolor='red', alpha=1)
                    plt.gca().add_patch(poly)
                if squares.index(real_idx) in levels[2]:
                    poly = Polygon(points, edgecolor='blue', alpha=1)
                    plt.gca().add_patch(poly)
                if squares.index(real_idx) in levels[3]:
                    poly = Polygon(points, edgecolor='red', alpha=1)
                    plt.gca().add_patch(poly)
                if squares.index(real_idx) in levels[4]:
                    poly = Polygon(points, edgecolor='blue', alpha=1)
                    plt.gca().add_patch(poly)
            real_idx += 1

    plt.colorbar()
    plt.title(file_name)

    plt.show()


def visualize_tree_classification(file_name):
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]
    conc = nc.variables['iceconc'][:][0]
    thic = nc.variables['icethic_cea'][:][0]
    nc.close()

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, thic, latlon=True, cmap='jet')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
               *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
               *list(range(139, 143))
               ]
    levels = [list(range(4, 7)), [3, 7, *list(range(20, 25))], [2, 8, 19, 25, *list(range(37, 44))],
              [1, 9, 18, 26, 36, 44, *list(range(53, 62))],
              [0, *list(range(10, 13)), 17, 27, 28, 29, 35, 45, 46, 62, *list(range(69, 76)), *list(range(80, 86))],
              [*list(range(14, 17)), *list(range(31, 34)), *list(range(49, 52))]]

    clf = joblib.load('samples/tree.pkl')
    real_idx = 0
    for y in range(0, 400, 50):
        for x in range(0, 1100, 50):
            if real_idx in squares:
                square_conc = conc[y:y + 50, x:x + 50]
                square_thic = thic[y:y + 50, x:x + 50]
                reshaped = np.append(square_conc.flatten(), square_thic.flatten())
                result = clf.predict([reshaped])
                predicted_index = result[0]

                result_x, result_y = m(lon[y + 15][x + 25], lat[y + 15][x + 25])
                plt.text(result_x, result_y, str(predicted_index), ha='center', size=7, color="yellow")
                result_x, result_y = m(lon[y + 30][x + 25], lat[y + 30][x + 25])
                plt.text(result_x, result_y, str(squares.index(real_idx)), ha='center', size=7, color="yellow")

                lat_poly = np.array([lat[y][x], lat[y][x + 49], lat[y + 49][x + 49], lat[y + 49][x]])
                lon_poly = np.array([lon[y][x], lon[y][x + 49], lon[y + 49][x + 49], lon[y + 49][x]])
                mapx, mapy = m(lon_poly, lat_poly)
                points = np.zeros((4, 2), dtype=np.float32)
                for j in range(0, 4):
                    points[j][0] = mapx[j]
                    points[j][1] = mapy[j]

                if predicted_index != squares.index(real_idx):
                    out = True
                    for level in levels:
                        if predicted_index in level and squares.index(real_idx) in level:
                            out = False

                    if out:
                        poly = Polygon(points, facecolor='red', alpha=0.6)
                        plt.gca().add_patch(poly)
                    else:
                        poly = Polygon(points, facecolor='yellow', alpha=0.6)
                        plt.gca().add_patch(poly)
                else:
                    poly = Polygon(points, facecolor='green', alpha=0.6)
                    plt.gca().add_patch(poly)

            real_idx += 1

    plt.colorbar()
    plt.title(file_name)

    plt.show()


def test_detector():
    good_dir = "samples/ice_tests/good/2013"
    bad_dir = "samples/ice_tests/bad/"

    samples = []

    # label good data
    for file_name in glob.iglob(good_dir + "**/*.nc", recursive=True):
        samples.append([os.path.normpath(file_name), 1])

    print(len(samples))
    # label bad data
    for file_name in glob.iglob(bad_dir + "**/*.nc", recursive=True):
        samples.append([os.path.normpath(file_name), 0])

    print(len(samples))

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    K.set_session(tf.Session(config=config))

    detector = IceDetector(0.1, "09")
    results = np.zeros((len(samples), 2))
    idx = 0
    for sample in samples:
        print(sample[0])
        pred, val = detector.detect(sample[0])
        print(str(pred) + " " + str(val))
        results[idx][0] = val
        results[idx][1] = sample[1]
        idx += 1

    # TODO: add function for roc calculation
    tpr = []
    fpr = []

    for barrier in np.arange(0.0, 1.0, 0.01):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index in range(0, len(results)):
            if results[index][0] > barrier:
                if results[index][1] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if results[index][1] == 0:
                    tn += 1
                else:
                    fn += 1
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    print(tpr)
    print(fpr)
    # fpr, tpr, _ = roc_curve(results[:, 1], results[:, 0])
    roc_auc = auc(fpr, tpr)
    print('AUC: %f' % roc_auc)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve with AUC(%0.2f)' % roc_auc)

    # plt.show()
    plt.savefig("test_150.png", dpi=500)
    K.clear_session()


def samples_distribution_by_cnn():
    good_dir = "samples/ice_tests/good/2013"
    bad_dir = "samples/ice_tests/bad/"

    good = []
    bad = []

    # label good data
    for file_name in glob.iglob(good_dir + "**/*.nc", recursive=True):
        good.append([os.path.normpath(file_name), 1])

    print(len(good))
    # label bad data
    for file_name in glob.iglob(bad_dir + "**/*.nc", recursive=True):
        bad.append([os.path.normpath(file_name), 0])

    print(len(bad))

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    K.set_session(tf.Session(config=config))

    detector = IceDetector(0.1, "09")
    good_results = []

    for sample in good:
        pred, val = detector.detect(sample[0])
        good_results.append(val)

    good_count = []
    idx = -1
    for barrier in np.arange(0, 1.0, 0.1):
        idx += 1
        good_count.append(0)
        for result in good_results:
            if result > barrier:
                good_count[idx] += 1

    bad_results = []

    for sample in bad:
        pred, val = detector.detect(sample[0])
        bad_results.append(val)

    bad_count = []
    idx = -1
    for barrier in np.arange(0, 1.0, 0.1):
        idx += 1
        bad_count.append(0)
        for result in bad_results:
            if result < barrier:
                bad_count[idx] += 1

    print(good_count)
    print(bad_count)


def samples_distribution_by_sat():
    good_dir = "samples/ice_tests/good/2013"
    bad_dir = "samples/ice_tests/bad/"

    good = []
    bad = []

    # label good data
    for file_name in glob.iglob(good_dir + "**/*.nc", recursive=True):
        good.append([os.path.normpath(file_name), 1])

    print(len(good))
    # label bad data
    for file_name in glob.iglob(bad_dir + "**/*.nc", recursive=True):
        bad.append([os.path.normpath(file_name), 0])

    print(len(bad))

    good_results = []

    for sample in good:
        val = sat_validate(sample[0])
        print(sample[0], val)
        good_results.append(val)

    good_count = []
    idx = -1
    for barrier in np.arange(0, 1.0, 0.1):
        idx += 1
        good_count.append(0)
        for result in good_results:
            if result > barrier:
                good_count[idx] += 1

    bad_results = []

    for sample in bad:
        val = sat_validate(sample[0])
        print(sample[0], val)
        bad_results.append(val)

    bad_count = []
    idx = -1
    for barrier in np.arange(0, 1.0, 0.1):
        idx += 1
        bad_count.append(0)
        for result in bad_results:
            if result < barrier:
                bad_count[idx] += 1

    print(good_count)
    print(bad_count)


def compare_nn_and_sat():
    bad_dir = "samples/ice_tests/bad/"
    bad = []

    # label bad data
    for file_name in glob.iglob(bad_dir + "**/*.nc", recursive=True):
        bad.append([os.path.normpath(file_name), 1])

    sat_results = []

    for sample in bad:
        val = sat_validate(sample[0])
        print(sample[0], val)
        sat_results.append(val)
    print("SAT results: ", sat_results)

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    K.set_session(tf.Session(config=config))

    detector = IceDetector(0.1, "09")
    cnn_results = []

    for sample in bad:
        pred, val = detector.detect(sample[0])
        print(sample[0], val)
        cnn_results.append(val)
    print("CNN results: ", cnn_results)

    cnn_treshold = 0.6
    sat_treshold = 0.5

    cnn_correct_predicted = 0
    sat_correct_predicted = 0

    for val in cnn_results:
        if val > cnn_treshold:
            cnn_correct_predicted += 1

    for val in sat_results:
        if val > sat_treshold:
            sat_correct_predicted += 1

    print("CNN: % bad correct predicted", cnn_correct_predicted / len(bad))
    print("SAT: % bad correct predicted", sat_correct_predicted / len(bad))


def tree_classification():
    good_dir = "samples/ice_tests/good/2013"
    bad_dir = "samples/ice_tests/bad/"

    files = []

    # label good data
    for file_name in glob.iglob(good_dir + "**/*.nc", recursive=True):
        files.append([os.path.normpath(file_name), 0])

    # label bad data
    for file_name in glob.iglob(bad_dir + "**/*.nc", recursive=True):
        files.append([os.path.normpath(file_name), 1])

    squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
               *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
               *list(range(139, 143))
               ]
    levels = [list(range(4, 7)), [3, 7, *list(range(20, 25))], [2, 8, 19, 25, *list(range(37, 44))],
              [1, 9, 18, 26, 36, 44, *list(range(53, 62))],
              [0, *list(range(10, 13)), 17, 27, 28, 29, 35, 45, 46, 62, *list(range(69, 76)),
               *list(range(80, 86))],
              [*list(range(14, 17)), *list(range(31, 34)), *list(range(49, 52))]]

    clf = joblib.load('samples/tree.pkl')

    real = []
    score = []

    for file in files:
        nc = NCFile(file[0])
        conc = nc.variables['iceconc'][:][0]
        thic = nc.variables['icethic_cea'][:][0]
        nc.close()

        samples = []
        labels = []

        real_idx = 0
        for y in range(0, 400, 50):
            for x in range(0, 1100, 50):
                if real_idx in squares:
                    square_conc = conc[y:y + 50, x:x + 50]
                    square_thic = thic[y:y + 50, x:x + 50]
                    reshaped = np.append(square_conc.flatten(), square_thic.flatten())
                    samples.append(reshaped)
                    labels.append(squares.index(real_idx))
                real_idx += 1

        predicted = clf.predict(samples)
        good_amount = 0
        for idx in range(len(samples)):
            predicted_index = predicted[idx]
            real_idx = labels[idx]

            good = False
            if predicted_index == real_idx:
                good = True
            else:
                for level in levels:
                    if predicted_index in level and real_idx in level:
                        good = True

            if good:
                good_amount += 1
            # out = True
            # if predicted_index != real_idx:
            #     for level in levels:
            #         if predicted_index in level and real_idx in level:
            #             out = False
            # else:
            #     out = False
            # if out:
            #     out_amount += 1

        val = good_amount / len(samples)
        real.append(file[1])
        score.append(val)
        # score.append(1.0)

    fpr, tpr, _ = roc_curve(real, score)
    roc_auc = auc(fpr, tpr)
    print('AUC: %f' % roc_auc)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve with AUC(%0.2f)' % roc_auc)

    plt.show()


def fit_tree():
    data_dir = "samples/ice_data/"

    squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
               *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
               *list(range(139, 143))
               ]

    samples = []
    labels = []
    for nc_file in os.listdir(data_dir):
        print(data_dir + nc_file)
        nc = NCFile(data_dir + nc_file)
        conc = nc.variables['iceconc'][:][0]
        thic = nc.variables['icethic_cea'][:][0]
        nc.close()
        real_idx = 0
        for y in range(0, 400, 50):
            for x in range(0, 1100, 50):
                if real_idx in squares:
                    square_conc = conc[y:y + 50, x:x + 50]
                    square_thic = thic[y:y + 50, x:x + 50]
                    reshaped = np.append(square_conc.flatten(), square_thic.flatten())
                    samples.append(reshaped)
                    labels.append(squares.index(real_idx))
                real_idx += 1

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(samples, labels)
    joblib.dump(clf, 'samples/tree.pkl')


def reduce_conc(conc):
    conc[conc < 0.4] = 0.0

    return conc


def load_mask():
    mask_file = NCFile("bathy_meter_mask.nc")
    mask = mask_file.variables['Bathymetry'][:]
    mask_file.close()

    mask = 1 - mask

    return mask


def count_predictions(model, month):
    data_dir = "samples/conc_satellite/"

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    set_session(tf.Session(config=config))

    model.load_weights("samples/sat_with_square_sizes/150/conc" + month + "_model.h5")

    count = dict()
    mask = load_mask()

    for nc_file in glob.iglob(data_dir + "*/" + month + "/*.nc", recursive=True):
        nc = NCFile(nc_file)
        conc = nc.variables['ice_conc'][:].filled(0) / 100.0
        conc = conc[0]
        conc = reduce_conc(conc)
        nc.close()
        square = 0
        print(nc_file)
        for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
            for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
                if is_inside(x, y):
                    conc_square = conc[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]
                    if square in [15, 16, 17, 18]:
                        conc_square = conc_square * mask[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]
                        # conc_square = np.full((100, 100), fill_value=0.0)
                    sample = np.zeros((1, SQUARE_SIZE, SQUARE_SIZE, 1))
                    combined = np.stack(
                        arrays=[conc_square],
                        axis=2)
                    sample[0] = combined
                    result = model.predict(sample)
                    predicted_index = np.argmax(result[0])
                    if square not in count.keys():
                        count[square] = dict()
                        count[square][predicted_index] = 1
                    else:
                        if predicted_index not in count[square].keys():
                            count[square][predicted_index] = 1
                        else:
                            count[square][predicted_index] += 1
                    square += 1
    # if month == "09":
    #     nemo_dir = "samples/NEMO_good_for_September/"
    #     for nc_file in glob.iglob(nemo_dir + "**/*.nc", recursive=True):
    #         nc = NCFile(nc_file)
    #         conc = nc.variables['iceconc'][:][0]
    #         nc.close()
    #         square = 0
    #         print(nc_file)
    #         for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
    #             for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
    #                 if is_inside(x, y):
    #                     sample = np.zeros((1, SQUARE_SIZE, SQUARE_SIZE, 1))
    #                     combined = np.stack(
    #                         arrays=[conc[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]],
    #                         axis=2)
    #                     sample[0] = combined
    #                     result = model.predict(sample)
    #                     predicted_index = np.argmax(result[0])
    #                     if square not in count.keys():
    #                         count[square] = dict()
    #                         count[square][predicted_index] = 1
    #                     else:
    #                         if predicted_index not in count[square].keys():
    #                             count[square][predicted_index] = 1
    #                         else:
    #                             count[square][predicted_index] += 1
    #                     square += 1

    filtered = count.copy()
    for key in filtered.keys():
        filtered[key] = []

    for key in count.keys():
        sorted_count = sorted(count[key].items(), key=operator.itemgetter(1))
        sorted_count.reverse()
        for pair in sorted_count:
            if pair[1] >= 50:
                filtered[key].append(pair[0])

        print(str(key), " : ", str(sorted_count))
        print("filtered : ", str(key), str(filtered[key]))

    for key in filtered.keys():
        if key not in filtered[key]:
            filtered[key].append(key)

    # save map to file
    np.save("samples/sat_with_square_sizes/150/zones_" + month + ".npy", filtered)


def load_zones(month):
    zones = np.load("samples/sat_with_square_sizes/150/zones_" + month + ".npy").item()
    return zones


def select_month(file_name):
    # ARCTIC_1h_ice_grid_TUV_YYYYMMDD-YYYYMMDD
    date = file_name.split("-")[1]
    if "satellite" in file_name:
        date = file_name.split("/")[4].split("_")[5]
    month = date[4:6]
    return month


def vis():
    nc = NCFile("samples/conc_satellite/2013/09/ice_conc_nh_ease2-250_cdr-v2p0_201309181200.nc")
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]

    # conc = nc.variables['iceconc'][:][0]
    conc = nc.variables['ice_conc'][:].filled(0) / 100.0
    conc = conc[0]
    # thic = nc.variables['icethic_cea'][:][0]
    nc.close()

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]
    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, conc, latlon=True, cmap='RdYlBu_r', vmax=1)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    squares = [*list(range(1, 8)), *list(range(12, 19)), *list(range(24, 30))]
    zones = [[1, 2, 3, 4, 9], [0, 8, 14, 15, 10, 11, 5, 18], [6, 13, 12, 19, 17, 16]]
    real_idx = 0
    for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
        for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
            if real_idx in squares:
                y_offset = int(SQUARE_SIZE / 2)
                x_offset = int(SQUARE_SIZE / 2)

                result_x, result_y = m(lon[y + y_offset][x + x_offset], lat[y + y_offset][x + x_offset])
                plt.text(result_x, result_y, str(squares.index(real_idx)), ha='center', size=10, color="yellow",
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

                lat_poly = np.array(
                    [lat[y][x], lat[y][x + SQUARE_SIZE - 1], lat[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lat[y + SQUARE_SIZE - 1][x]])
                lon_poly = np.array(
                    [lon[y][x], lon[y][x + SQUARE_SIZE - 1], lon[y + SQUARE_SIZE - 1][x + SQUARE_SIZE - 1],
                     lon[y + SQUARE_SIZE - 1][x]])
                mapx, mapy = m(lon_poly, lat_poly)
                points = np.zeros((4, 2), dtype=np.float32)
                for j in range(0, 4):
                    points[j][0] = mapx[j]
                    points[j][1] = mapy[j]

                if squares.index(real_idx) in zones[0]:
                    poly = Polygon(points, color='red', fill=False, linewidth=3)
                    plt.gca().add_patch(poly)
                if squares.index(real_idx) in zones[1]:
                    poly = Polygon(points, color='yellow', fill=False, linewidth=3)
                    plt.gca().add_patch(poly)
                if squares.index(real_idx) in zones[2]:
                    poly = Polygon(points, color='blue', fill=False, linewidth=3)
                    plt.gca().add_patch(poly)

            real_idx += 1

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(cax=cax, label="Ice concentration")
    plt.savefig("samples/conc_squares.png", dpi=500)


def test_full_year():
    dir = "samples/ice_test_bad_full_year/"

    for nc_file in glob.iglob(dir + "**/*.nc", recursive=True):
        draw_ice_ocean_only(nc_file)


def pick_same_sat_image(file_name):
    from pathlib import Path
    # ARCTIC_1h_ice_grid_TUV_YYYYMMDD-YYYYMMDD.nc
    date = file_name.split("-")[1].split(".")[0]
    year = date[0:4]
    month = date[4:6]

    sat_file_name = \
        "samples/conc_satellite/" + year + "/" + month + "/ice_conc_nh_ease2-250_cdr-v2p0_" + date + "1200.nc"

    file = Path(sat_file_name)

    if not file.exists():
        print(sat_file_name + " doesn't exists")

    return sat_file_name


def conc_dif(src_file, sat_file):
    src = NCFile(src_file)
    lat = src.variables['nav_lat'][:]
    lon = src.variables['nav_lon'][:]
    src_conc = src.variables['iceconc'][:][0]
    src.close()

    sat = NCFile(sat_file)
    sat_conc = sat.variables['ice_conc'][:].filled(0) / 100.0
    sat.close()
    dif = abs(src_conc - sat_conc[0])
    dif_field = np.full((400, 1100), fill_value=0.0)

    squares = [*list(range(1, 8)), *list(range(12, 19)), *list(range(24, 30))]
    real_idx = 0

    for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
        for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
            if real_idx in squares:
                square_dif = np.sum(dif[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]) / (100.0 * 100.0)
                dif_field[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE] = np.full((SQUARE_SIZE, SQUARE_SIZE),
                                                                          fill_value=square_dif)
            real_idx += 1

    # lat_left_bottom = lat[-1][-1]
    # lon_left_bottom = lon[-1][-1]
    # lat_right_top = lat[0][0]
    # lon_right_top = lon[0][0]
    # lat_center = 90
    # # 110, 119
    # lon_center = 110
    # m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
    #             llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
    #             urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)
    #
    # m.pcolormesh(lon, lat, dif_field, latlon=True, cmap='RdYlBu_r', vmax=0.5)
    # m.drawcoastlines()
    # m.drawcountries()
    # m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    # ax = plt.gca()
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    #
    # plt.colorbar(cax=cax, label="Ice conc diff")
    # plt.savefig("samples/conc_diff.png", dpi=500)

    return dif_field


def sat_validate(file_name):
    sat_image = pick_same_sat_image(file_name)

    dif = conc_dif(file_name, sat_image)

    squares = [*list(range(1, 8)), *list(range(12, 19)), *list(range(24, 30))]
    real_idx = 0
    good_amount = 0
    treshold = 0.25
    for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
        for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
            if real_idx in squares:
                if dif[y + int(SQUARE_SIZE / 2)][x + int(SQUARE_SIZE / 2)] < treshold:
                    good_amount += 1

            real_idx += 1
    return good_amount / len(squares)


# draw_ice_levels("samples/ice_tests/good/2013/ARCTIC_1h_ice_grid_TUV_20130925-20130925.nc_1.nc")
# draw_ice_ocean_only("samples/ice_tests/good/2013/ARCTIC_1h_ice_grid_TUV_20130928-20130928.nc_1.nc")
# construct_ice_dataset()

# draw_ice_data("samples/ice_data/bad/ARCTIC_1h_ice_grid_TUV_20130902-20130902.nc")
# construct_ice_dataset_with_small_grid()
# draw_ice_small_grid("samples/ice_bad/ARCTIC_1h_ice_grid_TUV_20120830-20120830.nc")

# construct_ice_dataset_ocean_only()

# detector = IceDetector(0.5)
# print(detector.detect("samples/ice_bad_bad/4/ARCTIC_1h_ice_grid_TUV_20010806-20010806.nc"))

#
# test_detector()

# fit_tree()
# tree_classification()
# visualize_tree_classification("samples/ice_tests/good/2013/ARCTIC_1h_ice_grid_TUV_20130925-20130925.nc_1.nc")

# construct_ice_dataset()

# draw_ice_zones("samples/avg/SAT_201312_avg.nc")

# sat_dataset_full_year()

# show_detection_results("samples/ice_tests/good/2013/ARCTIC_1h_ice_grid_TUV_20130907-20130907.nc_1.nc")
# count_predictions()

# test_detector()

# vis()
# test_full_year()
# draw_ice_ocean_only("samples/ARCTIC_1h_ice_grid_TUV_20121112-20121112.nc")

# print(sat_validate("samples/ice_tests/bad/3/ARCTIC_1h_ice_grid_TUV_20120906-20120906.nc"))

# samples_distribution_by_cnn()
# samples_distribution_by_sat()

# compare_nn_and_sat()

# sat_dataset_full_year("samples/sat_csvs/")
## construct_ice_sat_dataset('09', 'samples/sat_with_square_sizes/100/')
# sat_dataset_full_year("samples/sat_with_square_sizes/150/")

# zones = load_zones("09")
# print(zones)

test_detector()
