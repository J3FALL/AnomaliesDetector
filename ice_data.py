import csv
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset as NCFile


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
    def __init__(self, nc_file, index, size, time, label):
        self.nc_file = nc_file
        self.index = index
        self.size = size
        self.time = time

        self.x, self.y = self.get_borders()

        # 0 - not-outlier
        self.label = label

    def get_borders(self):
        x, y = divmod(self.index - 1, 22)
        return x, y

    def ice_conc(self, var):
        # nc = NCFile(self.nc_file)
        ice = var[self.time][self.x * self.size:self.x * self.size + self.size,
              self.y * self.size:self.y * self.size + self.size]
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
        return [str(self.nc_file), str(self.index), str(self.size), str(self.time), str(self.label)]

    @staticmethod
    def from_raw_data(raw):
        return IceSample(raw[0], int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4]))


def construct_ice_dataset():
    dataset = Dataset("samples/ice_samples.csv")

    data_dir = "samples/ice_data/"

    size = 100
    squares_amount = 44
    times_amount = 24

    for nc_file in os.listdir(data_dir):
        # open NetCDF, slice it to samples with size = (100, 100)
        # each square contains data for [0..24] hours
        for square_index in range(1, squares_amount + 1):
            for time in range(times_amount):
                dataset.samples.append(IceSample(data_dir + nc_file, square_index, size, time, 0))
    dataset.dump_to_csv()


def construct_ice_dataset_with_small_grid():
    dataset = Dataset("samples/ice_samples_small_grid.csv")

    data_dir = "samples/ice_data/"

    size = 50
    squares_amount = 176
    times_amount = 24

    for nc_file in os.listdir(data_dir):
        # open NetCDF, slice it to samples with size = (100, 100)
        # each square contains data for [0..24] hours
        for square_index in range(1, squares_amount + 1):
            for time in range(times_amount):
                dataset.samples.append(IceSample(data_dir + nc_file, square_index, size, time, 0))
    dataset.dump_to_csv()


def construct_ice_dataset_ocean_only():
    dataset = Dataset("samples/ice_samples_ocean_only.csv")

    data_dir = "samples/ice_data/"

    size = 50
    times_amount = 24
    squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
               *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
               *list(range(139, 143))
               ]

    for nc_file in os.listdir(data_dir):
        # open NetCDF, slice it to samples with size = (100, 100)
        # each square contains data for [0..24] hours
        for square_index in squares:
            for time in range(times_amount):
                dataset.samples.append(IceSample(data_dir + nc_file, square_index + 1, size, time, 0))
    dataset.dump_to_csv()


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


def draw_ice_ocean_only(file_name):
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

    model = load_model("samples/ocean_only_model.h5")

    squares = [*list(range(2, 19)), *list(range(24, 41)), *list(range(45, 63)),
               *list(range(68, 85)), *list(range(92, 103)), *list(range(114, 121)),
               *list(range(139, 143))
               ]
    real_idx = 0
    for y in range(0, 400, 50):
        for x in range(0, 1100, 50):
            if real_idx in squares:
                sample = np.zeros((1, 50, 50, 2))
                combined = np.stack(
                    arrays=[conc[y:y + 50, x:x + 50], thic[y:y + 50, x:x + 50]],
                    axis=2)
                sample[0] = combined
                result = model.predict(sample)
                predicted_index = np.argmax(result[0])
                result_x, result_y = m(lon[y + 15][x + 25], lat[y + 15][x + 25])
                plt.text(result_x, result_y, str(predicted_index), ha='center', size=7, color="yellow")
                result_x, result_y = m(lon[y + 30][x + 25], lat[y + 30][x + 25])
                plt.text(result_x, result_y, str(squares.index(real_idx)), ha='center', size=7, color="yellow")
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
              [0, *list(range(10, 13)), 17, 27, 28, 29, 35, 45, 46, 62, *list(range(69, 76)), *list(range(80, 86))]]
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

# draw_ice_levels("samples/ice_data/ARCTIC_1h_ice_grid_TUV_20000827-20000827.nc_1.nc")
# raw_ice_ocean_only("samples/ice_data/ARCTIC_1h_ice_grid_TUV_20000827-20000827.nc_1.nc")
# construct_ice_dataset()

# draw_ice_data("samples/ice_data/bad/ARCTIC_1h_ice_grid_TUV_20130902-20130902.nc")
# construct_ice_dataset_with_small_grid()
# draw_ice_small_grid("samples/ice_bad/ARCTIC_1h_ice_grid_TUV_20120830-20120830.nc")

# construct_ice_dataset_ocean_only()
