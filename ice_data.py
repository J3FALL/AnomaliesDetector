import csv
import os

import keras
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
        x, y = divmod(self.index, 11)
        return x, y

    def ice_conc(self, var):
        # nc = NCFile(self.nc_file)
        ice = var[self.time][self.x:self.x + self.size, self.y:self.y + self.size]
        # nc.close()

        return ice

    def ice_thic(self):
        nc = NCFile(self.nc_file)
        thic = nc.variables['icethic_cea'][:][self.time][self.x:self.x + self.size, self.y:self.y + self.size]
        nc.close()

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


construct_ice_dataset()
