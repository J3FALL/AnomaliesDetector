import os

import numpy as np
from netCDF4 import Dataset


def save_square_to_file(square, file_name):
    np.save(file_name + '.npy', square)


def load_square_from_file(file_name):
    square = np.load(file_name + '.npy')
    return square


class NCImage:
    def __init__(self, file_name):
        self.file_name = file_name
        self.dataset = Dataset(filename=file_name, mode='r')

    def extract_variable(self, var):
        return np.array(self.dataset.variables[var])

    def extract_square(self, var_name, x_offset, y_offset, size, depth=0, time=0):
        return self.extract_variable(var_name)[time, depth, x_offset:x_offset + size, y_offset:y_offset + size]

    def extract_square_from_variable(self, var, x_offset, y_offset, size, depth=0, time=0):
        return var[time, depth, x_offset:x_offset + size, y_offset:y_offset + size]


def generate_square_name(file_name, var, x, y):
    return '_'.join((file_name.split('.')[0], var, str(x), str(y)))


def slice_uv_squares(input_dir):
    index = 1
    amount = len(os.listdir(input_dir))

    square_size = 100

    output_dir = "samples/out/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        image = NCImage(input_dir + file_name)
        square_index = 1
        squares_amount = 44
        for x in range(0, 1100, square_size):
            for y in range(0, 400, square_size):
                square = image.extract_square('vomecrty', x, y, square_size)
                square_name = generate_square_name(file_name, 'vomecrty', x, y)
                save_square_to_file(square, output_dir + square_name)
                square = image.extract_square('vozocrtx', x, y, square_size)
                square_name = generate_square_name(file_name, 'vozocrtx', x, y)
                save_square_to_file(square, output_dir + square_name)
                print("squares: " + str(square_index) + "/" + str(squares_amount) + " done")
                square_index += 1
        # TODO: improve logging
        print("image: " + str(index) + "/" + str(amount) + " done")
        index += 1
