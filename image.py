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
    u = []
    v = []

    output_dir = "samples/out/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        image = NCImage(input_dir + file_name)
        square_index = 1
        squares_amount = 44
        for x in range(0, 1100, square_size):
            for y in range(0, 400, square_size):
                square = image.extract_square('u', x, y, square_size)
                square_name = generate_square_name(file_name, 'u', x, y)
                save_square_to_file(square, output_dir + square_name)
                u.append(square)
                square = image.extract_square('v', x, y, square_size)
                square_name = generate_square_name(file_name, 'v', x, y)
                save_square_to_file(square, output_dir + square_name)
                v.append(square)
                print("squares: " + str(square_index) + "/" + str(squares_amount) + " done")
                square_index += 1
        # TODO: improve logging
        print("image: " + str(index) + "/" + str(amount) + " done")
        index += 1

    return [u, v]

slice_uv_squares("samples/data/")
# image = NCImage("samples/uv_sample_with_outliers.nc")

# square = image.extract_square('vozocrtx', 0, 0, 50, 0, 0)
# save_square_to_file(square, 'samples/test')

# test = load_square_from_file('samples/test')
