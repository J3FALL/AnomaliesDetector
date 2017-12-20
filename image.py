import numpy as np
from netCDF4 import Dataset


class NCImage:
    def __init__(self, file_name):
        self.file_name = file_name
        self.dataset = Dataset(filename=file_name, mode='r')

    def extract_variable(self, var):
        return np.array(self.dataset.variables[var])

    def extract_square(self, var, x_offset, y_offset, size, depth=0, time=0):

        return self.extract_variable(var)[time, depth, x_offset:x_offset + size, y_offset:y_offset + size]


image = NCImage("samples/uv_sample_with_outliers.nc")

square = image.extract_square('vozocrtx', 0, 0, 50, 0, 0)
print(square)
