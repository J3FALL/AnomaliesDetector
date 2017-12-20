import numpy as np
from netCDF4 import Dataset


class NCImage:
    def __init__(self, file_name):
        self.file_name = file_name
        self.dataset = Dataset(filename=file_name, mode='r')

    def extract_variable(self, var):
        return np.array(self.dataset.variables[var])


image = NCImage("samples/uv_sample_with_outliers.nc")

print(image.extract_variable('vozocrtx'))

