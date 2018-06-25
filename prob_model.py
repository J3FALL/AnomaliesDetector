import glob

import matplotlib

matplotlib.use('agg')
import seaborn as sns
import numpy as np
from netCDF4 import Dataset as NCFile

from ice_data import Dataset
from ice_data import IMAGE_SIZE
from ice_data import IceSample
from ice_data import SQUARE_SIZE
from ice_data import is_inside


def create_dataset(dir, month, square):
    dataset = Dataset(dir + month + "#" + str(square) + ".csv")
    data_dir = "samples/conc_satellite/"

    for nc_file in glob.iglob(data_dir + "*/" + month + "/*.nc", recursive=True):
        idx = 0
        for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
            for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
                if is_inside(x, y):
                    if idx == square:
                        dataset.samples.append(IceSample(nc_file, square, SQUARE_SIZE, 0, 0, x, y))
                    idx += 1

    dataset.dump_to_csv()


def avg_of_square(dataset_file):
    dataset = Dataset.from_csv(dataset_file)

    avg = []

    for sample in dataset.samples:

        if dataset.samples.index(sample) % 50 == 0:
            print("%d/%d done" % (dataset.samples.index(sample), len(dataset.samples)))
        nc = NCFile(sample.nc_file)
        conc = nc.variables['ice_conc'][:].filled(0) / 100.0
        conc_square = sample.ice_conc(conc)
        avg.append(np.average(conc_square))

    return avg


def plot_distribution():
    avg = avg_of_square("samples/pm/01#0.csv")

    plot = sns.distplot(avg)
    plot.figure.savefig("samples/pm/01#0" + "_distr.png")


plot_distribution()
