import glob

import matplotlib.pyplot as plt
# matplotlib.use('agg')
import numpy as np
import scipy.stats as st
from netCDF4 import Dataset as NCFile

import plot_utils
from copula import Copula
from dist import best_fit_distribution
from ice_data import Dataset
from ice_data import IMAGE_SIZE
from ice_data import IceSample
from ice_data import SQUARE_SIZE
from ice_data import is_inside


def create_dataset(dir, month, squares):
    dataset = Dataset(dir + month + "#" + str(squares[0]) + "-" + str(squares[-1]) + ".csv")
    data_dir = "D:/ice_recovered_from_hybrid/conc_satellite/"
    # data_dir = "samples/conc_satellite/"
    for nc_file in glob.iglob(data_dir + "*/" + month + "/*.nc", recursive=True):
        idx = 0
        for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
            for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
                if is_inside(x, y):
                    if idx in squares:
                        dataset.samples.append(IceSample(nc_file, idx, SQUARE_SIZE, 0, 0, x, y))
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
    # avg_0 = np.asarray(avg_of_square("samples/pm/01#0.csv"), dtype=np.float32)
    # avg_1 = np.asarray(avg_of_square("samples/pm/01#1.csv"), dtype=np.float32)

    x = np.random.randint(100, size=1000)
    y = np.random.randint(100, size=1000)

    print(x.shape)
    print(y.shape)

    print(x.size is y.size)
    # print(avg_0.size)
    # print(avg_1.size)

    frank = Copula(x, y, family='frank')
    uf, vf = frank.generate_uv(1000)

    # plt.figure()
    plt.scatter(uf, vf, marker='.', color='blue')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title('Frank copula')
    plt.savefig('samples/pm/test_copula.png')

    # plot = sns.distplot(avg)
    # plot.figure.savefig("samples/pm/01#0" + "_distr.png")


def conditional_probs(dataset_file):
    dataset = Dataset.from_csv(dataset_file)

    x_idx, y_idx = 3, 9

    intervals = np.linspace(0, 1, 11)
    x_intervals = [(intervals[idx], intervals[idx + 1]) for idx in range(0, len(intervals) - 1)]

    samples_grouped = dict()
    for sample in dataset.samples:
        if sample.index in (x_idx, y_idx):
            nc = NCFile(sample.nc_file)
            conc = nc.variables['ice_conc'][:].filled(0) / 100.0
            avg = np.average(sample.ice_conc(conc))
            if sample.nc_file not in samples_grouped:
                samples_grouped[sample.nc_file] = dict()
                samples_grouped[sample.nc_file][sample.index] = avg
            else:
                samples_grouped[sample.nc_file][sample.index] = avg

    x_count = []
    y_count = [[] for _ in range(len(x_intervals))]
    for file in samples_grouped.keys():
        x_value = samples_grouped[file][x_idx]
        x_count.append(x_value)

        y_value = samples_grouped[file][y_idx]

        interval = interval_idx(x_value, x_intervals)
        y_count[interval].append(y_value)

    x_dist_params = best_fit_distribution(x_count)
    y_dist_params = []

    for idx in range(len(y_count)):
        y_dist_params.append(best_fit_distribution(y_count[idx]))

    return x_dist_params, y_dist_params

    # x_best_dist = getattr(st, x_dist[0])
    # x_arg = x_dist[1]
    # print(x_best_dist.pdf(x_count, *x_arg))
    # print(y_dist)


def dump_dist_params(name, dist_x, dist_y):
    np.save("samples/pm/" + name + ".npy", np.asarray([dist_x, dist_y]))


def load_dist_params(name):
    dists = np.load("samples/pm/" + name + ".npy")

    return dists[0], dists[1]


def interval_idx(value, intervals):
    for idx in range(len(intervals)):
        left, right = intervals[idx]

        if left <= value <= right:
            return idx


def join_distribution(params_x, params_y, x, y):
    intervals = np.linspace(0, 1, 11)
    x_intervals = [(intervals[idx], intervals[idx + 1]) for idx in range(0, len(intervals) - 1)]

    dist_x = getattr(st, params_x[0])
    idx = interval_idx(x, x_intervals)
    dist_y = getattr(st, params_y[idx][0])

    pdf = dist_x.pdf(x, *params_x[1]) * dist_y.pdf(y, *params_y[idx][1])

    return pdf


def plot_surface():
    params_x, params_y = load_dist_params("sept_dist_another")
    dist_x = getattr(st, params_x[0])

    u = np.linspace(0.1, 0.9, 10)
    UU = np.meshgrid(u, u)
    U2 = np.reshape(UU[0], (UU[0].shape[0] * UU[0].shape[1], 1))
    U1 = np.reshape(UU[1], (UU[1].shape[0] * UU[1].shape[1], 1))
    U = np.concatenate((U1, U2), axis=1)

    z = []
    for point in U:
        z.append(join_distribution(params_x=params_x, params_y=params_y, x=point[0], y=point[1]))

    X = UU[0]
    Y = UU[1]
    Z = np.reshape(z, UU[0].shape)
    plot_utils.plot_3d(X, Y, Z, 'September PDF')

    # start_x = dist_x.ppf(0.01, *params_x[1])
    # end_x = dist_x.ppf(0.99, *params_x[1])
    #
    # start_dist_y = getattr(st, params_y[-2][0])
    # end_dist_y = getattr(st, params_y[-1][0])
    #
    # start_y = start_dist_y.ppf(0.01, *params_y[-2][1])
    # end_y = end_dist_y.ppf(0.99, *params_y[-1][1])
    # x = np.linspace(start_x, end_x, 10000)
    # y = np.linspace(start_y, end_y, 10000)


def pdf_on_good():
    data_dir = "D:/ice_recovered_from_hybrid/ice_tests/good/"
    sq3 = []
    sq9 = []

    for nc_file in glob.iglob(data_dir + "2013/" + "*201309*.nc", recursive=True):
        idx = 0
        for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
            for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
                if is_inside(x, y):
                    if idx == 3:
                        nc = NCFile(nc_file)
                        conc = nc.variables['iceconc'][:][0]
                        avg = np.average(conc[y:y + 100, x:x + 100])
                        sq3.append(avg)
                        nc.close()
                    if idx == 9:
                        nc = NCFile(nc_file)
                        nc.variables['iceconc'][:][0]
                        avg = np.average(conc[y:y + 100, x:x + 100])
                        sq9.append(avg)
                        nc.close()
                    idx += 1
    pdf = []
    for i in range(len(sq3)):
        params_x, params_y = load_dist_params("sept_dist_another")
        pdf.append(join_distribution(params_x=params_x, params_y=params_y, x=sq3[i], y=sq9[i]))

    pdf_good_avg = np.average(pdf)
    print(pdf_good_avg)


def pdf_on_bad():
    data_dir = "D:/ice_recovered_from_hybrid/ice_tests/bad/"
    sq3 = []
    sq9 = []

    for nc_file in glob.iglob(data_dir + "*/" + "*201309*.nc", recursive=True):
        idx = 0
        for y in range(0, IMAGE_SIZE['y'], SQUARE_SIZE):
            for x in range(0, IMAGE_SIZE['x'], SQUARE_SIZE):
                if is_inside(x, y):
                    if idx == 3:
                        nc = NCFile(nc_file)
                        conc = nc.variables['iceconc'][:][0]
                        avg = np.average(conc[y:y + 100, x:x + 100])
                        sq3.append(avg)
                        nc.close()
                    if idx == 9:
                        nc = NCFile(nc_file)
                        nc.variables['iceconc'][:][0]
                        avg = np.average(conc[y:y + 100, x:x + 100])
                        sq9.append(avg)
                        nc.close()
                    idx += 1
    pdf = []
    for i in range(len(sq3)):
        params_x, params_y = load_dist_params("sept_dist_another")
        pdf.append(join_distribution(params_x=params_x, params_y=params_y, x=sq3[i], y=sq9[i]))

    pdf_bad_avg = np.average(pdf)
    print(pdf_bad_avg)


plot_surface()

# pdf_on_good()
# pdf_on_bad()

# x, y = conditional_probs("samples/pm/01#0-1.csv")

# join_distribution(params_x=x, params_y=y, x=0.9, y=0.9)

# create_dataset("samples/pm/", "09", [3, 9])
# x, y = conditional_probs("samples/pm/09#3-9.csv")
# dump_dist_params("sept_dist_another", x, y)
# avg_of_square("samples/pm/01#0.csv")
# plot_distribution()
