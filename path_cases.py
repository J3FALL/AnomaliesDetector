import os
import random

import keras.backend as K
import matplotlib
import tensorflow as tf

from break_my_ice import break_my_ice
from ice_data import IceDetector

matplotlib.use('agg')
import matplotlib.pyplot as plt


def init_session():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    K.set_session(tf.Session(config=config))


def full_ice(file_name):
    good = []
    for month in range(1, 13):
        init_session()
        detector = IceDetector(0.5, str(month)) if month >= 10 else IceDetector(0.5, "0" + str(month))
        pred, val = detector.detect(file_name)
        good.append(val * 100.0)
        print(month, val)

    return good


def generate_holes(size):
    dir = "samples/conc_satellite/2013/"
    holes_dir = "samples/pathological_cases/holes/" + str(size) + "/"
    for month in range(1, 13):
        month_str = str(month) if month >= 10 else "0" + str(month)
        file = random.choice(os.listdir(dir + month_str + "/"))
        from_dir = dir + month_str + "/" + file
        to_dir = holes_dir + file + '_break.nc'
        os.system('cp ' + from_dir + ' ' + to_dir)
        print(to_dir)
        break_my_ice(to_dir, 1100, 400, 'hole_' + str(size))


def generate_spots():
    dir = "samples/conc_satellite/2013/"
    holes_dir = "samples/pathological_cases/spots/"
    for month in range(1, 13):
        month_str = str(month) if month >= 10 else "0" + str(month)
        file = random.choice(os.listdir(dir + month_str + "/"))
        from_dir = dir + month_str + "/" + file
        to_dir = holes_dir + file + '_break.nc'
        os.system('cp ' + from_dir + ' ' + to_dir)
        print(to_dir)
        break_my_ice(to_dir, 1100, 400, 'spots')


def generate_noise():
    dir = "samples/conc_satellite/2013/"
    holes_dir = "samples/pathological_cases/noise/"
    for month in range(1, 13):
        month_str = str(month) if month >= 10 else "0" + str(month)
        file = random.choice(os.listdir(dir + month_str + "/"))
        from_dir = dir + month_str + "/" + file
        to_dir = holes_dir + file + '_break.nc'
        os.system('cp ' + from_dir + ' ' + to_dir)
        print(to_dir)
        break_my_ice(to_dir, 1100, 400, 'noise')


def detect_holes(size):
    holes_dir = "samples/pathological_cases/holes/" + str(size) + "/"
    files = []
    for file in os.listdir(holes_dir):
        files.append(holes_dir + file)

    files = sorted(files)

    good = []

    idx = 1
    for file in files:
        init_session()
        detector = IceDetector(0.5, str(idx)) if idx >= 10 else IceDetector(0.5, "0" + str(idx))
        pred, val = detector.detect(file)
        good.append(val * 100.0)
        print(idx, val)
        idx += 1

    return good


def detect_spots():
    holes_dir = "samples/pathological_cases/spots/"
    files = []
    for file in os.listdir(holes_dir):
        files.append(holes_dir + file)

    files = sorted(files)

    good = []

    idx = 1
    for file in files:
        init_session()
        detector = IceDetector(0.5, str(idx)) if idx >= 10 else IceDetector(0.5, "0" + str(idx))
        pred, val = detector.detect(file)
        good.append(val * 100.0)
        print(idx, val)
        idx += 1

    return good


def detect_noise():
    holes_dir = "samples/pathological_cases/noise/"
    files = []
    for file in os.listdir(holes_dir):
        files.append(holes_dir + file)

    files = sorted(files)

    good = []

    idx = 1
    for file in files:
        init_session()
        detector = IceDetector(0.5, str(idx)) if idx >= 10 else IceDetector(0.5, "0" + str(idx))
        pred, val = detector.detect(file)
        good.append(val * 100.0)
        print(idx, val)
        idx += 1

    return good


def plot_holes():
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    months = [i for i in range(1, 13)]
    good = detect_holes(50)
    plt.plot(months, good, marker='o', c="c")

    good = detect_holes(100)
    plt.plot(months, good, marker='o', c="y")

    good = detect_holes(200)
    plt.plot(months, good, marker='o', c="r")

    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.set_xticks(months)
    ax.set_xticklabels(labels)

    plt.xlabel('Month')
    plt.ylabel('Squares recognized as correct, %')
    plt.title('Prediction results for ice with holes')
    legend = plt.legend(['holes amount = 50', 'holes amount = 100', 'holes amount = 200'],
                        bbox_to_anchor=(1.05, 1), loc=2,
                        borderaxespad=0.)

    plt.savefig("samples/pathological_cases/holes_results.png", bbox_extra_artists=(legend,), bbox_inches='tight',
                dpi=500)


def plot_spots():
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    months = [i for i in range(1, 13)]
    good = detect_spots()
    plt.plot(months, good, marker='o', c="c")

    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.set_xticks(months)
    ax.set_xticklabels(labels)

    plt.xlabel('Month')
    plt.ylabel('Squares recognized as correct, %')
    plt.title('Prediction results for ice with spots')
    # legend = plt.legend(['holes amount = 50', 'holes amount = 100', 'holes amount = 200'],
    #                     bbox_to_anchor=(1.05, 1), loc=2,
    #                     borderaxespad=0.)

    plt.savefig("samples/pathological_cases/spots_results.png", dpi=500)


def plot_noise():
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    months = [i for i in range(1, 13)]
    good = detect_noise()
    plt.plot(months, good, marker='o', c="c")

    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.set_xticks(months)
    ax.set_xticklabels(labels)

    plt.xlabel('Month')
    plt.ylabel('Squares recognized as correct, %')
    plt.title('Prediction results for ice with noise')
    # legend = plt.legend(['holes amount = 50', 'holes amount = 100', 'holes amount = 200'],
    #                     bbox_to_anchor=(1.05, 1), loc=2,
    #                     borderaxespad=0.)

    plt.savefig("samples/pathological_cases/noise_results.png", dpi=500)


def plot_full_ice():
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    months = [i for i in range(1, 13)]

    good = full_ice("samples/pathological_cases/no_ice.nc")
    plt.plot(months, good, marker='o', c="c")

    good = full_ice("samples/pathological_cases/full_ice_02.nc")
    plt.plot(months, good, marker='o', c="m")

    good = full_ice("samples/pathological_cases/full_ice_05.nc")
    plt.plot(months, good, marker='o', c="g")

    good = full_ice("samples/pathological_cases/full_ice_08.nc")
    plt.plot(months, good, marker='o', c="y")

    good = full_ice("samples/pathological_cases/full_ice_10.nc")
    plt.plot(months, good, marker='o', c='r')

    m = [i for i in range(1, 13)]
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.set_xticks(m)
    ax.set_xticklabels(labels)

    plt.xlabel('Month')
    plt.ylabel('Sub-areas recognized as correct, %')
    plt.title('Prediction results for full ice case')
    legend = plt.legend(['conc = 0.0', 'conc = 0.2', 'conc = 0.5', 'conc = 0.8', 'conc = 1.0'],
                        bbox_to_anchor=(1.05, 1), loc=2,
                        borderaxespad=0.)
    plt.savefig("samples/pathological_cases/full_ice_results.png", bbox_extra_artists=(legend,), bbox_inches='tight',
                dpi=500)


def plot_path_cases():
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    months = [i for i in range(1, 13)]
    noise = detect_noise()
    plt.plot(months, noise, marker='o', c="c", label='Noise')

    holes50 = detect_holes(50)
    plt.plot(months, holes50, marker='o', c="g", label='Holes with amount = 50')

    holes100 = detect_holes(100)
    plt.plot(months, holes100, marker='o', c="r", label='Holes with amount = 100')

    holes100 = detect_holes(200)
    plt.plot(months, holes100, marker='o', c="b", label='Holes with amount = 200')

    spots = detect_spots()
    plt.plot(months, spots, marker='o', c="y", label='Spots')

    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.set_xticks(months)
    ax.set_xticklabels(labels)

    plt.xlabel('Month')
    plt.ylabel('Sub-areas recognized as correct, %')
    plt.title('Prediction results for pathological cases')
    plt.legend(loc='lower right', fontsize='medium')
    # legend = plt.legend(['holes amount = 50', 'holes amount = 100', 'holes amount = 200'],
    #                     bbox_to_anchor=(1.05, 1), loc=2,
    #                     borderaxespad=0.)

    plt.savefig("samples/pathological_cases/path_results.png", dpi=500)


# plot_full_ice()

# for size in [50, 100, 200]:
#     generate_holes(size)

# plot_holes()
# generate_spots()
# plot_spots()
# generate_noise()
# plot_noise()
# plot_path_cases()
plot_full_ice()

