import csv
import os
import re
from math import pow
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

import image as img


def magnitude(a, b):
    return sqrt(pow(a, 2.0) + pow(b, 2.0))


def calculate_velocity_magnitude_matrix(x, y):
    mgn = np.zeros((len(x), len(x[0])), dtype=np.float32)

    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            mgn[i][j] = magnitude(x[i][j], y[i][j])

    return mgn


def press(event):
    if event.key == 'y':
        bad_samples.append('1')
        plt.close()

    elif event.key == 'n':
        bad_samples.append('0')
        plt.close()


def show_velocity_square(vel):
    plt.connect('key_press_event', press)
    plt.imshow(vel)
    plt.colorbar()
    plt.show()


def generate_squares_global():
    img.slice_uv_squares("samples/data/")


def label_good_samples():
    file_name = "samples/good_samples.csv"
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['file, label'])

        images_amount = 30
        vel_dir = "samples/good/"
        if not os.path.exists(vel_dir):
            os.makedirs(vel_dir)

        for image_index in range(0, images_amount):
            square_index = 1
            for x in range(0, 1100, 100):
                for y in range(0, 400, 100):
                    u_file_name = "samples/out/rea" + str(image_index) + "_u_" + str(x) + "_" + str(y)
                    v_file_name = "samples/out/rea" + str(image_index) + "_v_" + str(x) + "_" + str(y)
                    u = img.load_square_from_file(u_file_name)
                    v = img.load_square_from_file(v_file_name)
                    # remove nan-values
                    u[u < -30000.0] = 0.0
                    v[v < -30000.0] = 0.0
                    vel = calculate_velocity_magnitude_matrix(u, v)
                    velocity_file_name = vel_dir + "rea" + str(image_index) + "_" + str(x) + "_" + str(y)
                    img.save_square_to_file(vel, velocity_file_name)

                    writer.writerow([velocity_file_name, '0'])
                    print("squares: " + str(square_index) + "/44 done")
                    square_index += 1
            print("images: " + str(image_index + 1) + "/" + str(images_amount) + " done")


bad_samples = []


def label_bad_samples():
    file_name = "samples/bad_samples.csv"
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['file, label'])

        vel_dir = "samples/bad/vel/"
        if not os.path.exists(vel_dir):
            os.makedirs(vel_dir)

        for file_name in os.listdir("samples/arctic"):
            arctic_file = file_name.split(".")[0]
            for x in range(0, 1100, 100):
                for y in range(0, 400, 100):
                    u_file_name = "samples/bad/" + arctic_file + "_vozocrtx_" + str(x) + "_" + str(y)
                    v_file_name = "samples/bad/" + arctic_file + "_vomecrty_" + str(x) + "_" + str(y)
                    u = img.load_square_from_file(u_file_name)
                    v = img.load_square_from_file(v_file_name)
                    u[u < -30000.0] = 0.0
                    v[v < -30000.0] = 0.0
                    vel = calculate_velocity_magnitude_matrix(u, v)

                    show_velocity_square(vel)

                    velocity_file_name = vel_dir + arctic_file + "_" + str(x) + "_" + str(y)
                    img.save_square_to_file(vel, velocity_file_name)
                    result = bad_samples[len(bad_samples) - 1]
                    writer.writerow([velocity_file_name, result])


def add_characteristics_of_samples(dataset_file_name):
    rows = []
    with open(dataset_file_name, 'r', newline='') as old_csv_file:
        reader = csv.reader(old_csv_file, delimiter=',')
        for row in reader:
            rows.append(row)

    # Add min && max && average values of samples
    for row in rows:
        square = img.load_square_from_file(row[0])
        row.append(np.min(square))
        row.append(np.max(square))
        row.append(np.average(square))
    with open(dataset_file_name, 'w', newline='') as new_csv_file:
        writer = csv.writer(new_csv_file, delimiter=',')
        for row in rows:
            writer.writerow(row)


def get_min_max_of_dataset(dataset_file_name):
    min_vel = 1000.0
    max_vel = -1000.0
    with open(dataset_file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            square = img.load_square_from_file(row[0])
            min_vel = min([min_vel, np.min(square)])
            max_vel = max([max_vel, np.max(square)])

    return min_vel, max_vel


def show_hist_samples_distribution(dataset_file_name):
    plt.style.use('seaborn-white')

    average_vel = []
    min_vel = []
    max_vel = []
    with open(dataset_file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            min_vel.append(np.float(row[2]))
            max_vel.append(np.float(row[3]))
            average_vel.append(np.float(row[4]))

    bins = np.linspace(min(average_vel), max(average_vel), num=500)
    counts = np.zeros_like(bins)
    i = np.searchsorted(bins, average_vel)
    np.add.at(counts, i, 1)

    plt.title("Average velocity distribution")
    plt.hist(average_vel, bins, histtype='stepfilled', normed=True)
    plt.show()


def extend_datasets():
    add_characteristics_of_samples("samples/good_samples.csv")
    add_characteristics_of_samples("samples/bad_samples.csv")
    add_characteristics_of_samples("samples/rest_bad_samples.csv")
    add_characteristics_of_samples("samples/valid_samples.csv")


def extract_square_index(square_name):
    try:
        found = re.search('_\d*_\d*', square_name).group(0)[1:]
    except AttributeError:
        found = ''

    return found


def group_squares(reader):
    groups, dic_name = [], {}
    for row in reader:
        square_index = extract_square_index(row[0])
        if square_index in dic_name:
            groups[dic_name[square_index]].extend([row])
        else:
            groups.append([row])
            dic_name[square_index] = len(dic_name)

    return groups


def show_average_vel_by_squares(dataset_file_name):
    with open(dataset_file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        groups = group_squares(reader)
        for index in range(0, len(groups)):
            squares_vel = np.zeros((len(groups[index]), 100, 100), dtype=float)
            for sample_index in range(0, len(groups[index])):
                square = img.load_square_from_file(groups[index][sample_index][0])
                squares_vel[sample_index] = square
            average_vel = np.average(squares_vel, axis=0)
            print(average_vel)
            plt.imshow(average_vel)
            plt.colorbar()
            plt.show()


def show_distribution_by_classes(dataset_file_name):
    with open(dataset_file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        classes = {}
        for row in reader:
            label = int(row[1])
            if label in classes:
                classes[label] += 1
            else:
                classes[label] = 0
        print(classes)


# show_distribution_by_classes("samples/bad_samples.csv")
# show_average_vel_by_squares("samples/bad_samples.csv")
# print(show_average_vel_by_squares("samples/bad_samples.csv"))
# print(extract_square_index("samples/bad/vel/ARCTIC_1h_UV_grid_UV_20130101-20130101_0_0"))
# extend_datasets()
# show_hist_samples_distribution("samples/good_samples.csv")
# show_hist_samples_distribution("samples/bad_samples.csv")
# show_hist_samples_distribution("samples/valid_samples.csv")

# img.slice_uv_squares("samples/data/")
# img.slice_uv_squares("samples/arctic/")

# img.slice_uv_squares("samples/arctic/", "samples/bad/", mode="arctic")

'''
for image_index in range(0, 1):
    for x in range(0, 1100, 100):
        for y in range(0, 400, 100):
            u_file_name = "samples/out/rea0" + "_u_" + str(x) + "_" + str(y)
            v_file_name = "samples/out/rea0" + "_v_" + str(x) + "_" + str(y)
            u = img.load_square_from_file(u_file_name)
            v = img.load_square_from_file(v_file_name)
            u[u < -30000.0] = 0.0
            v[v < -30000.0] = 0.0
            vel = calculate_velocity_magnitude_matrix(u, v)

            show_velocity_square(vel)


'''
