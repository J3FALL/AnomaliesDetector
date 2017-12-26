import csv
from math import pow
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import os

import image as img


def magnitude(a, b):
    return sqrt(pow(a, 2.0) + pow(b, 2.0))


def calculate_velocity_magnitude_matrix(x, y):
    mgn = np.zeros((len(x), len(x[0])), dtype=np.float32)

    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            mgn[i][j] = magnitude(x[i][j], y[i][j])

    return mgn


def show_velocity_square(vel):
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


#generate_squares_global()
#label_good_samples()

square = img.load_square_from_file("samples/out/rea0_u_500_0")
print(square)
'''
# img.slice_uv_squares("samples/data/")
img.slice_uv_squares("samples/test/")
# TODO: average by depth

for image_index in range(0, 1):
    for x in range(0, 1100, 100):
        for y in range(0, 400, 100):
            u_file_name = "samples/out/ARCTIC_1h_UV_grid_UV_20121017-20121017" + "_vomecrty_" + str(x) + "_" + str(y)
            v_file_name = "samples/out/ARCTIC_1h_UV_grid_UV_20121017-20121017" + "_vozocrtx_" + str(x) + "_" + str(y)
            u = img.load_square_from_file(u_file_name)
            v = img.load_square_from_file(v_file_name)
            u[u == -32767.0] = 0.0
            v[v == -32767.0] = 0.0
            vel = calculate_velocity_magnitude_matrix(u, v)

            show_velocity_square(vel)

'''
