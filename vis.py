import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset as NetCDFFile

import data
import image as img


def draw():
    nc = NetCDFFile('ARCTIC_1h_ice_grid_TUV_20130707-20130707.nc')
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]
    ice = nc.variables['iceconc'][:][0]
    plt.figure(figsize=(20, 15))

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]

    lat_center = 90
    lon_center = 119
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='h',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, ice, latlon=True)

    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    plt.colorbar()
    plt.show()


def draw_velocity_map(nc_file_name):
    nc = NetCDFFile(nc_file_name, 'r+')
    lat = nc.variables['nav_lat_grid_U'][:]
    lon = nc.variables['nav_lon_grid_U'][:]
    vel_dir = "samples/bad/vel/"
    nc_name = nc_file_name.split("/")[4].split(".")[0]
    velocity = np.zeros((400, 1100), dtype=np.float32)
    for file_name in os.listdir(vel_dir):
        if nc_name in file_name:
            square = img.load_square_from_file(vel_dir + file_name.split(".")[0])
            print(file_name)
            print(str(np.min(square)) + "; " + str(np.max(square)))
            square_index = data.extract_square_index(vel_dir + file_name.split(".")[0])
            x = int(square_index.split("_")[0])
            y = int(square_index.split("_")[1])
            print(str(x) + " " + str(y))
            velocity[y:y + 100, x:x + 100] = square

    nc.close()
    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]

    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, velocity, latlon=True, cmap='jet', vmax=0.6)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    plt.colorbar()
    plt.title(nc_file_name)
    model = load_model("samples/model.h5")
    with open("samples/valid_samples.csv", 'r', newline='') as csvfile:
        samples = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if "samples/bad/vel/" + nc_name in row[0]:
                print(row)
                samples.append(row)
                square_index = data.extract_square_index(row[0])
                x = int(square_index.split("_")[0])
                y = int(square_index.split("_")[1])

                sample = np.zeros((1, 100, 100, 1), dtype=np.float32)
                square = np.expand_dims(img.load_square_from_file(row[0]), axis=2)
                sample[0] = square
                result = model.predict(sample)
                result_x, result_y = m(lon[y + 50][x + 50], lat[y + 50][x + 50])

                plt.text(result_x, result_y, str(result[0][0]), ha='center', size=10, color="yellow")
                if row[1] == "1":
                    print("outlier!")
                    lat_poly = np.array([lat[y][x], lat[y][x + 99], lat[y + 99][x + 99], lat[y + 99][x]])
                    lon_poly = np.array([lon[y][x], lon[y][x + 99], lon[y + 99][x + 99], lon[y + 99][x]])
                    mapx, mapy = m(lon_poly, lat_poly)
                    points = np.zeros((4, 2), dtype=np.float32)
                    for j in range(0, 4):
                        points[j][0] = mapx[j]
                        points[j][1] = mapy[j]

                    if result[0][0] > 0.5:
                        poly = Polygon(points, facecolor='yellow', alpha=0.4)
                        plt.gca().add_patch(poly)
                    else:
                        poly = Polygon(points, facecolor='red', alpha=0.4)
                        plt.gca().add_patch(poly)
                else:
                    if result[0][0] > 0.5:
                        lat_poly = np.array([lat[y][x], lat[y][x + 99], lat[y + 99][x + 99], lat[y + 99][x]])
                        lon_poly = np.array([lon[y][x], lon[y][x + 99], lon[y + 99][x + 99], lon[y + 99][x]])
                        mapx, mapy = m(lon_poly, lat_poly)
                        points = np.zeros((4, 2), dtype=np.float32)
                        for j in range(0, 4):
                            points[j][0] = mapx[j]
                            points[j][1] = mapy[j]
                        poly = Polygon(points, facecolor='green', alpha=0.4)
                        plt.gca().add_patch(poly)

                print(result)

    plt.show()


def draw_velocity_map_with_level(nc_file_name, level):
    nc = NetCDFFile(nc_file_name, 'r+')
    lat = nc.variables['nav_lat_grid_U'][:]
    lon = nc.variables['nav_lon_grid_U'][:]

    velocity = np.zeros((400, 1100), dtype=np.float32)

    time = 0

    x_vel = nc.variables['vozocrtx'][:][time][level]
    y_vel = nc.variables['vomecrty'][:][time][level]

    velocity = data.calculate_velocity_magnitude_matrix(x_vel, y_vel)

    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]

    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, velocity, latlon=True, cmap='jet', vmax=0.6)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    plt.colorbar()
    plt.title(nc_file_name)

    model = load_model("samples/model.h5")

    for y in range(0, 400, 100):
        for x in range(0, 1100, 100):
            sample = np.zeros((1, 100, 100, 1), dtype=np.float32)
            sample[0] = np.expand_dims(velocity[y:y + 100, x:x + 100], axis=2)
            # print(sample)
            result = model.predict(sample)
            result_x, result_y = m(lon[y + 50][x + 50], lat[y + 50][x + 50])
            max_x, max_y = m(lon[y + 70][x + 50], lat[y + 70, x + 50])
            plt.text(result_x, result_y, str(result[0][0]), ha='center', size=10, color="yellow")
            plt.text(max_x, max_y, np.max(sample[0]), ha='center', size=10, color="yellow")
            if result[0][0] > 0.5:
                lat_poly = np.array([lat[y][x], lat[y][x + 99], lat[y + 99][x + 99], lat[y + 99][x]])
                lon_poly = np.array([lon[y][x], lon[y][x + 99], lon[y + 99][x + 99], lon[y + 99][x]])
                mapx, mapy = m(lon_poly, lat_poly)
                points = np.zeros((4, 2), dtype=np.float32)
                for j in range(0, 4):
                    points[j][0] = mapx[j]
                    points[j][1] = mapy[j]
                poly = Polygon(points, facecolor='green', alpha=0.4)
                plt.gca().add_patch(poly)

    plt.show()


draw_velocity_map_with_level("samples/valid!/samples/arctic/ARCTIC_1h_UV_grid_UV_20130320-20130320.nc", 15)
# draw_velocity_map("samples/valid!/samples/arctic/ARCTIC_1h_UV_grid_UV_20130320-20130320.nc")
