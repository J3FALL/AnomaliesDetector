import glob
import os

import matplotlib

matplotlib.use('agg')

import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from netCDF4 import Dataset as NCFile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from ice_data import IceDetector
from scipy.ndimage import uniform_filter

month_pair = {
    "01": ["09"],
    "03": ["09"],
    "06": ["09"],
    "09": ["01", "03", "06"]
}


def init_session():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    K.set_session(tf.Session(config=config))


def plot_roc(results, month):
    tpr = []
    fpr = []

    for barrier in np.arange(0.0, 1.0, 0.01):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index in range(0, len(results)):
            if results[index][0] > barrier:
                if results[index][1] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if results[index][1] == 0:
                    tn += 1
                else:
                    fn += 1
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    print(tpr)
    print(fpr)
    roc_auc = auc(fpr, tpr)
    print('AUC: %f' % roc_auc)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve with AUC(%0.2f)' % roc_auc)

    plt.savefig("roc_for_month_" + month + ".png", dpi=500)


def roc_for_month(month):
    bad_months = month_pair[month]

    nemo_dir = "samples/month_experiments/NEMO/good/"
    sat_dir = "samples/month_experiments/SAT/"

    samples = []

    # good data
    for file_name in glob.iglob(nemo_dir + "*/" + month + "/*.nc", recursive=True):
        samples.append([os.path.normpath(file_name), 1])

    # bad data
    for bad_month in bad_months:
        print(bad_month)
        for file_name in glob.iglob(nemo_dir + "*/" + bad_month + "/*.nc", recursive=True):
            samples.append([os.path.normpath(file_name), 0])
        for file_name in glob.iglob(sat_dir + "*/" + bad_month + "/*.nc", recursive=True):
            samples.append([os.path.normpath(file_name), 0])

    print(len(samples))

    init_session()

    detector = IceDetector(0.1, month)
    results = np.zeros((len(samples), 2))
    idx = 0
    for sample in samples:
        print(sample[0])
        pred, val = detector.detect(sample[0])
        print(str(pred) + " " + str(val))
        results[idx][0] = val
        results[idx][1] = sample[1]
        idx += 1

    plot_roc(results, month)
    K.clear_session()


def check_month():
    in_dir = "samples/conc_satellite/"
    out_dir = "samples/aug_check/"

    samples = []
    for file in glob.iglob(in_dir + "*/" + "08" + "/*0814*.nc", recursive=True):
        samples.append(file)

    for sample in samples:
        year = sample.split("/")[2]
        print(year)
        nc = NCFile(sample)
        lat = nc.variables['nav_lat'][:]
        lon = nc.variables['nav_lon'][:]
        conc = nc.variables['ice_conc'][:].filled(0) / 100.0
        conc = conc[0]

        nc.close()
        lat_left_bottom = lat[-1][-1]
        lon_left_bottom = lon[-1][-1]
        lat_right_top = lat[0][0]
        lon_right_top = lon[0][0]
        lat_center = 90

        lon_center = 110
        m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                    llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                    urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

        m.pcolormesh(lon, lat, conc, latlon=True, cmap='RdYlBu_r', vmax=1)
        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color='#cc9966', lake_color='#99ffff')

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(cax=cax, label="Ice concentration")
        plt.savefig(out_dir + year + ".png", dpi=500)


# check_month()


def reduce_conc(conc):
    conc[conc < 0.4] = 0.0

    return conc


def check_filter():
    file_name = "samples/conc_satellite/2013/08/ice_conc_nh_ease2-250_cdr-v2p0_201308181200.nc"
    nc = NCFile(file_name)
    lat = nc.variables['nav_lat'][:]
    lon = nc.variables['nav_lon'][:]
    conc = nc.variables['ice_conc'][:].filled(0) / 100.0
    conc = conc[0]
    conc = reduce_conc(conc)
    initial_conc = conc

    nc.close()

    for filter in range(1, 2, 1):
        conc = uniform_filter(initial_conc, filter)
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

        m.pcolormesh(lon, lat, conc, latlon=True, cmap='RdYlBu_r', vmax=1)
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(cax=cax, label="Ice concentration")
        plt.savefig("samples/filters/" + str(filter) + ".png", dpi=500)


roc_for_month("09")


# check_filter()

def load_mask():
    mask_file = NCFile("bathy_meter_mask.nc")
    mask = mask_file.variables['Bathymetry'][:]
    mask_file.close()

    mask = 1 - mask

    return mask


# load_mask()
