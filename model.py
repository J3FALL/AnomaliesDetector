import csv

import numpy as np
from sklearn.model_selection import train_test_split


def read_samples(file_name):
    samples = []
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            row[1] = int(row[1])
            samples.append(row)

    for sample in samples:
        label = np.zeros(2, dtype=np.int8)
        label[sample[1]] = 1
        sample[1] = label

    return samples


def split_data():
    good_samples_file = "samples/good_samples.csv"
    good_samples = read_samples(good_samples_file)

    train_good, test_good = train_test_split(good_samples, test_size=0.2)

    bad_samples_file = "samples/bad_samples.csv"
    bad_samples = read_samples(bad_samples_file)

    train_bad, test_bad = train_test_split(bad_samples, test_size=0.2)

    train = train_good + train_bad
    test = test_good + test_bad

    return train, test


split_data()
