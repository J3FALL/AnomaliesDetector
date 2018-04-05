import csv

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

import image


def read_samples(file_name):
    samples = []
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            row[1] = int(row[1])

            if row[2] == 0.0 and row[3] == 0.0:
                next(reader)

            samples.append(row)

    a, b = divmod(len(samples), 50)

    return samples[0:len(samples) - b]


def train_data():
    bad_samples_file = "samples/bad_samples.csv"
    bad_samples = read_samples(bad_samples_file)

    return bad_samples


def test_data():
    valid_samples_file = "samples/valid_samples.csv"
    samples = read_samples(valid_samples_file)

    return samples


def split_data(samples, test_size):
    train, test = train_test_split(samples, test_size=test_size)

    return train, test


def data_generator(samples, batch_size):
    while 1:
        for sample_index in range(0, len(samples), batch_size):
            x = np.zeros((batch_size, 100, 100, 1), dtype=np.float32)
            y = np.zeros((batch_size,))
            for index in range(sample_index, sample_index + batch_size):
                file_name = samples[index][0]
                square = image.load_square_from_file(file_name)
                expanded = np.expand_dims(square, axis=2)
                x[index - sample_index] = expanded
                y[index - sample_index] = samples[index][1]
            yield (x, y)


def init_model():
    input_shape = (100, 100, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.05])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve with AUC(%0.2f)' % roc_auc)
    print('AUC: %f' % roc_auc)

    # plt.show()

    pr, rc, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(rc, pr)
    # plt.plot(pr, rc)
    # plt.xlim([0.0, 1.05])
    # plt.ylim([0.0, 1.05])
    # plt.title('Precision-Recall curve with AUC(%0.2f)' % pr_auc)
    # plt.xlabel('Precision')
    # plt.ylabel('Recall')
    print('AUC: %f' % pr_auc)

    # plt.show()

    return roc_auc, pr_auc


def calc_batch_size(batch_len, min_size, max_size):
    for size in range(min_size, max_size):
        a, b = divmod(batch_len, size)
        if b == 0:
            return size


def run_metrics_experiment():
    r = read_samples("samples/rest_bad_samples.csv")
    roc = np.zeros((19, 3), dtype=np.float32)
    pr = np.zeros((19, 3), dtype=np.float32)
    t_size = []
    idx = 0
    for test_size in np.arange(0.05, 1, 0.05):
        size = round(test_size, 2)
        train, test = split_data(r, size)
        print("test_part: " + str(size) + " train_size: " + str(len(train)) + " test_size: " + str(len(test)))

        history = AccuracyHistory()
        train_middle = int(len(train) / 50)
        test_middle = int(len(test) / 25)
        train_batch_size = calc_batch_size(len(train), int(train_middle - 0.5 * train_middle),
                                           int(train_middle + 0.5 * train_middle))
        test_batch_size = calc_batch_size(len(test), int(test_middle - 0.5 * test_middle),
                                          int(test_middle + 0.5 * test_middle))
        # train_batch_size = int(len(train) / 50)
        # test_batch_size = int(len(test) / 25)
        epochs = 5
        print(train_batch_size)
        print(test_batch_size)
        roc_tmp = []
        pr_tmp = []
        for _ in range(5):
            K.clear_session()
            model = init_model()
            model.fit_generator(data_generator(train, train_batch_size),
                                steps_per_epoch=train_batch_size,
                                callbacks=[history],
                                epochs=epochs)

            # model = load_model("samples/model.h5")
            scores = model.predict_generator(data_generator(test, test_batch_size),
                                             steps=int(len(test) / test_batch_size))
            print(scores)
            real = np.zeros((len(test),), dtype=np.float32)
            for i in range(0, len(test)):
                real[i] = test[i][1]

            roc_auc, pr_auc = generate_results(real, scores)
            roc_tmp.append(roc_auc)
            pr_tmp.append(pr_auc)

        roc[idx][0] = np.average(roc_tmp)
        roc[idx][1] = np.min(roc_tmp)
        roc[idx][2] = np.max(roc_tmp)
        pr[idx][0] = np.average(pr_tmp)
        pr[idx][1] = np.min(pr_tmp)
        pr[idx][2] = np.max(pr_tmp)

        t_size.append(test_size)
        idx += 1

    plt.plot(t_size, roc[:, 0])
    plt.fill_between(t_size, roc[:, 1], roc[:, 2], alpha=0.3)
    plt.plot(t_size, pr[:, 0])
    plt.fill_between(t_size, pr[:, 1], pr[:, 2], alpha=0.5)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("test_part_size")
    plt.ylabel("AUC")
    plt.show()


def run_default_model():
    train = train_data()
    test = test_data()
    history = AccuracyHistory()
    train_batch_size = int(len(train) / 50)
    test_batch_size = int(len(test) / 25)

    print(train_batch_size)
    print(test_batch_size)
    epochs = 10

    model = init_model()

    model.fit_generator(data_generator(train, train_batch_size),
                        steps_per_epoch=train_batch_size,
                        callbacks=[history],
                        epochs=3)

    model.save("samples/model.h5")

    # model = load_model("samples/model.h5")
    scores = model.predict_generator(data_generator(test, test_batch_size), steps=25)
    print(scores)
    real = np.zeros((len(test),), dtype=np.float32)
    for i in range(0, len(test)):
        real[i] = test[i][1]

    print(scores)

    generate_results(real, scores)


# run_metrics_experiment()
run_default_model()
