import csv

import image

'''
batch_size = 10
for sample_index in range(0, 100, batch_size):
    print("yo" + str(sample_index))
    x = np.zeros((batch_size, 100, 100, 1), dtype=np.float32)
    for index in range(sample_index, sample_index + batch_size):
        print("index=" + str(index))
        x[index - sample_index] = 1
'''
samples = []
with open("samples/valid_samples.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    # next(reader)
    k = 0
    for row in reader:
        square = image.load_square_from_file(row[0])
        if square.max() == 0.0 and square.min() == 0.0:
            k += 1
    print(k)
print(len(samples))
