import csv

from netCDF4 import Dataset as NCFile


class Dataset:
    def __init__(self, file_name):
        self.file_name = file_name

        self.samples = []

    def dump_to_csv(self):
        with open(self.file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for sample in self.samples:
                writer.writerow(sample.raw_data())

    @staticmethod
    def from_csv(file_name):
        samples = []
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                samples.append(IceSample.from_raw_data(row))

        dataset = Dataset(file_name)
        dataset.samples = samples

        return dataset


class IceSample:
    def __init__(self, nc_file, index, size, time):
        self.nc_file = nc_file
        self.index = index
        self.size = size
        self.time = time

        self.x, self.y = self.get_borders()

        # 0 - not-outlier
        self.label = 0

    def get_borders(self):
        x, y = divmod(self.index, 11)
        return x, y

    def ice_conc(self):
        nc = NCFile(self.nc_file)
        ice = nc.variables['iceconc'][:][self.time][self.x:self.x + self.size, self.y:self.y + self.size]
        nc.close()

        return ice

    def ice_thic(self):
        nc = NCFile(self.nc_file)
        thic = nc.variables['icethic_cea'][:][self.time][self.x:self.x + self.size, self.y:self.y + self.size]
        nc.close()

        return thic

    def raw_data(self):
        return [str(self.nc_file), str(self.index), str(self.time), str(self.label)]

    @staticmethod
    def from_raw_data(raw):
        return IceSample(raw[0], int(raw[1]), int(raw[2]), int(raw[3]))


data = Dataset("samples/test.csv")
data.samples.append(IceSample("samples/ice_data/ARCTIC_1h_ice_grid_TUV_20000731-20000731.nc_1.nc", 1, 100, 0))
data.samples.append(IceSample("samples/ice_data/ARCTIC_1h_ice_grid_TUV_20000731-20000731.nc_1.nc", 10, 100, 0))
data.samples.append(IceSample("samples/ice_data/ARCTIC_1h_ice_grid_TUV_20000731-20000731.nc_1.nc", 15, 100, 0))
data.samples.append(IceSample("samples/ice_data/ARCTIC_1h_ice_grid_TUV_20000731-20000731.nc_1.nc", 34, 100, 0))
data.dump_to_csv()

data = Dataset.from_csv("samples/test.csv")

print(data.samples)
