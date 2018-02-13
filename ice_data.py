from netCDF4 import Dataset as NCFile


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


sample = IceSample("samples/ice_data/ARCTIC_1h_ice_grid_TUV_20000731-20000731.nc_1.nc", 1, 100, 0)
print(sample.ice_conc())
print(sample.ice_thic())
print(sample.raw_data())
