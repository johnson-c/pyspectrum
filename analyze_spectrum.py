# This script calls both spectrum.py to which is designed to hold the spectral
# data least_squares_gen.py holds the Fit_spectra class
# both of these class can operate seperately but this script allows them to be
# used in tandum to make a run day easier.
# This  will also have definitions that are need to help out durring the
# run day
from spectrum import Spectrum
from least_squares_gen import Fit_spectra
import matplotlib.pyplot as plt

spec_1 = Spectrum()
spec_1.populate(16052711, 2, 'black_comet_200_600', 'txt_file')

fit = Fit_spectra()
#
fit.wavelength = spec_1.raw.wavelength
fit.intensity = spec_1.raw.intensity
fit.min_val = 2300
fit.find_peak_indexes()

p = fit.setup_p()
fit.set_parameter_limits()
a = fit.fit_dat_stuff(p)
g = fit.gaussian_model(fit.wavelength, a[0])
gs = fit.gaussian_model_split(fit.wavelength, a[0])
