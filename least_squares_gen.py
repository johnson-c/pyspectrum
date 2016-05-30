import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize, leastsq
import peakutils
import pdb
# from spectrum

# def gauassian_models(x,gaussian_parameters)

#

#
# Fit_spectra
#        This is where data is stored for the fits and where the fits are done
# this script will take a (x,y)  arrays and then fit gaussians to the data
# based on the number of peaks that the program finds. Once the fit is found,
# fit parameters are saved in the class
#


class Fit_spectra:

    def __init__(self):
        self.wavelength = np.zeros(1)
        self.intensity = np.zeros(1)
        self.centriods = np.zeros(1)
        self.amplitudes = np.zeros(1)
        self.sigmas = np.zeros(1)
        self.amplitude_ranges = np.zeros(1)
        self.centroid_ranges = np.zeros(1)
        self.sigma_ranges = np.zeros(1)
        self.threshold = 0.01
        self.min_distance = 1
        self.min_val = 100
        self.centroid_indexes = np.zeros(1)

# Finds the peaks in y by taking its first order difference. By using
# thres and min_dist parameters, it is possible to reduce the number of
# detected peaks.
    def find_peak_indexes(self):
        indexes = peakutils.indexes(self.intensity, thres=self.threshold,
                                    min_dist=self.min_distance)

        self.centroid_indexes = indexes[
            np.where(self.intensity[indexes] > self.min_val)]

        self.centroids = self.wavelength[self.centroid_indexes]
        self.amplitudes = self.intensity[self.centroid_indexes]
        self.sigmas = np.ones(len(self.wavelength[self.centroid_indexes]))

# setup the limits that the code will fit.
# maxium value for gaussian amplititude allowed to be 1.2x the max y value
# sigma range can at most can be 5x the x array step
#
    def set_parameter_limits(self):
        if(self.centroids[0] != 0):
            self.amplitude_ranges = np.array([0, self.intensity.max() * 1.2])
            self.sigma_ranges = np.array(
                [0, (self.wavelength[1] - self.wavelength[0]) * 5])
            self.centroid_ranges = np.array(
                [self.wavelength[0], self.wavelength[-1]])

# This will generate the residuals for least sqruares fit_dat_stuff
# just take the data - fit if any parameter is out of the set range
# set the residual high
    def resid(self, p, y, x):
        err = y - self.gaussian_function(x, p)
        for i in range(0, len(self.centroids)):

            if((p[i * 3] < self.amplitude_ranges[0]) or (p[i * 3] > self.amplitude_ranges[1])):
                err = err + 100000000
            if((p[i * 3 + 1] < self.centroid_ranges[0]) or (p[i * 3 + 1] > self.centroid_ranges[1])):
                err = err + 100000000
            if((p[i * 3 + 2] < self.sigma_ranges[0]) or (p[i * 3 + 2] > self.sigma_ranges[1])):
                err = err + 100000000
        return err

# this is the guassian function that will need to be fit
# it made up of many guassians and a constant offset
    def gaussian_function(self, x, p):
        function = 0
        for i in range(0, (len(p) - 1), 3):
            function = function + \
                p[i] * np.exp(-(x - p[i + 1]) ** 2 / 2 / p[i + 2] ** 2)
        function = function + p[-1]
        return function

    def setup_p(self):
        p = np.zeros(len(self.centroids) * 3 + 1)
        for i in range(0, len(self.centroids)):
            p[i * 3] = self.amplitudes[i]
            p[i * 3 + 1] = self.centroids[i]
            p[i * 3 + 2] = self.sigmas[i]
            p[i * 3 + 3] = 1000
        return p

    def fit_dat_stuff(self, p):
        a = leastsq(self.resid, p, args=(self.intensity, self.wavelength))
        return a

    def gaussian_model(self, x, p):
        g = 0
        for i in range(0, len(p) / 3):

            g = g + \
                p[i * 3] * np.exp(
                    -(x - p[i * 3 + 1]) ** 2 / 2 / p[i * 3 + 2] ** 2)
        g = g + p[-1]
        return g

    def gaussian_model_split(self, x, p):
        g = []
        for i in range(0, len(p) / 3):

            g.append(
                p[i * 3] * np.exp(-(x - p[i * 3 + 1]) ** 2 / 2 / p[i * 3 + 2] ** 2) + p[-1])
        return g

    def plot_residual(self, wave, spec, fit):
        plt.figure()
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(wave, spec, color='b')
        axarr[0].plot(wave, fit, color='r')
        axarr[1].plot(wave, spec - fit, color='g')


'''
fit = Fit_spectra()
hdulist = astropy.io.fits.open('swp18936.silo')
hdu = hdulist[0]
fit.intensity = np.sum(hdu.data[40:60,:],0)
fit.wavelength = np.array([1050.+i*1.67634 for i in range(640)])
'''

'''
fit = Fit_spectra()
spec = Spectrum()
spec.populate(15061744)
spec.calibrate()

fit.intensity = spec.processed.intensity
fit.wavelength = spec.processed.wavelength
fit.min_val = 1900
fit.find_peak_indexes()

p = fit.setup_p()
fit.set_parameter_limits()
a = fit.fit_dat_stuff(p)
g = gaussian_model(fit.wavelength,a[0])
gs = gaussian_model_split(fit.wavelength,a[0])

for i in range(0,len(gs)):
    plt.plot(fit.wavelength,gs[i])
plt.plot(fit.wavelength,fit.intensity,color='b')
plt.plot(fit.wavelength,g,color='r')
plt.scatter(fit.centroids,fit.amplitudes,color='g')
plt.show()

'''
