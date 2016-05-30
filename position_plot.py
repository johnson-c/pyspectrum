from least_squares_gen import Fit_spectra
from spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt


class Position_plot:

    def __init__(self):
        self.shot_list = []
        self.fitted_gaus_params = []
        self.init_gaussians = []
        self.gaussians_all = []
        self.gaussians_ind = []
# take a list of shots and make 'spectrums' for them and return
# a list of those spectrums in the given list

    def make_shot_list(self, shot_l):
        for i in range(0, len(shot_l)):
            tmp_spec = Spectrum()
            tmp_spec.populate(int(shot_l[i][0:8]), int(shot_l[i][
                              9:]), 'black_comet_200_600', 'txt_file')
            self.shot_list.append(tmp_spec)

# fit all of the shots in the shot_list. make_shot_list must be run
# before so that there is data to be fit. Gaussians from the fits
# will be saved as well as the initial guesses for the fit

    def fit_shot_list(self):
        for i in range(0, len(self.shot_list)):
            tmp_fit = Fit_spectra()
            # wavelength and intensity from the current shot in the list
            tmp_fit.wavelength = self.shot_list[i].raw.wavelength
            tmp_fit.intensity = self.shot_list[i].raw.intensity
            # set the minimum val of the fit to be 1.1*background
            tmp_fit.min_val = self.shot_list[i].raw.intensity[0] * 1.3
            tmp_fit.find_peak_indexes()
            # set up the original guess for the guassians
            orig_gaus_params = tmp_fit.setup_p()
            self.init_gaussians.append(orig_gaus_params)
            tmp_fit.set_parameter_limits()
            self.fitted_gaus_params.append(
                tmp_fit.fit_dat_stuff(orig_gaus_params)[0])
            self.gaussians_all.append(tmp_fit.gaussian_model(
                tmp_fit.wavelength, self.fitted_gaus_params[i]))
            self.gaussians_ind.append(tmp_fit.gaussian_model_split(
                tmp_fit.wavelength, self.fitted_gaus_params[i]))

    def make_position_plot(self):
        # find the spect with most gaussians found use this as the base
        max_param_ind = 0
        # find the spectra with the most lines
        for i in range(0, len(self.fitted_gaus_params)):
            if(len(self.fitted_gaus_params[i][1::3]) > len(self.fitted_gaus_params[max_param_ind][1::3])):
                max_param_ind = i
        # create array that will hold the values of the lines amplitude to be used in the
        # position plot
        pp_array = np.zeros((len(self.fitted_gaus_params), len(
            self.fitted_gaus_params[max_param_ind][1::3])))
        # populate the array need to loop over the total number of lines found in the maximum
        # spectra. Also need to loop over the number of spectra in the shot
        # list
        for i in range(0, len(self.fitted_gaus_params[max_param_ind][1::3])):
            for j in range(0, len(self.fitted_gaus_params)):
                # find which index of the current spectra is closest to the line that is closest
                # to the line from the max_param_ind spectra
                min_arg_ind = np.abs(self.fitted_gaus_params[max_param_ind][
                                     1::3][i] - self.fitted_gaus_params[j][1::3]).argmin()
                # if the closest line's centroid is greater than 2sigmas of the max_param_ind
                # then dont include it in the position plot
                if(np.abs(self.fitted_gaus_params[max_param_ind][1::3][i] - self.fitted_gaus_params[j][1::3][min_arg_ind]) < np.average(self.fitted_gaus_params[max_param_ind][2::3]) * 2):

                    pp_array[j, i] = self.fitted_gaus_params[
                        j][0::3][min_arg_ind]
                else:
                    # if farther than 2sigma then just give it zero
                    pp_array[j, i] = 0
        return pp_array


'''
plot fits, initial conditions, and data

for i in range(0, len(pp.gaussians_all)):
    plt.figure()
    plt.plot(pp.shot_list[i].raw.wavelength, pp.shot_list[i].raw.intensity,color='b')
    plt.plot(pp.shot_list[i].raw.wavelength,pp.gaussians_all[i],color='g')
    plt.scatter(pp.init_gaussians[i][1::3],pp.init_gaussians[i][0::3][0:len(pp.init_gaussians[i][1::3])],color='r')
'''


'''
shot_list = np.array([15060806,15060807,15060808,15060809,15060810])
popt = []
for i in range(0,len(shot_list)):
    print(i)
    pop,wave,inten = fit1.fit_shot(int(shot_list[i]))
    popt.append(pop)



c = []
for i in range(0,len(popt[1][0::3])):
    a = np.abs(popt[0][0::3] - popt[1][0::3][i]).argmin()
    if ( np.abs(popt[0][0::3][a] - popt[1][0::3][i]) < 0.2):
        c.append(popt[0][0::3][a])
tmp = np.zeros( (len(c),len(popt)) )
tmp[:,0] = c


for j in range(1,len(popt)):
    for i in range(0,len(popt[j][0::3])):
        a = np.abs(popt[0][0::3] - popt[j][0::3][i]).argmin()
        if ( np.abs(popt[0][0::3][a] - popt[j][0::3][i]) < 0.2):
            temp = np.abs( tmp[:,0] - popt[j][0::3][i]).argmin()
            if(np.abs( tmp[temp,0] - popt[j][0::3][i]) < 0.2):

                tmp[temp,j] = popt[j][1::3][i]


for i in range(0,len(tmp)):
    plt.figure(tmp[i,0])
    plt.title('Line ' + str(tmp[i,0]))
    for j in range(0,len(tmp[0,:])):
        plt.scatter(j,tmp[i,j])
'''
