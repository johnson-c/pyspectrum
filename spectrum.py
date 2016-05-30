#
# Spectral_shot.py
# This script will be the main script for the class to hold the data
# This script will only serve as a data structure and it will be manipulate by
# other programs to do things like position plots ect.
# The 'spectrum' class will hold of this data
#
# This script need the spec_utils script as well as the fit_gaussians script
#
#

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from spec_utils_testing import *
import re
#
#
# Spectrum
#     This is where all of the data that a spectrometer shot can have
# the idea is to have all of the data in the one class and then manipulate it
# For multiple spectrometers we could create  a super class that would be made
# up of these individual Spectrum classes
#
#
#


class Spectrum:

    class info:  # info class will hold all of the strings associated with the shot

        def __init__(self):
            self.file = ''
            self.notes = ''
            self.spect_name = ''
            self.integration_ms = ''
            self.shot_num = ''
            self.exposure_num = ''
            self.computer_name = ''
            self.device = ''
            self.time_stamp = ''
            self.spec_loc = ''
            self.file_path = ''
            self.device = ''
            self.execution_time = ''
            self.start_exposure = ''
            self.exeternal_trig = ''
            self.ao = ''
            self.temp_comp = ''
            self.osl = ''
            self.nsa = ''
            self.drs = ''
            self.notes = ''

    class raw:  # This will hold the raw data, it will never be touch except to
                # get processed. Just for a sanity check if needed

        def __init__(self):
            self.intensity = np.zeros(1)
            self.wavelength = np.zeros(1)

    class processed:  # This will be the bulk of the data

        def __init__(self):
            self.calibration_coeff = 'none'  # coeffs used in the calibration
            self.wavelength = np.zeros(
                1)  # wavelength that will be used may or may not use cali
            self.intensity = np.zeros(
                1)  # intensity that will be used may or may not use cali
            self.gaussians = np.zeros(
                (1, 1))  # Holds the gaussian fit to the cali data
            self.background = 0
            self.min_val = 0
            self.new_points = np.array([-1])

    def __init__(self):
        self.raw = self.raw()
        self.info = self.info()
        self.processed = self.processed()

    def populate(self, shot=-1, exposure=-1, spec=-1, loc='', path=''):
        pop_spectrum(self, shot, exposure, spec, loc, path)
                     # See the pop_spectrum def in this script

    def calibrate(self, calibration_num=1, direction='right'):
        get_calibration(self, calibration_num)  # Get the calibration file
        cal_spectrum(self, direction='right')  # calibrate the data

    def fit_gaussians(self, plot='no', chunks=np.array([20, 1000, 1500, 2000]), new_points=np.array([-1])):
        if(self.processed.min_val == 0):
            print('No min value for peak specified')
        if(new_points[0] != -1):
            print('here')
            add_lines(self, new_points)

        if(self.processed.background == 0):
            find_background(self)

        self.processed.gaussians = np.reshape(
            fit_shot(self, chunks=chunks, types='ohmic', plot=plot, new_points=np.array([-1]), background=self.processed.background), (-1, 3))

    def save_spectra(self):
        save_spectrum(self)


def add_lines(self, new_points):
    self.processed.new_points = new_points
    print(self.processed.new_points)


def save_spectrum(self):

    f = open(self.info.file_path[0:49] + self.info.file_path[
             40:48] + '_cal/' + self.info.file_path[49:57] + '_cal.txt', 'w')
    f.write(self.info.file_path[20:] + '\n')
    f.write(self.info.file)
    f.write(self.info.other)
    np.savetxt(
        f, np.c_[self.processed.wavelength, self.processed.intensity], fmt='%10.3f')
    f.close()


#
#
# find_background,
#       Goes and take the first 10 pixels as the background level of entire spectra
#
#
def find_background(self):
    self.processed.background = np.average(self.processed.intensity[0:10])


#
#
# cal_spectrum, this grabs the data from processed.calibration_coeff and raw.wavelength
# if there is no calibration_coeff then processed.wavelength = raw.wavelength
# else use the calibration_coeff and shift in the specified direction
#
#
def cal_spectrum(self, direction):
    if(self.processed.calibration_coeff == 'none'):
        self.processed.wavelength = self.raw.wavelength
        self.processed.intensity = self.raw.intensity
        return

    if(direction == 'right'):
        for i in range(0, len(self.raw.wavelength)):
            quad_shift = quad(
                self.raw.wavelength[i], self.processed.calibration_coeff)
            self.processed.wavelength[i] = self.raw.wavelength[i] + quad_shift

    if(direction == 'left'):
        for i in range(0, len(self.raw.wavelength)):
            quad_shift = quad(
                self.raw.wavelength[i], self.processed.calibration_coeff)
            self.processed.wavelength[i] = self.raw.wavelength[i] - quad_shift

    self.processed.intensity = self.raw.intensity


#
#
# get_calibration goes and gets the calibration file for the spectrum
# sets the 'processed.calibration_coeff'
#
#
#

def get_calibration(self, calibration_num):

    cal_path = '/home/curtis/bonnie/_Users/curt/spectra/calibration_coeffs/'
    cal_file = cal_path + self.info.spec_name + str(calibration_num) + '.cal'

    self.processed.calibration_coeff = np.loadtxt(cal_file)


#
#
# pop_spectrum, go and get the nessary data from either bonnie or the tree
#
#
#
#

def pop_spectrum(self, shot, exposure, spec, loc, path):
    self.info.shot_num = shot
    self.info.exposure_num = exposure
    self.info.spec_name = spec

    if (loc == 'txt_file'):
        shot_str = str(shot)
        exposure_str = str(exposure)
        try:
            path = '/home/curtis/bonnie/_Users/curt/spectra/' + shot_str[0:2] + '_' + shot_str[
                2:4] + '_' + shot_str[4:6] + '/' + spec + '/' + shot_str[6:8] + '_' + exposure_str + '.txt'
            spec_file = open(path)

        except IOError:
            print('File was not found')
            shot_str = raw_input('Retry shot')
            exposure_str = raw_input('Retry exposure')

            spec = raw_input('Retry spectrometer name')
            path = '/home/curtis/bonnie/_Users/curt/spectra/' + shot_str[0:2] + '_' + shot_str[
                2:4] + '_' + shot_str[4:6] + '/' + spec + '/' + shot_str[6:8] + '_' + exposure_str + '.txt'
            spec_file = open(path)

        self.info.time_stamp = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.shot_num = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.exoposure_num = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.computer_name = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.spec_name = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.device = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.spec_loc = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.coeffs = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.integration_ms = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.execution_time = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.start_exposure = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.spec_delay = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.external_trig = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.ao = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.temp_comp = re.findall(
            '[^\r]*', spec_file.readline()[33:])[0]
        self.info.osl = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.nsa = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.drs = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        self.info.notes = re.findall('[^\r]*', spec_file.readline()[33:])[0]
        spec_file.readline()
        data_tmp = np.loadtxt(spec_file)
        self.raw.wavelength = data_tmp[:, 0]
        self.raw.intensity = data_tmp[:, 1]

    elif (loc == 'mdsplus'):
        self.info.path = 'MDSplus'

    else:
        print('youre dumb')


#
# Gets the nessary data from data files that are saved on bonnie
# Sets the 'file','other','integration','raw.wavelength','raw.intensity' of the
# Spectrum class
#
#


def spec_from_bonnie(self, spec_file):
    self.info.file = spec_file.readline()
                                        # get the first line of the .SSM data
                                        # file
    self.info.other = spec_file.readline()  # get the second line
    other = self.info.other
    self.info.integration_ms = int(
        other[other.index('Time:') + 5:  other.index('ms')])
    data_tmp = np.loadtxt(spec_file)
                           # remaining lines in the file are just wavelength,
                           # intensity
    self.raw.wavelength = data_tmp[:, 0]
    self.raw.intensity = data_tmp[:, 1]  # get the int time

    optimized_wavelength = float(
        other[16:22])  # Each new spec should have different optimized wavelength

    print(round(self.raw.wavelength[0], 2))
    if ((round(self.raw.wavelength[0], 2) == 190.78) or (round(self.raw.wavelength[0], 2) == 194.00)):
        self.info.spec_name = 'CTH Black Comet (HR-UV-14, 11041114, 200-600 nm)'
    elif (optimized_wavelength == 501.42):
        self.info.spec_name = 'Ivan Blue Wave (HR-VIS-, get serial from ivan, 400-600)'
    elif (round(self.raw.wavelength[0], 2) == 188.92):
        self.info.spec_name = 'CTH EPP-2000, (LHR-UV-3-7, 15072904, 200-300 nm)'
    else:
        self.info.spec_name = 'Unidentified spectrometer'


#
#
# Get the file from shot name.
# This assumes that the file was saved in the prescibed way that all spectra are saved
#
#
#
def path_from_shot(shot, spec):
    shot = str(shot)
    path = '/home/curtis/bonnie/_Users/curt/spectra/' + \
        shot[0:2] + '_' + shot[2:4] + '_' + shot[4:6] + '/' + shot + '.SSM'
    if(spec != -1):
        path = '/home/curtis/bonnie/_Users/curt/spectra/' + \
            shot[0:2] + '_' + shot[2:4] + '_' + shot[4:6] + \
            '/' + shot + '-c' + str(spec) + '.SSM'

    return path


a = Spectrum()
