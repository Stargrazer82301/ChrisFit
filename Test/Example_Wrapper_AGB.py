# Import smorgasbord
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.stats
import scipy.ndimage
import astropy.io.fits
import pandas as pd
import multiprocessing as mp
import hickle
import inspect
import matplotlib
matplotlib.use('Agg')
sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))))
import ChrisFit_AGB



# Create input dictionary for this source
target_name = 'IRC+10216'
target_dict = {'name':     target_name,
               'distance': 150.0, # Taken from De Beck+ (2012), not from NESS catalogue
               'redshift': 0.0}

# Create dictionary storing basic band information
bands_dict = {
               'Planck_10600':   {'wavelength':10600E-6, 'flux':6.19E-3, 'flux_unc':87E-3,  'calib_unc_rel':0.0035, 'limit':True}, # From aperture photometry in ap out to 0.5 FWHM radius, sky ann from 2-3 FWHM
               'Planck_6810':    {'wavelength':6810E-6,  'flux':3.79E-2, 'flux_unc':134E-3, 'calib_unc_rel':0.0026, 'limit':True}, # From aperture photometry in ap out to 0.5 FWHM radius, sky ann from 2-3 FWHM
               'Planck_4260':    {'wavelength':4260E-6,  'flux':0.119,   'flux_unc':101E-3, 'calib_unc_rel':0.0020, 'limit':True}, # From aperture photometry in ap out to 0.5 FWHM radius, sky ann from 2-3 FWHM
               'Planck_3000':    {'wavelength':3000E-6,  'flux':0.903,   'flux_unc':0.192,  'calib_unc_rel':0.0009, 'limit':True},
               'Planck_2100':    {'wavelength':2100E-6,  'flux':0.886,   'flux_unc':0.251,  'calib_unc_rel':0.0007, 'limit':True},
               'Planck_1380':    {'wavelength':1380E-6,  'flux':3.502,   'flux_unc':0.418,  'calib_unc_rel':0.0016, 'limit':True},
               'Planck_850':     {'wavelength':850E-6,   'flux':8.290,   'flux_unc':0.361,  'calib_unc_rel':0.0078, 'limit':False},
               'Planck_550':     {'wavelength':550E-6,   'flux':21.19,   'flux_unc':0.810,  'calib_unc_rel':0.061,  'limit':False},
               'Planck_350':     {'wavelength':350E-6,   'flux':59.91,   'flux_unc':2.770,  'calib_unc_rel':0.064,  'limit':False},
               'IRAS_12':        {'wavelength':12E-6,    'flux':4.75E4,  'flux_unc':3.80E3, 'calib_unc_rel':0.051,  'limit':True},
               'IRAS_25':        {'wavelength':25E-6,    'flux':2.31E4,  'flux_unc':1.85E3, 'calib_unc_rel':0.151,  'limit':True},
               'IRAS_60':        {'wavelength':60E-6,    'flux':5.65E3,  'flux_unc':452.0,  'calib_unc_rel':0.104,  'limit':True},
               'IRAS_100':       {'wavelength':100E-6,   'flux':922.0,   'flux_unc':46.1,   'calib_unc_rel':0.135,  'limit':False}
               }

# Convert dictionary to dataframe, and handle dtypes
bands_frame = pd.DataFrame.from_dict(bands_dict).transpose()
bands_frame['band'] = bands_frame.index.copy()
bands_frame['index'] = np.arange(0, bands_frame.shape[0])
bands_frame = bands_frame.set_index('index')
bands_frame['flux'] = bands_frame['flux'].astype(float)

# Compute errors by adding in quadrature the uncertainties on flux and calibration
bands_frame['calib_unc'] = bands_frame['flux'] * bands_frame['calib_unc_rel']
bands_frame['error'] = np.array((bands_frame['flux_unc']**2.0 + bands_frame['calib_unc']**2.0)**0.5).astype(float)

# State output directory
out_dir = 'output/'

# Call ChrisFit
sed_out = ChrisFit_AGB.Fit(target_dict,
                           bands_frame,
                           beta_vary = True,
                           beta = 2.0,
                           components = 1,
                           kappa_0 = 2.6,
                           kappa_0_lambda = 160E-6,
                           mcmc_n_threads = int(round(mp.cpu_count()*0.35)),
                           mcmc_n_walkers = 200,
                           mcmc_n_steps = 1000,
                           plot = out_dir,
                           simple_clean = 0.5,
                           mle_only = False,
                           full_posterior=True)

# Jubilate
print('And furthermore, Carthage must be destroyed!')