# Identify location
import socket
location = socket.gethostname()
if location == 'Orthanc':
    dropbox = 'E:\\Users\\Chris\\Dropbox\\'
if location == 'sputnik':
    dropbox = '/home/chris/Dropbox/'
if location in ['saruman','rosemary-pc']:
    dropbox = '/home/user/spx7cjc/Desktop/Herdata/Dropbox/'

# Import smorgasbord
import pdb
import os
import sys
import copy
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.stats
import scipy.ndimage
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.join(dropbox,'Work','Scripts','ChrisFit'))
import ChrisFit



# Read DustPedia photometry catalogue into dataframe
cat_frame = pd.read_csv('DustPedia_Combined_Photometry_2.2.csv')

# Create dataframe storing basic band information
bands_frame = pd.DataFrame({'band':         ['Spitzer_24','SOPHIA_53','IRAS_60','PACS_70','PACS_100','PACS_160','SPIRE_250','SPIRE_350','SPIRE_500'],
                            'wavelength':   np.array([24E-6, 53E-6, 60E-6, 70E-6, 100E-6,160E-6, 250E-6, 350E-6, 500E-6]),
                            'limit':        [True, True, False, False, False, False, False, False, False]})

# Construct function for SPIRE correlated uncertainty
def SpireCorrelUnc(prop, unc=0.04):
    if abs(prop) > (5*unc):
        return -np.inf
    else:
        x = np.linspace(-5*unc, 5*unc, 1E3)
        y = np.zeros([x.size])
        y[np.where(np.abs(x)<=unc)] = 1
        y = scipy.ndimage.filters.gaussian_filter1d(y, sigma=len(x)*(0.005/(x.max()-x.min())))
        return y[(np.abs(x-prop)).argmin()]

# Add correlated uncertainty information to band dataframe
correl_unc = [{'correl_bands':['SPIRE_250','SPIRE_350','SPIRE_500'],
               'correl_scale':0.04,
               'correl_distr':SpireCorrelUnc}]

# State output directory
out_dir = '/Output'

# Create input dictionary for this "galaxy"
gal_dict = {'name':'Test',
            'distance':10E6,
            'redshift':0.0025}

# Add empty columns to galaxy dictionary bands dataframe, to hold fluxes and uncertainties
bands_frame['flux'] = pd.Series(np.array([len(bands_frame)*np.NaN]), index=bands_frame.index)
bands_frame['error'] = pd.Series(np.array([len(bands_frame)*np.NaN]), index=bands_frame.index)

# Decide underlying properties of source
inject_temp = [24.0]#[26.0, 51.0]
inject_mass = [10.0**5.5]#[10.0**6.5, 10.0**3.0]
inject_beta = 2.1

# Generate artificial fluxes
inject_flux = ChrisFit.ModelFlux(bands_frame['wavelength'], inject_temp, inject_mass, gal_dict['distance'], kappa_0=0.051, kappa_0_lambda=500E-6, beta=inject_beta)

# Use calibration and observational uncertainties to generate artificial errors
obs_err = np.array([0.1, 0.2, 0.2, 0.15, 0.1, 0.3, 0.15, 0.15])
calib_unc = np.array([0.05, 0.07, 0.07, 0.07, 0.055, 0.055, 0.064, 0.055])
inject_err = np.sqrt(obs_err**2.0 + calib_unc**2.0)

# Use uncertainties to produce noisy fluxes with errors (with  extra emission added to bands that are just upper limits)
bands_frame['flux'] = inject_flux + (np.random.normal(loc=0.0, scale=inject_err) * inject_flux)
bands_frame['flux'] *= (1.0 + (bands_frame['limit'].values.astype(float) * np.abs(np.random.normal())))
bands_frame['error'] = bands_frame['flux'] * inject_err

# Call ChrisFit
posterior = ChrisFit.Fit(gal_dict,
                         bands_frame,
                         correl_unc = correl_unc,
                         beta_vary = True,
                         beta = 2.0,
                         components = 1,
                         kappa_0 = 0.051,
                         kappa_0_lambda = 500E-6,
                         mcmc_n_walkers = 12,
                         mcmc_n_steps = 10000,
                         plot = '',#os.path.join(dropbox,'Work'),
                         test = False)

