# Identify location
import socket
location = socket.gethostname()
if location == 'Orthanc':
    dropbox = 'E:\\Users\\Chris\\Dropbox\\'
if location == 'sputnik':
    dropbox = '/home/chris/Dropbox/'
if location in ['saruman','serpens']:
    dropbox = '/home/user/spx7cjc/Desktop/Herdata/Dropbox/'

# Import smorgasbord
import os
import sys
import copy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.stats
import scipy.ndimage
import pandas as pd
sys.path.append(os.path.join(dropbox,'Work','Scripts','ChrisFit'))
import ChrisFit

# Read DustPedia photometry catalogue as dataframe
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
            'distance':30E6,
            'redshift':0.00475}

# Construct flat temperature, mass, and beta priors
def TempPrior(temp):
    if (temp <= 10) or (temp > 100):
        return -np.inf
    else:
        return 0
def MassPrior(mass):
    if (mass <= 0) or (mass > 1E15):
        return -np.inf
    else:
        return 0
def BetaPrior(beta):
    if (beta <= 0) or (beta > 5):
        return -np.inf
    else:
        return 0

# Construct flat priors
priors = {'temp':[copy.deepcopy(TempPrior), copy.deepcopy(TempPrior)],
          'mass':[copy.deepcopy(MassPrior), copy.deepcopy(MassPrior)],
          'beta':[copy.deepcopy(BetaPrior)]}

## Use calibration and observational uncertainties to generate artificial errors
#calib_unc = np.array([0.05, 0.05, 0.2, 0.07, 0.07, 0.07, 0.055, 0.055, 0.055])
#obs_err = (1 + np.abs(np.random.normal(scale=2, size=calib_unc.shape))) * np.array([0.07, 0.15, 0.05, 0.05, 0.07, 0.05, 0.03, 0.03, 0.06])
#inject_err = np.sqrt(obs_err**2.0 + calib_unc**2.0)
#
## Add empty columns to galaxy dictionary bands dataframe, to hold fluxes and uncertainties
#bands_frame['flux'] = pd.Series(np.array([len(bands_frame)*np.NaN]), index=bands_frame.index)
#bands_frame['error'] = pd.Series(np.array([len(bands_frame)*np.NaN]), index=bands_frame.index)
#
## Decide underlying properties of source
#inject_temp = [21.7, 64.2]
#inject_mass = [10.0**8.5, 10.0**6.0]
#inject_beta = [1.9]
#inject_params = inject_temp+inject_mass+inject_beta
#
## Generate artificial fluxes
#inject_flux = ChrisFit.ModelFlux(bands_frame['wavelength'], inject_temp, inject_mass, gal_dict['distance'], kappa_0=0.051, kappa_0_lambda=500E-6, beta=inject_beta)
#
## Use uncertainties to produce noisy fluxes with errors (with  extra emission added to bands that are just upper limits)
#bands_frame['flux'] = inject_flux + (np.random.normal(loc=0.0, scale=inject_err) * inject_flux)
#bands_frame['flux'] *= 10.0**(bands_frame['limit'].values.astype(float) * np.abs(np.random.normal(loc=1.0, scale=1.0)))
#bands_frame['error'] = bands_frame['flux'] * inject_err
#
## Limit bands frame to the specific bands that are actually wanted for this run
#bands_use = ['Spitzer_24','PACS_70','PACS_100','PACS_160','SPIRE_250','SPIRE_350','SPIRE_500']
#bands_frame = bands_frame.loc[np.where(np.in1d(bands_frame['band'],bands_use))]



gal_dict = {'name':'Test',
            'distance':4.9E6,
            'redshift':0.00075}
bands_frame = pd.DataFrame({'band':         ['Spitzer_24','PACS_70','PACS_160','SPIRE_250','SPIRE_350','SPIRE_500'],
                            'wavelength':   np.array([24E-6,70E-6,160E-6, 250E-6, 350E-6, 500E-6]),
                            'limit':        [True, False, False, False, False, False],
                            'flux':         np.array([0.0287, 0.461, 0.611, 0.242, 0.0895, 0.0288]),
                            'error':        np.array([0.000431, 0.0132, 0.00895, 0.00232, 0.000873, 0.000361])})
calib_unc = 1.0 * np.array([0.05, 0.07, 0.07, 0.055, 0.055, 0.055])
bands_frame['error'] = np.sqrt(bands_frame['error']**2.0 + (bands_frame['flux']*calib_unc)**2.0)



# Call ChrisFit
output = ChrisFit.Fit(gal_dict,
                      bands_frame,
                      correl_unc = correl_unc,
                      beta_vary = True,
                      beta = 2.0,
                      components = 2,
                      kappa_0 = 0.051,
                      kappa_0_lambda = 500E-6,
                      mcmc_n_walkers = 500,
                      mcmc_n_steps = 500,
                      #mcmc_n_threads = 1,
                      simple_clean = 0.5,
                      plot = 'Output/',
                      test = False,
                      priors = None)

# Jubilate
print('All done!')