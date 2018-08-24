# Import smorgasbord
import sys
import pdb
import os
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
import inspect
sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))))
sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))))
import ChrisFit


# Create input dictionary for this galaxy
gal_dict = {'name':'Test Pixel In NGC628',
            'distance':10.6E6,
            'redshift':0.00219}

# Create dataframe storing basic band information
bands_frame = pd.DataFrame({'band':         ['WISE_22','PACS_70','PACS_100','PACS_160','SPIRE_250','SPIRE_350','SPIRE_500'],
                            'wavelength':   np.array([22E-6, 70E-6, 100E-6, 160E-6, 250E-6, 350E-6, 500E-6,]),
                            'flux':         np.array([1.319E-3, 0.01478, 0.02406, 0.03998, 0.02324, 0.01153, 0.004160]),
                            'error':        np.array([1.042E-4, 0.01129, 0.007339, 0.004859, 0.002094, 0.001200, 0.00051224]),
                            'limit':        [True, False, False, False, False, False, False]})

# Construct function for SPIRE correlated uncertainty
def SpireCorrelUnc(prop, unc=0.04):
    if abs(prop) > (5*unc):
        return -np.inf
    else:
        x = np.linspace(-5*unc, 5*unc, 5E3)
        y = np.zeros([x.size])
        y[np.where(np.abs(x)<=unc)] = 1
        y = scipy.ndimage.filters.gaussian_filter1d(y, sigma=len(x)*(0.005/(x.max()-x.min())))
        return y[(np.abs(x-prop)).argmin()]

# Add correlated uncertainty information to band dataframe
correl_unc = [{'correl_bands':['SPIRE_250','SPIRE_350','SPIRE_500'],
               'correl_scale':0.04,
               'correl_distr':SpireCorrelUnc}]

# State output directory
out_dir = 'Output/'

# Call ChrisFit
posterior = ChrisFit.Fit(gal_dict,
                         bands_frame,
                         correl_unc = correl_unc,
                         beta_vary = True,
                         beta = 2.0,
                         components = 2,
                         kappa_0 = 0.051,
                         kappa_0_lambda = 500E-6,
                         mcmc_n_walkers = 500,
                         mcmc_n_steps = 500,
                         plot = True,
                         simple_clean = 0.5,
                         test = False)

