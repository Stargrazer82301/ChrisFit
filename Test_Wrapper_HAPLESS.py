# Import smorgasbord
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
import ChrisFit



# Read DustPedia photometry catalogue into dataframe
cat_frame = pd.read_csv('DustPedia_Combined_Photometry_2.2.csv')

# State fitting parameters
beta_vary = True
beta = 2.0
components = 2
kappa_0 = 0.051
lambda_0 = 500E-6
plot_make = True
plot_dir = None

# Create dataframe storing basic band information
bands_frame = pd.DataFrame({'band':         ['WISE_22','Spitzer_24','IRAS_60','Spitzer_70','PACS_70','PACS_100','Spitzer_160','PACS_160','SPIRE_250','SPIRE_350','Planck_350','SPIRE_500','Planck_550', 'Planck_850', 'Planck_1380'],
                            'wavelength':   np.array([22E-6, 24E-6, 60E-6, 70E-6, 70E-6, 100E-6, 160E-6, 160E-6, 250E-6, 350E-6, 350E-6, 500E-6, 550E-6, 850E-6, 1380E-6]),
                            'limit':        [True, True, False, False, False, False, False, False, False, False, False, False, False, False, True]})

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

# State kwarg values
out_dir = 'Output/'
beta_vary = True
mcmc_n_steps = 30000

# Provide NGC4030-specific data
gal_dict = {'name':'NGC4030_nB_nC_P',
            'distance':27.16E6,
            'redshift':3E5/1465.0}
bands_frame['flux'] =  [1.94555,np.nan,18.780,np.nan,np.nan,61.02059,np.nan,69.35811,36.79235,14.85421,14.35777,5.13439,3.796851,0.62050974,0.331933]
bands_frame['error'] =  [0.32694,np.nan,3.75640,np.nan,np.nan,8.51913,np.nan,10.19806,2.77540,1.27614,0.982608782755,0.44821,0.20235413,0.097216487,0.06766073]

# Call ChrisFit
posterior = ChrisFit.Fit(gal_dict,
                         bands_frame,
                         correl_unc = None,
                         beta_vary = beta_vary,
                         beta = 2.0,
                         components = 2,
                         kappa_0 = 0.051,
                         kappa_0_lambda = 500E-6,
                         mcmc_n_walkers = 12,
                         mcmc_n_steps = mcmc_n_steps,
                         plot = out_dir)

# Provide NGC5584-specific data
gal_dict = {'name':'NGC5584_nB_nC_P',
            'distance':28.18E6,
            'redshift':3E5/1638.0}
bands_frame['flux'] =  [0.33132,np.nan,2.34,np.nan,np.nan,7.9419,np.nan,9.09188,6.34079,3.25695,2.9904351,1.35070,0.94377995,0.33604684,np.nan]
bands_frame['error'] =  [0.005587,np.nan,0.46996,np.nan,np.nan,1.49484,np.nan,1.58711,0.53389,0.30999,0.46691694068,0.13903,0.22293746,0.07334543,np.nan]

# Call ChrisFit
posterior = ChrisFit.Fit(gal_dict,
                         bands_frame,
                         correl_unc = None,
                         beta_vary = beta_vary,
                         beta = 2.0,
                         components = 2,
                         kappa_0 = 0.051,
                         kappa_0_lambda = 500E-6,
                         mcmc_n_walkers = 12,
                         mcmc_n_steps = mcmc_n_steps,
                         plot = out_dir)

