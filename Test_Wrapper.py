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

# State output directory
out_dir = 'Output/'

# List target galaxies (skipping galaxies already processed)
target_gals = ['NGC4030','NGC4559','NGC5496','NGC5584','NGC5658','NGC5690','NGC5691','NGC5705','NGC5719','NGC5740','NGC5746','NGC5750','UGC04684','UGC06879''UGC07396','UGC09299','UGC09470','UGC09482']
processed_gals = set([processed_gal.split('_')[:-1][0] for processed_gal in os.listdir(out_dir) if '.png' in processed_gal])
target_gals = list(set(target_gals) - processed_gals)

# Loop over galaxies
for g in np.random.permutation(cat_frame.index):
    cat_frame_gal = cat_frame.loc[g]
    if cat_frame_gal['name'] not in ['NGC4030','NGC4559']:#,'NGC5713','NGC5719','NGC5750','NGC5584','NGC5705','NGC5750','UGC09299']:
        continue
    bands_frame_gal = copy.deepcopy(bands_frame)

    # Create input dictionary for this galaxy
    gal_dict = {'name':cat_frame_gal['name'],
                'distance':1E6*cat_frame_gal['dist'],
                'redshift':3E5/cat_frame_gal['vel_helio']}

    # Add empty columns to galaxy dictionary bands dataframe, to hold fluxes and uncertainties
    bands_frame_gal['flux'] = pd.Series(np.array([len(bands_frame_gal)*np.NaN]), index=bands_frame_gal.index)
    bands_frame_gal['error'] = pd.Series(np.array([len(bands_frame_gal)*np.NaN]), index=bands_frame_gal.index)

    # Loop over bands, retrieving corresponding fluxes for this galaxy (where available)
    for b in range(len(bands_frame_gal['band'])):
        band = bands_frame_gal['band'][b]
        if band in cat_frame.columns:
            bands_frame_gal.loc[b,'flux'] = cat_frame.loc[:,band][g]
            bands_frame_gal.loc[b,'error'] = cat_frame.loc[:,band+'_err'][g]

        # Prune fluxes with major flags
        if isinstance(cat_frame.loc[g][band+'_flag'], str) and any(flag in cat_frame.loc[g][band+'_flag'] for flag in ['C','A','N']):
            bands_frame_gal.loc[b,'flux'] = np.NaN
            bands_frame_gal.loc[b,'error'] = np.NaN

    # Call ChrisFit
    posterior = ChrisFit.Fit(gal_dict,
                             bands_frame_gal,
                             correl_unc = correl_unc,
                             beta_vary = True,
                             beta = 2.0,
                             components = 2,
                             kappa_0 = 0.051,
                             kappa_0_lambda = 500E-6,
                             mcmc_n_walkers = 12,
                             mcmc_n_steps = 50000,
                             plot = out_dir)

