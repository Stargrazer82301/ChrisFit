# Import smorgasbord
import pdb
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ChrisFuncs
import ChrisFit





# Read DustPedia photometry catalogue into dataframe
cat_frame = pd.read_csv('DustPedia_Aperture_Photometry_2.2.csv')

# State fitting parameters
beta_vary = True
beta = 2.0
components = 2
kappa_0 = 0.051
lambda_0 = 500E-6
plot_make = True
plot_dir = None


# Create dataframe storing basic band information
bands_frame = pd.DataFrame({'band':         ['WISE_22','PACS_70','PACS_100','PACS_160','SPIRE_250','SPIRE_350','SPIRE_500'],
                            'wavelength':   np.array([22E-6, 70E-6, 100E-6, 160E-6, 250E-6, 350E-6, 500E-6]),
                            'limit':        [True, False, False, False, False, False, False]})

# Add correlated uncertainty information to band dataframe
covar_unc = [{'covar_bands':['SPIRE_250','SPIRE_350','SPIRE_500'],
              'covar_scale':0.04,
              'covar_distr':'flat'}]

# Initiate settings dictionary
settings_dict = {'plotting':True}

# Loop over galaxies
for g in cat_frame.index:
    cat_frame_gal = cat_frame.loc[g]
    if cat_frame_gal['name'] != 'NGC4030':
        continue
    bands_frame_gal = copy.deepcopy(bands_frame)

    # Create input dictionary for this galaxy
    gal_dict = {'name':cat_frame.loc[g],
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
        if isinstance(cat_frame.loc[g][band+'_flag'], str) and any(flag in cat_frame.loc[g][band+'_flag'] for flag in ['C','A','N','e']):
            bands_frame_gal.loc[b,'flux'] = np.NaN
            bands_frame_gal.loc[b,'error'] = np.NaN

    # Call ChrisFit
    out_dict = ChrisFit.Fit(gal_dict,
                            bands_frame_gal,
                            covar_unc = covar_unc,
                            beta_vary = True,
                            beta = 2.0,
                            components = 2,
                            kappa_0 = 0.077,#0.051,
                            kappa_0_lambda = 850E-6,#500E-6,
                            plot = True)

