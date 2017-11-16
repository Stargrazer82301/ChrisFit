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
covar_errors = [{'bands':['SPIRE_250','SPIRE_350','SPIRE_500'],'corr_err':0.05}]

# Initiate settings dictionary
settings_dict = {'plotting':True}

# Loop over galaxies
for g in cat_frame.index:
    cat_frame_gal = cat_frame.loc[g]
    if cat_frame_gal['name'] != 'NGC4030':
        continue

    # Create input dictionary for this galaxy
    gal_dict = {'name':cat_frame.loc[g],
                'distance':cat_frame_gal['dist'],
                'redshift':3E5/cat_frame_gal['vel_helio']}

    # Add empty columns to galaxy dictionary bands dataframe, to hold fluxes and uncertainties
    gal_dict['data']['flux'] = pd.Series(np.array([len(gal_dict['data'])*np.NaN]), index=gal_dict['data'].index)
    gal_dict['data']['error'] = pd.Series(np.array([len(gal_dict['data'])*np.NaN]), index=gal_dict['data'].index)

    # Loop over bands, retrieving corresponding fluxes for this galaxy (where available)
    for b in range(len(gal_dict['data']['band'])):
        band = gal_dict['data']['band'][b]
        if band in cat_frame.columns:
            gal_dict['data']['flux'][b] = cat_frame.loc[g][band]
            gal_dict['data']['error'][b] = cat_frame.loc[g][band+'_err']

        # Prune fluxes with major flags
        if isinstance(cat_frame.loc[g][band+'_flag'], str) and any(flag in cat_frame.loc[g][band+'_flag'] for flag in ['C','A','N','e']):
            gal_dict['data']['flux'] = np.NaN
            gal_dict['data']['error'] = np.NaN
    dsfdsf

    # Call ChrisFit
    out_dict = ChrisFit.Fir(gal_dict,
                            bands_frame,
                            covar_errors = covar_errors,
                            beta_vary = True,
                            beta = 2.0,
                            components = 2,
                            kappa_0 = 0.051,
                            kappa_0_lambda = 500E-6,
                            plot = True)

