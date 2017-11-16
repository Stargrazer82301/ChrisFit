# Import smorgasbord
from __future__ import print_function
import sys
import os
import numpy as np






def fit(gal_dict,
        bands_frame,
        beta_vary = True,
        beta = 2.0,
        components = 2,
        kappa_0 = 0.051,
        kappa_0_lambda = 500E-6,
        plot = True,
        covar_unc = None):
        """
        Function that runs the ChrisFit dust SED fitting routine.

        Arguments:
            gal_dict:           A dictionary, containing entries called 'name', 'distance', and 'redshift', giving the values for the target source in question
            bands_frame:        A dataframe, with columns called 'band', 'flux', and 'error', providing the relevant values for each band for the target source in question

        Keyword arguments:
            beta_vary:          A boolean, stating whether or not beta (the emissivity slope) should a free parameter, or fixed
            beta:               A float, stating the value of beta to use if beta_vary is set to true. If beta_vary is set to false, beta will provide starting position for MCMC
            components:         An integer, stating how many modified blackbody components should make up the model being fit
            kappa_0:            The value of the dust mass absorption coefficient, kappa_d, to use to caculate dust mass (uses Clark et al., 2016, value by default)
            kappa_0_lambda:     The reference wavelength for kappa_0; corresponding value of kappa_0 at other wavelengths extrapolated via (lambda_0/lambda)**beta
            plot:               A boolean, stating whether to generate plots of the resulting SED fit
            covar_unc:          A dictionary, describing band-covariant uncertainties (if any); eg, 5% Hershcel-SPIRE band covariance: covar_error = [{'bands':['SPIRE_250','SPIRE_350','SPIRE_500'],'corr_err':0.05}]
        """



