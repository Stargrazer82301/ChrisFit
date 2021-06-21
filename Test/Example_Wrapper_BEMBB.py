# Import smorgasbord
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.stats
import scipy.ndimage
import pandas as pd
import multiprocessing as mp
import inspect
import matplotlib
matplotlib.use('Agg')
sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))))
import ChrisFitBEMBB



# Create input dictionary for this galaxy
gal_dict = {'name':'Test_Pixel_In_NGC628',
            'distance':10.6E6,
            'redshift':0.00219}

# Create dataframe storing basic band information
bands_frame = pd.DataFrame({'band':         ['WISE_22','PACS_70','PACS_100','PACS_160','SPIRE_250','SPIRE_350','SPIRE_500'],
                            'wavelength':   np.array([22E-6, 70E-6, 100E-6, 160E-6, 250E-6, 350E-6, 500E-6,]),
                            'flux':         np.array([1.319E-3, 0.01478, 0.02406, 0.03998, 0.02324, 0.01153, 1.4*0.004160]),
                            'error':        np.array([1.042E-4, 0.01129, 0.007339, 0.004859, 0.002094, 0.001200, 0.00051224]),
                            'limit':        [True, True, True, False, False, False, False]})

# Construct function for SPIRE correlated uncertainty
def SpireCorrelUnc(prop, unc=0.04):
    if abs(prop) > (3*unc):
        return -np.inf
    else:
        x = np.linspace(-3*unc, 3*unc, 500)
        y = np.zeros([x.size])
        y[np.where(np.abs(x)<=unc)] = 1
        y = scipy.ndimage.filters.gaussian_filter1d(y, sigma=len(x)*(0.005/(x.max()-x.min())))
        return y[(np.abs(x-prop)).argmin()]

# Store correlated uncertainty information
correl_unc = [{'correl_bands':['SPIRE_250','SPIRE_350','SPIRE_500'],
               'correl_scale':0.04,
               'correl_distr':SpireCorrelUnc}]

# State output directory
out_dir = 'Output_BEMBB/'

# Call ChrisFit
posterior = ChrisFitBEMBB.FitBEMBB(gal_dict,
                                   bands_frame,
                                   correl_unc = correl_unc,
                                   beta = (2.0, 1.5),
                                   break_lambda = 225E-6,
                                   kappa_0 = 0.051,
                                   kappa_0_lambda = 500E-6,
                                   mcmc_n_threads = int(round(mp.cpu_count()*1.0)),
                                   mcmc_n_walkers = 250,#250,
                                   mcmc_n_steps = 500,#500,
                                   plot = out_dir,
                                   simple_clean = 0.50,
                                   map_only = False)

