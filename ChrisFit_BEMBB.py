# Import smorgasbord
from __future__ import print_function
import sys
import os
import copy
import re
import gc
import warnings
import time
warnings.filterwarnings('ignore')
from difflib import SequenceMatcher
import multiprocessing as mp
import numpy as np
import scipy.stats
import scipy.interpolate
import scipy.optimize
import scipy.ndimage
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import sklearn.neighbors
import progress.bar
import signal
import termcolor
import acor
import corner
import emcee

# Disable interactive plotting
plt.ioff()

# Define physical constants
c = 3E8
h = 6.64E-34
k = 1.38E-23





def FitBEMBB(gal_dict,
    bands_frame,
    beta = (2.0, 1.75),
    break_lambda = 175E-6,
    kappa_0 = 0.051,
    kappa_0_lambda = 500E-6,
    plot = True,
    correl_unc = None,
    priors = None,
    mcmc_n_walkers = 250,
    mcmc_n_steps = 2500,
    mcmc_n_threads = int(round(mp.cpu_count()*1.0)),
    simple_clean = 0.5,
    full_posterior = True,
    mle_only = False,
    map_only = False,
    danger = False,
    verbose = True):
    """
    Function that runs the ChrisFit dust SED fitting routine, adapted from the main function for a
    Broken-Emissivity Modifed Dlackbody (BEMBB) model

    Arguments:
        gal_dict:
                A dictionary, containing entries called 'name', 'distance', and 'redshift', giving the
                values for the target source in question (with distance in units of parsecs)
        bands_frame:
                A dataframe, with columns called 'band', 'wavelength, 'flux', 'error', and 'limit' providing the
                relevant values for each band for the target source in question

    Keyword arguments:
        beta:
                A list of two floats, stating the values of beta to use when initialising fitting
        break_lambda:
                A float, giving the wavelength (in m) of the value of the beta break wavelength with wich to
                initialise the fitting
        kappa_0:
                The value of the dust mass absorption coefficient, kappa_d, to use to calculate dust mass
                (uses Clark et al., 2016, value by default)
        kappa_0_lambda:
                The reference wavelength for kappa_0; corresponding value of kappa_0 at other
                wavelengths extrapolated via (kappa_0_lambda/lambda)**beta
        plot:
                A boolean, stating whether to generate plots of the resulting SED fit; or,
                alternatively, a string pointing to the desired plotting output directory
        correl_unc:
                A list, each element of which (if any) is a dictionary describing band-covariant
                uncertainties; for the 5% Hershcel-SPIRE band covariance, correl_unc would be:
                [{'correl_bands':['SPIRE_250','SPIRE_350','SPIRE_500'],
                'correl_scale':0.04,
                'correl_distr':'flat'}],
                where 'bands' describes the bands (as named in bands_frame) in question, 'correl_scale'
                describes the size of the covariant component of the flux uncertainty (as a fraction of
                measured source flux), and 'correl_distr' is the assumed distribution of the uncertainty
                (currently accepting either 'flat', 'normal', or a defined function)
        priors:
                A dictionary, of lists, of functions (yeah, I know); dictionary entries can be called
                'temp', 'mass', and 'beta'; note that the priors for any correlated uncertainty terms should
                be provided through the correl_unc kwarg instead
        mcmc_n_walkers:
                An int, stating how many emcee MCMC walkers should be used to explore the posterior; the higher the
                number, the less likely you are to find yourself stuck in a local minima
        mcmc_n_steps:
                An integer, stating how many steps each emcee MCMC walker should take whilst exploring the
                posterior; the more steps, the better the likelihood of convergence and good sampling
        mcmc_n_threads:
                An integer, stating how many CPU threads emcee should use for its MCMC sampling; if you have a
                small number of walkers, setting this to 1 might be faster
        simple_clean:
                A boolean or float, descrbing the type of chain cleaning to be performed. If False, a full cleaning
                will be performed, with convergance diagnostics to remove burn-in, metastability analysis to exclude
                bad chains. If a float in the range 0-1 is given, then all that is done is that fraction of the
                total is removed as burn-in; if you have enough of walkers (ie, hundreds) and steps (ie, thousands),
                then using this simple option and removing the first 50% all chains should give a good output
        full_posterior:
                A boolean, stating whether the full posterior distribution of each parameter should be
                returned, or just the summary of median, credible interval, etc
        mle_only:
                A boolean, which requests that only a quick Maximum Likelihood Estimation be performed.
        map_only:
                A boolean, which requests that only a quick Maximum A posteriori estimation be performed.
        danger:
                A boolean, which if True will prioritise speed over caution (enabling the emcee live_dangerously
                kwarg, and scatting the initial positions of the walkers a bit less)
        verbose:
                A boolean, stating whether ChrisFit should provide verbose output whilst operating
        """


    # State name of source, and enable fancy colour options if we're being run in a bona fide terminal environment
    name_bracket_prefix = '['+gal_dict['name']+']' + (' '*(23-len('['+gal_dict['name']+']')))
    if sys.stdout.isatty():
        name_bracket_prefix = termcolor.colored(name_bracket_prefix, 'cyan', attrs=['bold'])
    if verbose:
        print(name_bracket_prefix  + 'Commencing processing')

    # Add column to bands_frame, to record which fluxes are larger than their uncertainty
    bands_frame['det'] = bands_frame.loc[:,'flux'] > bands_frame.loc[:,'error']

    # Parse beta argument, so that each model component is assigned its own value (even if they are all the same)
    if len(beta) != 2:
        raise Exception('Beta argument must contain two values for fittin BEMBB model')

    # Parse correl_unc argument, so that if no value provided, an empty list is used throughout the rest of the function
    if not hasattr(correl_unc, '__iter__'):
        correl_unc = []

    # If no limit information provided in band frame, assume no bands are limit
    if 'limit' not in bands_frame.keys():
        bands_frame['limit'] = np.array([False]*len(bands_frame))

    # Bundle various fitting arguments in to a dictionary
    fit_dict = {'bands_frame':bands_frame,
                'gal_dict':gal_dict,
                'beta':beta,
                'break_lambda':break_lambda,
                'correl_unc':correl_unc,
                'bounds':False,
                'priors':priors,
                'mcmc_n_walkers':mcmc_n_walkers,
                'mcmc_n_steps':mcmc_n_steps,
                'distance':gal_dict['distance'],
                'kappa_0':kappa_0,
                'kappa_0_lambda':kappa_0_lambda,
                'danger':danger}

    # Determine number of parameters
    n_params = 5 + len(correl_unc)
    fit_dict['n_params'] = n_params

    # If all fluxes are negative, return null results
    if np.where(bands_frame.loc[:,'flux'] > 0)[0].shape[0] == 0:
        #raise Exception('All fluxes are negative; fitting not realistic')
        mcmc_samples = np.nan * np.zeros([1, n_params])
        results_dict = {'posterior':mcmc_samples}
        return results_dict

    # Read in colour-correction tables
    fit_dict['trans_dict'] = PrefetchColourCorrections()

    # No custom priors provided, construct priors ahead of time, but warn that this is slower
    if isinstance(fit_dict['priors'], dict):
        pass
    elif mcmc_n_threads == 1:
        fit_dict['priors'] = PriorsConstruct(fit_dict)
    else:
        if verbose and (True not in [mle_only, map_only]):
            print(name_bracket_prefix  + 'No custom priors provided; using (slower) default priors') #(Note that the multithreaded MCMC is *MUCH FASTER* when working with custom priors, as functions defined outsite the fitter can be handled more efficiently)


    # Generate initial guess values for maximum-likelihood estimation and maximum-a-posteriori estimation (which will then itself be used to initialise emcee's estimation)
    mle_fit_dict = copy.deepcopy(fit_dict)
    mle_fit_dict['bounds'] = True
    mle_fit_dict['correl_unc'] = False
    mle_initial = MaxLikeInitial(bands_frame, mle_fit_dict)



    # Find Maximum Likelihood Estimate (MLE)
    if verbose:
        if map_only:
            print(name_bracket_prefix + 'Performing maximum likelihood estimation')
        else:
            print(name_bracket_prefix + 'Performing MLE to initialise MCMC')
    NegLnLike = lambda *args: -LnLike(*args)
    """mle_opt = scipy.optimize.minimize(NegLnLike, mle_initial, args=(mle_fit_dict), method='Powell',
                                      tol=1E-5, options={'maxiter':5000,'maxfev':5000,'return_all':True})"""
    mle_bounds = np.array([[5, 50],
                           [mle_initial[1]/1000.0, mle_initial[1]*1000.0],
                           [0, 4],
                           [0, 3],
                           [250E-6, 300E-6]]) #!!! [100E-6, 600E-6]
    mle_done = False
    mle_fail_count = 0
    while not mle_done:
        try:
            signal.alarm(600)
            mle_opt = scipy.optimize.differential_evolution(NegLnLike, mle_bounds,
                                                            maxiter=250, popsize=5, polish=False,
                                                            seed=int(time.time()), args=(mle_fit_dict,))
            signal.alarm(0)
            mle_done = True
        except:
            if mle_fail_count == 5:
                breakpoint()
    mle_params = mle_opt.x

    # If only MLE fit was requested, return results now
    if mle_only:
        if map_only:
            raise Exception('Cannot have both mle_only and map_only kwargs set to true; chose one or the other')
        chi_squared = ChiSquared(mle_params, fit_dict)
        sed_fig = None
        if plot != False:
            print(name_bracket_prefix + 'Generating SED plot')
            sed_fig, sed_ax = SEDborn(mle_params, fit_dict)
            if isinstance(plot, str):
                sed_fig.savefig(os.path.join(plot,gal_dict['name']+'_SED.png'), dpi=300)
            else:
                sed_fig.savefig(gal_dict['name']+'_SED.png', dpi=300)
        return {'mle':mle_params,'sed':sed_fig,'chisq':chi_squared}

    # Re-introduce any correlated uncertainty parameters that were excluded from maximum-likelihood fit
    mle_params = np.array(mle_params.tolist()+([1E-5]*len(fit_dict['correl_unc'])))

    # Find Maximum A posteriori (MAP) estimate
    """if verbose:
        if map_only:
            print(name_bracket_prefix + 'Performing maximum-a-posteriori estimation')
        else:
            print(name_bracket_prefix + 'Performing MaP estimation to initialise MCMC')
    NegLnLike = lambda *args: -LnPost(*args)
    map_bounds = np.array([[5, 100],
                           [mle_initial[1]/1E10, mle_initial[1]*1E10],
                           [0, 10],
                           [0, 10],
                           [50E-6, 1000E-6],
                           [-0.25, 0.25],
                           [-0.2, 0.2]])
    map_opt = scipy.optimize.differential_evolution(NegLnLike, map_bounds, args=(fit_dict,),
                                                    strategy='rand2bin', maxiter=250, popsize=5)
    if map_opt['success'] == True:
        map_params = map_opt.x
    else:
        map_params = mle_params"""
    map_params = mle_params

    """# If only MAP fit was requested, return results now (with SED plot if needed)
    if map_only:
        if map_opt['success'] == False:
            raise Exception('MaP estimation failed; uncertainties may be too small?')
        chi_squared = ChiSquared(map_params, fit_dict)
        sed_fig = None
        if plot != False:
            print(name_bracket_prefix + 'Generating SED plot')
            sed_fig, sed_ax = SEDborn(map_params, fit_dict)
            if isinstance(plot, str):
                sed_fig.savefig(os.path.join(plot,gal_dict['name']+'_SED.png'), dpi=300)
            else:
                sed_fig.savefig(gal_dict['name']+'_SED.png', dpi=300)
        return {'map':map_params,'sed':sed_fig,'chisq':chi_squared}"""

    # Generate starting position for MCMC walkers, in small random cluster around maximum-likelihood position
    mcmc_initial = MCMCInitial(map_params, fit_dict)

    # Initialise multiprocessing pool
    mcmc_pool = mp.Pool(processes=mcmc_n_threads)

    # Initiate and run emcee affine-invariant ensemble sampler
    if verbose:
        print(name_bracket_prefix + 'Sampling posterior distribution using emcee')
    mcmc_sampler = emcee.EnsembleSampler(mcmc_n_walkers,
                                         n_params,
                                         LnPost,
                                         args=[fit_dict],
                                         pool=mcmc_pool,
                                         live_dangerously=danger)
    if verbose:
        mcmc_bar = progress.bar.Bar('Computing MCMC',
                                    max=mcmc_n_steps,
                                    fill='=',
                                    suffix='%(percent)d%% [%(elapsed_td)s -> %(eta_td)s]')
        for _, _ in enumerate(mcmc_sampler.sample(mcmc_initial, iterations=mcmc_n_steps)):
            mcmc_bar.next()
        mcmc_bar.finish()
    else:
        mcmc_sampler.run_mcmc(mcmc_initial, mcmc_n_steps)
    mcmc_chains = mcmc_sampler.chain

    # Identify and remove burn-in (and optionally, also search for meta-stability)
    if not danger:
        if verbose:
            print(name_bracket_prefix + 'Removing burn-in')
        mcmc_chains_clean = ChainClean(mcmc_chains, simple_clean=simple_clean)
    else:
        mcmc_chains_clean = mcmc_chains

    # Plot trace of MCMC chains
    if plot != False:
        if verbose:
            print(name_bracket_prefix + 'Generating trace plot')
        trace_fig, trace_ax = TracePlot(mcmc_chains_clean, fit_dict)
        if isinstance(plot, str):
            trace_fig.savefig(os.path.join(plot,gal_dict['name']+'_Trace.png'), dpi=300)
        else:
            trace_fig.savefig(gal_dict['name']+'_Trace.png', dpi=300)

    else:
        trace_fig = None

    # Combine MCMC chains into final posteriors for each parameter, excludig any samples that contain NaNs
    mcmc_samples = mcmc_chains_clean.reshape((-1, n_params))
    mcmc_samples = mcmc_samples[np.where(np.isnan(mcmc_samples[:,0])==False)[0],:]
    mcmc_samples = np.delete(mcmc_samples, list(set(np.where(np.isnan(mcmc_samples))[0].tolist())), axis=0)

    # Find median parameter estimates, and calculate corresponding chi-squared
    median_params = np.median(mcmc_samples, axis=0)
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(median_params, fit_dict)
    median_chi_squared = ChiSquared(median_params, fit_dict)

    # Plot posterior corner plot
    if plot != False:
        if verbose:
            print(name_bracket_prefix + 'Generating corner plot')
        corner_fig, corner_ax = CornerPlot(mcmc_samples.copy(), [np.nan]*n_params, fit_dict)
        if isinstance(plot, str):
            corner_fig.savefig(os.path.join(plot,gal_dict['name']+'_Corner.png'), dpi=300)
        else:
            corner_fig.savefig(gal_dict['name']+'_Corner.png', dpi=300)
    else:
        corner_fig = None

    # Plot SED
    if plot != False:
        if verbose:
            print(name_bracket_prefix + 'Generating SED plot')
        sed_fig, sed_ax = SEDborn(map_params, fit_dict, posterior=mcmc_samples)
        if isinstance(plot, str):
            sed_fig.savefig(os.path.join(plot,gal_dict['name']+'_SED.png'), dpi=300)
        else:
            sed_fig.savefig(gal_dict['name']+'_SED.png', dpi=300)
    else:
        sed_fig = None

    # Return results
    gc.collect()
    if mcmc_n_threads > 1:
        mcmc_pool.close()
    if verbose:
        print(name_bracket_prefix + 'Processing completed')
    results_dict = {'medians':median_params,'mle':mle_params,'map':map_params,'corner':corner_fig,'sed':sed_fig,'trace':trace_fig,'chisq':median_chi_squared}
    if full_posterior:
        results_dict.update({'posterior':mcmc_samples, 'trace':trace_fig})
    return results_dict




def LnLike(params, fit_dict):
    """ Function to compute ln-likelihood of some data, given the parameters of the proposed model """

    # Deal with parameter bounds, if they are required (for example, if we're doing a maximum-likelihood estimation)
    if fit_dict['bounds']:
        if not MaxLikeBounds(params, fit_dict):
            return -np.inf

    # Programatically extract dust temperature, dust mass, and beta (variable or fixed) parameter sub-vectors from params tuple
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # Extract bands_frame from fit_dict
    bands_frame = fit_dict['bands_frame']

    # Calculate predicted fluxes for each band, given SED parameters
    bands_flux_pred = ModelFlux(bands_frame['wavelength'], temp_vector, mass_vector, fit_dict['distance'],
                                kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'],
                                break_lambda=break_lambda_vector, beta=beta_vector)

    # Calculate colour corrections (doing this before correlated uncertainties, as colour corrections are calibrated assuming Neptune model is correct)
    bands_col_correct = ColourCorrect(bands_frame['wavelength'], bands_frame['band'], temp_vector, mass_vector,
                                      beta_vector, break_lambda_vector,
                                      kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'],
                                      verbose=False, fit_dict=fit_dict)

    # Apply colour corrections (but not for limit bands, as spectrum not necessarily constrained here)
    no_col_correct = np.where((fit_dict['bands_frame']['limit']==True) & (fit_dict['bands_frame']['flux']>bands_flux_pred))
    if no_col_correct[0].shape[0] > 0:
        bands_col_correct[no_col_correct] = 1.0
    bands_flux_pred *= bands_col_correct

    # If there are correlated uncertainty terms, reduce the flux uncertainties to uncorrelated (non-systematic) components, and update predicted fluxes
    bands_unc =  bands_frame['error'].values.copy()
    if len(correl_err_vector) > 0:
        for j in range(len(bands_frame)):
            b = bands_frame.index.values[j]
            for i in range(len(fit_dict['correl_unc'])):
                correl_param = fit_dict['correl_unc'][i]
                if bands_frame.loc[b,'band'] in correl_param['correl_bands']:
                    bands_unc[i] = bands_frame.loc[b,'flux'] * np.sqrt((bands_frame.loc[b,'error']/bands_frame.loc[b,'flux'])**2.0 - correl_param['correl_scale']**2.0)
                    bands_flux_pred[j] *= 1 + correl_err_vector[i]

    # Calculate ln-likelihood of each flux, given measurement uncertainties and proposed model (the scipy student's t distriubution needs factor of 1.3654 applied to sigma to account for fact that the -sigma > x > sigma interval only contains 50% the probability density, not 68.298)
    ln_like = np.log(scipy.stats.t.pdf(bands_flux_pred, 1, loc=bands_frame['flux'], scale=1.3654*bands_unc))

    # Factor in limits; for bands with limits if predicted flux is <= observed flux, it is assigned same ln-likelihood as if predicted flux == observed flux
    where_below_limit = np.where((bands_frame['limit'].values) & (bands_flux_pred<bands_frame['flux']))
    ln_like[where_below_limit] = np.log(scipy.stats.t.pdf(bands_frame['flux'], 1, loc=bands_frame['flux'], scale=1.3654*bands_unc))[where_below_limit]

    # Exclude the calculated ln-likelihood for bands where flux and/or uncertainty are NaN
    ln_like = ln_like[np.where((np.isnan(bands_frame['flux']) == False) & (np.isnan(bands_frame['error']) == False))]

    # Calculate and return final data ln-likelihood
    ln_like = np.sum(np.array(ln_like))
    return ln_like





def LnPrior(params, fit_dict):
    """ Function to compute prior ln-likelihood of the parameters of the proposed model """

    # Programatically extract dust temperature, dust mass, and beta (varible or fixed) parameter sub-vectors from params tuple
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # If priors already available, grab them; otherwise, generate now
    if isinstance(fit_dict['priors'], dict):
        priors = fit_dict['priors']
    else:
        priors = PriorsConstruct(fit_dict)

    # Declare empty list to hold ln-like of each parameter
    ln_like = []

    # Calculate ln-like for temperature
    ln_like.append(priors['temp'][0](temp_vector[0]))

    # Calculate ln-like for mass
    ln_like.append(priors['mass'][0](mass_vector[0]))

    # Calculate ln-like for beta
    for i in range(len(fit_dict['beta'])):
        ln_like.append(priors['beta'][i](beta_vector[i]))

    # Calculate ln-like for break_lambda
    ln_like.append(priors['break_lambda'][0](break_lambda_vector[0]))

    # Calculate ln-like for correlated uncertainties; if a user provides function in the kwarg, use that
    for i in range(len(correl_err_vector)):
        if callable(fit_dict['correl_unc'][i]['correl_distr']):
            ln_like.append(np.log(fit_dict['correl_unc'][i]['correl_distr'](correl_err_vector[i])))

        # If user has said in kwarg that uncertainty correlation has normal distributed, compute accordingly
        elif fit_dict['correl_unc'][i]['correl_distr'] == 'normal':
            ln_like.append(np.log(scipy.stats.norm.pdf(correl_err_vector[i], scale=fit_dict['correl_unc'][i]['correl_scale'], loc=0.0)))

        # If user has said in kwarg that uncertainty correlation has flat distribution, compite accordingly
        elif fit_dict['correl_unc'][i]['correl_distr'] == 'flat':
            if abs(correl_err_vector[i]) <= fit_dict['correl_unc'][i]['correl_scale']:
                ln_like.append(0)
            else:
                ln_like.append(-np.inf)

    # Calculate and return final prior log-likelihood
    ln_like = np.sum(np.array(ln_like))
    return ln_like





def LnPost(params, fit_dict):
    """ Funtion to compute posterior ln-likelihood of the parameters of the proposed model, given some data """

    # Calculate prior ln-likelihood of the proposed model parameters
    ln_prior = LnPrior(params, fit_dict)

    # Calculate the ln-likelihood of the data, given the proposed model parameters
    ln_like = LnLike(params, fit_dict)

    # Calculate posterior ln-likelihood of the proposed model parameters, given the data and the priors
    ln_post = ln_prior + ln_like

    # Check that posterior ln-likelihood is legitimate, and return it
    if np.isnan(ln_post):
        ln_post = -np.inf
    return ln_post





def ModelFlux(wavelength, temp, mass, dist, kappa_0=0.051, kappa_0_lambda=500E-6, beta=2.0, break_lambda=False):
    """
    Function to calculate flux at given wavelength(s) from dust component(s) of given mass and temperature, at a given
    distance, assuming modified blackbody ('greybody') emission.

    Arguments:
        wavelength:     A float, or list of floats, giving the wavelength(s) (in m) of interest
        temp:           A float, giving the dust temperature (in K)
        mass:           A float, giving the dust mass (in M_sol)
        dist:           A float, giving the distance to the target source (in pc)

    Keyword arguments:
        kappa_0:        A float, giving the dust mass absorption coefficient) (in m**2 kg**-1)
        kappa_0_lambda: A float, giving the reference wavelength (in m) corresponding to kappa_0
        beta:           A list of floats, giving the dust emissivity slope(s), beta_1 and beta_2
        break_lambda:   A float, giving the break wavelength (in m)

    If wavelength is given as a list, a list of output fluxes will be given, corresponding to the calculated flux at
    each wavelength.

    Temperature and mass can be set to be lists , corresponding to multiple dust components. For n components, both
    lists must be of length n.

    Optionally, a different dust mass absorption coefficient (ie, kappa) can be used for each component; this is done by
    giving lists of length n for kappa_0 and kappa_0_lambda.

    Optionally, a different dust emissivity slope (ie, beta) can be used for each component; this is done by giving a
    list of length n for beta.

    If a BEMBB is being used (ie, if break_lambda is set to a float), then two beta values must be given; the beta for
    the shorter and longer wavelengths (relative to the break) respectively.
    """

    # As needed, convert variables to arrays
    wavelength = Numpify(wavelength)
    temp = Numpify(temp)
    mass = Numpify(mass)
    kappa_0 = Numpify(kappa_0)
    kappa_0_lambda = Numpify(kappa_0_lambda)
    beta = Numpify(beta)
    break_lambda = Numpify(break_lambda)

    # Check that variables are the same length, when they need to be
    if np.std([len(temp), len(mass), len(beta), len(kappa_0), len(kappa_0_lambda)]) != 0:
        Exception('Number of dust components needs to be identical for temp/mass/beta/kappa_0/kappa_0_lambda variables')

    """ NB: Arrays have dimensions of n_comp rows by n_bands columns """

    # Convert wavelengths to frequencies (for bands of interest, and for kappa_0 reference wavelengths)
    nu = np.divide(c, wavelength)

    # Alternativey, if BEMBB resquested, first check that correct value types are given
    if len(beta) != 2:
        raise Exception('When fitting BEMBB model, two beta values must be provided')
    if (len(kappa_0) != 1) or (len(kappa_0_lambda) != 1):
        raise Exception('When a break_labmda value is given, all dust components must be given same kappa_0')

    # Shunt our reference kappa to the break wavelength, to stop a discontinuity appearing
    if kappa_0_lambda < break_lambda:
        kappa_break_lambda = kappa_0.mean() * (kappa_0_lambda/break_lambda)**beta[0]
    elif kappa_0_lambda >= break_lambda:
        kappa_break_lambda = kappa_0 * (kappa_0_lambda/break_lambda)**beta[1]
    break_nu = c / break_lambda

    # Now calculate kappa either side of the break
    short_kappa_nu_base = np.outer(break_nu**-1, nu)[0,:]
    short_kappa_nu_prefactor = np.power(short_kappa_nu_base, beta[0])
    short_kappa_nu = kappa_break_lambda * short_kappa_nu_prefactor
    long_kappa_nu_base = np.outer(break_nu**-1, nu)[0,:]
    long_kappa_nu_prefactor = np.power(long_kappa_nu_base, beta[1])
    long_kappa_nu = kappa_break_lambda * long_kappa_nu_prefactor
    kappa_nu = short_kappa_nu.copy()
    kappa_nu[np.where(wavelength >= break_lambda)] = long_kappa_nu[np.where(wavelength >= break_lambda)]
    kappa_nu = np.array([kappa_nu])

    # Calculate Planck function prefactor for each frequency
    B_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)

    # Calculate exponent term in Planck function, for each component, at each frequency
    B_exponent = np.array([ np.divide((h*nu),(k*temp)) ])

    # Calculate final value of Planck function for each, for each model component, at each frequency (output array will have n_comp rows, and n_freq columns)
    B_planck = B_prefactor * (np.e**B_exponent - 1)**-1.0

    # Convert mass and distance values to SI units
    mass_kilograms = mass * 2E30
    dist_metres = dist * 3.26 * 9.5E15

    # Calculate flux for each component, for each dust model component
    flux = 1E26 * kappa_nu * dist_metres**-2.0 * mass_kilograms * B_planck

    # Return calculated flux (denumpifying it if is only single value)
    if flux.size == 0:
        flux = flux[0]
    else:
        flux = flux.astype(float)
    if len(flux.shape) == 2:
        flux = flux[0,:]
    return flux





def PriorsConstruct(fit_dict):
    """ Function to automatically construct a set of default priors (if no specific priors were passed to ChrisFit),
    given the basic parameters of the model as described by the ChrisFit input """

    # If priors have already been provided, then don't bother doing anything
    if fit_dict['priors'] != None:
        return fit_dict['priors']

    # Initialize dictionary to hold priors
    priors = {'temp':[],
              'mass':[],
              'beta':[],
              'break_lambda':[]}

    # Define function to find scaling factor for gamma distribution of given mode, alpha, and location
    GammaScale = lambda mode, alpha, phi: (mode-phi)/(alpha-1.0)

    # Create temperature priors, using gamma distribution (with kwarg in lambda to make iterations evaluate separately)
    temp_alpha = 2.5
    temp_mode = 20.0
    temp_phi = 5.0
    temp_scale = GammaScale(temp_mode, temp_alpha, temp_phi)
    temp_like = lambda temp, temp_alpha=temp_alpha, temp_phi=temp_phi, temp_scale=temp_scale: scipy.stats.gamma.pdf(temp, temp_alpha, loc=temp_phi, scale=temp_scale)
    temp_x = np.linspace(0, 500, num=1000)
    temp_y = temp_like(temp_x)
    temp_norm = np.trapz(temp_y, x=temp_x)
    temp_like_norm = lambda temp, temp_like=temp_like, temp_norm=temp_norm: temp_like(temp) / temp_norm
    temp_ln_like = lambda temp, temp_like_norm=temp_like_norm: np.log(temp_like_norm(temp))
    priors['temp'].append(temp_ln_like)

    # Use flux and distance to estimate likely cold dust mass, based on empirical relation
    bands_frame = fit_dict['bands_frame']
    fluxes_submm = bands_frame.where((bands_frame['wavelength']>=150E-6)&(bands_frame['wavelength']<1E-3))['flux']
    peak_flux = fluxes_submm.max()
    if np.isnan(peak_flux):
        peak_flux = bands_frame.loc[np.argmax(bands_frame['wavelength']),'flux']
    peak_lum = peak_flux * fit_dict['distance']**2.0
    peak_mass = 10**(np.log10(peak_lum)-8)

    # Use likely cold dust mass to construct mass priors, using log-t distribution (with kwarg in lambda to make iterations evaluate separately)
    mass_mode = peak_mass
    mass_mode *= 0.051 / (fit_dict['kappa_0'] * (fit_dict['kappa_0_lambda'] / 500E-6)**fit_dict['beta'][0])
    mass_mode = np.log10(mass_mode)
    mass_sigma = 10.0
    mass_like = lambda mass, mass_mode=mass_mode, mass_sigma=mass_sigma: 10.0**scipy.stats.t.pdf(np.log10(mass), 1, loc=mass_mode, scale=mass_sigma)
    mass_x = np.logspace(-10, 20, num=10000)
    mass_y = mass_like(mass_x)
    mass_norm = np.trapz(mass_y, x=mass_x)
    mass_like_norm = lambda mass, mass_like=mass_like, mass_norm=mass_norm: mass_like(mass) / mass_norm
    mass_ln_like = lambda mass, mass_like_norm=mass_like_norm: np.log(mass_like_norm(mass))
    priors['mass'].append(mass_ln_like)

    # Create beta priors, using gamma distribution
    for i in range(len(fit_dict['beta'])):
        beta_mode = fit_dict['beta'][i]
        beta_like = lambda beta: scipy.stats.gamma.pdf(beta, beta_mode+1.0, loc=0, scale=1)
        beta_x = np.linspace(0, 20, num=2000)
        beta_y = beta_like(beta_x)
        beta_norm = np.trapz(beta_y, x=beta_x)
        beta_like_norm = lambda beta, beta_like=beta_like: beta_like(beta) / beta_norm
        beta_ln_like = lambda beta, beta_like_norm=beta_like_norm: np.log(beta_like_norm(beta))
        priors['beta'].append(beta_ln_like)

    # Create break_lambda prior, using log-t distribution
    """break_lambda_mode = np.log10(fit_dict['break_lambda'])
    break_lambda_sigma = 0.45
    break_lambda_like = lambda break_lambda: 10.0**scipy.stats.t.pdf(np.log10(break_lambda), 1, loc=break_lambda_mode, scale=break_lambda_sigma)
    break_lambda_x = np.logspace(np.log10(50E-6), np.log10(1000E-6), num=2000)
    break_lambda_y = break_lambda_like(break_lambda_x)
    break_lambda_norm = np.trapz(break_lambda_y, x=break_lambda_x)
    break_lambda_like_norm = lambda break_lambda, break_lambda_like=break_lambda_like, break_lambda_norm=break_lambda_norm: break_lambda_like(break_lambda) / break_lambda_norm
    break_lambda_ln_like = lambda break_lambda, break_lambda_like_norm=break_lambda_like_norm: np.log(break_lambda_like_norm(break_lambda))"""
    break_lambda_mode = fit_dict['break_lambda']
    break_lambda_alpha = 2.5
    break_lambda_phi = 50E-6
    break_lambda_scale = GammaScale(break_lambda_mode, break_lambda_alpha, break_lambda_phi)
    break_lambda_like = lambda break_lambda: scipy.stats.gamma.pdf(break_lambda, break_lambda_alpha, loc=break_lambda_phi, scale=break_lambda_scale)
    break_lambda_x = np.logspace(np.log10(50E-6), np.log10(1000E-6), num=2000)
    break_lambda_y = break_lambda_like(break_lambda_x)
    break_lambda_norm = np.trapz(break_lambda_y, x=break_lambda_x)
    break_lambda_like_norm = lambda break_lambda, beta_like=break_lambda_like: beta_like(break_lambda) / break_lambda_norm
    break_lambda_ln_like = lambda break_lambda, break_lambda_like_norm=break_lambda_like_norm: np.log(break_lambda_like_norm(break_lambda))
    priors['break_lambda'].append(break_lambda_ln_like)
    """# Construct function for SPIRE correlated uncertainty
    def Boxcar(break_lambda):
        x = np.linspace(0, 10000E-6, num=5000)
        y = np.array([(-1 * np.inf)] * x.size)
        y[np.where((x>100E-6) & (x<650E-6))] = 1
        return y[(np.abs(x-break_lambda)).argmin()]
    break_lambda_ln_like = Boxcar"""
    """plt.plot(1E6*break_lambda_x, break_lambda_y)
    plt.savefig('/Users/cclark/Desktop/prior.png')"""

    # Return priors dictionary
    return priors





def ParamsExtract(params, fit_dict):
    """ Function to extract SED parameters from params vector (a tuple). For FitBEMBB, we know that the parameters
    will be in the order: (temp, mass, beta_1, beta_2, break_lambda, correl_err_1, correl_err_2, ..., correl_err_n)"""

    # Initiate and populate dust parameter sub-vectors
    temp_vector = [params[0]]
    mass_vector = [params[1]]
    beta_vector = [params[2], params[3]]
    break_lambda_vector = [params[4]]

    # Initiate and populate correlated uncertainty parameter sub-vecto
    correl_err_vector = []
    if hasattr(fit_dict['correl_unc'], '__iter__'):
        correl_err_vector = params[5:len(params)]

    # Return parameters tuple
    return (tuple(temp_vector), tuple(mass_vector), tuple(beta_vector), tuple(break_lambda_vector), tuple(correl_err_vector))





def ParamsLabel(fit_dict):
    """ Function to generate list of plot-ready labels for free parameters """

    # Initiate list to hold labels
    labels = []

    # Add temperature and mass labels to list
    labels.append(r'$T_{d}$')
    labels.append(r'$M_{d}$')

    # Generate beta labels
    labels.append(r'$\beta_{1}$')
    labels.append(r'$\beta_{2}$')

    # Generate break_lambda label
    labels.append(r'$\lambda_{\it break}$')

    # Generate correlated uncertainty labels (if necessary) and add them to list
    if hasattr(fit_dict['correl_unc'], '__iter__'):
        for i in range(len(fit_dict['correl_unc'])):
            band_first = re.compile('[^a-zA-Z]').sub('',fit_dict['correl_unc'][i]['correl_bands'][0])
            band_last = re.compile('[^a-zA-Z]').sub('',fit_dict['correl_unc'][i]['correl_bands'][-1:][0])
            band_match = SequenceMatcher(None, band_first, band_last).find_longest_match(0, len(band_first), 0, len(band_last))
            if (band_match.size == 0) or (len(band_first) + len(band_last) == 0):
                instr = str(i+1)
            else:
                instr = band_first[band_match.a:band_match.size]
            labels.append(r'$\upsilon_{'+instr+'}$')

    # Return list of labels
    return labels





def MaxLikeInitial(bands_frame, fit_dict):
    """ Function to generate initial guess values for maximum-likelihood fitting """

    # Declare list to hold guesses
    guess = []

    # Temperature guesses for 18K if one MBB; 18K and 50K if two MBB; equally spaced therebetween for 3 or more
    temp_guess = 20.0
    guess.append(temp_guess)

    # Use flux and distance to estimate likely cold dust mass, based on empirical relation
    bands_frame = fit_dict['bands_frame']
    peak_flux = bands_frame.where((bands_frame['wavelength']>=150E-6)&(bands_frame['wavelength']<1E-3))['flux'].max()
    if np.isnan(peak_flux):
        Warning('Ideally need fluxes with rest wavelenghts in 150-1000 um range to obtain meaningful constrains')
        peak_flux = bands_frame.loc[np.argmax(bands_frame['wavelength']),'flux']
    peak_lum = peak_flux * fit_dict['distance']**2.0
    peak_mass = 10**(np.log10(peak_lum)-8)

    # Mass guesses are based on empirical relation, then scale for kappa
    mass_guess = peak_mass
    mass_guess *= 0.051 / (fit_dict['kappa_0'] * (fit_dict['kappa_0_lambda'] / 500E-6)**fit_dict['beta'][0])
    guess.append(mass_guess)

    # Beta guess taken from kwarg; defaults of 2.0 and 1.75 used if nothing provided by user
    beta_guess = np.array(fit_dict['beta'])
    guess += beta_guess.tolist()

    # Break_lambda guess taken from swarg; default of 175um used if nothing provided by user
    break_lambda_guess = fit_dict['break_lambda']
    guess.append(break_lambda_guess)

    # Correlated uncertainties are always guessed to have a value of 0
    if hasattr(fit_dict['correl_unc'], '__iter__'):
        correl_err_guess = np.array([0.0] * len(fit_dict['correl_unc']))
        guess += correl_err_guess.tolist()

    # Return tuple of guesses
    return tuple(guess)





def MCMCInitial(mle_params, fit_dict):
    """ Function to generate initial positions for MCMC walkers, in cluster around maximum likelihood estimate """

    # Extract parameter vectors
    mle_temp_vector, mle_mass_vector, mle_beta_vector, mle_break_lambda_vector, mle_correl_err_vector = ParamsExtract(mle_params, fit_dict)

    # Loop over walkers
    mcmc_initial = []
    for i in range(fit_dict['mcmc_n_walkers']):

        # Keep generating proposed starting positions until one is accepted
        accepted = False
        while not accepted:
            accepted = True

            # Create copy of MLE params for this iteration, and shut masses into log space for peterubations
            params = copy.deepcopy(mle_params)
            params[len(mle_temp_vector):len(mle_temp_vector+mle_mass_vector)] = np.log10(params[len(mle_temp_vector):len(mle_temp_vector+mle_mass_vector)])

            # Generate permutations for walker initial positions of +/- 5%, with additional +/- 0.001 (or 0.5%, whichever is smaller) random shift
            walker_scale = 0.05 * params * np.random.rand(len(params))
            walker_offset = 1E-3 * np.random.rand(len(params))
            walker_offset = np.min(np.array([walker_offset, 0.005 * params * np.random.rand(len(params))]).transpose(), axis=1)

            # Apply initial position permutations
            walker_initial = (params + walker_scale + walker_offset)

            # Extract parameters
            temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(walker_initial, fit_dict)

            # Check that temperature terms are all physical (ie, temp > 0 kelvin)
            if np.where(np.array(temp_vector)<=0)[0].size > 0:
                accepted = False

            # Check that mass terms are all physical (ie, mass > 0 Msol); if so, convert back to linar space
            if np.where(np.array(10.0**np.array(mass_vector))<0)[0].size > 0:
                accepted = False
            else:
                walker_initial[len(temp_vector):len(temp_vector+mass_vector)] = 10**walker_initial[len(temp_vector):len(temp_vector+mass_vector)]

            # Check that beta terms are all physical (ie, beta > 0)
            if (np.where(np.array(beta_vector)<1)[0].size > 0) or (np.where(np.array(beta_vector)>4)[0].size > 0):
                accepted = False

            # Check that break_lambda terms are all reasonable (ie, )

        # If proposed position is valid, add it to list of initial conditions
        mcmc_initial.append(walker_initial)

    # Return generated initial positions
    return mcmc_initial





def MaxLikeBounds(params, fit_dict):
    """ Function to check whether parameters for a proposed model violate standard boundary conditions. This is for
    maximum likelihood estimations, where there are no priors to go by. If the parameters are good, the function returns
    a value of True; else, it returns a value of False."""

    # Extract parameter vectors
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # Check if there are correlated uncertainty terms; if there are, loop over them
    if len(correl_err_vector) > 0:
        for i in range(len(fit_dict['correl_unc'])):
            correl_param = fit_dict['correl_unc'][i]

            # Make sure proposed correlated uncertainty value doesn't exceed bounds, in case of flat distribution
            if correl_param['correl_distr'] == 'flat':
                if correl_err_vector[i] > abs(correl_param['correl_scale']):
                    return False

    # Check that temperature terms are in order
    if len(temp_vector) > 1:
        for i in range(1,len(temp_vector)):
            if temp_vector[i] < temp_vector[i-1]:
                return False

    # Check that temperature terms are all physical (ie, 5 < temp < 100 kelvin)
    if np.where(np.array(temp_vector)<5)[0].size > 0 or np.where(np.array(temp_vector)>300)[0].size > 0:
        return False

    # Check that mass terms are all physical (ie, mass > 0 Msol)
    if np.where(np.array(mass_vector)<0)[0].size > 0:
        return False

    # Check that beta terms are all physical (ie, 1 < beta < 4)
    if (np.where(np.array(beta_vector)<1)[0].size > 0) or (np.where(np.array(beta_vector)>4)[0].size > 0):
        return False


    if beta_vector[0] < beta_vector[1]: #!!!
        return False#!!!


    # Check that break_lambda value is reasonable (ie, 100um < break_lambda < 1000um)
    if (break_lambda_vector[0] < 100E-6) or (break_lambda_vector[0] > 1000E-6):
        return False

    # If we've gotten this far, then everything is fine
    return True



def ChainClean(mcmc_chains, simple_clean=False):
    """ Function to identify and remove chains, and portions of chains, exhibiting non-convergence """

    # If we're doing this the simple way, just lop some fraction off the start of every chain
    if isinstance(simple_clean, float):
        burnin =int(simple_clean * mcmc_chains.shape[1])
        mcmc_chains[:,:burnin,:] = np.nan
        return mcmc_chains
    else:

        # Create copy of chains to work with
        mcmc_chains = mcmc_chains.copy()

        # Loop over chains and parameters, to find Geweke z-score for each
        for i in range(mcmc_chains.shape[0]):
            burnin = int(0.05 * mcmc_chains.shape[2])
            for j in range(mcmc_chains.shape[2]):
                geweke = Geweke(mcmc_chains[i,:,j])

                # Find where Geweke score crosses 0 for first time; assume this represents burn-in
                if (np.min(geweke[:,1]) >= 0) or (np.max(geweke[:,1]) <= 0):
                    burnin = mcmc_chains.shape[1]
                elif geweke[0,1] > 0:
                    burnin = max(burnin, int(2.0 * geweke[np.where(geweke[:,1]<0)[0].min(), 0]))
                elif geweke[0,1] < 0:
                    burnin = max(burnin, int(2.0 * geweke[np.where(geweke[:,1]>0)[0].min(), 0]))

            # Set burn-in steps to be NaN
            mcmc_chains[i,:burnin,:] = np.nan

        # Loop over chains and parameters, to check for metatstability
        end_frac = 0.4
        bad_chains = np.array([False]*mcmc_chains.shape[0])
        for i in range(mcmc_chains.shape[0]):
            for j in range(mcmc_chains.shape[2]):

                # To check for metastability, first compute the median values of the final portion of each chain
                test_chain = mcmc_chains[i,-int(0.25*mcmc_chains.shape[1]):,j]
                test_median = np.median(test_chain)
                comp_indices = np.array(range(mcmc_chains.shape[0]))
                comp_indices = comp_indices[np.where(comp_indices!=i)]
                comp_chains = mcmc_chains[comp_indices,-int(end_frac*mcmc_chains.shape[1]):,j]
                comp_medians = np.median(comp_chains, axis=1)
                comp_medians_median = np.median(comp_medians)

                # Perform nonparametric bootstrap resample of medians
                comp_medians_bs = np.array([sklearn.utils.resample(comp_medians) for k in range(10000)])

                # Subtract median of each set of bootstrapped values from themselves (ie, find deviations from the median of each set of medians)
                comp_medians_bs_dev = comp_medians_bs - np.array([np.median(comp_medians_bs, axis=1)]*comp_medians_bs.shape[1]).transpose()

                # Find the RMS deviation within each bootstrapped set of deviations, and use this to inform the rejection threshold
                comp_medians_bs_rms = np.mean(np.abs(comp_medians_bs_dev), axis=1)
                comp_medians_bs_thresh = 1.0 * np.median(comp_medians_bs_rms)

                # If median temp of current chain section is more than the determined threshold, call it metastable
                if abs(test_median - comp_medians_median) > comp_medians_bs_thresh:
                    bad_chains[i] = True

                # Also, access which chains have end regions with low variation (likely due to high rejection rates)
                comp_stds = np.nanstd(mcmc_chains[comp_indices,-int(end_frac*mcmc_chains.shape[1]):,j], axis=1)
                comp_stds_std = np.nanstd(comp_stds)
                comp_stds_mean = np.nanmean(comp_stds)
                comp_stds_thresh = 2.0 * comp_stds_std

                # Marks chains with high rejection rates as bad
                comp_stds_bad = np.where((comp_stds<(comp_stds_mean-comp_stds_thresh)) | (comp_stds>(comp_stds_mean+comp_stds_thresh)))
                bad_chains[comp_stds_bad] = True

                # Set all values, for all parameters, in suspected metastable chains to be NaN
                mcmc_chains[bad_chains,:,:] = np.nan

        """# Calculate the Gelman-Rubin criterion of each parameter from different start points
        gr_times = np.linspace(0, mcmc_chains.shape[1]-1, num=100).astype(int)[1:]
        gr = np.zeros([gr_times.shape[0], mcmc_chains.shape[2]])
        for i in range(len(gr_times)):
            t = gr_times[i]
            gr[i,:] = GelmanRubin(mcmc_chains[:,-t:,:])"""

        # Return cleaned chain
        return mcmc_chains





def GelmanRubin(mcmc_chains):
    """ Function to calculate Gelman-Rubin (1992) MCMC convergance criterion, adapted from Jorg Dietrich's blog; a G-R
    criterion of >1.1 is typically considered evidence for non-convergance. Chains of form (n_walkers, n_steps) """

    # Evaluate variance within chains
    variance = np.nanvar(mcmc_chains, axis=1, ddof=1)
    within_variance = np.nanmean(variance, axis=0)

    # Evaluate variance between chains
    theta_b = np.nanmean(mcmc_chains, axis=1)
    theta_bb = np.nanmean(theta_b, axis=0)
    walkers = float(mcmc_chains.shape[0])
    steps = float(mcmc_chains.shape[1])
    between_variance = steps / (walkers - 1) * np.sum((theta_bb - theta_b)**2, axis=0)
    var_theta = (((steps - 1.) / steps) * within_variance) + (((walkers + 1.) / (walkers * steps)) * between_variance)

    # Calculate  and return R-hat
    r_hat = np.sqrt(var_theta / within_variance)
    return r_hat





def Geweke(mcmc_chain, test_intrv=100, comp_frac=0.4):
    """ Function to compute Geweke z-scores for a whole bunch of small sections of chain, with all being compared to the
    a given portion of the end of the entire chain """

    # Extract benchmark section of chain, and calculate necessary values
    comp_chain = mcmc_chain[int(comp_frac*len(mcmc_chain)):len(mcmc_chain)]
    comp_mean = np.mean(comp_chain)
    comp_var = np.var(comp_chain)

    # Identify increments to examine in test region
    test_end = int(len(mcmc_chain) - comp_frac*len(mcmc_chain))
    test_steps = np.arange(test_intrv, test_end, test_intrv)

    # Loop over test increments, computing and recording geweke z-scores
    geweke = np.zeros([len(test_steps), 2])
    geweke[:,0] = test_steps
    for i in range(len(test_steps)):
        test_chain = mcmc_chain[test_steps[i]-test_intrv:test_steps[i]]
        test_mean = np.mean(test_chain)
        test_var = np.var(test_chain)
        geweke[i,1] = (test_mean - comp_mean) / np.sqrt(test_var + comp_var)

    # Return geweke scores
    return geweke





def PrefetchColourCorrections():
    """ Function to read in a data file containing instrumental response curves, and descritions of reference spectra,
    containging this information for all the bands where colour corrections are desired. The format should be:
    The first row for a given band takes form 'band,[BAND NAME]'.
    The following row for a given band provides a description of that band's calibration reference spectrum, taking
        the form 'ref,[SPECTRUM_DESCRIPTION]'; the spectrum description can be either nu_X, where X is replaced by a
        number giving the index of some frequency-dependent power law spectrum; of BB_T, where T is replaced by a number
        giving the temperature of a blackbody spectrum. Note that this line can be ommitted entirely
    All subsequent rows for a given band then provide the actual transmission data for a large number of increments in
        wavelength (in microns), and take the form '[SOME WAVELENGTH IN MICRONS],[TRANSMISSION FRACTION]'.
    This format can be repeated to fit transmissions data for any number of bands in one file."""

    # Set location of ChrisFuncs.py to be current working directory, recording the old CWD to switch back to later
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Read in transmission curves file, and loop over lines
    trans_dict = {'refs':{}}
    with open('Transmissions.dat') as curve_file:
        curve_list = curve_file.readlines()
    for i in range(0,len(curve_list)):
        curve_line = curve_list[i]

        # Check if this line indicates start of a new band; if so, start recording new band
        if curve_line[:4] == 'band':
            band = curve_line[5:].replace('\n','')
            trans_dict[band] = []

        # Check if this line contains reference spectrun information; if so, place in the 'refs' dictionary entry
        elif curve_line[:3] == 'ref':
            trans_dict['refs'][band] = curve_line[4:].replace('\n','')

        # Check if this line contains regular transmission information; if so, record
        else:
            trans_dict[band].append(curve_line.replace('\n','').split(','))

    # Loop over filters in filter dict, setting them to be arrays
    for curve in trans_dict.keys():
        if curve != 'refs':
            trans_dict[curve] = np.array(trans_dict[curve]).astype(float)

    # Restore old CWD, and colour
    os.chdir(old_cwd)
    return trans_dict





def ColourCorrect(wavelengths, bands, temp, mass, beta, break_lambda,
                  kappa_0=0.051, kappa_0_lambda=500E-6, verbose=False, fit_dict=None):
    """ Function to calculate colour-correction FACTOR appropriate to a given underlying spectrum. Will work for any
    instrument for which file 'Color_Corrections_INSTRUMENTNAME.csv' is found in the same directory as this script. """

    # Check if a fit dict has been provided; if it has, grab the transmission dict from it
    if fit_dict != None:
        trans_dict = fit_dict['trans_dict']
    else:
        trans_dict = PrefetchColourCorrections()

    # Make sure bands and wavelengths are in list format, for easy iterating later
    if isinstance(wavelengths, pd.core.series.Series):
        wavelengths = wavelengths.values.tolist()
    if isinstance(bands, pd.core.series.Series):
        bands = bands.values.tolist()

    # Construct source SED given current proposed model
    source_spec_lambda = np.logspace(-6,-2,5000)
    source_spec = np.array([source_spec_lambda, ModelFlux(source_spec_lambda, temp, mass, 1E6,
                                                          kappa_0=kappa_0, kappa_0_lambda=kappa_0_lambda,
                                                          break_lambda=break_lambda, beta=beta)]).transpose()

    # Loop over bands (if only one band has been submitted, stick it in a list to enable looping)
    factor_result = []
    if not hasattr(wavelengths, '__iter__'):
        single = True
        wavelengths, bands = [wavelengths], [bands]
    else:
        single = False
    for b in range(len(bands)):
        band = bands[b]
        wavelength = wavelengths[b]

        # Check that requested filter is actually in dictionary; if it is grab it, if it isn't return correction factor of 1.0
        if band not in trans_dict:
            factor_result.append(1.0)
            continue
        else:
            band_filter = trans_dict[band].copy()
            band_filter[:,0] /= 1E6

        # Check if reference spectrum present in transmission dictionary; if it is, construct spectrum array
        if (band in trans_dict.keys()) and (band in trans_dict['refs'].keys()):
            nu = (c / band_filter[:,0])

            # If reference is a power law, turn into corresponding array of values at same wavelength intervals as filter curve
            if trans_dict['refs'][band][:2] == 'nu':
                index = float(trans_dict['refs'][band][3:])
                ref_spec = np.zeros(band_filter.shape)
                ref_spec[:,0] = band_filter[:,0]
                ref_spec[:,1] = nu**index
                #ref_spec[:,1] /= np.max(ref_spec[:,1])

            # If reference spectrum is a blackbody, turn into a corresponding array of values at same wavelength intervals as filter curve
            if trans_dict['refs'][band][:2] == 'BB':
                temp = float(trans_dict['refs'][band][3:])
                ref_spec = np.zeros(band_filter.shape)
                ref_spec[:,0] = band_filter[:,0]
                planck_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
                planck_exponent = (h * nu) / (k * temp)
                ref_spec[:,1] = planck_prefactor * (np.e**planck_exponent - 1)**-1.0
                #ref_spec[:,1] /= np.max(ref_spec[:,1])

        # If reference spectrum not in Transmission.dat, nor provided by user, raise exception
        else:
            raise Exception('Reference spectrum not given, nor found in dictionary of band transmissions; please provide reference spectrum')

        # Normalise source and reference SEDs to have observed flux at (interpolated) nominal wavelength
        source_spec[:,1] /= np.interp(wavelength, source_spec[:,0], source_spec[:,1])
        ref_spec[:,1] /= np.interp(wavelength, ref_spec[:,0], ref_spec[:,1])

        # Filter SEDs by response curve (appropriately resampled in wavelength intervals), to record observed flux
        source_obs = source_spec[:,1] * np.interp(source_spec[:,0], band_filter[:,0], band_filter[:,1])
        ref_obs = ref_spec[:,1] * np.interp(ref_spec[:,0], band_filter[:,0], band_filter[:,1])

        # Integrate observed filtered SEDs (in intervals of freqency, as Jy are in terms of Hz)
        source_int = np.trapz(source_obs, x=(c/source_spec[:,0]))
        ref_int = np.trapz(ref_obs, x=(c/ref_spec[:,0]))

        # Calculate and return colour correction factor from integrals
        factor = ref_int / source_int

        # Append results to output lists
        factor_result.append(factor)

    # Return results (grabbing single values if only one band is being processed)
    if single:
        return factor_result[0]
    else:
        return np.array(factor_result)





def Numpify(var, n_target=False):
    """ Function for checking if variable is a list, and (if necessary) converting to a n_target length list of identical entries """

    # If variable is not iterable (ie, a list/array/etc), convert into an appropriate-length list
    if not hasattr(var, '__iter__'):
        if not n_target:
            var = [var]
        else:
            var = [var]*n_target

    # If necessary Convert a single-element iterable into a list of length n_targets
    elif len(var) == 1 and n_target > 1:
        var = [var[0]]*n_target

    # Object to mis-matched list lengths
    elif len(var) > 1 and len(var) != n_target:
        Exception('Variable list must either be of length 1, or of length n_targets')

    # If variable is not a numpy array, turn it into one
    if not isinstance(var, np.ndarray):
        var = np.array(var)

    # Return freshly-numpified variable
    return var





def ChiSquared(params, fit_dict):
    """ Function to calculate residuals, and thence chi-squared, of model, accounting for colour corrections """

    # Extract required parameters
    bands_frame = fit_dict['bands_frame']
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # Calculate predicted fluxes in each band, in order to establish if colour corrections should be applied to bands which are upper limits
    pred_fluxes = ModelFlux(bands_frame['wavelength'], temp_vector, mass_vector, fit_dict['distance'],
                            kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'],
                            break_lambda=break_lambda_vector, beta=beta_vector)

    # Apply colour corrections for observed fluxes, given model
    flux_corr = bands_frame['flux'].copy()
    for b in bands_frame.index:
        if (bands_frame.loc[b,'limit']) and (pred_fluxes[b] < bands_frame.loc[b,'flux']):
            colour_corr_factor = 1.0
        else:
            colour_corr_factor = ColourCorrect(bands_frame.loc[b,'wavelength'], bands_frame.loc[b,'band'],
                                               temp_vector, mass_vector, beta_vector, break_lambda_vector,
                                               kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'], verbose=False)
        flux_corr[b] = bands_frame.loc[b,'flux'] * colour_corr_factor

    # Calcylate residuals, and thence chi-squared
    flux_resid = flux_corr - ModelFlux(bands_frame['wavelength'], temp_vector, mass_vector, fit_dict['distance'],
                                       kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'],
                                       beta=beta_vector, break_lambda=break_lambda_vector)
    chi = (flux_resid / bands_frame['error'])[bands_frame['limit']==False]
    chi_squared = chi**2

    # Return result
    return chi_squared





def SEDborn(params, fit_dict, posterior=False, font_family='sans'):
    """ Function to plot an SED, with the same information used to produce fit """

    # Enable seaborn for easy, attractive plots
    plt.ioff()
    sns.set(context='talk') # Possible context settings are 'notebook' (default), 'paper', 'talk', and 'poster'
    sns.set_style('ticks', {'font.sans-serif':'DejaVu Sans'})
    matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'



    # Initialise figure
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])

    # Extract band dataframe and parameter vectors
    bands_frame = fit_dict['bands_frame']
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # Select only bands of interest
    bands_frame = bands_frame.loc[np.isnan(bands_frame['flux']) == False]

    # Generate fit fluxes
    fit_wavelengths = np.logspace(-5, -2, num=2000)
    fit_fluxes = ModelFlux(fit_wavelengths, temp_vector, mass_vector, fit_dict['distance'],
                                kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'],
                                beta=beta_vector, break_lambda=break_lambda_vector)
    fit_fluxes_tot = fit_fluxes

    # Plot fits (if no full posterior provided)
    if not isinstance(posterior, np.ndarray):
        ax.plot(fit_wavelengths*1E6, fit_fluxes_tot, ls='-', lw=1.5, c='red')

    # Calculate predicted fluxes in each band, in order to establish if colour corrections should be applied to bands which are upper limits
    pred_fluxes = ModelFlux(bands_frame['wavelength'], temp_vector, mass_vector, fit_dict['distance'],
                            kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'],
                            break_lambda=break_lambda_vector, beta=beta_vector)

    # Colour-correct fluxes according to model being plotted (excluding upper limits where flux is less than limit)
    bands_frame.loc[:,'flux_corr'] = bands_frame['flux'].copy()
    for b in bands_frame.index:
        if (bands_frame.loc[b,'limit']) and (pred_fluxes[b] < bands_frame.loc[b,'flux']):
            colour_corr_factor = 1.0
        else:
            colour_corr_factor = ColourCorrect(bands_frame.loc[b,'wavelength'], bands_frame.loc[b,'band'],
                                               temp_vector, mass_vector, beta_vector, break_lambda_vector,
                                               kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'], verbose=False)
        bands_frame.loc[b,'flux_corr'] = bands_frame.loc[b,'flux'] * colour_corr_factor



    # Create flux and error columns, for plotting with
    flux_plot = bands_frame['flux_corr'].values.copy()
    error_plot = bands_frame['error'].values.copy()
    errorbar_up, errorbar_down = bands_frame['error'].values, bands_frame['error'].values.copy()

    # Format errorbar sizes deal with negative fluxes
    errorbar_up[np.where(flux_plot <= 0)] = flux_plot[np.where(flux_plot <= 0)] + errorbar_up[np.where(flux_plot <= 0)]
    flux_plot[np.where(flux_plot <= 0)] = 1E-50

    # Format errobars to account for non-detections
    errorbar_down[np.where(errorbar_down > flux_plot)] = 0.99999 * flux_plot[np.where(errorbar_down > flux_plot)]

    # Plot datapoints
    if np.sum(bands_frame['limit']) == 0:
        ax.errorbar(bands_frame['wavelength']*1E6, flux_plot, yerr=[errorbar_down, errorbar_up], ecolor='black', elinewidth=1.5, capthick=0, marker='x', color='black', markersize=6.25, markeredgewidth=1.5, linewidth=0)
    else:
        ax.errorbar(bands_frame['wavelength'][bands_frame['limit']==False]*1E6, flux_plot[bands_frame['limit']==False], yerr=[errorbar_down[bands_frame['limit']==False], errorbar_up[bands_frame['limit']==False]], ecolor='black', elinewidth=1.5, capthick=0, marker='x', color='black', markersize=6.25, markeredgewidth=1.5, linewidth=0)
        ax.errorbar(bands_frame['wavelength'][bands_frame['limit']]*1E6, flux_plot[bands_frame['limit']], yerr=[errorbar_down[bands_frame['limit']], errorbar_up[bands_frame['limit']]], ecolor='gray', elinewidth=1.5, capthick=0, marker='x', color='gray', markersize=6.25, markeredgewidth=1.5, linewidth=0)



    # If parameter distribution provided, produce a thinned version to compute uncertainties
    if isinstance(posterior, np.ndarray):
        post = posterior[np.random.choice(range(posterior.shape[0]), size=min(2000,posterior.shape[0]), replace=False), :]

        # Generate SEDs for each sample of the thinned posterior, for individual components and for combined model
        post_wavelengths = np.logspace(-5, -2, num=500)
        post_fluxes_indv = np.zeros([post.shape[0], post_wavelengths.shape[0]])
        post_fluxes_tot = np.zeros([post.shape[0], post_wavelengths.shape[0]])
        for j in range(post.shape[0]):
            post_temp_vector, post_mass_vector, post_beta_vector, post_break_lambda_vector, post_correl_err_vector = ParamsExtract(post[j,:], fit_dict)
            post_fluxes_indv[j,:] = ModelFlux(post_wavelengths, post_temp_vector,
                                              post_mass_vector, fit_dict['distance'],
                                              kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'],
                                              beta=post_beta_vector, break_lambda=break_lambda_vector)
        post_fluxes_tot = post_fluxes_indv

        # Plot translucent SEDs of thinned posterior samples
        for j in range(post.shape[0]):
            ax.plot(post_wavelengths*1E6, post_fluxes_tot[j,:], ls='-', lw=1.0, c='red', alpha=0.0025)

        # Work out 16th, 50th, and 84th percentile fluxes at each wavelength
        lim_fluxes_tot = np.zeros([2, post_wavelengths.shape[0]])
        lim_fluxes_tot[0,:] = np.percentile(post_fluxes_tot, 16, axis=0)
        lim_fluxes_tot[1,:] = np.percentile(post_fluxes_tot, 84, axis=0)

        # Work out the median flux at a large number of closely-spaced wavelength intervals
        med_wavelengths = post_wavelengths[::5]
        med_fluxes_tot = np.median(post_fluxes_tot, axis=0)[::5]
        med_scatter_tot = np.std(post_fluxes_tot, axis=0)[::5]
        med_fit_dict = copy.deepcopy(fit_dict)
        med_fit_dict['bounds'] = True

        # Take the fluxes at the small wavelength intervals, and create a fit_dict with them
        med_bands_frame = pd.DataFrame(columns=fit_dict['bands_frame'].columns)
        for i in range(len(med_fluxes_tot)-1):
            med_bands_frame.loc[i,'band'] = 'SLICE'+str(int(round(med_wavelengths[i]*1E6)))
            med_bands_frame.loc[i,'wavelength'] = med_wavelengths[i]
            med_bands_frame.loc[i,'flux'] = med_fluxes_tot[i]
            med_bands_frame.loc[i,'error'] = med_scatter_tot[i]
            med_bands_frame.loc[i,'limit'] = False
            med_bands_frame.loc[i,'det'] = True
        med_bands_frame.loc[:,'wavelength'] = med_bands_frame['wavelength'].astype(float)
        med_bands_frame.loc[:,'flux'] = med_bands_frame['flux'].astype(float)
        med_bands_frame.loc[:,'error'] = med_bands_frame['error'].astype(float)
        med_fit_dict['bands_frame'] = med_bands_frame
        med_initial = params.copy()

        # Fit a model to the median fluxes in each small interval of wavelength
        NegLnLike = lambda *args: -LnLike(*args)
        med_opt = scipy.optimize.minimize(NegLnLike, med_initial, args=(med_fit_dict), method='Powell', options={'maxiter':500})
        med_params = params.copy() #!!! med_opt.x

        # Generate fit components for this "median model"
        med_temp_vector, med_mass_vector, med_beta_vector, med_break_lambda_vector, med_correl_err_vector = ParamsExtract(med_params, med_fit_dict)
        med_fit_wavelengths = np.logspace(-5, -2, num=2000)
        med_fit_fluxes = ModelFlux(med_fit_wavelengths, med_temp_vector, med_mass_vector, med_fit_dict['distance'],
                                   kappa_0=med_fit_dict['kappa_0'], kappa_0_lambda=med_fit_dict['kappa_0_lambda'],
                                   beta=med_beta_vector, break_lambda=break_lambda_vector)
        med_fit_fluxes_tot = med_fit_fluxes
        temp_vector, mass_vector, beta_vector = med_temp_vector, med_mass_vector, med_beta_vector

        # Plot "median model"
        ax.plot(med_fit_wavelengths*1E6, med_fit_fluxes_tot, ls='-', lw=1.5, c='red')



    # Construct strings containing parameter values
    temp_string = r'T$_{d}$ = '+str(np.around(temp_vector[0], decimals=3))[0:5] + r' K'
    mass_string = r',   M$_{d}$ = '+str(np.around(np.log10(mass_vector[0]), decimals=2))[0:5] + r' log$_{10}$M$_{\odot}$'
    beta_1_string = r'$\beta_{1}$ = '+str(np.around(beta_vector[0], decimals=2))[0:4]
    beta_2_string = r',   $\beta_{2}$ = '+str(np.around(beta_vector[1], decimals=2))[0:4]
    break_lambda_string = r',   $\lambda_{\it break}$ = '+str(np.around(break_lambda_vector[0]*1E6, decimals=0))[0:3] + r' $\mu$m'

    # Place text on figure
    string_x_base = 0.015
    string_y_base = 0.945
    string_y_step = 0.055
    ax.text(string_x_base, string_y_base, fit_dict['gal_dict']['name'], fontsize=15, fontweight='bold', transform=ax.transAxes, family=font_family)
    ax.text(string_x_base, string_y_base-(1*string_y_step), temp_string+mass_string, fontsize=14, transform=ax.transAxes, family=font_family)
    ax.text(string_x_base, string_y_base-(2*string_y_step), beta_1_string+beta_2_string+break_lambda_string, fontsize=14, transform=ax.transAxes, family=font_family)
    #ax.text(string_x_base, string_y_base-(3*string_y_step), break_lambda_string, fontsize=14, transform=ax.transAxes, family=font_family)



    # Scale x-axes to account for wavelengths provided
    xlim_min = 1E6 * 10.0**( np.log10( 0.66 * np.min( bands_frame['wavelength'] ) ) )#**( np.floor( np.log10( np.min( bands_frame['wavelength'] ) ) ) )
    xlim_max = 1E6 * 10.0**( np.log10( 1.33 * np.max( bands_frame['wavelength'] ) ) )#**( np.ceil( np.log10( np.max( bands_frame['wavelength'] ) ) ) )
    if (np.floor(np.log10(xlim_max)) - np.ceil(np.log10(xlim_min))) == 0:
        if (np.log10(xlim_max) - np.floor(np.log10(xlim_max))) > (np.ceil(np.log10(xlim_min)) - np.log10(xlim_min)):
            xlim_max = 1.33 * 10.0**(1.0 + np.floor(np.log10(xlim_max)))
        else:
            xlim_min = 0.66 * 10.0**(np.ceil(np.log10(xlim_min)) - 1.0)
    ax.set_xlim(xlim_min,xlim_max)

    # Scale y-axes to account for range of values and non-detections
    if sum(bands_frame['det']) > 0:
        ylim_min = 10.0**( -1.0 + np.round( np.nanmin( np.log10( flux_plot[bands_frame['det']] - error_plot[bands_frame['det']] ) ) ) )
        ylim_max = 10.0**( 1.0 + np.ceil( np.log10( 1.1 * np.max( flux_plot[bands_frame['det']] + error_plot[bands_frame['det']] ) ) ) )
    else:
        flux_where_pos = np.where(flux_plot > 0)
        ylim_min = 10.0**( -1.0 + np.round( np.nanmin( np.log10( flux_plot[flux_where_pos] / 2.0 ) ) ) )
        ylim_max = 10.0**( 1.0 + np.ceil( np.log10( 1.1 * np.max( flux_plot + error_plot ) ) ) )
    ax.set_ylim(ylim_min,ylim_max)

    # Format figure axes and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Wavelength ($\mu$m)', fontname=font_family, fontsize=20.0)
    ax.set_ylabel('Flux Density (Jy)', fontname=font_family, fontsize=20.0)

    # Format font of tick labels
    for xlabel in ax.get_xticklabels():
        xlabel.set_fontproperties(matplotlib.font_manager.FontProperties(family=font_family, size=17.5))
    for ylabel in ax.get_yticklabels():
        ylabel.set_fontproperties(matplotlib.font_manager.FontProperties(family=font_family, size=17.5))

    # Return figure and axis objects
    return fig, ax





def CornerPlot(mcmc_samples, params_highlight, fit_dict):
    """ Function to produce corner plot of posterior distribution, replacing histograms with KDEs, and with the maximum
    likelihood solution shown """

    # Enable seaborn for easy, attractive plots
    plt.ioff()
    sns.set(context='talk') # Possible context settings are 'notebook' (default), 'paper', 'talk', and 'poster'
    sns.set_style('ticks', {'font.sans-serif':'DejaVu Sans'})
    matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'

    # Generate label strings for parameter names
    labels = ParamsLabel(fit_dict)

    # Convert mass parameters into logarithmic space
    for i in range(len(labels)):
        if labels[i][:2] == '$M':
            mcmc_samples[:,i] = np.log10(mcmc_samples[:,i])
            params_highlight[i] = np.log10(params_highlight[i])

    # Remove NaN samples
    mcmc_samples_good = np.where(np.isnan(np.median(mcmc_samples, axis=1))==False)[0]
    mcmc_samples = mcmc_samples[mcmc_samples_good,:]

    # Convert break wavelengths to microns
    mcmc_samples[:,4] *= 1E6

    # Plot posterior corner diagrams (with histograms hidden)
    try:
        fig = corner.corner(mcmc_samples, labels=labels, quantiles=[0.16,0.5,0.84], range=[0.995]*len(labels),
                            show_titles=True, truths=params_highlight, hist_kwargs={'edgecolor':'none'})
    except:
        fig = corner.corner(mcmc_samples, labels=labels, quantiles=[0.16,0.5,0.84],
                            show_titles=True, truths=params_highlight, hist_kwargs={'edgecolor':'none'})

    # Loop over variables and subplots, finding histogram subplot corresponding to each variable
    for i in range(len(labels)):
        label = labels[i]
        for ax in fig.get_axes():
            if ax.get_title()[:len(label)] == label:

                # Now we've found the correct subplot for this variable, plot the KDE (with twice Freedman-Diaconis bandwidth)
                values = np.array(mcmc_samples[:,i])[:,np.newaxis]
                bandwidth = 2.0 * ( 2.0 * (np.percentile(values,75)-np.percentile(values,25)) ) / values.size**0.333#np.ptp(np.histogram(trace.get_values(varname),bins='fd')[1][:2])
                bandwidth = max(bandwidth, 0.01)
                kde = sklearn.neighbors.KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(values)
                line_x = np.linspace(np.nanmin(values), np.nanmax(values), 10000)[:,np.newaxis]
                line_y = kde.score_samples(line_x)
                line_y = np.exp(line_y)
                line_y = line_y * 0.9 * max(ax.get_ylim()) * line_y.max()**-1
                ax.plot(line_x,line_y, color='black')

                # Also, make column titles smaller to fit in long variable names, and set tick marks to point inside plots
                ax.set_title(ax.get_title(), fontsize=15)
            ax.tick_params(direction='in')

    # Return final figure and axes objects
    return fig, ax





def Autocorr(mcmc_chains, fit_dict):
    """ Function to plot the autocorrelation of the MCMC chains """

    # Enable seaborn for easy, attractive plots
    plt.ioff()
    sns.set(context='talk') # Possible context settings are 'notebook' (default), 'paper', 'talk', and 'poster'
    sns.set_style('ticks', {'font.sans-serif':'DejaVu Sans'})
    matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'

    # Define colour palettes to use for different parameters (up to three components)
    temp_palettes = ['PuBu']
    mass_palettes = ['BuPu']
    beta_palettes = ['GnBu', 'YlGnBu']
    correl_err_palettes = ['YlOrBr', 'PuRd']#, 'YlOrRd']
    break_lambda_palettes = ['OrRd']

    # Create dummy parameter vectors, just to find out how many parameters there are
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(mcmc_chains[0,0,:], fit_dict)

    # Assign colour palettes to components, and bundle into combined palettes list
    temp_palettes = np.repeat(temp_palettes, int(np.ceil(float(len(temp_vector))/float(len(temp_palettes))))).tolist()[:len(temp_vector)]
    mass_palettes = np.repeat(mass_palettes, int(np.ceil(float(len(mass_vector))/float(len(mass_palettes))))).tolist()[:len(mass_vector)]
    beta_palettes = np.repeat(beta_palettes, int(np.ceil(float(len(fit_dict['beta']))/float(len(beta_palettes))))).tolist()[:len(fit_dict['beta'])]
    correl_err_palettes = np.repeat(correl_err_palettes, int(np.ceil(float(len(correl_err_vector))/float(len(correl_err_palettes))))).tolist()[:len(correl_err_vector)]
    break_lambda_palettes = np.repeat(break_lambda_palettes, int(np.ceil(float(len(break_lambda_vector))/float(len(break_lambda_palettes))))).tolist()[:len(break_lambda_vector)]
    palettes = temp_palettes + mass_palettes + beta_palettes + correl_err_palettes + break_lambda_palettes

    # Generate figure, with subplot for each parameter
    labels = ParamsLabel(fit_dict)
    fig, ax = plt.subplots(nrows=mcmc_chains.shape[2], ncols=1,
                           figsize=(8,(1.5*fit_dict['n_params'])), sharex=True, squeeze=True)

    # Initiate record-keeping variables, and loop over parameters, then loop over chains
    #autocorr_time = np.zeros([mcmc_chains.shape[2], mcmc_chains.shape[0]])
    for i in range(mcmc_chains.shape[2]):
        for j in range(mcmc_chains.shape[0]):

            # Compute autocorrelation function and time and burn-in for chains
            print ('##### CHANGE THIS TO EMCEE\'S BUILT-IN AUTOCORRELATION FUNCTION, AS IT WORKS IN PYTHON 2 & 3 #####')
            autocorr_func = acor.function(mcmc_chains[j, :, i])
            #autocorr_time[i,j] = acor.acor(mcmc_chains[j, :, i])[0]

            # Plot autocorrelation function and autocorrelation time
            autocorr_func_conv = scipy.ndimage.filters.gaussian_filter(autocorr_func, 10)
            palette = sns.color_palette(palettes[i]+'_d', mcmc_chains.shape[0])
            ax[i].plot(range(mcmc_chains[j, :, i].shape[0]), autocorr_func_conv, color=palette[j], alpha=0.65, lw=1.0)
            #ax[i].plot([autocorr_time[i,j],autocorr_time[i,j]], [1E5,-1E5], c=palette[j], ls=':')

        # Format axis
        ax[i].set_ylim(-1, 1)
        ax[i].set_ylabel('${\it ACF\, (}$'+labels[i]+'${\it )}$')

    # Plot autocorrelation time on axes, and format accordingly
    autocorr_xlim = min(mcmc_chains.shape[1], 5000. * np.ceil((0.6*mcmc_chains.shape[1]) / 5000.))
    for i in range(mcmc_chains.shape[2]):
        ax[i].set_xlim(0, autocorr_xlim)
        ax[i].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, min_n_ticks=5, prune='lower'))

    # Perform final formatting, and return figure and axes objects
    ax[-1:][0].set_xlabel('MCMC Step')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax





def TracePlot(mcmc_chains, fit_dict):
    """ Function to produce a trace plot showing the MCMC chains, and use it to identify burn-in """

    # Enable seaborn for easy, attractive plots
    plt.ioff()
    sns.set(context='talk') # Possible context settings are 'notebook' (default), 'paper', 'talk', and 'poster'
    sns.set_style('ticks')

    # Define colour palettes to use for different parameters (up to three components)
    temp_palettes = ['PuBu']
    mass_palettes = ['BuPu']
    beta_palettes = ['GnBu', 'YlGnBu']
    break_lambda_palettes = ['RdPu']
    correl_err_palettes = ['YlOrRd', 'OrRd', 'YlOrBr']

    # Create dummy parameter vectors, just to find out how many parameters there are
    temp_vector, mass_vector, beta_vector, break_lambda_vector, correl_err_vector = ParamsExtract(mcmc_chains[0,0,:], fit_dict)

    # Assign colour palettes to components, and bundle into combined palettes list
    temp_palettes = np.repeat(temp_palettes, int(np.ceil(float(len(temp_vector))/float(len(temp_palettes))))).tolist()[:len(temp_vector)]
    mass_palettes = np.repeat(mass_palettes, int(np.ceil(float(len(mass_vector))/float(len(mass_palettes))))).tolist()[:len(mass_vector)]
    beta_palettes = np.repeat(beta_palettes, int(np.ceil(float(len(fit_dict['beta']))/float(len(beta_palettes))))).tolist()[:len(fit_dict['beta'])]
    break_lambda_palettes = np.repeat(break_lambda_palettes, int(np.ceil(float(len(break_lambda_vector))/float(len(break_lambda_palettes))))).tolist()[:len(break_lambda_vector)]
    correl_err_palettes = np.repeat(correl_err_palettes, int(np.ceil(float(len(correl_err_vector))/float(len(correl_err_palettes))))).tolist()[:len(correl_err_vector)]
    palettes = temp_palettes + mass_palettes + beta_palettes + correl_err_palettes + break_lambda_palettes

    # Generate figure, with subplot for each parameter
    labels = ParamsLabel(fit_dict)
    fig, ax = plt.subplots(nrows=mcmc_chains.shape[2], ncols=1, figsize=(8,(1.5*fit_dict['n_params'])), sharex=True, squeeze=True)

    # Put masses into log space
    mcmc_chains = mcmc_chains.copy()
    for i in range(len(labels)):
        if labels[i][:2] == '$M':
            mcmc_chains[:,:,i] = np.log10(mcmc_chains[:,:,i])

    # Convert break wavelengths to microns
    mcmc_chains[:,:,4] *= 1E6

    # Initiate record-keeping variables, and loop over parameters, then loop over chains
    for i in range(mcmc_chains.shape[2]):
        for j in range(mcmc_chains.shape[0]):

            # Plot chains
            palette = sns.color_palette(palettes[i]+'_d', mcmc_chains.shape[0])
            ax[i].plot(range(mcmc_chains[j, :, i].shape[0]), mcmc_chains[j, :, i], color=palette[j], alpha=0.3, lw=0.8)

        # Format axis
        ax[i].set_ylabel(labels[i])
        ax[i].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, min_n_ticks=5, prune='both'))
        ax[i].set_ylim(np.nanpercentile(mcmc_chains[:, :, i], 0.1), np.nanpercentile(mcmc_chains[:, :, i], 99.9))

    # Perform final formatting, and return figure and axes objects
    ax[-1:][0].set_xlabel('MCMC Step')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax
