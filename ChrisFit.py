# Import smorgasbord
from __future__ import print_function
import pdb
import sys
import os
import copy
import dill
import re
from difflib import SequenceMatcher
import multiprocessing as mp
import numpy as np
import scipy.stats
import scipy.interpolate
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import corner
import emcee

# Disable interactive plotting
plt.ioff()

# Define physical constants
c = 3E8
h = 6.64E-34
k = 1.38E-23





def Fit(gal_dict,
        bands_frame,
        beta_vary = True,
        beta = 2.0,
        components = 2,
        kappa_0 = 0.051,
        kappa_0_lambda = 500E-6,
        plot = True,
        correl_unc = None,
        priors = None,
        full_posterior = False,
        verbose = True):
        """
        Function that runs the ChrisFit dust SED fitting routine.

        Arguments:
            gal_dict:           A dictionary, containing entries called 'name', 'distance', and 'redshift', giving the
                                values for the target source in  question
            bands_frame:        A dataframe, with columns called 'band', 'flux', and 'error', providing the relevant
                                values for each band for the target source in question

        Keyword arguments:
            beta_vary:          A boolean, stating whether or not beta (the emissivity slope) should a free parameter,
                                or fixed
            beta:               A float, or list of floats, stating the value(s) of beta to use. If only a single float
                                is given, this is used for all components. If beta_vary is set to true, beta will
                                provide starting position for MCMC
            components:         An integer, stating how many modified blackbody components should make up the model
                                being fit
            kappa_0:            The value of the dust mass absorption coefficient, kappa_d, to use to cacculate dust mass
                                (uses Clark et al., 2016, value by default)
            kappa_0_lambda:     The reference wavelength for kappa_0; corresponding value of kappa_0 at other
                                wavelengths extrapolated via (kappa_0_lambda/lambda)**beta
            plot:               A boolean, stating whether to generate plots of the resulting SED fit; or,
                                alternatively, a string pointing to the desired plotting output directory
            correl_unc:          A list, each element of which (if any) is a dictionary describing band-covariant
                                uncertainties; for the 5% Hershcel-SPIRE band covariance, correl_unc would be:
                                [{'correl_bands':['SPIRE_250','SPIRE_350','SPIRE_500'],
                                'correl_scale':0.04,
                                'correl_distr':'flat'}],
                                where 'bands' describes the bands (as named in bands_frame) in question, 'correl_scale'
                                describes the size of the covariant component of the flux uncertainty (as a fraction of
                                measured source flux), and 'correl_distr' is the assumed distribution of the uncertainty
                                (currently accepting either 'flat', 'normal', or a defined function)
            priors:             A dictionary, of lists, of functions (yeah, I know); dictionary entries can be called
                                'temp', 'mass', and 'beta', each entry being an n-length list, where n is the number of
                                components, with the n-th list element being a function giving the ln-like prior for the
                                parameter in question (ie, temperature, mass, or beta) of the n-th model component; note
                                that the priors for any correlated uncertainty terms should be provided through the
                                correl_unc kwarg instead
            full_posterior:     A boolean, stating whether the full posterior distribution of each paramter should be
                                returned, or just the summary of median, credible interval, etc
            verbose:            A boolean, stating whether ChrisFit should provide verbose output whilst operating
            """


        # Add column to bands_frame, to record which fluxes are larger than their uncertainty
        bands_frame['det'] = bands_frame.loc[:,'flux'] > bands_frame.loc[:,'error']

        # Parse beta argument, so that each model component is assigned its own value (even if they are all the same)
        if not hasattr(beta, '__iter__'):
            beta = np.array([beta])
        if len(beta) == 1 and components > 1:
            beta = np.array(beta)
        elif (len(beta) > 1) and (len(beta)<components) and (components > 1):
            Exception('Either provide a single value of beta, or a list of values of length the number of components')

        # Parse correl_unc argument, so that if no value provided, an empty list is used throughout the rest of the function
        if not hasattr(correl_unc, '__iter__'):
            correl_unc = []

        # Bundle various fitting arguments in to a dictionary
        fit_dict = {'bands_frame':bands_frame,
                    'gal_dict':gal_dict,
                    'components':components,
                    'beta_vary':beta_vary,
                    'beta':beta,
                    'correl_unc':correl_unc,
                    'bounds':False,
                    'priors':priors,
                    'distance':gal_dict['distance'],
                    'kappa_0':kappa_0,
                    'kappa_0_lambda':kappa_0_lambda}

        # Determine number of parameters
        n_params = (2 * int(components)) + (int(fit_dict['beta_vary']) * len(fit_dict['beta'])) + len(correl_unc)
        PriorsConstruct(fit_dict)
        # Generate initial guess values for maximum-likelihood estimation (which will then itself be used to initialise emcee's estimation)
        max_like_fit_dict = copy.deepcopy(fit_dict)
        max_like_fit_dict['bounds'] = True
        max_like_fit_dict['correl_unc'] = False
        max_like_initial = MaxLikeInitial(max_like_fit_dict)#(20.0, 50.0, 5E-9*fit_dict['distance']**2.0, 5E-12*fit_dict['distance']**2.0, 2.0, 2.0, 0.0)

        # Find maximum-likelihood solution
        NegLnLike = lambda *args: -LnLike(*args)
        #max_like_opt = scipy.optimize.basinhopping(NegLnLike, max_like_initial, minimizer_kwargs={'args':(max_like_fit_dict)})
        max_like_opt = scipy.optimize.minimize(NegLnLike, max_like_initial, args=(max_like_fit_dict), method='powell', tol=1E-5)
        max_like = max_like_opt.x

        # Re-introduce any correlated uncertainty parameters that were excluded from maximum-likelihood fit
        max_like = np.array(max_like.tolist()+([0.0]*len(fit_dict['correl_unc'])))

        # Generate starting position for MCMC walkers, in small Gaussian cluster around maximum-likelihood position
        mcmc_n_walkers = 200
        mcmc_n_steps = 5000
        mcmc_initial = [max_like + 1e-4*np.random.randn(len(max_like)) for i in range(mcmc_n_walkers)]

        # Initiate and run emcee affine-invariant ensemble sampler
        mcmc_sampler = emcee.EnsembleSampler(mcmc_n_walkers, n_params, LnPost, args=[fit_dict], threads=mp.cpu_count())
        mcmc_sampler.run_mcmc(mcmc_initial, mcmc_n_steps)

        # Examine autocorrelation of MCMC chains, to identify burn-in
        mcmc_n_burn = 0.2 * mcmc_n_steps

        # Combine MCMC chains into final posteriors for each parameter
        mcmc_samples = mcmc_sampler.chain[:, mcmc_n_burn:, :].reshape((-1, n_params))
        dill.dump(mcmc_samples, open('/home/saruman/spx7cjc/MCMC.dj','wb'))
        #mcmc_samples = dill.load(open('/home/saruman/spx7cjc/MCMC.dj','rb'))

        # Plot posterior corner plot
        corner_fig, corner_ax = CornerPlot(mcmc_samples.copy(), max_like.copy(), fit_dict)
        if plot == True:
            corner_fig.savefig(gal_dict['name']+'_Corner.png', dpi=150)
        elif plot != False:
            if isinstance(plot, str):
                if os.path.exists(plot):
                    corner_fig.savefig(os.path.join(plot,gal_dict['name']+'_Corner.png'), dpi=150)

        # Plot median SED
        sed_fig, sed_ax = SEDborn(np.median(mcmc_samples, axis=0), fit_dict)
        if plot == True:
            sed_fig.savefig(gal_dict['name']+'_SED.png', dpi=150)
        elif plot != False:
            if isinstance(plot, str):
                if os.path.exists(plot):
                    sed_fig.savefig(os.path.join(plot,gal_dict['name']+'_SED.png'), dpi=150)





def LnLike(params, fit_dict):
    """ Function to compute ln-likelihood of some data, given the parameters of the proposed model """

    # Deal with parameter bounds, if they are required (for example, if we're doing a maximum-likelihood estimation)
    if fit_dict['bounds']:
        if not LikeBounds(params, fit_dict):
            return -np.inf

    # Programatically dust temperature, dust mass, and beta (variable or fixed) parameter sub-vectors from params tuple
    temp_vector, mass_vector, beta_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # Extract bands_frame from fit_dict
    bands_frame = fit_dict['bands_frame']

    # Loop over fluxes, to calculate the ln-likelihood of each, given the proposed model
    ln_like = []
    for b in bands_frame.index.values:

        # Skip this band if flux or uncertainty are nan
        if True in np.isnan([bands_frame.loc[b,'error'],bands_frame.loc[b,'flux']]):
            continue

        # Calculate predicted flux, given SED parameters
        band_flux_pred = ModelFlux(bands_frame.loc[b,'wavelength'], temp_vector, mass_vector, fit_dict['distance'],
                                   kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'], beta=beta_vector)

        # Update predicted flux value, to factor in colour correction (do this before correlated uncertainties, as colour corrections are calibrated assuming Neptune model is correct)
        col_correct_factor = ColourCorrect(bands_frame.loc[b,'wavelength'], bands_frame.loc[b,'band'].split('_')[0],
                                           temp_vector, mass_vector, beta_vector,
                                           kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'], verbose=False)
        band_flux_pred *= col_correct_factor[0]

        # If there is a correlated uncertainty term, reduce the flux uncertainty to its uncorrelated (non-systematic) component, and update predicted flux
        band_unc = bands_frame.loc[b,'error']
        if len(correl_err_vector) > 0:
            for i in range(len(fit_dict['correl_unc'])):
                correl_param = fit_dict['correl_unc'][i]
                if bands_frame.loc[b,'band'] in correl_param['correl_bands']:
                    band_unc = bands_frame.loc[b,'flux'] * np.sqrt((bands_frame.loc[b,'error']/bands_frame.loc[b,'flux'])**2.0 - correl_param['correl_scale']**2.0)
                    band_flux_pred *= 1 + correl_err_vector[i]

        # Calculate ln-likelihood of flux, given measurement uncertainties and proposed model
        band_ln_like = np.log(scipy.stats.norm.pdf(band_flux_pred, loc=bands_frame.loc[b,'flux'], scale=band_unc))

        # Factor in limits; for bands with limits if predicted flux is <= observed flux, it is assinged same ln-likelihood as if predicted flux == observed flux
        if bands_frame.loc[b,'limit']:
            if band_flux_pred < bands_frame.loc[b,'flux']:
                band_ln_like = np.log(scipy.stats.norm.pdf(bands_frame.loc[b,'flux'], loc=bands_frame.loc[b,'flux'], scale=band_unc))

        # Record ln-likelihood for this band
        ln_like.append(band_ln_like)

    # Calculate and return final data ln-likelihood
    ln_like = np.sum(np.array(ln_like))
    return ln_like





def LnPrior(params, fit_dict):
    """ Function to compute prior ln-likelihood of the parameters of the proposed model """

    # Programatically extract dust temperature, dust mass, and beta (varible or fixed) parameter sub-vectors from params tuple
    temp_vector, mass_vector, beta_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # If a prior kwarg has been given, use that; otherwise, construce a set of default priors
    if isinstance(fit_dict['priors'], dict):
        priors = fit_dict['priors']
    else:
        priors = PriorsConstruct(fit_dict)

    # Declare empty list to hold ln-like of each parameter
    ln_like = []

    # Calculate ln-like for temperature
    for i in range(fit_dict['components']):
        ln_like.append(priors['temp'][i](temp_vector[i]))

    # Calculate ln-like for mass
    for i in range(fit_dict['components']):
        ln_like.append(priors['mass'][i](mass_vector[i]))

    # Calculate ln-like for beta
    if fit_dict['beta_vary']:
        for i in range(len(fit_dict['beta'])):
            ln_like.append(priors['beta'][i](beta_vector[i]))

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





def ModelFlux(wavelength, temp, mass, dist, kappa_0=0.051, kappa_0_lambda=500E-6, beta=2.0):
    """
    Function to calculate flux at given wavelength(s) from dust component(s) of given mass and temperature, at a given
    distance, assuming modified blackbody ('greybody') emission.

    Arguments:
        wavelength:     A float, or list of floats, giving the wavelength(s) (in m) of interest
        temp:           A float, or list of floats, giving the temperature(s) (in K) of each dust component
        mass:           A float, or list of floats, giving the mass(es) (in M_sol) of each dust component
        dist:           A float, giving the distance to the target source (in pc)

    Keyword arguments:
        kappa_0:        A float, or list of floats, giving the dust mass absorption coefficient(s) (in m**2 kg**-1),
                        kappa, of each dust component; reference wavelengths given by kwarg kappa_0_lambda
        kappa_0_lambda: A float, or list of floats, giving the reference wavelength (in m) corresponding to each value
                        of kappa_0
        beta:           A float, or list of floats, giving the dust emissivity slope(s), beta, of each dust component

    If wavelength is given as a list, a list of output fluxes will be given, corresponding to the calculated flux at
    each wavelength.

    Temperature and mass can be set to be lists , corresponding to multiple dust components. For n components, both
    lists must be of length n.

    Optionally, a different dust mass absorption coefficient (ie, kappa) can be used for each component; this is done by
    giving lists of length n for kappa_0 and kappa_0_lambda.

    Optionally, a different dust emissivity slope (ie, beta) can be used for each component; this is done by giving a
    list of length n for beta.
    """
    # Establish the number of model components
    if hasattr(temp, '__iter__') and hasattr(mass, '__iter__'):
        if len(temp) != len(mass):
            Exception('Number of dust components needs to be identical for temp and mass variables')
        else:
            n_comp = len(temp)
    elif not hasattr(temp, '__iter__') and not hasattr(mass, '__iter__'):
        n_comp = 1
    else:
        Exception('Number of dust components needs to be identical for temp and mass variables')

    # As needed, convert variables to arrays
    wavelength = Numpify(wavelength)
    temp = Numpify(temp)
    mass = Numpify(mass, n_target=n_comp)
    kappa_0 = Numpify(kappa_0, n_target=n_comp)
    kappa_0_lambda = Numpify(kappa_0_lambda, n_target=n_comp)
    beta = Numpify(beta, n_target=n_comp)

    # Check that variables are the same length, when they need to be
    if np.std([len(temp), len(mass), len(beta), len(kappa_0), len(kappa_0_lambda)]) != 0:
        Exception('Number of dust components needs to be identical for temp/mass/beta/kappa_0/kappa_0_lambda variables')

    """ NB: Arrays have dimensons of n_comp rows by n_bands columns """

    # Convert wavelengths to frequencies (for bands of interest, and for kappa_0 reference wavelengths)
    nu = np.divide(c, wavelength)
    nu_0 = np.divide(c, kappa_0_lambda)

    # Calculate kappa for the frequency of each band of interest
    kappa_nu_base = np.outer(nu_0**-1, nu) # This being the array-wise equivalent of nu/nu_0
    kappa_nu_prefactor = np.array([ np.power(kappa_nu_base[m,:],beta[m]) for m in range(n_comp) ]) # This expontiates each model component's base term to its corresponding beta
    kappa_nu = np.array([ np.multiply(kappa_0[m],kappa_nu_prefactor[m,:]) for m in range(n_comp) ])

    # Caclulate Planck function prefactor for each frequency
    B_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)

    # Calculate exponent term in Planck function, for each component, at each frequency
    B_exponent = np.array([ np.divide((h*nu),(k*temp[m])) for m in range(n_comp) ])

    # Calculate final value of Planck function for each, for each model component, at each frequency (output array will have n_comp rows, and n_freq columns)
    B_planck = B_prefactor * (np.e**B_exponent - 1)**-1.0

    # Convert mass and distance values to SI unuts
    mass_kilograms = mass * 2E30
    dist_metres = dist * 3.26 * 9.5E15

    # Calculate flux for each component, for each dust model component
    flux = 0.0
    for m in range(n_comp):
        flux += 1E26 * kappa_nu[m,:] * dist_metres**-2.0 * mass_kilograms[m] * B_planck[m,:]

    # Return calculated flux (denumpifying it if is only single value)
    if flux.size == 0:
        flux = flux[0]
    #flux([250E-6,350E-6,500E-6], [21.7,64.1], [3.92*(10**7.93),3.92*(10**4.72)], 25E6, kappa_0=[0.051,0.051], kappa_0_lambda=[500E-6,500E-6], beta=[2.0,2.0])
    return flux





def ParamsExtract(params, fit_dict):
    """ Function to extract SED parameters from params vector (a tuple). Parameter vector is structured:
    (temp_1, temp_2, ..., temp_n, mass_1, mass_2, ..., mass_n,
    correl_err_1, correl_err_2, ..., correl_err_n, beta_1, beta_2, ..., beta_n);
    note that beta values are only included if fit_dict['beta_vary'] == True. """

    # Initiate and populate dust temperature and dust mass parameter sub-vectors
    temp_vector = []
    mass_vector = []
    [ temp_vector.append(params[i]) for i in range(fit_dict['components']) ]
    [ mass_vector.append(params[i]) for i in range(fit_dict['components'], 2*fit_dict['components']) ]

    # Initiate and populate beta parameter sub-vector (from params if beta variable, else from fit_dict otherwise)
    beta_vector = []
    if fit_dict['beta_vary']:
        beta_index_range_lower = 2 * fit_dict['components']
        if len(fit_dict['beta']) == 1:
            beta_index_range_upper = beta_index_range_lower + 1
        elif len(fit_dict['beta']) == fit_dict['components']:
            beta_index_range_upper = 3 * fit_dict['components']
        [ beta_vector.append(params[i]) for i in range(beta_index_range_lower, beta_index_range_upper) ]
    else:
        beta_index_range_lower = 2 * fit_dict['components']
        beta_index_range_upper = 2 * fit_dict['components']
        if len(fit_dict['beta']) == 1:
            beta_vector = ([fit_dict['beta'][0]] * fit_dict['components'])
        elif len(fit_dict['beta']) == fit_dict['components']:
            beta_vector = tuple(fit_dict['components'].tolist())
    if (len(beta_vector) == 1) and (fit_dict['components'] > 1):
        beta_vector = [beta_vector[0]] * fit_dict['components']

    # Initiate and populate correlated uncertainty parameter sub-vector
    correl_err_vector = []
    if hasattr(fit_dict['correl_unc'], '__iter__'):
        correl_err_index_range_lower = beta_index_range_upper
        correl_err_index_range_upper = beta_index_range_upper + len(fit_dict['correl_unc'])
        [ correl_err_vector.append(params[i]) for i in range(correl_err_index_range_lower, correl_err_index_range_upper) ]

    # Return parameters tuple
    return (tuple(temp_vector), tuple(mass_vector), tuple(beta_vector), tuple(correl_err_vector))





def ParamsLabel(fit_dict):
    """ Function to generate list of plot-ready labels for free parameters """

    # Initiate list to hold labels
    labels = []

    # Depending on how many components there are, generate subscripts for mass and temperature parameters
    if fit_dict['components'] == 1:
        subs = ['']
    elif fit_dict['components'] == 2:
        subs = ['_{c}','_{w}']
    else:
        subs = range(1,fit_dict['components']+1)
        subs = ['_{'+str(subs[i])+'}' for i in range(len(subs))]

    # Add temperature and mass labels to list
    for i in range(fit_dict['components']):
        labels.append(r'$T'+subs[i]+'$')
    for i in range(fit_dict['components']):
        labels.append(r'$M'+subs[i]+'$')

    # Generate beta labels (depending on how many beta parameters there are) and add them to list
    if fit_dict['beta_vary']:
        if len(fit_dict['beta']) == 1:
            labels.append(r'$\beta$')
        elif len(fit_dict['beta']) == fit_dict['components']:
            for i in range(fit_dict['components']):
                labels.append(r'$\beta'+subs[i]+'$')

    # Generate correlated uncertainty labels (if necessary) and add them to list
    if hasattr(fit_dict['correl_unc'], '__iter__'):
        for i in range(len(fit_dict['correl_unc'])):
            band_first = re.compile('[^a-zA-Z]').sub('',fit_dict['correl_unc'][i]['correl_bands'][0])
            band_last = re.compile('[^a-zA-Z]').sub('',fit_dict['correl_unc'][i]['correl_bands'][-1:][0])
            band_match = SequenceMatcher(None, band_first, band_last).find_longest_match(0, len(band_first), 0, len(band_last))
            if band_match.size == 0:
                instr = band_first
            else:
                instr = band_first[band_match.a:band_match.size]
                labels.append(r'$\upsilon_{'+instr+'}$')

    # Return list of labels
    return labels





def MaxLikeInitial(fit_dict):
    """ Function to generate initial guess values for maximum-likelihood fitting """

    # Declare list to hold guesses
    guess = []

    # Temperature guesses for 18K if one MBB; 18K and 50K if two MBB; equally spaced therebetween for 3 or more
    temp_guess = np.linspace(18.0, 35.0, num=fit_dict['components'])
    guess += temp_guess.tolist()

    # Use flux and distance to estimate likely cold dust mass, based on empirical relation
    bands_frame = fit_dict['bands_frame']
    peak_flux = bands_frame.where((bands_frame['wavelength']>=200E-6)&(bands_frame['wavelength']<1E-3))['flux'].max()
    peak_lum = peak_flux * fit_dict['distance']**2.0
    peak_mass = 10**(np.log10(peak_lum)-8)

    # Mass guesses are based on empirical relation, then scale for kappa
    mass_guess = np.array([peak_mass] * fit_dict['components'])
    mass_guess *= 0.051 / (fit_dict['kappa_0'] * (fit_dict['kappa_0_lambda'] / 500E-6)**fit_dict['beta'][0])
    mass_guess *= 10**((temp_guess-18)/-15)
    guess += mass_guess.tolist()

    # Beta is always guessed to have a value of 2
    if fit_dict['beta_vary']:
        beta_guess = np.array(fit_dict['beta'])
        guess += beta_guess.tolist()

    # Correlated uncertainties are always guessed to have a value of 0
    if hasattr(fit_dict['correl_unc'], '__iter__'):
        correl_err_guess = np.array([0.0] * len(fit_dict['correl_unc']))
        guess += correl_err_guess.tolist()

    # Return tuple of guesses
    return tuple(guess)





def PriorsConstruct(fit_dict):
    """ Function to auitomatically construct a set of default priors, given the basic parameters of the model as
    described by the ChrisFit input """

    # Initialise dictionary to hold priors
    priors = {'temp':[],
              'mass':[],
              'beta':[]}

    # Define function to find scaling factor for gamma distribution of given mode, alpha, and location
    GammaScale = lambda mode, alpha, phi: (mode-phi)/(alpha-1.0)

    # Create temperature priors, using gamma distribution (with kwarg in lambda to make iterations evaluate separately)
    temp_alpha = np.linspace(2.5, 3.0, num=fit_dict['components'])
    temp_mode = np.linspace(18.0, 35.0, num=fit_dict['components'])
    temp_phi = np.linspace(5.0, 15.0, num=fit_dict['components'])
    for i in range(fit_dict['components']):
        temp_scale = GammaScale(temp_mode[i],temp_alpha[i],temp_phi[i])
        temp_ln_like = lambda temp, temp_alpha=temp_alpha[i], temp_phi=temp_phi[i], temp_scale=temp_scale: np.log(scipy.stats.gamma.pdf(temp, temp_alpha, loc=temp_phi, scale=temp_scale))
        priors['temp'].append(temp_ln_like)

    # Use flux and distance to estimate likely cold dust mass, based on empirical relation
    bands_frame = fit_dict['bands_frame']
    peak_flux = bands_frame.where((bands_frame['wavelength']>=200E-6)&(bands_frame['wavelength']<1E-3))['flux'].max()
    peak_lum = peak_flux * fit_dict['distance']**2.0
    peak_mass = 10**(np.log10(peak_lum)-8)

    # Use likely cold dust mass to construct mass priors, using log-t distribution (with kwarg in lambda to make iterations evaluate separately)
    mass_mode = np.array([peak_mass] * fit_dict['components'])
    mass_mode *= 0.051 / (fit_dict['kappa_0'] * (fit_dict['kappa_0_lambda'] / 500E-6)**fit_dict['beta'][0])
    mass_mode *= 10**((temp_mode-18)/-15)
    mass_mode = np.log10(mass_mode)
    mass_sigma = np.array([2.0] * fit_dict['components'])
    for i in range(fit_dict['components']):
        mass_ln_like = lambda mass, mass_mode=mass_mode[i], mass_sigma=mass_sigma[i]: np.log(10.0**scipy.stats.t.pdf(np.log10(mass), 1, loc=mass_mode, scale=mass_sigma))
        priors['mass'].append(mass_ln_like)

    # Create beta priors, using gamma distribution
    if fit_dict['beta_vary']:
        beta_ln_like = lambda beta: np.log(scipy.stats.gamma.pdf(beta, 3, loc=0, scale=1))
        priors['beta'] = [beta_ln_like] * len(fit_dict['beta'])

    # Return comleted priors dictionary
    return priors





def LikeBounds(params, fit_dict):
    """ Function to check whether parameters for a proposed model violate standard boundary conditions. This is for
    maximum likelihood estimations, where there are no priors to go by. If the parameters are good, the function returns
    a value of True; else, it returns a value of False."""

    # Extract parameter vectors
    temp_vector, mass_vector, beta_vector, correl_err_vector = ParamsExtract(params, fit_dict)

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

    # Check that temperature terms are all physical (ie, temp > 5 kelvin)
    if np.where(np.array(temp_vector)<5)[0].size > 0:
        return False

    # Check that mass terms are all physical (ie, mass > 0 Msol)
    if np.where(np.array(mass_vector)<0)[0].size > 0:
        return False

    # Check that beta terms are all physical (ie, 1 < beta < 4)
    if (np.where(np.array(beta_vector)<1)[0].size > 0) or (np.where(np.array(beta_vector)>4)[0].size > 0):
        return False

    # If we've gotten this far, then everything is fine
    return True





def ColourCorrect(wavelength, instrument, temp, mass, beta, kappa_0=0.051, kappa_0_lambda=500E-6, verbose=False):
    """ Function to calculate colour-correction FACTOR appropriate to a given underlying spectrum. Will work for any
    instrument for which file 'Color_Corrections_INSTRUMENTNAME.csv' is found in the same directory as this script. """

    # Set location of ChrisFuncs.py to be current working directory, recording the old CWD to switch back to later
    old_cwd = os.getcwd()
    os.chdir(str(os.path.dirname(os.path.realpath(sys.argv[0]))))

    # Identify instrument and wavelength, and read in corresponding colour-correction data
    unknown = False
    try:
        try:
            data_table = np.genfromtxt('Colour_Corrections_'+instrument+'.csv', delimiter=',', names=True)
        except:
            data_table = np.genfromtxt(os.path.join('ChrisFit','Colour_Corrections_'+instrument+'.csv'), delimiter=',', names=True)
        data_index = data_table['alpha']
        data_column = 'K'+str(int(wavelength*1E6))
        data_factor = data_table[data_column]
    except:
        unknown = True
        if verbose == True:
            print(' ')
            print('Instrument \''+instrument+'\' not recognised, no colour correction applied.')

    # If instrument successfully identified, perform colour correction; otherwise, cease
    if unknown==True:
        factor = 1.0
        index = np.NaN
    elif unknown==False:

        # Calculate relative flux at wavelengths at points at wavelengths 1% to either side of target wavelength (no need for distance or kappa, as absolute value is irrelevant)
        lambda_plus = wavelength*1.01
        lambda_minus = wavelength*0.99
        flux_plus = ModelFlux(lambda_plus, temp, mass, 1E6, kappa_0=kappa_0, kappa_0_lambda=kappa_0_lambda, beta=beta)
        flux_minus = ModelFlux(lambda_minus, temp, mass, 1E6, kappa_0=kappa_0, kappa_0_lambda=kappa_0_lambda, beta=beta)

        # Determine spectral index (remembering to convert to frequency space)
        index = ( np.log10(flux_plus) - np.log10(flux_minus) ) / ( np.log10(lambda_plus) - np.log10(lambda_minus) )
        index= -1.0 * index

        # Use cubic spline interpolation to estimate colour-correction divisor at calculated spectral index
        interp = scipy.interpolate.interp1d(data_index, data_factor, kind='linear', bounds_error=None, fill_value='extrapolate')
        factor = interp(index)[0]

    # Restore old cwd, and return results
    os.chdir(old_cwd)
    return factor, index





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





def SEDborn(params, fit_dict, params_dist=False, font_family='sans'):
    """ Function to plot an SED, with the same information used to produce fit """

    # Enable seaborn for easy, attractive plots
    plt.ioff()
    sns.set(context='talk') # Possible context settings are 'notebook' (default), 'paper', 'talk', and 'poster'
    sns.set_style('ticks')



    # Initialise figure
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])

    # Extract band dataframe and parameter vectors
    bands_frame = fit_dict['bands_frame']
    temp_vector, mass_vector, beta_vector, correl_err_vector = ParamsExtract(params, fit_dict)

    # Select only bands of interest
    bands_frame = bands_frame.loc[np.isnan(bands_frame['flux']) == False]

    # Generate fit components
    fit_wavelengths = np.linspace(10E-6, 10000E-6, num=10000)
    fit_fluxes = np.zeros([fit_dict['components'], len(fit_wavelengths)])
    for i in range(fit_dict['components']):
        fit_fluxes[i,:] = ModelFlux(fit_wavelengths, temp_vector[i], mass_vector[i], fit_dict['distance'],
                                    kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'], beta=beta_vector[i])
    fit_fluxes_tot = np.sum(fit_fluxes, axis=0)

    # Plot fits
    for i in range(fit_dict['components']):
        ax.plot(fit_wavelengths*1E6, fit_fluxes[i,:], ls='--', lw=1.0, c='black')
    ax.plot(fit_wavelengths*1E6, fit_fluxes_tot, ls='-', lw=1.5, c='red')

    # Colour-correct fluxes according to model being plotted
    bands_frame['flux_corr'] = bands_frame['flux'].copy()
    for b in bands_frame.index:
        colour_corr_factor = ColourCorrect(bands_frame.loc[b,'wavelength'], bands_frame.loc[b,'band'].split('_')[0],
                                           temp_vector, mass_vector, beta_vector,
                                           kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'], verbose=False)
        bands_frame.loc[b,'flux_corr'] = bands_frame.loc[b,'flux'] * colour_corr_factor[0]



    # Create flux and error columns, for plotting with
    flux_plot = bands_frame['flux_corr'].values
    error_plot = bands_frame['error'].values
    errorbar_up, errorbar_down = bands_frame['error'].values, bands_frame['error'].values

    # Format errorbar sizes deal with negative fluxes
    error_plot[np.where(flux_plot <= 0)] -= flux_plot[np.where(flux_plot <= 0)]
    flux_plot[np.where(flux_plot <= 0)] = 1E-50

    # Format errobars to account for non-detections
    errorbar_down[np.where(errorbar_down > flux_plot) ] = 0.999 * flux_plot[np.where(errorbar_down > flux_plot)]

    # Plot datapoints
    if np.sum(bands_frame['limit']) == 0:
        ax.errorbar(bands_frame['wavelength']*1E6, flux_plot, yerr=[errorbar_up, errorbar_down], ecolor='black', elinewidth=1.5, capthick=1.5, marker='x', color='black', markersize=6.25, markeredgewidth=1.5, linewidth=0)
    else:
        ax.errorbar(bands_frame['wavelength'][bands_frame['limit']==False]*1E6, flux_plot[bands_frame['limit']==False], yerr=[errorbar_down[bands_frame['limit']==False], errorbar_up[bands_frame['limit']==False]], ecolor='black', elinewidth=1.5, capthick=1.5, marker='x', color='black', markersize=6.25, markeredgewidth=1.5, linewidth=0)
        ax.errorbar(bands_frame['wavelength'][bands_frame['limit']]*1E6, flux_plot[bands_frame['limit']], yerr=[errorbar_down[bands_frame['limit']], errorbar_up[bands_frame['limit']]], ecolor='gray', elinewidth=1.5, capthick=1.5, marker='x', color='gray', markersize=6.25, markeredgewidth=1.5, linewidth=0)



    # Calculate residuals
    flux_resid = flux_plot - ModelFlux(bands_frame['wavelength'], temp_vector, mass_vector, fit_dict['distance'], kappa_0=fit_dict['kappa_0'], kappa_0_lambda=fit_dict['kappa_0_lambda'], beta=beta_vector[i])
    chi = (flux_resid / bands_frame['error'])[bands_frame['limit']==False]
    chi_squared = chi**2



    # Construct strings containing parameter values
    temp_1_value_string = 'T$_{c}$ = '+str(np.around(temp_vector[0], decimals=3))[0:5]
    mass_1_value_string = ',   M$_{c}$ = '+str(np.around(np.log10(mass_vector[0]), decimals=3))[0:5]
    if fit_dict['components'] == 1:
        temp_2_value_string = ''
        mass_2_value_string = ''
        mass_tot_value_string = ''
    elif fit_dict['components'] == 2:
        temp_2_value_string = 'T$_{w}$ = '+str(np.around(temp_vector[1], decimals=3))[0:5]
        mass_2_value_string = ',   M$_{w}$ = '+str(np.around(np.log10(mass_vector[1]), decimals=3))[0:5]
        mass_tot_value_string = ',   M$_{d}$ = '+str(np.around(np.log10(np.sum(mass_vector)), decimals=3))[0:5]
    if (fit_dict['beta_vary'] == True) and (len(fit_dict['beta']) == 1):
        beta_1_value_string = ',   $\\beta$ = '+str(np.around(beta_vector[0], decimals=2))[0:4]
    else:
        beta_1_value_string = ''

    # Construct strings for present uncertainties (if available)
    temp_1_error_string = ''
    mass_1_error_string = ''
    temp_2_error_string = ''
    mass_2_error_string = ''
    mass_tot_error_string = ''
    beta_1_error_string = ''
    """if isinstance(params_dist, np.ndarray):
        temp_1_error_string = ' $\pm$ '+str(np.around(bs_T_c_sigma, decimals=3))[0:5]
        mass_1_error_string = ' $\pm$ '+str(np.around(bs_M_c_sigma_log, decimals=3))[0:5]
        if fit_dict['components'] == 2:
            temp_2_error_string = ' $\pm$ '+str(np.around(bs_T_w_sigma, decimals=3))[0:5]
            mass_2_error_string = ' $\pm$ '+str(np.around(bs_M_w_sigma_log, decimals=3))[0:5]
            mass_tot_error_string = ' $\pm$ '+str(np.around(bs_M_d_sigma_log, decimals=3))[0:5]
        if (fit_dict['beta_vary'] == True) and (len(fit_dict['beta']) == 1):
            beta_1_error_string = ' $\pm$ '+str(np.around( bs_beta_sigma, decimals=3))[0:4]"""

    # Assemble combined results strings
    temp_1_string = temp_1_value_string + temp_1_error_string + ' K'
    mass_1_string = mass_1_value_string + mass_1_error_string + ' log$_{10}$M$_{\odot}$'
    temp_2_string = temp_2_value_string + temp_2_error_string + ' K'
    mass_2_string = mass_2_value_string + mass_2_error_string + ' log$_{10}$M$_{\odot}$'
    mass_tot_string = mass_tot_value_string + mass_tot_error_string + ' log$_{10}$M$_{\odot}$'
    beta_1_string = beta_1_value_string + beta_1_error_string

    # Calculate chi-squared and produce corresponding string
    chi_squared_string = '$\chi^{2}$ = '+str(np.around(np.sum(chi_squared), decimals=3))[0:5]

    # Place text on figure
    string_x_base = 0.035
    string_y_base = 0.925
    string_y_step = 0.06
    ax.text(string_x_base, string_y_base, fit_dict['gal_dict']['name'], fontsize=15, fontweight='bold', transform=ax.transAxes, family=font_family)
    ax.text(string_x_base, string_y_base-(1*string_y_step), temp_1_string+mass_1_string, fontsize=14, transform=ax.transAxes, family=font_family)
    ax.text(string_x_base, string_y_base-(2*string_y_step), temp_2_string+mass_2_string, fontsize=14, transform=ax.transAxes, family=font_family)
    ax.text(string_x_base, 0.805-(fit_dict['components']-1)*(0.805-0.745), chi_squared_string+beta_1_string+mass_tot_string, fontsize=14, transform=ax.transAxes, family=font_family)



    # Scale x-axes to account for wavelengths provided
    xlim_min = 1E6 * 10.0**( np.floor( np.log10( np.min( bands_frame['wavelength'] ) ) ) )
    xlim_max = 1E6 * 10.0**( np.ceil( np.log10( np.max( bands_frame['wavelength'] ) ) ) )
    ax.set_xlim(xlim_min,xlim_max)

    # Scale y-axes to account for range of values and non-detections
    ylim_min = 10.0**( -1.0 + np.round( np.log10( np.min( bands_frame['flux'].where(bands_frame['det']) - error_plot[bands_frame['det']] ) ) ) )
    ylim_max = 10.0**( 1.0 + np.ceil( np.log10( 1.1 * np.max( bands_frame['flux'].where(bands_frame['det']) + error_plot[bands_frame['det']] ) ) ) )
    ax.set_ylim(ylim_min,ylim_max)

    # Format figure axes and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Wavelength ($\mu$m)', fontname=font_family)#, fontsize=17.5)
    ax.set_ylabel('Flux Density (Jy)', fontname=font_family)#, fontsize=17.5)

    # Format font of tick labels
    for xlabel in ax.get_xticklabels():
        xlabel.set_fontproperties(matplotlib.font_manager.FontProperties(family=font_family, size=15))
    for ylabel in ax.get_yticklabels():
        ylabel.set_fontproperties(matplotlib.font_manager.FontProperties(family=font_family, size=15))

    # Return figure and axis objects
    return fig, ax





def CornerPlot(mcmc_samples, max_like, fit_dict):
    """ Function to produce corner plot of posterior distribution, replacing histograms with KDEs, and with the maximum
    likelihood solution shown """

    # Enable seaborn for easy, attractive plots
    plt.ioff()
    sns.set(context='talk') # Possible context settings are 'notebook' (default), 'paper', 'talk', and 'poster'
    sns.set_style('ticks')

    # Generate label strings for parameter names
    labels = ParamsLabel(fit_dict)

    # Convert mass parameters into logarithmic space
    for i in range(len(labels)):
        if labels[i][:2] == '$M':
            mcmc_samples[:,i] = np.log10(mcmc_samples[:,i])
            max_like[i] = np.log10(max_like[i])

    # Plot posterior corner diagrams (with histograms hidden)
    fig = corner.corner(mcmc_samples,
                        labels=labels,
                        quantiles=[0.16,0.5,0.84],
                        range=[0.9999]*len(labels),
                        show_titles=True,
                        truths=max_like,
                        hist_kwargs={'edgecolor':'none'})

    # Loop over variables and subplots, finding histogram subplot corresponding to each variable
    for i in range(len(labels)):
        label = labels[i]
        for ax in fig.get_axes():
            if ax.get_title()[:len(label)] == label:

                # Now we've found the correct subplot for this variable, plot the KDE (with twice Freedman-Diaconis bandwidth)
                values = np.array(mcmc_samples[:,i])[:,np.newaxis]
                bandwidth = 2.0 * ( 2.0 * (np.percentile(values,75)-np.percentile(values,25)) ) / values.size**0.333#np.ptp(np.histogram(trace.get_values(varname),bins='fd')[1][:2])
                kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(values)
                line_x = np.linspace(np.nanmin(values), np.nanmax(values), 10000)[:,np.newaxis]
                line_y = kde.score_samples(line_x)
                line_y = np.exp(line_y)
                line_y = line_y * 0.9 * max(ax.get_ylim()) * line_y.max()**-1
                ax.plot(line_x,line_y, color='black')

            # Also, set tick marks to point inside plots
            ax.tick_params(direction='in')

    # Return final figure and axes objects
    return fig, ax