# Import smorgasbord
from __future__ import print_function
import pdb
import copy
import numpy as np
import emcee

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
        covar_unc = None,
        priors = None,
        full_posterior = False):
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
            kappa_0:            The value of the dust mass absorption coefficient, kappa_d, to use to caculate dust mass
                                (uses Clark et al., 2016, value by default)
            kappa_0_lambda:     The reference wavelength for kappa_0; corresponding value of kappa_0 at other
                                wavelengths extrapolated via (lambda_0/lambda)**beta
            plot:               A boolean, stating whether to generate plots of the resulting SED fit
            covar_unc:          A list, each element of which (if any) is a dictionary describing band-covariant
                                uncertainties; for the 5% Hershcel-SPIRE band covariance, covar_unc would be:
                                [{'covar_bands':['SPIRE_250','SPIRE_350','SPIRE_500'],
                                'covar_scale':0.04,
                                'covar_distr':'flat'}],
                                where 'bands' describes the bands (as named in bands_frame) in question, 'covar_scale'
                                describes the size of the covariant component of the flux uncertainty (as a fraction of
                                measured source flux), and 'covar_dist' is the distribution of the uncertainty
                                (currently accepting either 'flat' or 'normal')
            priors:             A dictionary, of lists, of functions (yeah, I know); dictionary entries can be called
                                'temp', 'mass', and 'beta', each entry being an n-length list, where n is the number of
                                components, with the n-th list element being a function giving the prior for the
                                parameter in question (ie, temperature, mass, or beta) of the n-th model component
            full_posterior:     A boolean, stating whether the full posterior distribution of each paramter should be
                                returned, or just the summary of median, credible interval, etc
            """


        def LnLike(params, bands_frame, fit_dict):
            """ Funtion to compute log-likelihood of some data, given the parameters of the proposed model """

            # Programatically dust temperature, dust mass, and beta (varible or fixed) parameter sub-vectors from params tuple
            temp_vector, mass_vector, beta_vector = ParamsExtract(params, fit_dict)

            # Loop over fluxes, to calculate the log-likelihood of each, given the proposed model
            for b in range(bands_frame['wavelength'].size):
                pdb.set_trace()


            ### REMEMBER TO HANDLE LIMITS - JUST MAKE IT SO THAT LOG-LIKELIHOODS BENEATH MOST-LIKELY VALUE ARE ALL SAME AS THE MOST LIKELY VALUE ###

            # Return data log-likelihood
            return



        def LnPrior(params, fit_dict):
            """ Function to compute prior log-likelihood of the parameters of the proposed model """

            # Programatically dust temperature, dust mass, and beta (varible or fixed) parameter sub-vectors from params tuple
            temp_vector, mass_vector, beta_vector = ParamsExtract(params, fit_dict)

            # Return prior log-likelihood
            return



        def LnPost(params, bands_frame, fit_dict):
            """ Funtion to compute posterior log-likelihood of the parameters of the proposed model, given some data """

            # Caculate prior log-likelihood of the proposed model parameters
            ln_prior = LnPrior(params, fit_dict)

            # Caculate the log-likelihood of the data, given the proposed model parameters
            ln_like = LnLike(params, bands_frame, fit_dict)

            # Calculate and return the posterior log-likelihood of the proposed model parameters, given the data
            ln_prob = ln_prior + ln_like
            return ln_prob



        # Parse beta argument, so that each model component is assigned its own value (even if they are all the same)
        if not hasattr(beta, '__iter__'):
            beta = np.array([beta])
        if len(beta) == 1 and components > 1:
            beta = np.array([beta[0]] * int(components))
        elif len(beta) != int(components):
            Exception('Either provide a single value of beta, or a list of values of length the number of components')

        # Bundle various fitting argumnts in to a dictionary
        fit_dict = {'components':components,
                    'beta_vary':beta_vary,
                    'beta':beta,
                    'covar_unc':covar_unc}

        # Determine number of parameters
        n_params = (2 * int(components)) + int(fit_dict['beta_vary'])

        # Arbitrary test model
        test = LnLike((21.7, 64.1, 3.92*(10**7.93), 3.92*(10**4.72), 2.0, 2.0), bands_frame, fit_dict)

        # Initiate emcee affine-invariant ensemble sampler
        sampler = emcee.EnsembleSampler(n_walkers, n_params, LnProb, args=(bands_frame, fit_dict))










def ModelFlux(wavelength, temp, mass, dist, kappa_0=0.051, kappa_0_lambda=500E-6, beta=2.0):
    """
    Function to caculate flux at given wavelength(s) from dust component(s) of given mass and temperature, at a given
    distance, assuming modified blackbody ('greybody') emission.

    Arguments:
        wavelength:     A float, or list of floats, giving the wavelength(s) (in m) of interest
        temp:           A float, or list of floats, giving the temperature(s) (in K) of each dust component
        mass:           A float, or list of floats, giving the mass(es) (in M_sol) of each dust component
        dist:           A float, giving the distance to the target source (in pc)

    Keyword arguments:
        kappa_0:        A float, or list of floats, giving the dust mass absorption coefficient(s) (in m**2 kg**-1),
                        kappa, of each dust component; reference wavelengths given by kwarg kappa_0_lambda
        kappa_0_lambda:       A float, or list of floats, giving the reference wavelength (in m) coresponding to each value
                        of kappa_0
        beta:           A float, or list of floats, giving the dust emissivity slope(s), beta, of each dust component

    If wavelenghth is given as a list, a list of output fluxes will be given, corresponding to the calculated flux at
    each wavelength.

    Temperature and mass can be set to be lists , corresponding to multiple dust components. For n components, both
    lists must be of length n.

    Optionally, a different dust mass absorption coefficient (ie, kappa) can be used for each component; this is done by
    giving lists of length n for kappa_0 and kappa_0_lambda.

    Optionally, a different dust emissivity slope (ie, beta) can be used for each component; this is done by giving a
    list of length n for beta.
    """


    # Record number of model components, and number of bands of interest
    n_comp = len(temp)

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





def Numpify(var, n_target=False):
    """ Function for checking if variable is a list, and (if necessary) converting to a n_target length list of identical entries """

    # If variable is not iterable (ie, a list/array/etc), convert into an appropriate-length list
    if not hasattr(var, '__iter__'):
        if not n_target:
            var = list(var)
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



def ParamsExtract(params, fit_dict):
    """ Function to extract SED parameters from params vector (a tuple). Parameter vector is structured:
    (temp_1, temp_2, ..., temp_n, mass_1, mass_2, ..., mass_n, beta_1, beta_2, ..., beta_n);
    note that beta values are only included if fit_dict['beta_vary'] == True. """

    # Initiate and populate dust temperature and dust mass parameter sub-vectors
    temp_vector = []
    mass_vector = []
    [ temp_vector.append(params[i]) for i in range(fit_dict['components']) ]
    [ mass_vector.append(params[i]) for i in range(fit_dict['components'], 2*fit_dict['components']) ]

    # Initiate and populate beta parameter sub-vector (from params if beta variable, else from fit_dict otherwise)
    if fit_dict['beta_vary']:
        beta_vector = []
        [ beta_vector.append(params[i]) for i in range(2*fit_dict['components'], 3*fit_dict['components']) ]
    else:
        beta_vector = copy.deepcopy(fit_dict['beta'])

    return (tuple(temp_vector), tuple(mass_vector), tuple(beta_vector))


