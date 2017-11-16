# Import smorgasbord
from __future__ import print_function
import pdb
import sys
import os
import numpy as np

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




def ModelFlux(wavelength, temp, mass, dist, kappa_0=0.051, lambda_0=500E-6, beta=2.0):
    """
    Function to caculate flux at given wavelength(s) from dust component(s) of given mass and temperature, at a given distance, assuming modified blackbody ('greybody') emission

    Arguments:
        wavelength:     A float, or list of floats, giving the wavelength(s) (in m) of interest
        temp:           A float, or list of floats, giving the temperature(s) (in K) of each dust component
        mass:           A float, or list of floats, giving the mass(es) (in M_sol) of each dust component
        dist:           A float, giving the distance to the target source (in pc)

    Keyword arguments:
        kappa_0:        A float, or list of floats, giving the dust mass absorption coefficient(s) (in m**2 kg**-1), kappa, of each dust component; reference wavelengths given by kwarg lambda_0
        lambda_0:       A float, or list of floats, giving the reference wavelength (in m) coresponding to each value of kappa_0
        beta:           A float, or list of floats, giving the dust emissivity slope(s), beta, of each dust component

    If wavelenghth is given as a list, a list of output fluxes will be given, corresponding to the calculated flux at each wavelength.

    Temperature and mass can be set to be lists , corresponding to multiple dust components. For n components, both lists must be of length n.

    Optionally, a different dust mass absorption coefficient can be used for each component; this is done by giving lists of length n for kappa_0 and lambda_0.

    Optionally, a different dust emissivity slope can be used for each component; this is done by giving a list of length n for beta.
    """

    # Define function for checking if variable is a list, and (if necessary) converting to a n_target length list of identical entries
    def Numpify(var, n_target=False):
        if not hasattr(var, '__iter__'):
            if not n_target:
                var = list(var)
            else:
                var = [var]*n_target
        else:
            if len(var)==1 and n_target>1:
                var = [var[0]]*n_target
        if not isinstance(var, np.ndarray):
            var = np.array(var)
        return var

    # Record number of model components, and number of bands of interest
    n_comp = len(temp)
    n_band = len(wavelength)

    # As needed, convert variables to arrays
    wavelength = Numpify(wavelength)
    temp = Numpify(temp)
    mass = Numpify(mass, n_target=n_comp)
    kappa_0 = Numpify(kappa_0, n_target=n_comp)
    lambda_0 = Numpify(lambda_0, n_target=n_comp)
    beta = Numpify(beta, n_target=n_comp)

    # Check that variables are the same length, when they need to be
    if np.std([len(temp), len(mass), len(beta), len(kappa_0), len(lambda_0)]) != 0:
        Exception('Number of dust components needs to be identical for temp/mass/beta/kappa_0/lambda_0 variables')

    """ NB: Arrays have dimensons of n_comp rows by n_bands columns """

    # Convert wavelengths to frequencies (for bands of interest, and for kappa_0 reference wavelengths)
    nu = np.divide(c, wavelength)
    nu_0 = np.divide(c, lambda_0)

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
    flux = np.zeros(n_band)
    for m in range(n_comp):
        flux += 1E26 * kappa_nu[m,:] * dist_metres**-2.0 * mass_kilograms[m] * B_planck[m,:]
    #flux = ( 1E26 * kappa_nu * dist_metres**-2.0 * mass_kilograms * B_planck )

    return flux



flux([250E-6,350E-6,500E-6], [15.0,25.0], [1E8,1E5], 25E6, kappa_0=[0.051,0.077], lambda_0=[500E-6,850E-6], beta=[1.5,2.0])