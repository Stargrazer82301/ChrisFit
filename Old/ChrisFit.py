# Import smorgasbord
from __future__ import division
from __future__ import print_function
import sys
import os
if sys.version_info[0] >= 3:
    from builtins import str
    from builtins import range
    from past.utils import old_div
current_module = sys.modules[__name__]
sys.path.append(str(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pdb
import numpy as np
import scipy
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
import lmfit
import ChrisFuncs
plt.ioff()

# Fiddle with fonts
matplotlib.rc('mathtext', fontset='stix')#sans

# Define constants
c = 3E8
h = 6.64E-34
k = 1.38E-23





# Function to perform a optionally-colour-corrected, optionally-bootstrapped, optionally-plotted, one- or two-component modified blackbody fit to a set of fluxes
# Input: Name of source, array of wavelengths (m), array of fluxes (Jy), array of uncertainties (Jy), list of camera used at each band
    # number of greybody components to fit, distance (pc), array of booleans stating whether points are upper limits, beta value if fixed or 'free' if free, kappa_0 (m^2 kg^-1), lambda_0 (m), initial guess for dust mass (Msol) (else boolean default),
    # redshift, boolen for whether to colour-correct (if so, colour-correction datfiles required), boolen for whether to output plot (if so, plots written to subdir called 'Output'),
    # boolean for whether to bootstrap for errors, boolean for verbosity, string of name of scipy minimisation algorithm to use, string of output directory or boolean of default, value for percentile uncertainties or boolean of default
# Output: List of [chi-squared, list of [cold dust temp, cold dust mass, warm dust temp, warm dust mass, total dust mass, beta], list of [cold dust temp err, cold dust mass err, warm dust temp err, warm dust mass err, total dust mass err, beta err],
    # corrected fluxes, list of residuals, list of plot [fig, ax], list of [cold dust temp median, cold dust mass median, warm dust temp median, warm dust mass median, total dust mass median, beta median],
    # list of [cold dust temp boostrapped values, cold dust mass boostrapped values, warm dust temp boostrapped values, warm dust mass boostrapped values, total dust mass boostrapped values, beta boostrapped values],
    # list of [lower confidence interval, upper confidence interval] for [cold dust temp, cold dust mass, warm dust temp, warm dust mass, total dust mass, beta]]
def ChrisFit(source_name,
             wavelengths,
             fluxes,
             errors,
             instruments,
             components,
             distance,
             limits = [False],
             beta = 2.0,
             kappa_0 = 0.051,
             lambda_0 = 500E-6,
             guess_mass = False,
             redshift = 0.0,
             col_corr = True,
             min_temp = 5.0,
             plotting = True,
             plot_pdf = True,
             bootstrapping = False,
             verbose = True,
             algorithm = 'leastsq',
             output_dir = False,
             percentile = False):



    # Announce the name of the source being processed
    if verbose==True:
        print(' ')
        print('Fitting source: '+str(source_name))

    # Set boolean depending upon number of components in fit
    components = int(components)
    if components==1:
        warm_boolean = False
    elif components==2:
        warm_boolean = True

    # Deal with free or fixed beta
    if beta=='free':
        beta_boolean = True
        beta = 2.0
    else:
        beta_boolean = False
        beta = float(beta)

    # Ensure input is in numpy arrays where necessary
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)
    errors = np.array(errors)

    # Use provided guess mass, or crudely estimate sensible initial guess for dust mass
    if guess_mass!=False:
        M_c_guess = float(guess_mass)
    else:
        M_c_guess = 5E-9 * distance**2.0

    # Package parameters for initial fit
    params = lmfit.Parameters()
    params.add('beta', value=beta, vary=beta_boolean)
    params.add('D', value=distance, vary=False)
    params.add('lambda_0', value=lambda_0, vary=False)
    params.add('kappa_0', value=kappa_0, vary=False)
    params.add('components', value=components, vary=False, min=1, max=2)
    if warm_boolean==False:
        params.add('T_c', value=20.0, vary=True, min=10.0, max=200.0)
        params.add('T_w', value=0, vary=False)
        params.add('M_c', value=M_c_guess, vary=True, min=np.divide(M_c_guess,1E6))
        params.add('M_w', value=0, vary=False)
    elif warm_boolean==True:
        params.add('T_c', value=20.0, vary=True, min=min_temp, max=200.0)
        params.add('T_offset', value=30.0, vary=True, min=0.0, max=50.0)
        params.add('T_w', expr='T_c + T_offset')
        params.add('M_c', value=M_c_guess, vary=True, min=np.divide(M_c_guess,1E4))
        params.add('M_ratio', value=1E-2, vary=True, min=1E-6, max=1E4)
        params.add('M_w', expr='M_c * M_ratio')

    # Perform initial, using LMfit
    if verbose==True:
        print('Performing initial fit...')
    if algorithm=='leastsq':
        result = lmfit.minimize(ChrisFit_2GB_LMfit, params, args=(wavelengths, fluxes, errors, limits), method=algorithm, maxfev=1000000, xtol=1E-14, ftol=1E-14)
    else:
        result = lmfit.minimize(ChrisFit_2GB_LMfit, params, args=(wavelengths, fluxes, errors, limits), method=algorithm)



    # If required, use initial fit to perform colour corrections, and then re-fit to corrected fluxes
    if col_corr==False:
        fluxes_corr = np.copy(fluxes)
    if col_corr==True:
        if verbose==True:
            print('Performing colour-corrected fit...')

        # Loop over each wavelength, and use initial fit to colour-correct fluxes
        fluxes_corr = np.empty([wavelengths.shape[0]])
        for w in range(0, wavelengths.shape[0]):
            corr_output = ChrisFit_ColourCorrection(wavelengths[w], instruments[w], result.params['T_w'].value, result.params['T_c'].value, result.params['M_w'].value, result.params['M_c'].value, beta=result.params['beta'].value)
            fluxes_corr[w] = np.divide(fluxes[w], corr_output[0])
            #print 'Band: '+str(1E6*wavelengths[w])+'um;   Correction: '+str(100.0*(1.0-(1.0/corr_output[0])))[:6]+'%'

        # Perform colour-corrected fit, using LMfit
        if algorithm=='leastsq':
            result = lmfit.minimize(ChrisFit_2GB_LMfit, result.params, args=(wavelengths, fluxes_corr, errors, limits), method=algorithm, maxfev=1000000, xtol=1E-14, ftol=1E-14)
        else:
            result = lmfit.minimize(ChrisFit_2GB_LMfit, result.params, args=(wavelengths, fluxes_corr, errors, limits), method=algorithm)



    # Extract best-fit values, and make sure that warm and cold components are ordered correctly
    beta = result.params['beta'].value
    T_order, M_both = np.array([result.params['T_w'].value, result.params['T_c'].value]), np.array([result.params['M_w'].value, result.params['M_c'].value])
    if components==1:
        T_w = np.min(T_order)
        T_c = np.max(T_order)
        M_w = 0.0
        M_c = M_both[ np.where( T_order==T_c ) ][0]
        M_d = M_c
    elif components==2:
        T_w = np.max(T_order)
        T_c = np.min(T_order)
        M_w = M_both[ np.where( T_order==T_w ) ][0]
        M_c = M_both[ np.where( T_order==T_c ) ][0]
        M_d = M_w + M_c
    if verbose==True:
        print(' ')
        print('Best-fit cold dust temp of: '+str(T_c)[0:5]+' K')
        print('Best-fit cold dust mass of: '+str(np.log10(M_c))[0:5]+' log10 Msol')
        if components==2:
            print('Best-fit warm dust temp of: '+str(T_w)[0:5]+' K')
            print('Best-fit warm dust mass of: '+str(np.log10(M_w))[0:5]+' log10 Msol')
        if beta=='free':
            print('Best-fit beta of: '+str(beta)[0:4])

    # Calculate chi-squared of fit
    fit = ChrisFit_2GB_Flux(wavelengths, T_w, T_c, M_w, M_c, distance, kappa_0=kappa_0, lambda_0=lambda_0, beta=beta)
    chi_squared = ( np.divide((fluxes_corr-fit)**2.0, errors**2.0) )
    if (True in limits)==True:
        chi_squared[ np.where( (np.array(limits)==True) & (fit-fluxes<0) ) ] = 0.0

    # Calculate residuals
    residuals = np.zeros([wavelengths.shape[0]])
    for w in range(0, wavelengths.shape[0]):
        residuals[w] = ( ChrisFit_2GB_Flux(wavelengths[w], T_w, T_c, M_w, M_c, distance, kappa_0=kappa_0, lambda_0=lambda_0, beta=beta) - fluxes_corr[w] )# / errors[w]



    # Commence bootstrapping, if required
    if bootstrapping!=False:
        if verbose==True:
            print(' ')
            print('Bootstrapping fit...')
        if str(bootstrapping)=='True':
            bs_iter = 1000
            bootstrapping = True
        else:
            bs_iter = int(bootstrapping)
            bootstrapping = True

        # Generate peturbation values
        bs_peturbs = np.zeros([fluxes_corr.shape[0], bs_iter])
        for w in range(0, fluxes_corr.shape[0]):
            bs_peturbs[w,:] = np.array(np.random.normal(loc=0.0, scale=errors[w], size=bs_iter))

        # Start bootstrap iterations
        bs_T_w_array, bs_T_c_array, bs_M_w_array, bs_M_c_array, bs_beta_array = np.zeros([bs_iter]), np.zeros([bs_iter]), np.zeros([bs_iter]), np.zeros([bs_iter]), np.zeros([bs_iter])
        for b in range(0, bs_iter):
            if np.mod(b, 100)==0:
                if verbose==True:
                    print('Bootstrap iterations: '+str(b)+' - '+str(b+100))

            # Peturb corrected fluxes within errors
            bs_fluxes = np.copy(fluxes_corr)
            for w in range(0, fluxes_corr.shape[0]):
                bs_fluxes[w] += bs_peturbs[w,b]

            # Repackage variables for bootstrap
            bs_params = lmfit.Parameters()
            bs_params.add('beta', value=beta, vary=beta_boolean)
            bs_params.add('D', value=distance, vary=False)
            bs_params.add('lambda_0', value=lambda_0, vary=False)
            bs_params.add('kappa_0', value=kappa_0, vary=False)
            bs_params.add('components', value=components, vary=False, min=1, max=2)
            if warm_boolean==False:
                bs_params.add('T_c', value=20.0, vary=True, min=10.0, max=200.0)
                bs_params.add('T_w', value=0, vary=False)
                bs_params.add('M_c', value=M_c_guess, vary=True, min=np.divide(M_c_guess,1E6))
                bs_params.add('M_w', value=0, vary=False)
            elif warm_boolean==True:
                bs_params.add('T_c', value=20.0, vary=True, min=min_temp, max=200.0)
                bs_params.add('T_offset', value=30.0, vary=True, min=10.0, max=50.0)
                bs_params.add('T_w', expr='T_c + T_offset')
                bs_params.add('M_c', value=M_c_guess, vary=True, min=np.divide(M_c_guess,1E6))
                bs_params.add('M_ratio', value=1E-2, vary=True, min=1E-6, max=1.0)
                bs_params.add('M_w', expr='M_c * M_ratio')

            # Perform bootstrap fit
            if algorithm=='leastsq':
                bs_result = lmfit.minimize(ChrisFit_2GB_LMfit, bs_params, args=(wavelengths, bs_fluxes, errors, limits), method=algorithm, maxfev=1000, xtol=2E-9, ftol=2E-9)
            else:
                bs_result = lmfit.minimize(ChrisFit_2GB_LMfit, bs_params, args=(wavelengths, bs_fluxes, errors, limits), method=algorithm)

            # Retrieve output values, and ensure they are in correct order
            bs_T_order, bs_M_order = np.array([bs_result.params['T_w'].value, bs_result.params['T_c'].value]), np.array([bs_result.params['M_w'].value, bs_result.params['M_c'].value])
            if components==1:
                bs_T_w_array[b] = np.min(bs_T_order)
                bs_T_c_array[b] = np.max(bs_T_order)
                bs_M_w_array[b] = 0.0
                bs_M_c_array[b] = bs_M_order[ np.where( bs_T_order==bs_T_c_array[b] ) ][0]
            elif components==2:
                bs_T_w_array[b] = np.max(bs_T_order)
                bs_T_c_array[b] = np.min(bs_T_order)
                bs_M_w_array[b] = bs_M_order[ np.where( bs_T_order==bs_T_w_array[b] ) ][0]
                bs_M_c_array[b] = bs_M_order[ np.where( bs_T_order==bs_T_c_array[b] ) ][0]
            bs_beta_array[b] = bs_result.params['beta'].value

        # Sigma-clip temperature and beta output
        bs_T_w_clip = ChrisFuncs.SigmaClip(bs_T_w_array, median=True)
        bs_T_w_sigma, bs_T_w_mu = bs_T_w_clip[0], bs_T_w_clip[1]
        bs_T_c_clip = ChrisFuncs.SigmaClip(bs_T_c_array, median=True)
        bs_T_c_sigma, bs_T_c_mu = bs_T_c_clip[0], bs_T_c_clip[1]
        bs_beta_clip = ChrisFuncs.SigmaClip(bs_beta_array, median=True)
        bs_beta_sigma, bs_beta_mu = bs_beta_clip[0], bs_beta_clip[1]

        # Find sigma-clip bootstrapped dust masses
        bs_M_w_sigma = ChrisFuncs.SigmaClip(np.log10(bs_M_w_array), median=True)[0]
        bs_M_c_sigma = ChrisFuncs.SigmaClip(np.log10(bs_M_c_array), median=True)[0]
        bs_M_w_mu = 10.0**ChrisFuncs.SigmaClip(np.log10(bs_M_w_array), median=True)[1]
        bs_M_c_mu = 10.0**ChrisFuncs.SigmaClip(np.log10(bs_M_c_array), median=True)[1]
        bs_M_d_mu = 10.0**ChrisFuncs.SigmaClip(np.log10(bs_M_w_array+bs_M_c_array), median=True)[1]
        if components==1:
            bs_M_d_sigma = bs_M_c_sigma
        elif components==2:
            bs_M_d_sigma = ChrisFuncs.SigmaClip(np.log10(bs_M_w_array+bs_M_c_array), median=True)[0]

        # Calculate uncertainties relative to best-fit values
        bs_T_w_sigma_up, bs_T_w_sigma_down = np.abs( T_w - ( bs_T_w_mu + bs_T_w_sigma ) ), np.abs( T_w - ( bs_T_w_mu - bs_T_w_sigma ) )
        bs_T_c_sigma_up,bs_T_c_sigma_down  = np.abs( T_c - ( bs_T_c_mu + bs_T_c_sigma ) ), np.abs( T_c - ( bs_T_c_mu - bs_T_c_sigma ) )
        bs_M_w_sigma_up, bs_M_w_sigma_down = np.abs( M_w - ( bs_M_w_mu + bs_M_w_sigma ) ), np.abs( M_w - ( bs_M_w_mu - bs_M_w_sigma ) )
        bs_M_c_sigma_up, bs_M_c_sigma_down = np.abs( M_c - ( bs_M_c_mu + bs_M_c_sigma ) ), np.abs( M_c - ( bs_M_c_mu - bs_M_c_sigma ) )
        bs_M_d_sigma_up, bs_M_d_sigma_down = np.abs( M_d - ( bs_M_d_mu + bs_M_d_sigma ) ), np.abs( M_d - ( bs_M_d_mu - bs_M_d_sigma ) )
        bs_beta_sigma_up, bs_beta_sigma_down = np.abs( beta - ( bs_beta_mu + bs_beta_sigma ) ), np.abs( beta - ( bs_beta_mu - bs_beta_sigma ) )

        # Translate mass uncertainties into log space
        bs_M_w_sigma_log = ChrisFuncs.SigmaClip(np.log10(bs_M_w_array), median=True)[0]
        bs_M_c_sigma_log = ChrisFuncs.SigmaClip(np.log10(bs_M_c_array), median=True)[0]
        if components==1:
            bs_M_d_sigma_log = bs_M_c_sigma_log
        elif components==2:
            bs_M_d_sigma_log = ChrisFuncs.SigmaClip(np.log10(bs_M_w_array+bs_M_c_array), median=True)[0]

        # Calculate uncertainties as a percentile, if requested
        if percentile>0:
            bs_T_w_sigma = ( np.sort( np.abs( ChrisFuncs.Nanless(bs_T_w_array) - T_w ) ) )[ int( (np.divide(float(percentile),100.0)) * ChrisFuncs.Nanless(bs_T_w_array).shape[0] ) ]
            bs_T_c_sigma = ( np.sort( np.abs( ChrisFuncs.Nanless(bs_T_c_array) - T_c ) ) )[ int( (np.divide(float(percentile),100.0)) * ChrisFuncs.Nanless(bs_T_c_array).shape[0] ) ]
            bs_M_w_sigma = ( np.sort( np.abs( ChrisFuncs.Nanless(bs_M_w_array) - M_w ) ) )[ int( (np.divide(float(percentile),100.0)) * ChrisFuncs.Nanless(bs_M_w_array).shape[0] ) ]
            bs_M_c_sigma = ( np.sort( np.abs( ChrisFuncs.Nanless(bs_M_c_array) - M_c ) ) )[ int( (np.divide(float(percentile),100.0)) * ChrisFuncs.Nanless(bs_M_c_array).shape[0] ) ]
            bs_beta_sigma = ( np.sort( np.abs( ChrisFuncs.Nanless(bs_beta_array) - beta ) ) )[ int( (np.divide(float(percentile),100.0)) * ChrisFuncs.Nanless(bs_beta_array).shape[0] ) ]
            if components==1:
                bs_M_d_sigma = bs_M_c_sigma
            elif components==2:
                bs_M_d_array = bs_M_w_array + bs_M_c_array
                bs_M_d_sigma = ( np.sort( np.abs( ChrisFuncs.Nanless(bs_M_d_array) - M_d ) ) )[ int( (np.divide(float(percentile),100.0)) * ChrisFuncs.Nanless(bs_M_d_array).shape[0] ) ]
            bs_M_w_sigma_log = np.log10( np.divide(( bs_M_w_sigma + M_w ), M_w) )
            bs_M_c_sigma_log = np.log10( np.divide(( bs_M_c_sigma + M_c ), M_c) )
            bs_M_d_sigma_log = np.log10( np.divide(( bs_M_d_sigma + M_d ), M_d) )

        # Reporty uncertainties
        if verbose==True:
            print(' ')
            print('Cold dust temp uncertainty of: '+str(bs_T_c_sigma)[0:5]+' K')
            print('Cold dust mass uncertainty of: '+str(bs_M_c_sigma_log)[0:5]+' log10 Msol')
            if components==2:
                print('Warm dust temp uncertainty of: '+str(bs_T_w_sigma)[0:5]+' K')
                print('Warm dust mass uncertainty of: '+str(bs_M_w_sigma_log)[0:5]+' log10 Msol')
            if beta=='free':
                print('Beta uncertainty of: '+str(bs_beta_sigma)[0:4])

    # Return NaN values if bootstrapping not requested
    elif bootstrapping==False:
        bs_T_w_sigma, bs_T_c_sigma, bs_M_w_sigma, bs_M_c_sigma, bs_M_d_sigma, bs_beta_sigma = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        bs_T_w_mu, bs_T_c_mu, bs_M_w_mu, bs_M_c_mu, bs_M_d_mu, bs_beta_mu = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        bs_T_w_array, bs_T_c_array, bs_M_w_array, bs_M_c_array, bs_M_d_array, bs_beta_array = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        bs_T_w_sigma_down, bs_T_w_sigma_up, bs_T_c_sigma_down, bs_T_c_sigma_up, bs_M_w_sigma_down, bs_M_w_sigma_up, bs_M_c_sigma_down, bs_M_c_sigma_up, bs_M_d_sigma_down, bs_M_d_sigma_up, bs_beta_sigma_down, bs_beta_sigma_up = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        bs_M_w_sigma_log, bs_M_c_sigma_log, bs_M_d_sigma_log = np.NaN, np.NaN, np.NaN



    # Carry out plotting, if required
    if plotting==False:
        fig, ax = [], []
    if plotting==True:
        plt.close('all')
        font_family = 'serif'
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([0.125, 0.125, 0.825, 0.825])

        # Generate fit components
        fit_wavelengths = np.linspace(10E-6, 10000E-6, num=10000)
        fit_fluxes_w = ChrisFit_2GB_Flux(fit_wavelengths, T_w, 0.0, M_w, 0.0, distance, kappa_0=kappa_0, lambda_0=lambda_0, beta=beta)
        fit_fluxes_c = ChrisFit_2GB_Flux(fit_wavelengths, 0.0, T_c, 0.0, M_c, distance, kappa_0=kappa_0, lambda_0=lambda_0, beta=beta)
        fit_fluxes_tot = ChrisFit_2GB_Flux(fit_wavelengths, T_w, T_c, M_w, M_c, distance, kappa_0=kappa_0, lambda_0=lambda_0, beta=beta)

        # Plot fits
        ax.plot(fit_wavelengths*1E6, fit_fluxes_w, ls='--', lw=1.0, c='black')
        ax.plot(fit_wavelengths*1E6, fit_fluxes_c, ls='--', lw=1.0, c='black')
        ax.plot(fit_wavelengths*1E6, fit_fluxes_tot, ls='-', lw=1.5, c='red')

        # Assemble strings for plot text in various circumstances
        chi_squared_string = '$\chi^{2}$ = '+str(np.around(np.sum(chi_squared), decimals=3))[0:5]
        if bootstrapping==False:
            T_c_string = 'T$_{c}$ = '+str(np.around(T_c, decimals=3))[0:5]+' K'
            M_c_string = ',   M$_{c}$ = '+str(np.around(np.log10(M_c), decimals=3))[0:5]+' log$_{10}$M$_{\odot}$'
            T_w_string = ''
            M_w_string = ''
            M_d_string = ''
            if components==2:
                T_w_string = 'T$_{w}$ = '+str(np.around(T_w, decimals=3))[0:5]+' K'
                M_w_string = ',   M$_{w}$ = '+str(np.around(np.log10(M_w), decimals=3))[0:5]+' log$_{10}$M$_{\odot}$'
                M_d_string = ',   M$_{d}$ = '+str(np.around(np.log10(M_d), decimals=3))[0:5]+' log$_{10}$M$_{\odot}$'
            if beta_boolean==True:
                beta_string = ',   $\\beta$ = '+str(np.around(beta, decimals=2))[0:4]
        elif bootstrapping==True:
            T_c_string = 'T$_{c}$ = ('+str(np.around(T_c, decimals=3))[0:5]+' $\pm$ '+str(np.around(bs_T_c_sigma, decimals=3))[0:5]+') K'
            M_c_string = ',   M$_{c}$ = ('+str(np.around(np.log10(M_c), decimals=3))[0:5]+' $\pm$ '+str(np.around(bs_M_c_sigma_log, decimals=3))[0:5]+') log$_{10}$M$_{\odot}$'
            T_w_string = ''
            M_w_string = ''
            M_d_string = ''
            if components==2:
                T_w_string = 'T$_{w}$ = ('+str(np.around(T_w, decimals=3))[0:5]+' $\pm$ '+str(np.around(bs_T_w_sigma, decimals=3))[0:5]+') K'
                M_w_string = ',   M$_{w}$ = ('+str(np.around(np.log10(M_w), decimals=3))[0:5]+' $\pm$ '+str(np.around(bs_M_w_sigma_log, decimals=3))[0:5]+') log$_{10}$M$_{\odot}$'
                M_d_string = ',   M$_{d}$ = ('+str(np.around(np.log10(M_d), decimals=3))[0:5]+' $\pm$ '+str(np.around(bs_M_d_sigma_log, decimals=3))[0:5]+') log$_{10}$M$_{\odot}$'
            if beta_boolean==True:
                beta_string = ',   $\\beta$ = '+str(np.around(beta, decimals=2))[0:4]+' $\pm$ '+str(np.around( bs_beta_sigma, decimals=3))[0:4]
        if beta_boolean==False:
                beta_string = ''

        # Place text on figure
        ax.text(0.035, 0.925, source_name, fontsize=15, fontweight='bold', transform=ax.transAxes, family=font_family)
        if components==1:
            ax.text(0.035, 0.865, T_c_string+M_c_string, fontsize=14, transform=ax.transAxes, family=font_family)
            ax.text(0.035, 0.805, chi_squared_string+beta_string+M_d_string, fontsize=14, transform=ax.transAxes, family=font_family)
        if components==2:
            ax.text(0.035, 0.865, T_c_string+M_c_string, fontsize=14, transform=ax.transAxes, family=font_family)
            ax.text(0.035, 0.805, T_w_string+M_w_string, fontsize=14, transform=ax.transAxes, family=font_family)
            ax.text(0.035, 0.745, chi_squared_string+beta_string+M_d_string, fontsize=14, transform=ax.transAxes, family=font_family)

        # Set up figure axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=17.5, fontname=font_family)
        ax.set_ylabel('Flux Density (Jy)', fontsize=17.5, fontname=font_family)

        # Format font of tick labels
        for xlabel in ax.get_xticklabels():
            xlabel.set_fontproperties(matplotlib.font_manager.FontProperties(family=font_family, size=15))
        for ylabel in ax.get_yticklabels():
            ylabel.set_fontproperties(matplotlib.font_manager.FontProperties(family=font_family, size=15))

        # Create seperature flux and error arrays for plot
        fluxes_plot, errors_plot = np.copy(fluxes_corr), np.copy(errors)
        errors_up, errors_down = np.copy(errors), np.copy(errors)

        # Format errorbars deal with negative fluxes
        errors_plot[ np.where( fluxes_plot<=0 ) ] -= fluxes_plot[ np.where( fluxes_plot<=0 ) ]
        fluxes_plot[ np.where( fluxes_plot<=0 ) ] = 1E-50

        # Format errobars to account for non-detections
        det = np.where(fluxes_plot>errors_plot)
        errors_down[ np.where( errors_down > fluxes_plot ) ] = 0.999 * fluxes_plot[ np.where( errors_down > fluxes_plot ) ]

        # Plot datapoints
        if (True in limits)==False:
            ax.errorbar(wavelengths*1E6, fluxes_plot, yerr=[errors_down, errors_up], ecolor='black', elinewidth=1.15, capthick=1.15, marker='x', color='black', markersize=5.0, markeredgewidth=1.15, linewidth=0)
        else:
            lim_true, lim_false = np.where( np.array(limits)==True ), np.where( np.array(limits)==False )
            ax.errorbar(wavelengths[lim_false]*1E6, fluxes_plot[lim_false], yerr=[errors_down[lim_false], errors_up[lim_false]], ecolor='black', elinewidth=1.15, capthick=1.15, marker='x', color='black', markersize=5.0, markeredgewidth=1.15, linewidth=0)
            ax.errorbar(wavelengths[lim_true]*1E6, fluxes_plot[lim_true], yerr=[errors_down[lim_true], errors_up[lim_true]], ecolor='gray', elinewidth=1.15, capthick=1.15, marker='x', color='gray', markersize=5.0, markeredgewidth=1.15, linewidth=0)

        # Scale x-axes to account for wavelengths provided
        xlim_min = 1E6 * 10.0**( np.floor( np.log10( np.min( wavelengths ) ) ) )
        xlim_max = 1E6 * 10.0**( np.ceil( np.log10( np.max( wavelengths ) ) ) )
        ax.set_xlim(xlim_min,xlim_max)

        # Scale y-axes to account for range of values and non-detections
        ylim_min = 10.0**( -1.0 + np.round( np.log10( np.min( fluxes_plot[det] - errors_plot[det] ) ) ) )
        ylim_max = 10.0**( 1.0 + np.ceil( np.log10( 1.1 * np.max( fluxes_plot[det] + errors_plot[det] ) ) ) )
        ax.set_ylim(ylim_min,ylim_max)

        # Save figures to designated'Output' folder
        comp_strings = ['Eh', 'One', 'Two']
        if output_dir==False:
            if not os.path.exists('Output'):
                os.mkdir('Output')
            fig.savefig( os.path.join('Output',source_name+' '+comp_strings[components]+' Component.png'), dpi=175.0 )
            if plot_pdf==True:
                fig.savefig( os.path.join('Output',source_name+' '+comp_strings[components]+' Component.pdf') )
        if output_dir!=False:
            fig.savefig( os.path.join(output_dir,source_name+' '+comp_strings[components]+' Component.png'), dpi=175.0 )
            if plot_pdf==True:
                fig.savefig( os.path.join(output_dir,source_name+' '+comp_strings[components]+' Component.pdf') )



    # Enter placeholder value for one-component warm output, and return output
    if components==1:
        T_w = np.NaN
        M_w = np.NaN
        bs_T_w_sigma = np.NaN
        bs_M_w_sigma = np.NaN
    if verbose==True:
        print(' ')
        print(' ')
    return chi_squared,\
    [T_c, M_c, T_w, M_w, M_d, beta],\
    [bs_T_c_sigma, bs_M_c_sigma_log, bs_T_w_sigma, bs_M_w_sigma_log, bs_M_d_sigma_log, bs_beta_sigma],\
    fluxes_corr,\
    residuals,\
    [fig, ax],\
    [bs_T_c_mu, bs_M_c_mu, bs_T_w_mu, bs_M_w_mu, bs_M_d_mu, bs_beta_mu],\
    [bs_T_c_array, bs_M_c_array, bs_T_w_array, bs_M_w_array, bs_M_w_array+bs_M_c_array, bs_beta_array],\
    [[bs_T_c_sigma_down, bs_T_c_sigma_up],[bs_M_c_sigma_down, bs_M_c_sigma_up], [bs_T_w_sigma_down, bs_T_w_sigma_up], [bs_M_w_sigma_down, bs_M_w_sigma_up], [bs_M_d_sigma_down, bs_M_d_sigma_up], [bs_beta_sigma_down, bs_beta_sigma_up]]





# Function to calculate flux at a given wavelength of a two-component modified blackbody
# Input: Wavelength (m), warm dust temperature (K), cold dust temperature (K), warm dust mass (Msol), cold dust mass (Msol), distance (pc), kappa_0, lambda_0, beta
# Returns: Flux (Jy) at wavelength in question
def ChrisFit_2GB_Flux(wavelength, T_w, T_c, M_w, M_c, D, kappa_0=0.051, lambda_0=500E-6, beta=2.0):

    # Convert wavelength to frequency, and find kappa
    nu = np.divide(c, wavelength)
    nu_0 = np.divide(c, lambda_0)
    kappa_nu = kappa_0 * ( np.divide(nu, nu_0) )**beta

    # Evaluate hot dust Planck function
    if T_w>0:
        B_w_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
        B_w_e = np.divide((h * nu), (k * T_w))
        B_w = B_w_prefactor * (np.e**B_w_e - 1)**-1.0
    else:
        B_w = 0.0

    # Evaluate cold dust Planck function
    B_c_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_c_e = np.divide((h * nu), (k * T_c))
    B_c = B_c_prefactor * (np.e**B_c_e - 1)**-1.0

    # Calculate flux
    M_w_kilograms = M_w * 2E30
    M_c_kilograms = M_c * 2E30
    D_metres = D * 3.26 * 9.5E15
    flux = ( 1E26 * kappa_nu * D_metres**-2.0 * M_w_kilograms * B_w ) + ( 1E26 * kappa_nu * D_metres**-2.0 * M_c_kilograms * B_c )

    return flux



# Function to calculate flux at a given wavelength of a two-component modified blackbody with a general normalisation constant omega
# Input: Wavelength (m), warm dust temperature (K), cold dust temperature (K), warm omega, cold omega, beta
# Returns: Flux (Jy) at wavelength in question
def ChrisFit_2GB_Omega_Flux(wavelength, T_w, T_c, Omega_w, Omega_c, beta=2.0):

    # Convert wavelength to frequency
    nu = np.divide(c, wavelength)

    # Evaluate hot dust Planck function
    if T_w>0:
        B_w_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
        B_w_e = np.divide((h * nu), (k * T_w))
        B_w = B_w_prefactor * (np.e**B_w_e - 1)**-1.0
    else:
        B_w = 0.0

    # Evaluate cold dust Planck function
    B_c_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_c_e = np.divide((h * nu), (k * T_c))
    B_c = B_c_prefactor * (np.e**B_c_e - 1)**-1.0

    # Calculate flux
    flux = ( 1E26 * nu**beta * Omega_w * B_w ) + ( 1E26 * nu**beta * Omega_c * B_c )

    return flux



# Function to calculate the chi-sqiared between a set of fluxes and a two-component modified blackbody using the LMfit package
# Input: Parameter object containing {warm dust temperature (K), cold dust temperature (K), warm dust mass (Msol), cold dust mass (Msol), distance (pc), kappa_0, lambda_0, beta}, array of wavelengths (m), array of fluxes (Jy), array of uncertainties (Jy), array of booleans stating whether point is upper limit
# Returns: Chi-squared value of fit
def ChrisFit_2GB_LMfit(params, wavelengths, fluxes, errors, limits=[False]):

    # Extract parameters
    T_w = params['T_w'].value
    T_c = params['T_c'].value
    M_w = params['M_w'].value
    M_c = params['M_c'].value
    D = params['D'].value
    lambda_0 = params['lambda_0'].value
    kappa_0 = params['kappa_0'].value
    beta = params['beta'].value

    # Convert wavelength to frequency, and find kappa
    nu = np.divide(c, wavelengths)
    nu_0 = np.divide(c, lambda_0)
    kappa_nu = kappa_0 * ( np.divide(nu, nu_0) )**beta

    # Evaluate hot dust Planck function
    B_w_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_w_e = np.divide((h * nu), (k * T_w))
    B_w = B_w_prefactor * (np.e**B_w_e - 1)**-1.0

    # Evaluate cold dust Planck function
    B_c_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_c_e = np.divide((h * nu), (k * T_c))
    B_c = B_c_prefactor * (np.e**B_c_e - 1)**-1.0

    # Calculate fluxes of fit, and find chi_squared
    M_w_kilograms = M_w * 2E30
    M_c_kilograms = M_c * 2E30
    D_metres = D * 3.26 * 9.5E15
    fit = ( 1E26 * kappa_nu * D_metres**-2.0 * M_w_kilograms * B_w ) + ( 1E26 * kappa_nu * D_metres**-2.0 * M_c_kilograms * B_c )
    chi_squared = ( np.divide((fluxes-fit)**2.0, errors**2.0) )

    # Adjust chi-squared to account for limits
    if (True in limits)==True:
        chi_squared[ np.where( (np.array(limits)==True) & (fit-fluxes<0) ) ] = 0.0
    return chi_squared



# Function to calculate the chi-sqiared between a set of fluxes to be colour-corrected and a two-component modified blackbody using the LMfit package
# Input: Parameter object containing {warm dust temperature (K), cold dust temperature (K), warm dust mass (Msol), cold dust mass (Msol), distance (pc), kappa_0, lambda_0, beta}, array of wavelengths (m), array of fluxes (Jy), array of uncertainties (Jy), list of camera used at each band, array of booleans stating whether point is upper limit
# Returns: Chi-squared value of fit
def ChrisFit_2GB_ColCorr_LMfit(params, wavelengths, fluxes, errors, instruments, limits=[False]):

    # Perform initial fit, to produce colour-corrected fluxes
    corr_result = lmfit.minimize(ChrisFit_2GB_LMfit, params, args=(wavelengths, fluxes, errors), method='powell')
    params = corr_result.params

    # Loop over each wavelength, and use initial fit to colour-correct fluxes
    fluxes_corr = np.empty([wavelengths.shape[0]])
    for w in range(0, wavelengths.shape[0]):
        corr_output = ChrisFit_ColourCorrection(wavelengths[w], instruments[w], params['T_w'].value, params['T_c'].value, params['M_w'].value, params['M_c'].value, beta=params['beta'].value)
        fluxes_corr[w] = np.divide(fluxes[w], corr_output[0])

    # Extract parameters
    T_w = params['T_w'].value
    T_c = params['T_c'].value
    M_w = params['M_w'].value
    M_c = params['M_c'].value
    D = params['D'].value
    lambda_0 = params['lambda_0'].value
    kappa_0 = params['kappa_0'].value
    beta = params['beta'].value

    # Convert wavelength to frequency, and find kappa
    nu = np.divide(c, wavelengths)
    nu_0 = np.divide(c, lambda_0)
    kappa_nu = kappa_0 * ( np.divide(nu, nu_0) )**beta

    # Evaluate hot dust Planck function
    B_w_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_w_e = np.divide((h * nu), (k * T_w))
    B_w = B_w_prefactor * (np.e**B_w_e - 1)**-1.0

    # Evaluate cold dust Planck function
    B_c_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_c_e = np.divide((h * nu), (k * T_c))
    B_c = B_c_prefactor * (np.e**B_c_e - 1)**-1.0

    # Calculate fluxes of fit, and find chi_squared
    M_w_kilograms = M_w * 2E30
    M_c_kilograms = M_c * 2E30
    D_metres = D * 3.26 * 9.5E15
    fit = ( 1E26 * kappa_nu * D_metres**-2.0 * M_w_kilograms * B_w ) + ( 1E26 * kappa_nu * D_metres**-2.0 * M_c_kilograms * B_c )
    chi_squared = ( np.divide((fluxes_corr-fit)**2.0, errors**2.0) )

    # Adjust chi-squared to account for limits
    if (True in limits)==True:
        chi_squared[ np.where( (np.array(limits)==True) & (fit-fluxes<0) ) ] = 0.0

    return chi_squared



# Function to calculate the chi-sqiared between a set of fluxes and a two-component modified blackbody with a general normalisation constant omega using the LMfit package
# Input: Parameter object containing {warm dust temperature (K), cold dust temperature (K), warm omega, cold omega, beta}, array of wavelengths (m), array of fluxes (Jy), array of uncertainties (Jy), array of booleans stating whether point is upper limit
# Returns: Chi-squared value of fit
def ChrisFit_2GB_Omega_LMfit(params, wavelengths, fluxes, errors, limits=[False]):

    # Extract parameters
    T_w = params['T_w'].value
    T_c = params['T_c'].value
    Omega_w = params['Omega_w'].value
    Omega_c = params['Omega_c'].value
    beta = params['beta'].value

    # Convert wavelength to frequency
    nu = np.divide(c, wavelengths)

    # Evaluate hot dust Planck function
    B_w_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_w_e = np.divide((h * nu), (k * T_w))
    B_w = B_w_prefactor * (np.e**B_w_e - 1)**-1.0

    # Evaluate cold dust Planck function
    B_c_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_c_e = np.divide((h * nu), (k * T_c))
    B_c = B_c_prefactor * (np.e**B_c_e - 1)**-1.0

    # Calculate fluxes of fit, and find chi_squared
    fit = ( 1E26 * nu**beta * Omega_w * B_w ) + ( 1E26 * nu**beta * Omega_c * B_c )
    chi_squared = ( np.divide((fluxes-fit)**2.0, errors**2.0) )

    # Adjust chi-squared to account for limits
    if (True in limits)==True:
        chi_squared[ np.where( (np.array(limits)==True) & (fit-fluxes<0) ) ] = 0.0

    return chi_squared



# Function to calculate the chi-sqiared between a set of fluxes to be colour-corrected and a two-component modified blackbody with a general normalisation constant omega using the LMfit package
# Input: Parameter object containing {warm dust temperature (K), cold dust temperature (K), warm omega, cold omega, beta}, array of wavelengths (m), array of fluxes (Jy), array of uncertainties (Jy), array of booleans stating whether point is upper limit
# Returns: Chi-squared value of fit
def ChrisFit_2GB_Omega_ColCorr_LMfit(params, wavelengths, fluxes, errors, instruments, limits=False):

    # Perform initial fit, to produce colour-corrected fluxes
    corr_result = lmfit.minimize(ChrisFit_2GB_Omega_LMfit, params, args=(wavelengths, fluxes, errors), method='powell')
    params = corr_result.params

    # Loop over each wavelength, and use initial fit to colour-correct fluxes
    fluxes_corr = np.empty([wavelengths.shape[0]])
    for w in range(0, wavelengths.shape[0]):
        corr_output = ChrisFit_ColourCorrection(wavelengths[w], instruments[w], params['T_w'].value, params['T_c'].value, params['M_w'].value, params['M_c'].value, beta=params['beta'].value)
        fluxes_corr[w] = np.divide(fluxes[w], corr_output[0])

    # Extract parameters
    T_w = params['T_w'].value
    T_c = params['T_c'].value
    Omega_w = params['Omega_w'].value
    Omega_c = params['Omega_c'].value
    beta = params['beta'].value

    # Convert wavelength to frequency
    nu = np.divide(c, wavelengths)

    # Evaluate hot dust Planck function
    B_w_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_w_e = np.divide((h * nu), (k * T_w))
    B_w = B_w_prefactor * (np.e**B_w_e - 1)**-1.0

    # Evaluate cold dust Planck function
    B_c_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
    B_c_e = np.divide((h * nu), (k * T_c))
    B_c = B_c_prefactor * (np.e**B_c_e - 1)**-1.0

    # Calculate fluxes of fit, and find chi_squared
    fit = ( 1E26 * nu**beta * Omega_w * B_w ) + ( 1E26 * nu**beta * Omega_c * B_c )
    chi_squared = ( np.divide((fluxes-fit)**2.0, errors**2.0) )

    # Adjust chi-squared to account for limits
    if (True in limits)==True:
        chi_squared[ np.where( (np.array(limits)==True) & (fit-fluxes<0) ) ] = 0.0

    return chi_squared





# Function to calculate colour-correction divisor appropriate to a given spectral index for IRAS, PACS, and SPIRE
# Input: Wavelength (m), instrument, warm dust temperature (K), cold dust temperature (K), warm dust mass (Msol), cold dust mass (Msol)
# Output: Colour-correction divisor
def ChrisFit_ColourCorrection(wavelength, instrument, T_w, T_c, M_w, M_c, beta=2.0, verbose=True):

    # Set location of ChrisFuncs.py to be current working directory
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
        data_column = 'K'+str(int((wavelength*1E6)))
        data_factor = data_table[data_column]
    except:
        unknown = True
        if verbose==False:
            print(' ')
            print('Instrument \''+instrument+'\' not recognised, no colour correction applied.')

    # If instrument successfuly identified, perform colour correction; otherwise, cease
    if unknown==True:
        divisor = 1.0
        index = np.NaN
    elif unknown==False:

        # Calculate relative flux at wavelengths at points 1 um to either side of target wavelength (no need for distance or kappa, as absolute value is irrelevant)
        lambda_plus = wavelength+1E-6
        lambda_minus = wavelength-1E-6
        flux_plus = ChrisFit_2GB_Flux(lambda_plus, T_w, T_c, M_w, M_c, 1, beta=beta)
        flux_minus = ChrisFit_2GB_Flux(lambda_minus, T_w, T_c, M_w, M_c, 1, beta=beta)

        # Determine spectral index
        index = -1.0 * ( np.log10(flux_plus) - np.log10(flux_minus) ) / ( np.log10(lambda_plus) - np.log10(lambda_minus) )

        # Use cubic spline interpolation to estimate colour-correction divisor at calculated spectral index
        interp = scipy.interpolate.interp1d(data_index, data_factor, kind='linear')
        if index>np.max(data_index):
            extrap = ChrisFuncs.Extrap1D(interp)
            divisor = extrap([index])[0]
        elif index<np.min(data_index):
            extrap = ChrisFuncs.Extrap1D(interp)
            pdb.set_trace()
            divisor = extrap([index])[0]
        else:
            divisor = interp.__call__(index)

    # Restore old cwd, and return results
    os.chdir(old_cwd)
    return divisor, index


#data = np.array([2.795583251953125000e+03, 6.821301269531250000e+03, 2.734590332031250000e+03, 7.011021118164062500e+02, 1.355904998779296875e+02])
#data = data/1000.0
#ChrisFit('Test Pixel', [70E-6,160E-6,250E-6,350E-6,500E-6], data, data*0.12, ['PACS','PACS','SPIRE','SPIRE','SPIRE'], 2, 1.0, limits=[False], beta='free', kappa_0=0.077, lambda_0=850E-6, guess_mass=10.0**-8, redshift=0.0, col_corr=True, plotting=True, bootstrapping=False, verbose=True, algorithm='leastsq', output_dir=False, percentile=False)