# Use calibration and observational uncertainties to generate artificial errors
calib_unc = np.array([0.05, 0.05, 0.2, 0.07, 0.07, 0.07, 0.055, 0.055, 0.055])
obs_err = (1 + np.abs(np.random.normal(scale=2, size=calib_unc.shape))) * np.array([0.07, 0.15, 0.05, 0.05, 0.07, 0.05, 0.03, 0.03, 0.06])
inject_err = np.sqrt(obs_err**2.0 + calib_unc**2.0)
 
# Add empty columns to galaxy dictionary bands dataframe, to hold fluxes and uncertainties
bands_frame['flux'] = pd.Series(np.array([len(bands_frame)*np.NaN]), index=bands_frame.index)
bands_frame['error'] = pd.Series(np.array([len(bands_frame)*np.NaN]), index=bands_frame.index)
 
# Decide underlying properties of source
inject_temp = [18.5, 35.5]
inject_mass = [10.0**8.5, 10.0**5.5]
inject_beta = [1.9]
inject_params = inject_temp + inject_mass + inject_beta
 
# Generate artificial fluxes
inject_flux = ChrisFit.ModelFlux(bands_frame['wavelength'], inject_temp, inject_mass, gal_dict['distance'], kappa_0=0.051, kappa_0_lambda=500E-6, beta=inject_beta)
 
# Use uncertainties to produce noisy fluxes with errors (with  extra emission added to bands that are just upper limits)
bands_frame['flux'] = inject_flux + (np.random.normal(loc=0.0, scale=inject_err) * inject_flux)
bands_frame['flux'] *= 10.0**(bands_frame['limit'].values.astype(float) * np.abs(np.random.normal(loc=1.0, scale=0.3)))
bands_frame['error'] = bands_frame['flux'] * inject_err
 
# Limit bands frame to the specific bands that are actually wanted for this run
bands_use = ['WISE_22','Spitzer_24','PACS_70','PACS_160','SPIRE_250','SPIRE_350','SPIRE_500']
bands_frame = bands_frame.loc[np.where(np.in1d(bands_frame['band'],bands_use))]
 
# Call ChrisFit
output = ChrisFit.Fit(gal_dict,
                      bands_frame,
                      correl_unc = correl_unc,
                      beta_vary = True,
                      beta = 2.0,
                      components = 1,
                      kappa_0 = 0.051,
                      kappa_0_lambda = 500E-6,
                      mcmc_n_walkers = 500,
                      mcmc_n_steps = 500,
                      #mcmc_n_threads = 1,
                      simple_clean = 0.5,
                      plot = 'Output/',
                      test = False,
                      priors = None)
 
# Jubilate
print('All done!')
