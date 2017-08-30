# Identify location
import socket
location = socket.gethostname()
if location == 'Monolith':
    dropbox = 'E:\\Users\\Chris\\Dropbox\\'
if location == 'grima':
    dropbox = '/home/chris/Dropbox/'
if location == 'saruman':
    dropbox = '/home/herdata/spx7cjc/Dropbox/'

# Import smorgasbord
import os
import pdb
import sys
import gc
sys.path.append( os.path.join(dropbox,'Work','Scripts') )
sys.path.append( os.path.join(dropbox,'Work','Scripts','ChrisFit'))
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
#import pygame.mixer
import shutil
import ChrisFuncs
import ChrisFit



# Prepare output file
time_stamp = time.time()
output_header = '# ID T_COLD T_COLD_ERR M_COLD M_COLD_ERR T_WARM T_WARM_ERR M_WARM M_WARM_ERR M_DUST M_DUST_ERR BETA BETA_ERR CHISQ_DUST \n'
filepath = os.path.join(dropbox,'Work','Scripts','ChrisFit','Wrappers','Output','HAPLESS_Pieter_Greybody_Output_'+str(time_stamp).replace(',','-')+'.dat')
datfile = open(filepath, 'a')
datfile.write(output_header)
datfile.close()

# Read input catalogue
data = np.genfromtxt( os.path.join(dropbox+'Work','Tables','H-ATLAS','HAPLESS Ancillary','HAPLESS_Pieter_DR1_Old-Beam.csv'), delimiter=',', names=True)
#data = np.genfromtxt( os.path.join(dropbox,'Work','Tables','HAPLESS.csv'), delimiter=',', names=True)
names = data['HAPLESS_ID']
n_sources = data['HAPLESS_ID'].shape[0]
distances =  np.genfromtxt( os.path.join(dropbox+'Work','Tables','HAPLESS.csv'), delimiter=',', names=True)['DISTANCE']

# Declare wavebands and their instruments
in_wavelengths = np.array([22E-6, 60E-6, 100E-6, 160E-6, 250E-6, 350E-6, 500E-6])
in_instruments = ['WISE', 'IRAS', 'PACS', 'PACS', 'SPIRE', 'SPIRE', 'SPIRE']
in_limits = [True, 'blah', False, False, False, False, False]
#in_wavelengths = np.array([100E-6, 160E-6, 250E-6, 350E-6, 500E-6])
#in_instruments = ['PACS', 'PACS', 'SPIRE', 'SPIRE', 'SPIRE']
#in_limits = [False, False, False, False, False]

# Construct nice big arrays of fluxes & errors
in_fluxes = np.zeros([n_sources, in_wavelengths.shape[0]])
in_errors = np.zeros([n_sources, in_wavelengths.shape[0]])
in_fluxes[:,0], in_fluxes[:,1], in_fluxes[:,2], in_fluxes[:,3], in_fluxes[:,4], in_fluxes[:,5], in_fluxes[:,6] = data['w4'], data['60'], data['100'], data['160'], data['250'], data['350'], data['500']
in_errors[:,0], in_errors[:,1], in_errors[:,2], in_errors[:,3], in_errors[:,4], in_errors[:,5], in_errors[:,6] = data['w4err'], data['60err'], data['100err'], data['160err'], data['250err'], data['350err'], data['500err']
#in_fluxes[:,0], in_fluxes[:,1], in_fluxes[:,2], in_fluxes[:,3], in_fluxes[:,4], in_fluxes[:,5], in_fluxes[:,6] = ChrisFuncs.ABMagsToJy(data['W4_CAAPR']), data['F60_SCANPI'], data['F100_CAAPR'], data['F160_CAAPR'], data['F250_CAAPR'], data['F350_CAAPR'], data['F500_CAAPR']
#in_errors[:,0], in_errors[:,1], in_errors[:,2], in_errors[:,3], in_errors[:,4], in_errors[:,5], in_errors[:,6] = ChrisFuncs.ErrABMagsToJy(data['W4_CAAPR_ERR'], data['W4_CAAPR']), data['E60_SCANPI'], data['E100_CAAPR'], data['E160_CAAPR'], data['E250_CAAPR'], data['E350_CAAPR'], data['E500_CAAPR']

# Prepare array to hold output values
final = np.zeros([n_sources, 14])
col_corrs = np.zeros([n_sources, len(in_wavelengths)])

# Decide whether to generate plots
plotting = True

# Decide whether to bootstrap
bootstrapping = 100

# Decide whether to attempt both types of fit, or just a single-component fit
both_fits = True
residuals_test1_list = np.zeros(n_sources)
residuals_test2_list = np.zeros(n_sources)

# Ready timer variables
source_time_array = []
time_total = 0.0

# Loop over galaxies
for h in range(0, n_sources):
    H = int(data['HAPLESS_ID'][h])
    final[h,0] = H
    time_start = time.time()
    if h==24:
        continue

    # Read in source details
    distance = distances[h] * 1.0E6
    source_name = 'HAPLESS '+str(H)
    func_redshift = 0.0

    # Check if data present for each waveband, and construct appropriate input arrays for function
    func_wavelengths = []
    func_instruments = []
    func_fluxes = []
    func_errors = []
    func_limits = []
    for w in range(0, in_wavelengths.shape[0]):
#        if in_instruments[w]!='IRAS':
        if in_fluxes[h,w] > -90.0:
            if np.isnan(in_fluxes[h,w]) == False:
                if in_errors[h,w] > -90.0:
                    if np.isnan(in_errors[h,w]) == False:
                        func_wavelengths.append(float(in_wavelengths[w]))
                        func_instruments.append(in_instruments[w])
                        func_limits.append(in_limits[w])
                        func_fluxes.append(float(in_fluxes[h,w]))
                        func_errors.append(float(in_errors[h,w]))
                        if in_instruments[w]=='IRAS':
                            if float(in_fluxes[h,w])==0.0:
                                func_errors[len(func_errors)-1] *= 1.0

    # Deal with bands which are or are not limits depending upon fit type
    func_limits_1GB, func_limits_2GB = func_limits[:], func_limits[:]
    func_limits_1GB[ np.where(np.array(func_instruments)=='IRAS')[0] ] = True
    if func_limits_2GB[ np.where(np.array(func_instruments)=='IRAS')[0] ] != True:
        func_limits_2GB[ np.where(np.array(func_instruments)=='IRAS')[0] ] = False

    # Run data through ChrisFit functions for 1 and then 2 component greybodies, and move generated plots to appropriate directories
    output_1GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 1, distance, limits=func_limits_1GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=plotting, bootstrapping=bootstrapping, percentile=66.6, min_temp=10.0)
    if both_fits==True:
        output_2GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 2, distance, limits=func_limits_2GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=plotting, bootstrapping=bootstrapping, percentile=66.6, min_temp=10.0)

    # Record colour corrections
    if both_fits==True:
        func_indices = np.in1d(in_wavelengths, func_wavelengths)
        col_corrs[h,func_indices] = output_2GB[3] / np.array(func_fluxes)

    # Calculate for each fit the probability that the hull hypothesis is not satisfied
    prob_1GB = 0
    prob_2GB = 1E50

    # Deal with output if only attempting the one-component fit
    if both_fits==False:
        final[h,1] = output_1GB[1][0]
        final[h,2] = output_1GB[2][0]
        final[h,3] = output_1GB[1][1]
        final[h,4] = output_1GB[2][1]
        final[h,5] = output_1GB[1][2]
        final[h,6] = output_1GB[2][2]
        final[h,7] = output_1GB[1][3]
        final[h,8] = output_1GB[2][3]
        final[h,9] = output_1GB[1][4]
        final[h,10] = output_1GB[2][4]
        final[h,11] = output_1GB[1][5]
        final[h,12] = output_1GB[2][5]
        final[h,13] = np.sum(output_1GB[0])

    # Check if 1-greybody fit is superior, and process accordingly
    if both_fits==True:
        if prob_1GB>=prob_2GB:
            final[h,1] = output_1GB[1][0]
            final[h,2] = output_1GB[2][0]
            final[h,3] = output_1GB[1][1]
            final[h,4] = output_1GB[2][1]
            final[h,5] = output_1GB[1][2]
            final[h,6] = output_1GB[2][2]
            final[h,7] = output_1GB[1][3]
            final[h,8] = output_1GB[2][3]
            final[h,9] = output_1GB[1][4]
            final[h,10] = output_1GB[2][4]
            final[h,11] = output_1GB[1][5]
            final[h,12] = output_1GB[2][5]
            final[h,13] = np.sum(output_1GB[0])

        # Else check if 2-greybody fit is superior, and process accordingly
        if prob_2GB>prob_1GB:
            final[h,1] = output_2GB[1][0] # T_c
            final[h,2] = output_2GB[2][0] # T_c_sigma
            final[h,3] = output_2GB[1][1] # M_c
            final[h,4] = output_2GB[2][1] # M_c_sigma
            final[h,5] = output_2GB[1][2] # T_w
            final[h,6] = output_2GB[2][2] # T_w_sigma
            final[h,7] = output_2GB[1][3] # M_w
            final[h,8] = output_2GB[2][3] # M_w_sigma
            final[h,9] = output_2GB[1][4] # M_d
            final[h,10] = output_2GB[2][4] # M_d_sigma
            final[h,11] = output_2GB[1][5] # beta
            final[h,12] = output_2GB[2][5] # beta_sigma
            final[h,13] = np.sum(output_2GB[0]) # chisq
    # Record output for source to file
    datstring = str((final[h,:]).tolist())
    datstring = datstring.replace('[', '')
    datstring = datstring.replace(']', '')
    datstring = datstring.replace(',', '   ')
    datstring = datstring+' \n'
    datfile = open(filepath, 'a')
    datfile.write(datstring)
    datfile.close()
    plt.close('all')
    gc.collect()

    # Record time spent
    time_source = (time.time() - time_start)
    time_total += time_source
    time_source_mins = time_source/60.0
    source_time_array.append(time_source_mins)
    time_source_mean = np.mean(np.array(source_time_array))

    try:
        lambda_test1 = 350E-6
        index_test1 = np.where(np.array(func_wavelengths)==lambda_test1)[0][0]
        residuals_test1_list[h] = output_2GB[4][index_test1] / np.array(func_errors)[index_test1]
        index_test1 = np.where(np.array(in_wavelengths)==lambda_test1)[0][0]
    except:
        pass

    try:
        lambda_test2 = 500E-6
        index_test2 = np.where(np.array(func_wavelengths)==lambda_test2)[0][0]
        residuals_test2_list[h] = output_2GB[4][index_test2] / np.array(func_errors)[index_test2]
        index_test2 = np.where(np.array(in_wavelengths)==lambda_test2)[0][0]
    except:
        pass


print np.where( (in_fluxes[:,index_test1]>in_errors[:,index_test1]) & (np.isnan(residuals_test1_list)==False) )[0].shape[0]
print np.where( (in_fluxes[:,index_test2]>in_errors[:,index_test2]) & (np.isnan(residuals_test2_list)==False) )[0].shape[0]
print np.mean( residuals_test1_list[ np.where( (in_fluxes[:,index_test1]>in_errors[:,index_test1]) & (np.isnan(residuals_test1_list)==False) ) ] )
print np.mean( residuals_test2_list[ np.where( (in_fluxes[:,index_test2]>in_errors[:,index_test2]) & (np.isnan(residuals_test2_list)==False) ) ] )
print np.median(final[:,1][ np.where( (in_fluxes[:,index_test2]>in_errors[:,index_test2]) & (np.isnan(residuals_test2_list)==False) ) ] )
print np.median(final[:,11][ np.where( (in_fluxes[:,index_test2]>in_errors[:,index_test2]) & (np.isnan(residuals_test2_list)==False) ) ] )



# Jubilate
print 'All done!'


