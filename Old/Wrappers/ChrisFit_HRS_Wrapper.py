# Identify location
import socket
location = socket.gethostname()
if location == 'Monolith':
    dropbox = 'E:\\Users\\Chris\\Dropbox\\'
if location == 'Hobbitslayer':
    dropbox = 'C:\\Users\\spx7cjc\\Dropbox\\'
if location == 'saruman':
    dropbox = '/home/user/spx7cjc/Desktop/Herdata/Dropbox/'

# Import smorgasbord
import os
import pdb
import sys
import gc
sys.path.append(dropbox+'Work\\Scripts\\')
sys.path.append(dropbox+'Work\\Scripts\\ChrisFit')
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pygame.mixer
import shutil
import readcol
import ChrisFuncs
import ChrisFit

def AddInQuad(Err, Flux, Calib):
    return ( Err**2.0 + (Flux*Calib)**2.0 )**0.5



# Prepare output file
time_stamp = time.time()
output_header = '# ID T_COLD T_COLD_ERR M_COLD M_COLD_ERR T_WARM T_WARM_ERR M_WARM M_WARM_ERR M_DUST M_DUST_ERR BETA BETA_ERR CHISQ_DUST \n'
filedir = dropbox+'Work\\HAPLESS\\SEDs\\HRS\\'
filename = 'HRS_Greybody_Output_'+str(time_stamp).replace('.','')+'.dat'
filepath = filedir+filename
datfile = open(filepath, 'w')
datfile.write(output_header)
datfile.close()

# Read input catalogue
data = np.genfromtxt(dropbox+'Work\\Tables\\HRS.csv', delimiter=',', names=True)
names = readcol.readcol(dropbox+'Work\\Tables\\HRS.csv', fsep=',', names=True)[1][:,0]
in_redshifts = ( data['Dist'] * 67.30 ) / 3E5 # STRICTLY PLACEHOLDER
n_sources = data['HRS'].shape[0]

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
in_fluxes[:,0], in_fluxes[:,1], in_fluxes[:,2], in_fluxes[:,3], in_fluxes[:,4], in_fluxes[:,5], in_fluxes[:,6] =  data['S22'], data['S60'], 1.01*data['S100'], 0.97*data['S160'], data['S250'], data['S350'], data['S500']
in_errors[:,0], in_errors[:,1], in_errors[:,2], in_errors[:,3], in_errors[:,4], in_errors[:,5], in_errors[:,6] = AddInQuad(data['E22'],data['S22'],0.2), AddInQuad(data['E60'],data['S60'],0.2), 1.01*AddInQuad(data['E100'],data['S100'],0.12), 0.97*AddInQuad(data['E160'],data['S160'],0.12), AddInQuad(data['E250'],data['S250'],0.07), AddInQuad(data['E350'],data['S350'],0.07), AddInQuad(data['E500'],data['S500'],0.07)
#in_fluxes[:,0], in_fluxes[:,1], in_fluxes[:,2], in_fluxes[:,3], in_fluxes[:,4] = data['S100'], data['S160'], data['S250'], data['S350'], data['S500']
#in_errors[:,0], in_errors[:,1], in_errors[:,2], in_errors[:,3], in_errors[:,4] = data['E100'], data['E160'], data['E250'], data['E350'], data['E500']

# Prepare array to hold output values
final = np.zeros([n_sources, 14])

# Ready timer variables
source_time_array = []
time_total = 0.0

# Decide whether to attempt both types of fit, or just a single-component fit
both_fits = True
residuals_100_list = np.zeros(n_sources)
residuals_500_list = np.zeros(n_sources)


# Loop over galaxies
for h in range(0, n_sources):#range(0, n_sources):
    H = int(data['HRS'][h])
    final[h,0] = H
    time_start = time.time()

    # Read in source details
    distance = data['Dist'][h] * 1.0E6
    source_name = 'HRS '+str(H)
    func_redshift = 0.0#in_redshifts[h]

    # Check if data present for each waveband, and construct appropriate input arrays for function
    func_wavelengths = []
    func_instruments = []
    func_fluxes = []
    func_errors = []
    func_limits = []
    for w in range(0, in_wavelengths.shape[0]):
        if (in_fluxes[h,w] > -90.0) and (np.isnan(in_fluxes[h,w]) == False):
            if (in_errors[h,w] > -90.0) and (np.isnan(in_errors[h,w]) == False):
                if (in_fluxes[h,w] >= 0.0) and (in_errors[h,w] > 0.0):
                    func_wavelengths.append(float(in_wavelengths[w]))
                    func_instruments.append(in_instruments[w])
                    func_limits.append(in_limits[w])
                    func_fluxes.append(float(in_fluxes[h,w]))
                    func_errors.append(float(in_errors[h,w]))
                    if in_instruments[w]=='IRAS':
                            if float(in_fluxes[h,w])==0.0:
                                func_errors[len(func_errors)-1] *= 3.0

        # Deal with 3-sigma upper limits as presented in the HRS data
        if (in_fluxes[h,3] > 0.0) and (np.isnan(in_fluxes[h,3]) == False):
            if (np.isnan(in_errors[h,w]) == True) and (in_fluxes[h,w] > -90.0) and (np.isnan(in_fluxes[h,w]) == False):
                func_wavelengths.append(float(in_wavelengths[w]))
                func_instruments.append(in_instruments[w])
                func_limits.append(True)
                func_fluxes.append(float(in_fluxes[h,w]))
                func_errors.append(float(in_fluxes[h,w]/3.0))

    # If all bands are limits, skip to next source, nothing to see here
    if (False in func_limits)==False:
        continue

    # Deal with bands which are or are not limits depending upon fit type
    func_limits_1GB, func_limits_2GB = func_limits[:], func_limits[:]
    if 'IRAS' in func_instruments:
        func_limits_1GB[ np.where(np.array(func_instruments)=='IRAS')[0] ] = True
        func_limits_2GB[ np.where(np.array(func_instruments)=='IRAS')[0] ] = False

    # Run data through ChrisFit functions for 1 and then 2 component greybodies, and move generated plots to appropriate directories
    output_1GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 1, distance, limits=func_limits_1GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=True, bootstrapping=5000, percentile=66.6)
    if both_fits==True:
        output_2GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 2, distance, limits=func_limits_2GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=True, bootstrapping=5000, percentile=66.6)

    # Move generated plots to appropriate destinations
    shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\One Component\\'+source_name+' One Component.png')
    shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\One Component\\'+source_name+' One Component.eps')
    if both_fits==True:
        shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Two Component\\'+source_name+' Two Component.png')
        shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Two Component\\'+source_name+' Two Component.eps')

    # Calculate for each fit the probability that the hull hypothesis is not satisfied
    func_array = np.array(func_instruments)
    DoF_1GB = ( np.where(func_array=='PACS')[0].shape[0] + np.where(func_array=='SPIRE')[0].shape[0] ) - 2.0 - 1.0
    DoF_2GB = ( np.where(func_array=='PACS')[0].shape[0] + np.where(func_array=='SPIRE')[0].shape[0] + np.where(func_array=='IRAS')[0].shape[0] ) - 4.0 - 1.0
    prob_1GB = 0.0
    prob_2GB = 1E50 # So 2GB always gets favoured
#    if both_fits==True:
#        prob_1GB = 1.0 - scipy.stats.chi2.cdf(np.sum(output_1GB[0]), DoF_1GB)
#        prob_2GB = 1.0 - scipy.stats.chi2.cdf(np.sum(output_2GB[0]), DoF_2GB)
#        if np.isnan(prob_1GB)==True:
#            prob_1GB = 0.0
#        if np.isnan(prob_2GB)==True:
#            prob_2GB = 0.0

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
        shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Best\\'+source_name+' One Component.png')
        shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Best\\'+source_name+' One Component.eps')

    # If attempting both types of fit, check if 1-greybody fit is superior, and process accordingly
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
            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Best\\'+source_name+' One Component.png')
            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Best\\'+source_name+' One Component.eps')
            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png')
            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.eps')

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
            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Best\\'+source_name+' Two Component.png')
            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\HRS\\Best\\'+source_name+' Two Component.eps')
            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png')
            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps')

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





    index_100 = np.where(in_wavelengths==100E-6)[0][0]
    index_500 = np.where(in_wavelengths==500E-6)[0][0]
#    if prob_1GB>prob_2GB:
    residuals_100_list[h]=output_1GB[4][0]
    residuals_500_list[h]=output_1GB[4][1]
#    elif prob_2GB>=prob_1GB:
#        residuals_100_list[h]=output_2GB[4][0]
#        residuals_500_list[h]=output_2GB[4][1]
print np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) )[0].shape[0]
print np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) )[0].shape[0]
print np.mean( residuals_100_list[ np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) ) ] )
print np.mean( residuals_500_list[ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) ) ] )
print np.median(final[:,1][ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) ) ] )
print np.median(final[:,11][ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) ) ] )

print ' '
HRS = np.genfromtxt(dropbox+'Work\\Tables\\HRS.csv', names=True, delimiter=',')
print np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) & (HRS['FUV_MINUS_KS']<3.5) )[0].shape[0]
print np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (HRS['FUV_MINUS_KS']<3.5) )[0].shape[0]
print np.mean( residuals_100_list[ np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) & (HRS['FUV_MINUS_KS']<3.5) ) ] )
print np.mean( residuals_500_list[ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (HRS['FUV_MINUS_KS']<3.5) ) ] )
print np.median( final[:,1][ np.where( (HRS['FUV_MINUS_KS']<3.5) & (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) ) ] )
print np.median( final[:,11][ np.where( (HRS['FUV_MINUS_KS']<3.5) & (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) ) ] )




# Jubilate
print 'All done!'
pygame.mixer.init()
alert=pygame.mixer.Sound(dropbox+'Work\\Scripts\\R2-D2.wav')
alert.play()
