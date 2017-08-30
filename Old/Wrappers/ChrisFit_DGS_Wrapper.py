# Identify location
location = 'saruman'
if location == 'Monolith':
    dropbox = 'E:\\Users\\Chris\\Dropbox\\'
if location == 'Hobbitslayer':
    dropbox = 'C:\\Users\\spx7cjc\\Dropbox\\'
if location == 'saruman':
    dropbox = '/home/herdata/spx7cjc/Dropbox/'

# Import smorgasbord
import os
import pdb
import sys
import gc
sys.path.append(dropbox+'Work/Scripts')
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import shutil
import ChrisFuncs
import ChrisFit

def AddInQuad(Err, Flux, Calib):
    return ( Err**2.0 + (Flux*Calib)**2.0 )**0.5



# Prepare output file
time_stamp = time.time()
output_header = '# ID T_COLD T_COLD_ERR M_COLD M_COLD_ERR T_WARM T_WARM_ERR M_WARM M_WARM_ERR M_DUST M_DUST_ERR BETA BETA_ERR CHISQ_DUST \n'
filedir = dropbox+'Work/HAPLESS/SEDs/DGS/'
filename = 'DGS_Greybody_Output_'+str(time_stamp)+'.dat'
try:
    os.remove(filedir+filename)
    filepath = filedir+filename
except:
    filepath = filedir+filename
datfile = open(filepath, 'a')
datfile.write(output_header)
datfile.close()

# Read input catalogue
data = np.genfromtxt(filedir+'DGS_Pieter_2.csv', names=True, delimiter=',', dtype=None)
names = data['nme']
n_sources = data['nme'].shape[0]

# Declare wavebands and their instruments
in_wavelengths = np.array([70E-6, 100E-6, 160E-6, 250E-6, 350E-6, 500E-6])
in_instruments = ['PACS', 'PACS', 'PACS', 'SPIRE', 'SPIRE', 'SPIRE']
in_limits = [False, False, False, False, False, False]
#in_wavelengths = np.array([100E-6, 160E-6, 250E-6, 350E-6, 500E-6])
#in_instruments = ['PACS', 'PACS', 'SPIRE', 'SPIRE', 'SPIRE']
#in_limits = [False, False, False, False, False]

# Construct nice big arrays of fluxes & errors
in_fluxes = np.zeros([n_sources, in_wavelengths.shape[0]])
in_errors = np.zeros([n_sources, in_wavelengths.shape[0]])
in_fluxes[:,0], in_fluxes[:,1], in_fluxes[:,2], in_fluxes[:,3], in_fluxes[:,4], in_fluxes[:,5] = data['PCS_70_2'], data['PCS_100_2'], data['PCS_160_2'], data['SPIRE_250_2'], data['SPIRE_350_2'], data['SPIRE_500_2']
in_errors[:,0], in_errors[:,1], in_errors[:,2], in_errors[:,3], in_errors[:,4], in_errors[:,5] = data['PCS_70_err_2'], data['PCS_100_err_2'], data['PCS_160_err_2'], data['SPIRE_250_err_2'], data['SPIRE_350_err_2'], data['SPIRE_500_err_2']

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
    final[h,0] = h + 1
    time_start = time.time()

    # Read in source details
    distance = data['DistanceMpc'][h] * 1.0E6
    source_name = names[h]
    func_redshift = 0.0

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

        # Deal with upper limits as presented in the DGS data
        if (in_fluxes[h,w] < 0.0) and (in_fluxes[h,w] >= -100.0) and (np.isnan(in_fluxes[h,w]) == False):
            func_wavelengths.append(float(in_wavelengths[w]))
            func_instruments.append(in_instruments[w])
            func_limits.append(True)
            func_fluxes.append(abs(float(in_fluxes[h,w])))
            func_errors.append(abs(float(in_fluxes[h,w]/5.0)))

    # If all bands are limits, skip to next source, nothing to see here
    if (False in func_limits)==False:
        continue

    # Skip sources with insufficient fluxes
    if sum(func_limits)>1:
        continue
    if len(func_fluxes)<5:
        continue

    # Deal with bands which are or are not limits depending upon fit type
    func_limits_1GB, func_limits_2GB = func_limits[:], func_limits[:]

    # Run data through ChrisFit functions for 1 and then 2 component greybodies, and move generated plots to appropriate directories
    output_2GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 2, distance, limits=func_limits_2GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=True, bootstrapping=1000, percentile=66.6, algorithm='leastsq', output_dir=filedir+'Plots/')
    output_1GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 1, distance, limits=func_limits_1GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=True, bootstrapping=1000, percentile=66.6, algorithm='leastsq', output_dir=filedir+'Plots/')
#    if both_fits==True:
#        output_2GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 2, distance, limits=func_limits_2GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=True, bootstrapping=False, percentile=66.6, output_dir=filedir+'Plots/', guess_mass=output_1GB[1][1])


#    # Move generated plots to appropriate destinations
#    shutil.copy(filedir+'Output/'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\One Component\\'+source_name+' One Component.png')
#    shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\One Component\\'+source_name+' One Component.eps')
#    if both_fits==True:
#        shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Two Component\\'+source_name+' Two Component.png')
#        shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Two Component\\'+source_name+' Two Component.eps')

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
#        shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Best\\'+source_name+' One Component.png')
#        shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Best\\'+source_name+' One Component.eps')

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
#            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Best\\'+source_name+' One Component.png')
#            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Best\\'+source_name+' One Component.eps')
#            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png')
#            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.eps')

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
#            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Best\\'+source_name+' Two Component.png')
#            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.eps', dropbox+'Work\\HAPLESS\\SEDs\\DGS\\Best\\'+source_name+' Two Component.eps')
#            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png')
#            os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.eps')

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



# Jubilate
print 'All done!'
pygame.mixer.init()
alert=pygame.mixer.Sound(dropbox+'Work\\Scripts\\R2-D2.wav')
alert.play()
