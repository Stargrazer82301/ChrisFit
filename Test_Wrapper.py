# Import smorgasbord
import pdb
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ChrisFuncs
import ChrisFit





# Read input catalogue
phot_raw = pd.read_csv('DustPedia_Aperture_Photometry_2.2.csv')

band_dict = {'WISE_22':{'wavelength':22E-6,'limit':True},
             'PACS_70':{'wavelength':70E-6,'limit':False},
             'PACS_100':{'wavelength':100E-6,'limit':False},
             'PACS_160':{'wavelength':160E-6,'limit':False},
             'SPIRE_250':{'wavelength':250E-6,'limit':False},
             'SPIRE_350':{'wavelength':350E-6,'limit':False},
             'SPIRE_500':{'wavelength':500E-6,'limit':False}}

# Loop over table fields, to collate columns of interest
band_cols = ['name']
for col in phot_raw.columns:
    if col in band_dict.keys():
        band_cols.append(col)
        band_cols.append(col+'_err')
phot_in = phot_raw[band_cols]
sfdf

"""
# Prune fluxes with major flags
any(substring in string for substring in substring_list)
"""

# Initiate settings dictionary
settings_dict = {'plotting':True}

# Loop over galaxies
for h in range(0, n_sources):
    H = int(data['HAPLESS_ID'][h])
    final[h,0] = H
    time_start = time.time()

    # Read in source details
    distance = data['DISTANCE'][h] * 1.0E6
    source_name = 'HAPLESS '+str(H)+' ('+str(names[h])+')'
    func_redshift = 0.0#in_redshifts[h]

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
    output_1GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 1, distance, limits=func_limits_1GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=plotting, bootstrapping=500, percentile=66.6)
    if both_fits==True:
        output_2GB = ChrisFit.ChrisFit(source_name, func_wavelengths, func_fluxes, func_errors, func_instruments, 2, distance, limits=func_limits_2GB, beta=2.0, kappa_0=0.077, lambda_0=850E-6, redshift=func_redshift, col_corr=True, plotting=plotting, bootstrapping=1000, percentile=66.6)

    # Record colour corrections
    if both_fits==True:
        func_indices = np.in1d(in_wavelengths, func_wavelengths)
        col_corrs[h,func_indices] = output_2GB[3] / np.array(func_fluxes)

    # Move generated plots to appropriate destinations
    if plotting==True:
        shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\One Component\\'+source_name+' One Component.png')
        shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.pdf', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\One Component\\'+source_name+' One Component.pdf')
        if both_fits==True:
            shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Two Component\\'+source_name+' Two Component.png')
            shutil.copy(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.pdf', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Two Component\\'+source_name+' Two Component.pdf')

    # Calculate for each fit the probability that the hull hypothesis is not satisfied
    func_array = np.array(func_instruments)
    DoF_1GB = ( np.where(func_array=='PACS')[0].shape[0] + np.where(func_array=='SPIRE')[0].shape[0] ) - 2.0 - 1.0
    DoF_2GB = ( np.where(func_array=='PACS')[0].shape[0] + np.where(func_array=='SPIRE')[0].shape[0] + np.where(func_array=='IRAS')[0].shape[0] ) - 4.0 - 1.0
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
        if plotting==True:
            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Best\\'+source_name+' One Component.png')
            shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.pdf', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Best\\'+source_name+' One Component.pdf')

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
            if plotting==True:
                shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Best\\'+source_name+' One Component.png')
                shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.pdf', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Best\\'+source_name+' One Component.pdf')
                os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png')
                os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.pdf')

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
            if plotting==True:
                shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.png', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Best\\'+source_name+' Two Component.png')
                shutil.move(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' Two Component.pdf', dropbox+'Work\\HAPLESS\\SEDs\\HAPLESS\\Best\\'+source_name+' Two Component.pdf')
                os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.png')
                os.remove(dropbox+'Work\\Scripts\\ChrisFit\\Output\\'+source_name+' One Component.pdf')

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
pygame.mixer.init()
alert=pygame.mixer.Sound(dropbox+'Work\\Scripts\\R2-D2.wav')
alert.play()





"""
    index_100 = np.where(in_wavelengths==100E-6)[0][0]
    index_500 = np.where(in_wavelengths==500E-6)[0][0]
    residuals_100_list[h]=output_1GB[4][index_100]
    residuals_500_list[h]=output_1GB[4][index_500]

print np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) )[0].shape[0]
print np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) )[0].shape[0]
print np.mean( residuals_100_list[ np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )
print np.mean( residuals_500_list[ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )
print np.median(final[:,1][ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )
print np.median(final[:,11][ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )

print ' '
HAPLESS = np.genfromtxt(dropbox+'Work\\Tables\\HAPLESS.csv', names=True, delimiter=',')
print np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) & (HAPLESS['FUV_MINUS_KS']<3.5) & (data['HAPLESS_ID'].astype(int)!=13) )[0].shape[0]
print np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (HAPLESS['FUV_MINUS_KS']<3.5) & (data['HAPLESS_ID'].astype(int)!=13) )[0].shape[0]
print np.mean( residuals_100_list[ np.where( (in_fluxes[:,index_100]>in_errors[:,index_100]) & (np.isnan(residuals_100_list)==False) & (HAPLESS['FUV_MINUS_KS']<3.5) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )
print np.mean( residuals_500_list[ np.where( (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (HAPLESS['FUV_MINUS_KS']<3.5) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )
print np.median( final[:,1][ np.where( (HAPLESS['FUV_MINUS_KS']<3.5) & (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )
print np.median( final[:,11][ np.where( (HAPLESS['FUV_MINUS_KS']<3.5) & (in_fluxes[:,index_500]>in_errors[:,index_500]) & (np.isnan(residuals_500_list)==False) & (data['HAPLESS_ID'].astype(int)!=13) ) ] )
"""