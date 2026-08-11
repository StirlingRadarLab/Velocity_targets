
# Armando Marino 09/07/2025

# system library
# import sys
# sys.path.insert(0, 'C:\\Programms\\Python\\Libraries\\')
# this library is used to tell Python where our functions or libraries are. Since we are working with a single script, 
# we will not use this library now but you may want to use it in the future. You need to make sure that the 
# folder is the one containing your user libraries. 

import sys
sys.path.insert(0, '/home/am221/C/Programs/Python_Lib')

# Numpy is for Numerical manipulations
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift


import matplotlib.pyplot as plt
from scipy import signal

# is for manipulating files and filenames
import os
# a useful library for managing path with different OS
from pathlib import Path 

import SAR_Utilities as sar


import pandas as pd
  

import tqdm
import time


# Import the required libraries
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

import xml.etree.ElementTree as ET
import glob

plt.close('all')


#%%
def read_sentinel1_params(safe_folder):
    ann_files = glob.glob(os.path.join(safe_folder, "annotation", "*.xml"))
    if not ann_files:
        raise FileNotFoundError("No annotation XMLs found in SAFE package")

    tree = ET.parse(ann_files[0])
    root = tree.getroot()

    ns = {
        's1': "http://www.esa.int/safe/sentinel-1.0",
        's1sarl1': "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"
    }

    # Radar frequency
    center_freq = root.findtext(".//s1sarl1:radarFrequency", namespaces=ns)
    if center_freq is None:
        center_freq = root.findtext(".//radarFrequency")
    if center_freq is None:
        raise ValueError("radarFrequency not found in annotation file")
    center_freq = float(center_freq)

    # Try PRFs directly
    prfs = [x.text for x in root.findall(".//s1sarl1:pulseRepetitionFrequencyList/s1sarl1:pulseRepetitionFrequency", ns)]
    if not prfs:
        prfs = [x.text for x in root.findall(".//pulseRepetitionFrequencyList/pulseRepetitionFrequency")]
    prfs = [float(x) for x in prfs]

    # Fallback: compute from azimuthTimeInterval
    if not prfs:
        ati = root.findtext(".//s1sarl1:azimuthTimeInterval", namespaces=ns)
        if ati is None:
            ati = root.findtext(".//azimuthTimeInterval")
        if ati:
            prfs = [1.0 / float(ati)]

    return prfs[0], center_freq


#%% SUBDETECTORS
###########################################################################
def SubCoherence(data, flag, params):
    
    [Win] = params

    #if flag eq 'AZ' then data = transpose(data)
    if flag == 'AZ': data = np.transpose(data)
    
    #; Parameters SAR data (to modify accordingly!!!)
    #;----------------------------------------------------------------------------------------------------
    #; TSX
#    BW = rangeBandwidth              #; Range Bandwidth of the real data [Hz]
#    rg_spacing  = rangeSpacing         #; Range (row) spacing of the real data [s]
    
    dim = np.shape(data)     

    #; Obtain the sublooks (in range!)
    #;----------------------------------------------------------------------------------------------------
    #do fft in range = 1st dimension = along the columns = dim[0] (IDL) = dim[1] (Python) = axis nr. 1 (Python)
    #IDL's "shift" shifts complex array in column-wise (dim[0]) in frequency domain: 
    #    shift(fft(data, -1, dimension=1), dim[0]/2., 0) 
    #spectrum = np.roll(np.fft.fft(data, 1), dim[1]/2, axis=1)   
    #?? need a 1dim fft that returns a 2-dim array in python
    #spectrum = np.roll(np.fft.fft2(data), dim[1]/2, axis=1)
    dataFFT = np.fft.fftn(data, axes=[1]) #columns: IDL-dim1 = Py-axis1
    spectrum = np.roll(dataFFT, int(dim[1]/2), axis=1) #colums: IDL-dim1 = Py-axis1 | colums: IDL-dim1 = Py-axis1
#    spectrumVis = spectrum
    #spectrum = np.roll(np.fft.fft(data,1), dim[1]/2, axis=1)
    #; Unhamming process
    #spectrum_medio  = (total(abs(spectrum), 2))/(dim[1])
    spectrum_medio  = (np.sum(abs(spectrum), axis=0))/(dim[0]) #row: IDL-dim2 = Py-axis0, row: IDL-dim2 = Py-axis0
    #;Smoothing the signal to get a good "unhamming" function
    #for kkk=0, 10 do spectrum_medio = smooth(spectrum_medio, 7, /edge_truncate)
    convFilter = np.ones((7,))/7
    for kkk in range (0, 10): spectrum_medio = signal.fftconvolve(spectrum_medio, convFilter, mode='same')
    #;To force that the maximum of my "unhamming" function to be equal 1
    spectrum_medio  = spectrum_medio/max(spectrum_medio)
 
    #; for Doppler analysis
    if flag == 'AZ':
        #shifts the 0-frequency component to the spectrum's center
        #plt.plot(spectrum_medio, 'g')
        #spectrum_medio = np.fft.fftshift(spectrum_medio, axes=0)
 

        gf0 = np.where(spectrum_medio>0.1*max(spectrum_medio),0,spectrum_medio) #set all pixels bigger than threshold to 0
        gf = gf0.nonzero() #eliminate all pixels = 0
        
        s = np.shape(gf)  
        n_smp_band = s[1] #nb of pixels in gf | columns: IDL-size[1] = Py-axis1 = Py-shape[1]
        n_smp_extra = dim[1]-n_smp_band #columns: IDL-size[1]=IDL-dim[0] = Py-axis1
    
        gf0 = np.where(spectrum_medio<0.1*max(spectrum_medio),0,spectrum_medio) 
        gf = gf0.nonzero()     
        spectrum_medio = np.where(spectrum_medio<0.1*max(spectrum_medio),spectrum_medio,1)
        plt.plot(spectrum_medio, 'b')
        #spectrum_medio = np.fft.ifftshift(spectrum_medio, axes=0)
        #spectrum = np.roll(spectrum,shi,axis=1)
        #plt.plot(spectrum_medio, 'r')
        plt.show()

    #;Line by line, I correct the spectrum to get the original one (without windowing!!!)    
    spectrum_corr = np.zeros(dim, dtype=np.complex64) 
    #for jjj=0, dim[1]-1 do spectrum_corr(*,jjj) = spectrum(*,jjj)/spectrum_medio(*)    
    for jjj in range (0, dim[0]-1):
        #https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
        spectrum_corr[jjj:] = spectrum[jjj:]/spectrum_medio
        #spectrum_corr[jjj:] = np.divide(spectrum[jjj:], spectrum_medio)
        
#    spectrum_corrVis = spectrum_corr
#    #for Range analysis
#    if flag == 'RG':
#        # How many samples correspond to the bandwidth?
#        fact    = BW*rg_spacing
#        if (fact <= 1.): print('Sample frequency is NOT lower than bandwidth')
#        n_smp_band  = int(dim[0]*fact)                        #number of samples in bandwidth
#        n_smp_extra = dim[0] - n_smp_band                   #number of extra samples
#        if (n_smp_extra < 2): n_smp_extra = 2                #For index in the FOR loop
    
        
    spectrum1 = np.zeros(dim, dtype=np.complex64)
    spectrum2 = np.zeros(dim, dtype=np.complex64)    
    spectrum1[:,int(n_smp_extra/2):int(dim[1]/2)-1] = spectrum_corr[:,int(n_smp_extra/2):int(dim[1]/2)-1]  # [i:j,*] -> [:,i:j]
    spectrum2[:,int(dim[1]/2):int(dim[1])-int(n_smp_extra/2)-1] = spectrum_corr[:,int(dim[1]/2):int(dim[1]-n_smp_extra/2)-1]
    
    #make the 2 spectrums overlap
#    spectrum2 = np.roll(spectrum2, dim[1]/2)
  
    data1 = np.fft.ifftn(np.roll(spectrum1, int(dim[1]/4-n_smp_extra/4), axis=1), axes=[1]) # ?? problem: ifft produces 1dim array
    data2 = np.fft.ifftn(np.roll(spectrum2, -int(dim[1]/4+n_smp_extra/4), axis=1), axes=[1])
 
    winFilterUngarded = np.ones((Win,Win),np.float32)/(Win ** 2) #without guard windows

    SubNum = signal.convolve2d(data1*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
    SubDen1 = signal.convolve2d(data1*np.conj(data1), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
    SubDen2 = signal.convolve2d(data2*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
    
#    print SubNum.dtype, SubDen1.dtype

    SubCorr = abs(SubNum/np.sqrt(abs(SubDen1)*abs(SubDen2)))
    SubCohe = abs(SubNum/np.sqrt(abs(SubDen1)*abs(SubDen2)))



    return SubCohe, SubCorr

#%% retrivial using traditional processing

def velocity_estimation(img, lambda_c, prf):

    # Wavelength from frequency


    # Step 1: Compute azimuth FFT (along-track is typically the rows)
    azimuth_fft = np.fft.fftshift(np.fft.fft(img, axis=0), axes=0)
    # average along ramge lines to get a mean power spectrum 
    power_spectrum = np.mean(np.abs(azimuth_fft)**2, axis=1)
    
    
    # Step 2: Find Doppler centroid (frequency bin of peak energy)
    N_az = power_spectrum.shape[0]
    df = prf / N_az # azimuth frequency resolution [Hz]
    
    # create the frequancy axis
    freq_axis = (np.arange(N_az) - N_az//2) * df
    
    # find the maximum and call it centroid
    doppler_centroid = freq_axis[np.argmax(power_spectrum)]
    
    
    # Step 3: Convert Doppler centroid shift into azimuth velocity
    v_az = -(lambda_c/2.0) * doppler_centroid
    
    
    print(f"PRF: {prf:.2f} Hz")
    print(f"Center frequency: {center_freq/1e9:.3f} GHz (λ={lambda_c:.3f} m)")
    print(f"Estimated Doppler centroid: {doppler_centroid:.2f} Hz")
    # print(f"Estimated vessel along-track speed: {v_az:.2f} m/s")
    
    return freq_axis, doppler_centroid, v_az, power_spectrum


#%%


# defining paths where data are
path = Path("/home/am221/C/Data/S1/Velocity") 
path_save = Path("/home/am221/C/Data/S1/Velocity/Save")


# filtering  windows
win = [7,7]     # averagiung window for boxcar and algorithms

# this following window is useful if one want to do some extra multilook 
# and reduce the number of pixels (subsample) to reduce the size of images
# if you have a powerful machine you can keep it [1,1]
sub_win = [3,3]  # subsampling window for reducing size (partial multilook)


# # --- Constants for Sentinel-1 C-band ---
# lambda_c = 0.0555 # [m] wavelength at 5.405 GHz


manifest_path = "S1A_IW_SLC__1SDV_20240102T092331_20240102T092350_051926_06460F_974E.SAFE" # path to Sentinel-1 SAFE manifest
prf, center_freq = read_sentinel1_params(path / manifest_path)

# from previous runs
# center_freq = 5405000454.33435
# prf = 486.48631029955294


#%% Getting metadata and creating the cubes

# First we want to get the metadata
file_name = Path("S1A_IW_SLC__1SDV_20240102T092331_20240102T092350_051926_06460F_974E_Orb_Cal_Deb.tif")
fullpath_img = path / file_name
  
flag_pol = 'dual'

if flag_pol == 'dual':
    num_el = 4
elif flag_pol == 'quad':
    num_el = 8
    
    
with rasterio.open(fullpath_img) as src:
    
        # Get the metadata
        metadata = src.meta
        num_band = src.count # number of bands
        name = src.descriptions[1] # name of bands if it works... not with SNAP
     
# each acquisiiton has 4 layers (bands) therefore the total number of acquisitions is:          
num_acq = int(num_band/num_el)


# If you want to analyse the full image        
col_off = 0
row_off = 0
width = src.width
height = src.height


# part of the image        
col_off = 7700
row_off = 0
width = 1200
height = 1200

# # part of the image        
# col_off = 1800*sub_win[0]
# row_off = 450*sub_win[1]
# width = 200*sub_win[0]
# height = 200*sub_win[1]

# # part of the image        
# col_off = 8700
# row_off = 0
# width = 512
# height = 512





#%% opening the images
# we need a for loop to run through all the acquisitions in the time series
for i in range(num_acq):
    
    # the following command produces a print out that allows us to know how much is missing. 
    print('Pre-Processing date ' + str(i+1) + '....... ' + str(num_acq-i-1) + ' dates left.' )
    
    
    # this opens the layers (bands) of the geotiff, one by one
    with rasterio.open(fullpath_img) as src:
        # Read the image data
        VV_real = src.read(num_el*(i)+1)  # Reading the band at index "band"
        VV_imag = src.read(num_el*(i)+2)  # Reading the band at index "band"
        VH_real = src.read(num_el*(i)+3)  # Reading the band at index "band"
        VH_imag = src.read(num_el*(i)+4)  # Reading the band at index "band"
        
        # # Get the metadata
        # metadata = src.meta
    
    VV_full = VV_real + 1j*VV_imag
    VH_full = VH_real + 1j*VH_imag
    
    # we want to take a crop of the full image to avoid issues with the limited RAM    
    VV = VV_full[row_off:row_off+height, col_off:col_off+width]           
    VH = VH_full[row_off:row_off+height, col_off:col_off+width]
    # del C11Full, C22Full, C12reFull, C12imFull


plt.figure()
plt.imshow(np.abs(VV), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VV)))
plt.title("VV for area of interest")
plt.figure()
plt.imshow(np.abs(VH), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VH)))
plt.title("VH for area of interest")




#%% SUBLOOK DETECTION ANALSYSIS
##########################################


flag = 'Range'
# flag = "ciao"
params = [7]
data = VV
# [SubCohe, SubCorr] = SubCoherence(VV, flag, params)


[Win] = params

#if flag eq 'AZ' then data = transpose(data)
if flag == 'Range': data = np.transpose(data)

#; Parameters SAR data (to modify accordingly!!!)
#;----------------------------------------------------------------------------------------------------
#; TSX
#    BW = rangeBandwidth              #; Range Bandwidth of the real data [Hz]
#    rg_spacing  = rangeSpacing         #; Range (row) spacing of the real data [s]

dim = np.shape(data)     

#; Obtain the sublooks (in dimention 1)
# dataFFT = np.fft.fftn(data, axes=[1])
dataFFT = fftshift(fft(data, axis=1), axes=1)


#; Unhamming process
spectrum_medio  = (np.sum(abs(dataFFT), axis=0))/(dim[0]) 
# Smoothing the signal to get a good "unhamming" function

Kernel_Spectrum = np.ones((7,))/7
for kkk in range (0, 10): spectrum_medio = signal.fftconvolve(spectrum_medio, 
                                           Kernel_Spectrum, mode='same')
#;To force that the maximum of my "unhamming" function to be equal 1
spectrum_medio  = spectrum_medio/max(spectrum_medio)
# finding what is the last pixel of noise outside the band
gf0 = np.where(spectrum_medio[0:int(dim[1]/2)] < 0.1)
ind_noise = 0
if gf0[0] != []: 
    ind_noise = np.max(gf0)
# remove noise outside band in equalisation
spectrum_medio[spectrum_medio < 0.1] = 1  

 
# Line by line, we correct the spectrum to get the original one (without windowing!!!)    
spectrum_corr = np.zeros(dim, dtype=np.complex64) 
for jjj in range (0, dim[0]-1):
    spectrum_corr[jjj:] = dataFFT[jjj:]/spectrum_medio
    
#    spectrum_corrVis = spectrum_corr
#    #for Range analysis
#    if flag == 'RG':
#        # How many samples correspond to the bandwidth?
#        fact    = BW*rg_spacing
#        if (fact <= 1.): print('Sample frequency is NOT lower than bandwidth')
#        n_smp_band  = int(dim[0]*fact)                        #number of samples in bandwidth
#        n_smp_extra = dim[0] - n_smp_band                   #number of extra samples
#        if (n_smp_extra < 2): n_smp_extra = 2                #For index in the FOR loop

    
spectrum1 = np.zeros(dim, dtype=np.complex64)
spectrum2 = np.zeros(dim, dtype=np.complex64)    
spectrum1[:, ind_noise:int(dim[1]/2)] = spectrum_corr[:, ind_noise:int(dim[1]/2)]
spectrum2[:, int(dim[1]/2):int(dim[1])-ind_noise] = spectrum_corr[:,int(dim[1]/2):int(dim[1])-ind_noise]
# now cenering the spetrum so that they overlap in frequency
spectrum1 = np.roll(spectrum1, int(dim[1]/4-ind_noise/2), axis=1)
spectrum2 = np.roll(spectrum2, -int(dim[1]/4-ind_noise/2), axis=1)

data1 = ifft(ifftshift(spectrum1, axes=1), axis=1)
data2 = ifft(ifftshift(spectrum2, axes=1), axis=1)
# data2 = np.fft.ifftn(np.roll(spectrum2, -int(dim[1]/4+n_smp_extra/4), axis=1), axes=[1])
 
winFilterUngarded = np.ones((Win,Win),np.float32)/(Win ** 2) #without guard windows

SubNum = signal.convolve2d(data1*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
SubDen1 = signal.convolve2d(data1*np.conj(data1), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
SubDen2 = signal.convolve2d(data2*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)

SubCorr = SubNum 
SubCohe = abs(SubNum/np.sqrt(abs(SubDen1)*abs(SubDen2)))


if flag == 'Range': 
    data = np.transpose(data)
    data1 = np.transpose(data1)
    data2 = np.transpose(data2)
    SubCorr = np.transpose(SubCorr)
    SubCohe= np.transpose(SubCohe)


plt.figure()
plt.imshow(np.abs(dataFFT), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
plt.title("Magnitude spectrum of the data")

plt.figure()
plt.plot(spectrum_medio)
plt.title("Mean Spectrum of the entire image")

plt.figure()
plt.imshow(np.abs(spectrum_corr), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
plt.title("Magnitude spectrum AFTER removing Hamming")

plt.figure()
plt.imshow(np.abs(spectrum1), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
plt.title("Magnitude of FIRST portion of spectrum")

plt.figure()
plt.imshow(np.abs(spectrum2), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
plt.title("Magnitude of SECOND portion of spectrum")

plt.figure()
plt.imshow(np.abs(data1), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(data1)))
plt.title("Magnitude of FIRST subaperture")

plt.figure()
plt.imshow(np.abs(data2), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(data2)))
plt.title("Magnitude of SECOND subaperture")


plt.figure()
plt.imshow(np.abs(SubCohe), cmap = 'gray', vmin = 0, vmax = 1)
plt.title("VV Sub Coherence image")
plt.figure()
plt.imshow(np.abs(SubCorr), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(SubCorr)))
plt.title("VH Sub Correlation")















#%%
stop
flag = 'AZ'
params = [7]
[SubCohe, SubCorr] = SubCoherence(VV, flag, params)


plt.figure()
plt.imshow(np.abs(SubCohe), cmap = 'gray', vmin = 0, vmax = 1)
plt.title("VV Sub Coherence image")
plt.figure()
plt.imshow(np.abs(SubCorr), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(SubCorr)))
plt.title("VH Sub Correlation")


#%% SELECT SHIPS for velocity analysis
####################################################
####################################################
stop
size_imaget = 64
wd = int(size_imaget /2)
hg = int(size_imaget /2)


############# only seen in VH
# # SHIP 2       
# col_cr = 367
# row_cr =  908
#############
# # SHIP 3       
# col_cr = 1069
# row_cr =  828
# #############
# # LAND     
# col_cr = 938
# row_cr =  645
#############
# SHIP 1       
col_cr_ship = 725
row_cr_ship =  364
#


c = 299792458.0
lambda_c = c / center_freq

# creatingf the small imagets
imgVV_ship = VV[col_cr_ship-wd:col_cr_ship+wd, row_cr_ship-hg:row_cr_ship+hg]
imgVH_ship = VH[col_cr_ship-wd:col_cr_ship+wd, row_cr_ship-hg:row_cr_ship+hg]


#############
# SEA       
# col_cr = 1106
# row_cr =  991
col_cr_sea = col_cr_ship + size_imaget
row_cr_sea =  row_cr_ship + size_imaget


# creatingf the small imagets
imgVV_sea = VV[col_cr_sea-wd:col_cr_sea+wd, row_cr_sea-hg:row_cr_sea+hg]
imgVH_sea = VH[col_cr_sea-wd:col_cr_sea+wd, row_cr_sea-hg:row_cr_sea+hg]

plt.figure()
plt.imshow(np.abs(imgVV_ship), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VV)))
plt.title("VV for ship ROI")
plt.figure()
plt.imshow(np.abs(imgVH_ship), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VH)))
plt.title("VH for ship ROI")

plt.figure()
plt.imshow(np.abs(imgVV_sea), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VV)))
plt.title("VV for sea ROI")
plt.figure()
plt.imshow(np.abs(imgVH_sea), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VH)))
plt.title("VH for sea ROI")


[freq_axis, doppler_centroid_ship, v_az_ship, power_spectrum_ship] = velocity_estimation(imgVV_ship, lambda_c, prf)

[freq_axis, doppler_centroid_sea, v_az_sea, power_spectrum_sea] = velocity_estimation(imgVV_sea, lambda_c, prf)

vessel_velocity = -(lambda_c/2.0) * (doppler_centroid_ship - doppler_centroid_sea)

print(f"Estimated vessel along-track speed: {v_az_ship:.2f} m/s")
print(f"Estimated sea along-track speed: {v_az_sea:.2f} m/s")
print(f"Estimated vessel along-track speed: {vessel_velocity:.2f} m/s")



