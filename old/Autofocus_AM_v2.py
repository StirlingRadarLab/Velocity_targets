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
# matplotlib.pyplot is for making plots
from numpy.fft import fft, ifft, fftshift, ifftshift

import matplotlib.pyplot as plt
# scipy.signal is for more complex signal processing manipulations
from scipy import signal
# is for manipulating files and filenames
import os
# a useful library for managing path with different OS
from pathlib import Path 

import SAR_Utilities as sar


import pandas as pd
  

import tqdm

# for file copying
import shutil

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


# # part of the image      VIEW 1   
# col_off = 7700
# row_off = 0
# width = 1200
# height = 1200


# part of the image        VIEW 2
col_off = 9700
row_off = 0
width = 2000
height = 2000


# part of the image        VIEW 3
col_off = 11700
row_off = 0
width = 2000
height = 2000



# # part of the image        
# col_off = 20000
# row_off = 1000
# width = 2400
# height = 2400


# # part of the image        
# col_off = 1800*sub_win[0]
# row_off = 450*sub_win[1]
# width = 200*sub_win[0]
# height = 200*sub_win[1]




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


#%% SELECT SHIPS for analysis

size_imaget = 64
wd = int(size_imaget /2)
hg = int(size_imaget /2)




############# VIEW 1
# # LAND     
# col_cr = 938
# row_cr =  645
# #############
# # SHIP 1       
# col_cr_ship = 725
# row_cr_ship =  364
#
#############
# SHIP 2       
col_cr_ship = 788
row_cr_ship =  96
###################################################


############# VIEW 2
#############
# SHIP 1       
col_cr_ship = 134
row_cr_ship = 1055
####################################


############# VIEW 3
#############
# SHIP 1       
col_cr_ship = 547
row_cr_ship = 907
####################################


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
plt.imshow(np.abs(imgVV_ship), cmap = 'gray', vmin = 0, vmax = 7.5*np.nanmean(np.abs(VV)))
plt.title("VV for ship ROI")
plt.figure()
plt.imshow(np.abs(imgVH_ship), cmap = 'gray', vmin = 0, vmax = 7.5*np.nanmean(np.abs(VH)))
plt.title("VH for ship ROI")

# 
# plt.figure()
# plt.imshow(np.abs(imgVV_sea), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VV)))
# plt.title("VV for sea ROI")
# plt.figure()
# plt.imshow(np.abs(imgVH_sea), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VH)))
# plt.title("VH for sea ROI")

# %%




# #%% Retrieving the speed using the doppler centroid shift
# [freq_axis, doppler_centroid_ship, v_az_ship, power_spectrum_ship] = velocity_estimation(imgVV_ship, lambda_c, prf)

# [freq_axis, doppler_centroid_sea, v_az_sea, power_spectrum_sea] = velocity_estimation(imgVV_sea, lambda_c, prf)

# vessel_velocity = -(lambda_c/2.0) * (doppler_centroid_ship - doppler_centroid_sea)

# print(f"Estimated vessel along-track speed: {v_az_ship:.2f} m/s")
# print(f"Estimated sea along-track speed: {v_az_sea:.2f} m/s")
# print(f"Estimated vessel along-track speed: {vessel_velocity:.2f} m/s")

#%% Slant range distance
c = 299792458.0  # speed of light in m/s

# From metadata (example values)
slant_range_time = 5.830090502e-03  # seconds
range_pixel_spacing = 2.329560e+01  # meters

# Derived quantities
R0 = c * slant_range_time / 2       # slant range to first pixel


# %% Autofocus in azimuth
Win = 7

imgVV_FFT = fftshift(fft(imgVV_ship, axis=1), axes=1)
dim = np.shape(imgVV_FFT)

#; Unhamming process
spectrum_medio  = (np.sum(abs(imgVV_FFT), axis=0))/(dim[0]) 
# Smoothing the signal to get a good "unhamming" function

Kernel_Spectrum = np.ones((7,))/7
for kkk in range (0, 10): spectrum_medio = signal.fftconvolve(spectrum_medio, 
                                           Kernel_Spectrum, mode='same')
#;To force that the maximum of my "unhamming" function to be equal 1
spectrum_medio  = spectrum_medio/max(spectrum_medio)
# finding what is the last pixel of noise outside the band
gf0 = np.where(spectrum_medio[0:int(dim[1]/2)] < 0.1)
ind_noise = 0
if len(gf0) > 0: 
    ind_noise = np.max(gf0)
# remove noise outside band in equalisation
spectrum_medio[spectrum_medio < 0.1] = 1  

# plt.figure()
# plt.imshow(np.abs(imgVV_FFT), cmap = 'gray', vmin = 0, vmax = 5.5*np.nanmean(np.abs(imgVV_FFT)))
# plt.title("Magnitude spectrum of the data")

plt.figure()
plt.plot(spectrum_medio)
plt.title("Mean Spectrum of the entire image")

# plt.figure()
# plt.imshow(np.abs(spectrum_corr), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(imgVV_FFT)))
# plt.title("Magnitude spectrum AFTER removing Hamming")


# Function to compute entropy-based sharpness
def image_entropy(img, bins=256):
    mag = np.abs(img)
    hist, _ = np.histogram(mag, bins=bins, range=(mag.min(), mag.max()), density=False)
    hist = hist.astype(float)
    hist /= hist.sum()           # normalize so sum = 1
    hist = hist[hist > 0]        # remove zeros to avoid log(0)
    H = -np.sum(hist * np.log2(hist))
    return H

# Function to refocus an image given parameters
def refocus_image(img_FFT, max_val, centre):
    dim = img_FFT.shape
    x = np.linspace(0, 1, dim[0])
    quadratic = (max_val) * (1 - (x - centre)**2)
    phase_corr = np.exp(-1j * quadratic * (2*np.pi/lambda_c)/R0)
    # img_corr_FFT = img_FFT * phase_corr[:, np.newaxis]
    img_corr_FFT = img_FFT * phase_corr
    img_corr = ifft(ifftshift(img_corr_FFT, axes=1), axis=1)
    
    # plt.figure()
    # plt.plot(quadratic)
    # plt.title("Quadratic function")
    
    return img_corr


# ###############################################################
################## THIS BIT TESTS IT WITH A DUMMY QUADRATIC
min_val = 0        # start value
max_val = 50       # end value
centre = 50
# Quadratic array (parabolic increase)
# normalized from 0 -> 1, then scaled to min_val -> max_val
x = np.linspace(0, 1, dim[0])
quadratic = min_val + (max_val - min_val) * (1 - (x-centre)**2)

plt.figure()
plt.plot(quadratic)
plt.title("Quadratic function")

imgVV_FFT_corr = imgVV_FFT*np.exp(-1j*quadratic*(2*np.pi/lambda_c)/R0)

# imgVV_FFT_corr = imgVV_FFT*np.exp(-1j*quadratic*(2*np.pi/lambda_c))

imgVV_corr   = ifft(ifftshift(imgVV_FFT_corr, axes=1), axis=1)
imgVV_nocorr = ifft(ifftshift(imgVV_FFT, axes=1), axis=1)

plt.figure()
plt.imshow(np.abs(imgVV_corr), cmap = 'gray', vmin = 0, vmax = 7.5*np.nanmean(np.abs(imgVV_corr)))
plt.title("Magnitude of image AFTER focusing")

plt.figure()
plt.imshow(np.abs(imgVV_nocorr), cmap = 'gray', vmin = 0, vmax = 7.5*np.nanmean(np.abs(imgVV_nocorr)))
plt.title("Magnitude of image BEFORE focusing")

H_org = image_entropy(imgVV_nocorr, bins=256)
H_cor = image_entropy(imgVV_corr, bins=256)
print(f"Entropy for original: {H_org:.3f}")
print(f"Entropy for refocused: {H_cor:.3f}")


##########################################################


#%%
# from scipy.stats import entropy


# searcing parameters
grid_size = 50
max_vals_ext = 1.15
min_value = 0.9
centers_ext = 200
# vectors to swip maximum and centre of the quadratic
max_vals = np.linspace(0, max_vals_ext, grid_size)
centres = np.linspace(-centers_ext, centers_ext, grid_size)

best_entropy = np.inf
best_params = None

H = np.zeros([grid_size, grid_size])  

for i in range(len(max_vals)):
    mv = max_vals[i]
    for j in range(len(centres)):
        c = centres[j]
        img_corr = refocus_image(imgVV_FFT, mv, c)
        H[i, j] = image_entropy(np.abs(img_corr))  
        if H[i, j] < best_entropy:
            best_entropy = H[i, j]
            best_params = (mv, c)

print(f"Best parameters: max_val={best_params[0]:.3f}, centre={best_params[1]:.3f}, entropy={best_entropy:.3f}")

# Generate best-focused image
img_best = refocus_image(imgVV_FFT, *best_params)

# Show before/after
plt.figure()
plt.imshow(np.abs(imgVV_ship), cmap = 'gray', vmin = 0, vmax = 7.5*np.nanmean(np.abs(imgVV_ship)))
plt.title("Before focusing")

plt.figure()
plt.imshow(np.abs(img_best), cmap = 'gray', vmin = 0, vmax = 7.5*np.nanmean(np.abs(imgVV_ship)))
plt.title("After autofocusing")
plt.show()

plt.figure()
plt.imshow(H, 
           cmap='gray', 
           vmin=np.min(H), 
           vmax=np.max(H),
           extent=[centres[0], centres[-1], max_vals[0], max_vals[-1]],  # x_min, x_max, y_min, y_max
           origin='lower',   # ensures y-axis starts from min_val at bottom
           aspect='auto')    # avoids stretching
plt.colorbar(label='Entropy')
plt.xlabel('centre')
plt.ylabel('max_val')
plt.title('Entropy map')
plt.show()


# # %% sublook detectors 
# # Line by line, we correct the spectrum to get the original one (without windowing!!!)    
# spectrum_corr = np.zeros(dim, dtype=np.complex64) 
# for jjj in range (0, dim[0]-1):
#     spectrum_corr[jjj:] = imgVV_FFT[jjj:]/spectrum_medio
    
   
# spectrum1 = np.zeros(dim, dtype=np.complex64)
# spectrum2 = np.zeros(dim, dtype=np.complex64)    
# spectrum1[:, ind_noise:int(dim[1]/2)] = spectrum_corr[:, ind_noise:int(dim[1]/2)]
# spectrum2[:, int(dim[1]/2):int(dim[1])-ind_noise] = spectrum_corr[:,int(dim[1]/2):int(dim[1])-ind_noise]
# # now cenering the spetrum so that they overlap in frequency
# spectrum1 = np.roll(spectrum1, int(dim[1]/4-ind_noise/2), axis=1)
# spectrum2 = np.roll(spectrum2, -int(dim[1]/4-ind_noise/2), axis=1)

# data1 = ifft(ifftshift(spectrum1, axes=1), axis=1)
# data2 = ifft(ifftshift(spectrum2, axes=1), axis=1)
# # data2 = np.fft.ifftn(np.roll(spectrum2, -int(dim[1]/4+n_smp_extra/4), axis=1), axes=[1])
 
# winFilterUngarded = np.ones((Win,Win),np.float32)/(Win ** 2) #without guard windows

# SubNum = signal.convolve2d(data1*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
# SubDen1 = signal.convolve2d(data1*np.conj(data1), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
# SubDen2 = signal.convolve2d(data2*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)

# SubCorr = SubNum 
# SubCohe = abs(SubNum/np.sqrt(abs(SubDen1)*abs(SubDen2)))

# # plt.figure()
# # plt.imshow(np.abs(spectrum1), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(imgVV_FFT)))
# # plt.title("Magnitude of FIRST portion of spectrum")

# # plt.figure()
# # plt.imshow(np.abs(spectrum2), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(imgVV_FFT)))
# # plt.title("Magnitude of SECOND portion of spectrum")

# # plt.figure()
# # plt.imshow(np.abs(data1), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(data1)))
# # plt.title("Magnitude of FIRST subaperture")

# # plt.figure()
# # plt.imshow(np.abs(data2), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(data2)))
# # plt.title("Magnitude of SECOND subaperture")

# plt.figure()
# plt.imshow(np.abs(SubCohe), cmap = 'gray', vmin = 0, vmax = 1)
# plt.title("VV Sub Coherence image")
# plt.figure()
# plt.imshow(np.abs(SubCorr), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(SubCorr)))
# plt.title("VH Sub Correlation")