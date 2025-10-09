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

#############
# SHIP 1       
col_cr = 725
row_cr =  364


# creatingf the small imagets
imgVV = VV[col_cr-wd:col_cr+wd, row_cr-hg:row_cr+hg]
imgVH = VH[col_cr-wd:col_cr+wd, row_cr-hg:row_cr+hg]


plt.figure()
plt.imshow(np.abs(imgVV), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VV)))
plt.title("VV for ship ROI")
plt.figure()
plt.imshow(np.abs(imgVH), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(VH)))
plt.title("VH for ship ROI")




#%% retrivial using traditional processing


# Wavelength from frequency
c = 299792458.0
lambda_c = c / center_freq


# Step 1: Compute azimuth FFT (along-track is typically the rows)
azimuth_fft = np.fft.fftshift(np.fft.fft(imgVV, axis=0), axes=0)
power_spectrum = np.mean(np.abs(azimuth_fft)**2, axis=1)


# Step 2: Find Doppler centroid (frequency bin of peak energy)
N_az = power_spectrum.shape[0]
df = prf / N_az # azimuth frequency resolution [Hz]


freq_axis = (np.arange(N_az) - N_az//2) * df


doppler_centroid = freq_axis[np.argmax(power_spectrum)]


# Step 3: Convert Doppler centroid shift into azimuth velocity
v_az = -(lambda_c/2.0) * doppler_centroid


print(f"PRF: {prf:.2f} Hz")
print(f"Center frequency: {center_freq/1e9:.3f} GHz (λ={lambda_c:.3f} m)")
print(f"Estimated Doppler centroid: {doppler_centroid:.2f} Hz")
print(f"Estimated vessel along-track speed: {v_az:.2f} m/s")


# %% VISUALISATION

# 1. Power spectrum with Doppler centroid marker
plt.figure(figsize=(8,4))
plt.plot(freq_axis, 10*np.log10(power_spectrum/np.max(power_spectrum)), label="Power Spectrum (dB)")
plt.axvline(doppler_centroid, color='r', linestyle='--', label=f"Doppler Centroid = {doppler_centroid:.1f} Hz")
plt.xlabel("Azimuth Frequency [Hz]")
plt.ylabel("Normalized Power [dB]")
plt.title("Azimuth FFT Power Spectrum")
plt.legend()
plt.grid(True)

# 2. Show 2D FFT magnitude
plt.figure(figsize=(8,6))
plt.imshow(20*np.log10(np.abs(azimuth_fft)+1e-6), aspect='auto', cmap='viridis',
           extent=[0, VV.shape[1], freq_axis[0], freq_axis[-1]])
plt.colorbar(label="Magnitude [dB]")
plt.xlabel("Range Pixels")
plt.ylabel("Azimuth Frequency [Hz]")
plt.title("Azimuth FFT Magnitude (Range vs Azimuth Freq)")

# 3. Example azimuth cut (time domain vs FFT domain)
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(np.real(VV[:, VV.shape[1]//2]))
plt.title("Example Azimuth Line (Time Domain)")
plt.xlabel("Azimuth Samples")
plt.ylabel("Amplitude")

plt.subplot(1,2,2)
plt.plot(freq_axis, 10*np.log10(np.abs(azimuth_fft[:, VV.shape[1]//2])**2 + 1e-12))
plt.title("FFT of Example Azimuth Line")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")

plt.tight_layout()
plt.show()

