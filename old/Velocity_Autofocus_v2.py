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





# #%% Retrieving the speed using the doppler centroid shift
# [freq_axis, doppler_centroid_ship, v_az_ship, power_spectrum_ship] = velocity_estimation(imgVV_ship, lambda_c, prf)

# [freq_axis, doppler_centroid_sea, v_az_sea, power_spectrum_sea] = velocity_estimation(imgVV_sea, lambda_c, prf)

# vessel_velocity = -(lambda_c/2.0) * (doppler_centroid_ship - doppler_centroid_sea)

# print(f"Estimated vessel along-track speed: {v_az_ship:.2f} m/s")
# print(f"Estimated sea along-track speed: {v_az_sea:.2f} m/s")
# print(f"Estimated vessel along-track speed: {vessel_velocity:.2f} m/s")



# %% Autofocus in azimuth



# ----------------------
# Helpers
# ----------------------
def velocity_to_doppler(v_radial_m_s, wavelength_m):
    return 2.0 * v_radial_m_s / wavelength_m

def apply_phase_correction_time(tile, az_times, fd_hz, extra_phase_time=None):
    """
    Apply phase correction in slow-time (time-domain).
    - tile: (N_az, N_rg)
    - az_times: array length N_az
    - fd_hz: linear Doppler freq to remove (Hz)
    - extra_phase_time: optional func(az_times) -> phi(t) in radians to subtract
    """
    phase = -2.0 * np.pi * fd_hz * az_times   # negative because we "remove" fd
    if extra_phase_time is not None:
        phase = phase - extra_phase_time(az_times)
    corr = np.exp(1j * phase)[:, None]
    return tile * corr

def apply_phase_correction_freq(tile, prf, fd_hz=0.0, phi_freq_func=None):
    """
    Apply correction in Doppler (frequency) domain:
      1) FFT along azimuth (no shift or with shift, but handle consistently)
      2) multiply each Doppler bin by exp(-j * Psi(f))
      3) inverse FFT
    - tile: (N_az, N_rg)
    - prf: sampling rate in slow-time
    - fd_hz: optional linear Doppler removal (applied in time domain BEFORE FFT for accuracy)
    - phi_freq_func: function freq_axis -> Psi(f) [radians]. If None, no extra freq-phase is applied.
    """
    N_az = tile.shape[0]
    # optional: remove linear Doppler first (do time-domain multiplication)
    if fd_hz != 0.0:
        az_times = (np.arange(N_az) - N_az/2.0) / prf
        tile = apply_phase_correction_time(tile, az_times, fd_hz)

    # FFT (we'll use fftshifted freq axis for easier reasoning)
    S = np.fft.fftshift(np.fft.fft(tile, axis=0), axes=0)  # shape (N_az, N_rg)
    # frequency axis (centered)
    df = prf / N_az
    freq_axis = (np.arange(N_az) - N_az//2) * df  # Hz

    if phi_freq_func is not None:
        # compute phase for each Doppler bin (array length N_az)
        psi = phi_freq_func(freq_axis)  # radians, shape (N_az,)
        # multiply each doppler row by exp(-j*psi)
        S = S * np.exp(-1j * psi)[:, None]

    # inverse FFT
    corrected = np.fft.ifft(np.fft.ifftshift(S, axes=0), axis=0)
    return corrected

def entropy_metric(img, nbins=256, eps=1e-12):
    intensity = np.abs(img)**2
    flat = intensity.ravel()
    if flat.max() == 0:
        return np.inf
    hist, _ = np.histogram(flat, bins=nbins, density=True)
    p = hist + eps
    p /= p.sum()
    return -np.sum(p * np.log2(p))


# Simple time-domain search (remove linear Doppler only)
def search_velocity_time(tile, prf, wavelength, v_min, v_max, n_steps=61):
    N_az = tile.shape[0]
    az_times = (np.arange(N_az) - N_az/2.0) / prf
    v_grid = np.linspace(v_min, v_max, n_steps)
    ent = np.zeros_like(v_grid)
    for i, v in enumerate(v_grid):
        fd = velocity_to_doppler(v, wavelength)
        corr = apply_phase_correction_time(tile, az_times, fd_hz=fd)
        ent[i] = entropy_metric(corr)
    idx = np.argmin(ent)
    return v_grid[idx], v_grid, ent

# Frequency-domain search where we allow a quadratic spectral-phase correction Psi(f) = alpha * f^2
# We assume the dominant defocus is quadratic in time -> corresponds to quadratic in freq (up to constants).
def search_velocity_freq(tile, prf, wavelength, v_min, v_max, n_steps=61, alpha_max=1e-6):
    N_az = tile.shape[0]
    df = prf / N_az
    v_grid = np.linspace(v_min, v_max, n_steps)
    ent = np.zeros_like(v_grid)
    # freq axis for computing Psi(f) inside helper
    freq_axis = (np.arange(N_az) - N_az//2) * df

    for i, v in enumerate(v_grid):
        fd = velocity_to_doppler(v, wavelength)
        # choose a spectral-phase function Psi(f): try a quadratic with alpha chosen by scanning or a guess.
        # Here we pick alpha proportional to fd to show interplay; in practice you'd estimate Psi(f).
        def psi_of_f(f):
            alpha = alpha_max  # tune or make it another search dimension
            return alpha * (f**2)
        corr = apply_phase_correction_freq(tile, prf, fd_hz=0.0, phi_freq_func=psi_of_f)
        ent[i] = entropy_metric(corr)
    idx = np.argmin(ent)
    return v_grid[idx], v_grid, ent




#%%#########  SIMULATIONS ############################
# ----------------------
# Params (example)
# ----------------------
N_az = 64
N_rg = 128
prf = 486.48631029955294
c = 299792458.0
center_freq = 5.405e9
wavelength = c / center_freq
true_velocity = 8.5  # m/s

# ----------------------
# Simulation: multi-scatterer "ship" + quadratic defocus
# ----------------------
A = np.zeros((N_az, N_rg), dtype=np.complex64)
for offset in [-8, -3, 0, 3, 8]:
    A[N_az//2 + offset, N_rg//2] = 1.0 + 0j

az = np.arange(N_az)
rg = np.arange(N_rg)
az_win = np.exp(-((az - N_az//2)/(N_az*0.18))**2)
rg_win = np.exp(-((rg - N_rg//2)/(N_rg*0.06))**2)
psf = az_win[:, None] * rg_win[None, :]
clean = A * psf

az_times = (np.arange(N_az) - N_az/2.0) / prf
fd_true = velocity_to_doppler(true_velocity, wavelength)

# Create a QUADRATIC defocus in time (simulate motion error)
kappa = 400.0  # tune this to control blur strength; units such that pi*kappa*t^2 is radians
phase_error_time = (np.pi * kappa * az_times**2)  # radians
moved = clean * np.exp(1j * 2.0 * np.pi * fd_true * az_times)[:, None] * np.exp(1j * phase_error_time)[:, None]

rng = np.random.default_rng(1)
noise_level = 0.01
moved_noisy = moved + noise_level * (rng.standard_normal(moved.shape) + 1j*rng.standard_normal(moved.shape))

# ----------------------
# Search variants:
#   - time domain correction for linear fd (works for linear)
#   - freq domain correction for a quadratic spectral-phase (we'll craft Psi(f) corresponding to the time quadratic)
# ----------------------

# Run a couple of experiments:
v_time, vgrid_time, ent_time = search_velocity_time(moved_noisy, prf, wavelength, 0.0, 20.0, n_steps=101)
v_freq, vgrid_freq, ent_freq = search_velocity_freq(moved_noisy, prf, wavelength, 0.0, 20.0, n_steps=101, alpha_max=1e-6)

print("True v:", true_velocity)
print("Time-domain best v:", v_time)
print("Freq-domain best v (with quadratic Psi):", v_freq)

# Plot to inspect
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.title("Before (magnitude)")
plt.imshow(np.abs(moved_noisy), origin='lower', aspect='auto')
plt.colorbar()
plt.subplot(1,3,2)
plt.title("After time correction (v_time)")
az_times = (np.arange(N_az) - N_az/2.0) / prf
best_corr_time = apply_phase_correction_time(moved_noisy, az_times, velocity_to_doppler(v_time, wavelength))
plt.imshow(np.abs(best_corr_time), origin='lower', aspect='auto')
plt.colorbar()
plt.subplot(1,3,3)
plt.title("After freq correction (quadratic Psi)")
best_corr_freq = apply_phase_correction_freq(moved_noisy, prf, fd_hz=0.0,
                                            phi_freq_func=lambda f: 1e-6*(f**2))
plt.imshow(np.abs(best_corr_freq), origin='lower', aspect='auto')
plt.colorbar()
plt.show()

plt.figure()
plt.plot(vgrid_time, ent_time, label='time search')
plt.plot(vgrid_freq, ent_freq, label='freq search')
plt.axvline(true_velocity, color='k', linestyle='--', label='true v')
plt.legend()
plt.xlabel('Velocity (m/s)')
plt.ylabel('Entropy')
plt.grid(True)
plt.show()



#%%#########  REAL DATA ############################
# ----------------------
# Params (example)
# ----------------------
N_az = np.shape(imgVH_ship)[0]
N_rg = np.shape(imgVH_ship)[1]
# prf = 486.48631029955294
# c = 299792458.0
# center_freq = 5.405e9
wavelength = c / center_freq
# true_velocity = 8.5  # m/s

# # ----------------------
# # Simulation: multi-scatterer "ship" + quadratic defocus
# # ----------------------
# A = np.zeros((N_az, N_rg), dtype=np.complex64)
# for offset in [-8, -3, 0, 3, 8]:
#     A[N_az//2 + offset, N_rg//2] = 1.0 + 0j

# az = np.arange(N_az)
# rg = np.arange(N_rg)
# az_win = np.exp(-((az - N_az//2)/(N_az*0.18))**2)
# rg_win = np.exp(-((rg - N_rg//2)/(N_rg*0.06))**2)
# psf = az_win[:, None] * rg_win[None, :]
# clean = A * psf

# az_times = (np.arange(N_az) - N_az/2.0) / prf
# fd_true = velocity_to_doppler(true_velocity, wavelength)

# # Create a QUADRATIC defocus in time (simulate motion error)
# kappa = 400.0  # tune this to control blur strength; units such that pi*kappa*t^2 is radians
# phase_error_time = (np.pi * kappa * az_times**2)  # radians
# moved = clean * np.exp(1j * 2.0 * np.pi * fd_true * az_times)[:, None] * np.exp(1j * phase_error_time)[:, None]

# rng = np.random.default_rng(1)
# noise_level = 0.01
# moved_noisy = moved + noise_level * (rng.standard_normal(moved.shape) + 1j*rng.standard_normal(moved.shape))

# ----------------------
# Search variants:
#   - time domain correction for linear fd (works for linear)
#   - freq domain correction for a quadratic spectral-phase (we'll craft Psi(f) corresponding to the time quadratic)
# ----------------------

# Run a couple of experiments:
v_time, vgrid_time, ent_time = search_velocity_time(imgVH_ship, prf, wavelength, -20.0, 20.0, n_steps=101)
v_freq, vgrid_freq, ent_freq = search_velocity_freq(imgVH_ship, prf, wavelength, -20.0, 20.0, n_steps=101, alpha_max=1e-6)

print("True v:", true_velocity)
print("Time-domain best v:", v_time)
print("Freq-domain best v (with quadratic Psi):", v_freq)

# Plot to inspect
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.title("Before (magnitude)")
plt.imshow(np.abs(imgVH_ship), origin='lower', aspect='auto')
plt.colorbar()
plt.subplot(1,3,2)
plt.title("After time correction (v_time)")
az_times = (np.arange(N_az) - N_az/2.0) / prf

best_corr_time = apply_phase_correction_time(imgVH_ship, az_times, velocity_to_doppler(v_time, wavelength))
plt.imshow(np.abs(best_corr_time), origin='lower', aspect='auto')
plt.colorbar()
plt.subplot(1,3,3)
plt.title("After freq correction (quadratic Psi)")


best_corr_freq = apply_phase_correction_freq(imgVH_ship, prf, fd_hz=0.0,
                                            phi_freq_func=lambda f: 1e-6*(f**2))
# best_corr_freq = apply_phase_correction_freq(imgVH_ship, prf, fd_hz=0.0,
#                                             phi_freq_func=lambda f: 1e6*(f**2))

plt.imshow(np.abs(best_corr_freq), origin='lower', aspect='auto')
plt.colorbar()
plt.show()

plt.figure()
plt.plot(vgrid_time, ent_time, label='time search')
plt.plot(vgrid_freq, ent_freq, label='freq search')
plt.axvline(true_velocity, color='k', linestyle='--', label='true v')
plt.legend()
plt.xlabel('Velocity (m/s)')
plt.ylabel('Entropy')
plt.grid(True)
plt.show()
