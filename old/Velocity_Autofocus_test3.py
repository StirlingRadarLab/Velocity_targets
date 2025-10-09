



# # ----------------------------
# # If run as script, show the demo
# # ----------------------------
# if __name__ == "__main__":
#     img_f, img_r, info = demo_on_synthetic()
#     print("Fitted polynomial coeffs (highest->lowest):", info['poly_coeffs'])






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




import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt

# ----------------------------
# Sentinel-1 default params
# ----------------------------
C = 299792458.0
FC = 5.405e9          # Sentinel-1 C-band center (Hz)
LAMBDA = C / FC
PRF_DEFAULT = 1716.0  # Hz (approx)
# For a small imagette these defaults are OK; replace if you know exact numbers.

# ----------------------------
# Utility: polynomial fit to unwrapped phase
# ----------------------------
def fit_phase_polynomial(t, phase, order=2):
    """
    Fit a polynomial of given order to unwrapped phase vs time.
    Returns coefficients (highest->lowest) compatible with np.polyval.
    phase must be unwrapped (np.unwrap).
    """
    # ensure shapes
    t = np.asarray(t).ravel()
    phase = np.unwrap(np.asarray(phase).ravel())
    # Fit polynomial: phase ≈ p[0]*t^order + ... + p[-1]
    p = np.polyfit(t, phase, order)
    return p

# ----------------------------
# Main Doppler-parameter refocusing routine
# ----------------------------
def doppler_refocus(
    img,                     # complex 2D array: azimuth x range (small imagette)
    prf=PRF_DEFAULT,
    fc=FC,
    poly_order=2,            # fit order (2 => quadratic), increase if needed
    search_range_halfwidth=2,# range bins around peak to average phase (small int)
    undo_fft_azimuth=True,   # if True, try to invert simple FFT-based azimuth compression
    do_crop_peak=True        # if True, auto-find strongest range bin; else user must provide
):
    """
    Estimate Doppler phase law from img and remove it to refocus a moving target.
    Returns:
      img_refocused : complex 2D array (same shape as img)
      info dict with fitted polynomial and used slow-time vector
    Notes:
      - img must be complex-valued. If it's magnitude-only, this cannot work.
      - If the imagette was produced with a simple FFT azimuth compressor, set undo_fft_azimuth=True.
        If you have SLC (slow-time lines), provide img as SLC and set undo_fft_azimuth=False.
    """
    if not np.iscomplexobj(img):
        raise ValueError("img must be complex-valued (contain phase).")

    na, nr = img.shape
    t_az = (np.arange(na) - na//2) / prf  # slow-time vector centered

    # Step 1: (Optional) Undo simple FFT-based azimuth compression to estimate slow-time signal S_est
    # If the image was formed by img = ifft( fft(S_slc, axis=0) ) then S_est ~= fft(img, axis=0)
    if undo_fft_azimuth:
        # Recover approx slow-time signal
        S_est = fftshift(fft(ifftshift(img, axes=0), axis=0), axes=0)
    else:
        # assume input is already SLC slow-time x range (not azimuth-compressed)
        S_est = img.copy()

    # Step 2: find the target range bin (auto) or use provided
    # Use the azimuth center line magnitude to locate the strongest range bin
    center_line = np.abs(S_est[na//2, :])
    peak_idx = int(np.argmax(center_line))
    # Build list of range bins to average phase across (small neighborhood)
    r0 = peak_idx
    rmin = max(0, r0 - search_range_halfwidth)
    rmax = min(nr, r0 + search_range_halfwidth + 1)

    # Step 3: extract azimuth phase vs time by averaging over range neighborhood
    # compute complex azimuth signal by summing/averaging over range bins
    s_az = np.sum(S_est[:, rmin:rmax], axis=1) / (rmax - rmin)
    # if signal is weak, warn
    if np.max(np.abs(s_az)) < 1e-8:
        print("Warning: extracted azimuth signal is very small; check input SLC or img content.")

    phase_az = np.angle(s_az)         # wrapped phase
    phase_unwrapped = np.unwrap(phase_az)  # unwrap

    # Step 4: fit polynomial to unwrapped phase vs t
    p = fit_phase_polynomial(t_az, phase_unwrapped, order=poly_order)

    # Evaluate fitted phase law
    fitted_phase = np.polyval(p, t_az)  # radians

    # Step 5: construct phase correction (we want to remove fitted phase)
    # Note: the fitted phase is the total azimuth phase of the target; to remove only motion-induced error
    # you might subtract a reference (e.g., fitted_phase at center) to avoid range shift.
    # We'll remove the *variation* around center so the target stays in place:
    fitted_phase_centered = fitted_phase - fitted_phase[na//2]

    # phase correction factor per slow-time
    phase_corr = np.exp(-1j * fitted_phase_centered)   # multiply slow-time lines by this to remove phase

    # Step 6: apply phase correction to S_est (per azimuth scalar)
    S_corr = S_est * phase_corr[:, None]

    # Step 7: recompress azimuth (if we undid FFT earlier)
    if undo_fft_azimuth:
        # inverse of step 1: img_refocused = ifftshift(ifft(fftshift(S_corr, axes=0), axis=0), axes=0)
        img_refocused = fftshift(ifft(ifftshift(S_corr, axes=0), axis=0), axes=0)
    else:
        img_refocused = S_corr

    info = {
        "poly_coeffs": p,
        "fitted_phase": fitted_phase,
        "fitted_phase_centered": fitted_phase_centered,
        "t_az": t_az,
        "peak_range_index": peak_idx,
        "range_window": (rmin, rmax)
    }
    return img_refocused, info

# ----------------------------
# Example / demo for a 64x64 imagette
# ----------------------------
def refocus_sentinel1_imagette(img, prf=PRF_DEFAULT, fc=FC,
                                poly_order=2, search_range_halfwidth=1,
                                undo_fft_azimuth=True):
    """
    Doppler-parameter refocusing of a 2D complex Sentinel-1 imagette (64x64).

    Parameters:
    -----------
    img : 2D complex array (azimuth x range)
        Focused imagette from Sentinel-1 (complex values).
    prf : float
        Sentinel-1 azimuth PRF (Hz). Default 1716 Hz.
    fc : float
        Carrier frequency (Hz). Default Sentinel-1 C-band 5.405 GHz.
    poly_order : int
        Order of polynomial to fit Doppler phase (default 2).
    search_range_halfwidth : int
        Half-width of range bins around peak to average for phase extraction.
    undo_fft_azimuth : bool
        True if the imagette was compressed using simple FFT; False if SLC.

    Returns:
    --------
    img_refocused : 2D complex array (same shape as img)
        Doppler-refocused imagette.
    info : dict
        Dictionary with fitted polynomial, fitted phase, and used range bins.
    """
    if not np.iscomplexobj(img):
        raise ValueError("Input img must be complex-valued.")

    na, nr = img.shape
    t_az = (np.arange(na) - na//2) / prf  # slow-time vector

    # Step 1: (Optional) Undo FFT azimuth compression to estimate slow-time signal
    if undo_fft_azimuth:
        S_est = fftshift(fft(ifftshift(img, axes=0), axis=0), axes=0)
    else:
        S_est = img.copy()

    # Step 2: Find strongest range bin (assume target roughly centered)
    center_line = np.abs(S_est[na//2, :])
    peak_idx = int(np.argmax(center_line))
    rmin = max(0, peak_idx - search_range_halfwidth)
    rmax = min(nr, peak_idx + search_range_halfwidth + 1)

    # Step 3: Extract azimuth signal by averaging over range bins
    s_az = np.mean(S_est[:, rmin:rmax], axis=1)
    phase_unwrapped = np.unwrap(np.angle(s_az))

    # Step 4: Fit polynomial to azimuth phase
    p = np.polyfit(t_az, phase_unwrapped, poly_order)
    fitted_phase = np.polyval(p, t_az)
    fitted_phase_centered = fitted_phase - fitted_phase[na//2]

    # Step 5: Construct phase correction
    phase_corr = np.exp(-1j * fitted_phase_centered)
    S_corr = S_est * phase_corr[:, None]

    # Step 6: Recompress azimuth (if we undid FFT)
    if undo_fft_azimuth:
        img_refocused = fftshift(ifft(ifftshift(S_corr, axes=0), axis=0), axes=0)
    else:
        img_refocused = S_corr

    info = {
        "poly_coeffs": p,
        "fitted_phase": fitted_phase,
        "fitted_phase_centered": fitted_phase_centered,
        "peak_range_index": peak_idx,
        "range_window": (rmin, rmax),
        "t_az": t_az
    }

    return img_refocused, info




# img is your 64x64 complex Sentinel-1 patch
# img = ... (load from your data)

img_refocused, info = refocus_sentinel1_imagette(imgVV_ship)

# Plot comparison
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Imagette")
plt.imshow(np.abs(imgVV_ship), cmap='gray'); plt.axis('off')
plt.subplot(1,2,2)
plt.title("Refocused Imagette")
plt.imshow(np.abs(img_refocused), cmap='gray'); plt.axis('off')
plt.show()

print("Fitted polynomial coefficients (highest->lowest):", info['poly_coeffs'])



#%%
stop
# ---------------------------
# Helper: generate azimuth chirp
# ---------------------------
def generate_azimuth_chirp(na, PRF, az_bw):
    """
    Generate an approximate linear azimuth chirp for demonstration.
    na: number of azimuth samples
    PRF: pulse repetition frequency (Hz)
    az_bw: azimuth bandwidth (Hz)
    Returns complex 1D array of length na
    """
    t = (np.arange(na) - na/2) / PRF
    k = az_bw / (t[-1] - t[0])  # chirp rate (Hz/s)
    chirp = np.exp(1j * np.pi * k * t**2)
    return chirp

# ---------------------------
# Phase-only motion correction applied in azimuth
# ---------------------------
def apply_phase_correction(S_est, t_az, xp, yp, zp, x0, y0, z0, vx, vy, fc=5.405e9):
    c = 299792458.0
    lam = c / fc
    xt = x0 + vx * t_az
    yt = y0 + vy * t_az
    zt = np.ones_like(t_az) * z0
    R_t = np.sqrt((xp - xt)**2 + (yp - yt)**2 + (zp - zt)**2)
    R_ref = R_t[len(t_az)//2]
    phi_corr = (4*np.pi/lam) * (R_ref - R_t)
    az_phase = np.exp(1j*phi_corr)
    return S_est * az_phase[:, None]

# ---------------------------
# Main function for "undo/re-chirp" method
# ---------------------------
def refocus_imagette(img, vx, vy, PRF=1716.0, az_bw=20.0, Vp=7090.0, h=693000.0):
    """
    img: complex 2D imagette (azimuth x range)
    vx, vy: target velocities in m/s
    PRF: azimuth PRF (Hz)
    az_bw: azimuth bandwidth (Hz) for approximate chirp
    Vp: platform velocity (m/s)
    h: platform altitude (m)
    """
    na, nr = img.shape
    t_az = (np.arange(na) - na/2) / PRF

    # 1) Undo simple FFT-based azimuth compression
    S_est = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=0), axis=0), axes=0)

    # 2) Multiply by approximate azimuth chirp ("re-chirp")
    chirp = generate_azimuth_chirp(na, PRF, az_bw)
    S_chirped = S_est * chirp[:, None]

    # 3) Apply phase-only motion correction
    xp = Vp * t_az
    yp = np.zeros_like(t_az)
    zp = np.ones_like(t_az) * h
    S_mc = apply_phase_correction(S_chirped, t_az, xp, yp, zp, x0=0.0, y0=0.0, z0=0.0, vx=vx, vy=vy)

    # 4) De-chirp / recompress azimuth (inverse of step 2)
    S_dechirped = S_mc / chirp[:, None]

    # 5) Undo FFT-shift to get final image
    img_refocused = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(S_dechirped, axes=0), axis=0), axes=0)

    return np.abs(img_refocused)

# ---------------------------
# Demo usage
# ---------------------------
# Example small 64x64 complex imagette (replace with your real data)
img = imgVV_ship

# Test velocities
vx, vy = 0.0, 0.0  # m/s

# Apply "undo/re-chirp" phase correction
img_refocused = refocus_imagette(img, vx, vy)

# Show results
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(np.abs(img), cmap='gray')
plt.title("Original Imagette")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_refocused, cmap='gray')
plt.title(f"Refocused (vx={vx}, vy={vy})")
plt.axis('off')
plt.show()







#%%
stop
# import numpy as np
# import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Phase-only motion compensation (from previous message)
# ---------------------------------------------------------
# import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

def undo_simple_fft_azimuth_compression(img_focused):
    """
    Given img = ifft( fft(S), axis=0 ), recover an estimate of S:
      S_est = ifft( fft(img), axis=0 )
    (Because ifft(fft(·)) is an involution up to shift conventions used consistently.)
    """
    # assume complex img with shape (na, nr)
    S_est = ifft( fft(img_focused, axis=0), axis=0 )
    return S_est

def redo_simple_fft_azimuth_compression(S_slowtime):
    """
    Reapply the simple FFT-based azimuth compression used in the demo:
      img = ifft( fft(S_slowtime), axis=0 )
    """
    img = ifft( fft(S_slowtime, axis=0), axis=0 )
    return img

def phase_only_mc_on_focused_imagette(img, t_az, tau_range,
                                      x0=0.0, y0=0.0, z0=0.0,
                                      vx=0.0, vy=0.0,
                                      fc=5.405e9, c=299792458.0,
                                      xp=None, yp=None, zp=None,
                                      ref_time_index=None):
    """
    Attempt to apply phase-only motion compensation to an already-focused imagette `img`.
    This attempts to invert the simple demo FFT-based azimuth compression, apply the azimuth
    phase correction on slow-time lines, then re-apply the FFT-based compression.
    Works only if img was produced by the same simple FFT/ifft chain and img is complex.
    """
    na, nr = img.shape
    lam = c / fc

    # default sentinel-1 ephemeris like earlier demo (straight-line)
    if xp is None or yp is None or zp is None:
        Vp = 7090.0
        h  = 693000.0
        xp = Vp * t_az
        yp = np.zeros_like(t_az)
        zp = np.ones_like(t_az) * h
    if callable(xp): xp = xp(t_az)
    if callable(yp): yp = yp(t_az)
    if callable(zp): zp = zp(t_az)

    # STEP 1: try to recover slow-time data (estimate of S)
    S_est = undo_simple_fft_azimuth_compression(img)

    # STEP 2: compute the azimuth phase correction using vx,vy and x0,y0
    if ref_time_index is None:
        ref_time_index = na // 2
    xt = x0 + vx * t_az
    yt = y0 + vy * t_az
    zt = np.ones_like(t_az) * z0
    R_t = np.sqrt((xp - xt)**2 + (yp - yt)**2 + (zp - zt)**2)
    R_ref = R_t[ref_time_index]
    phi_corr = (4.0 * np.pi / lam) * (R_ref - R_t)   # radians (na,)
    az_phase = np.exp(1j * phi_corr)
    # apply per-azimuth scalar to all range bins
    S_mc = S_est * az_phase[:, None]

    # STEP 3: reapply simple FFT-based azimuth compression
    img_refocused = redo_simple_fft_azimuth_compression(S_mc)

    return img_refocused, S_est, S_mc

# ---------------------------------------------------------

# ---------------------------------------------------------
# Main demo using Sentinel-1 parameters
# ---------------------------------------------------------
# Suppose `img` is your 64×64 complex Sentinel-1 imagette (already loaded)
# For demonstration, we'll create a placeholder complex image:

img = imgVV_ship

# Dimensions
na, nr = img.shape
PRF = 1716.0                  # Sentinel-1 azimuth PRF (Hz)
fs_range = 42.5e6             # Sentinel-1 range sampling rate (Hz)
t_az = (np.arange(na) - na/2) / PRF        # slow-time vector (s)
tau_range = (np.arange(nr) - nr/2) / fs_range  # fast-time vector (s)

# Known ship motion (example values, adjust to your case)
# vx, vy are horizontal velocities in m/s (across-track, along-track)
vx = 100.0        # across-track (e.g., toward radar)
vy = 50.0        # along-track (parallel to flight)

# suppose `img` is your complex 64x64 imagette and you've created t_az, tau_range as before
img_refocused, S_est, S_mc = phase_only_mc_on_focused_imagette(
    img=img,
    t_az=t_az,
    tau_range=tau_range,
    x0=0.0, y0=0.0, z0=0.0,
    vx=0.0, vy=0.0,         # test zero velocity: result should be (almost) unchanged if original chain matches
    fc=5.405e9
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(np.abs(img), cmap='gray')
ax[0].set_title("Original Imagette (|img|)")
ax[1].imshow(np.abs(img_refocused), cmap='gray')
ax[1].set_title("Refocused Imagette (|img_refocused|)")
for a in ax: a.axis('off')
plt.tight_layout()
plt.show()







#%%
stop
from numpy.fft import fft, ifft, fftshift, ifftshift

def phase_only_motion_compensation(
    S,                 # complex array (na, nr) slow-time x range (range-compressed)
    t_az,              # slow-time array (na,) in seconds (one time per azimuth line)
    tau_range,         # fast-time array (nr,) in seconds (uniform)
    xp=None, yp=None, zp=None,   # platform pos arrays (na,) in meters OR callables xp(t_az)
    x0=0.0, y0=0.0, z0=0.0,      # target reference position at t=0 (meters)
    vx=0.0, vy=0.0,              # target ground-plane velocity (m/s)
    fc=5.3e9,                    # carrier frequency (Hz)
    c=299792458.0,
    ref_time_index=None,         # index in t_az chosen as reference (None -> center index)
    apply_azimuth_refocus=True,  # if True, produce a simple azimuth-refocused image for display
    autofocus_refine=False,      # if True, do a small grid search around (vx,vy) to improve focus
    af_search_radius=2.0,        # m/s search radius each side for autofocus
    af_search_steps=9            # steps per axis (odd)
):
    """
    Returns:
      S_mc : motion-compensated data (na, nr) (phase corrected, no range shifting)
      img_refocused : (na, nr) complex image after simple azimuth FFT refocus (or None)
      vx_used, vy_used : velocity actually used (after autofocus if active)
    """

    na, nr = S.shape
    assert t_az.shape[0] == na
    assert tau_range.shape[0] == nr
    lam = c / fc

    # -- default platform ephemeris (straight-line) if not provided --
    if xp is None or yp is None or zp is None:
        Vp = 150.0    # platform ground speed (m/s) - change to match your system
        h = 8000.0    # platform altitude (m)
        xp = Vp * t_az
        yp = np.zeros_like(t_az)
        zp = np.ones_like(t_az) * h
    if callable(xp): xp = xp(t_az)
    if callable(yp): yp = yp(t_az)
    if callable(zp): zp = zp(t_az)

    # helper to compute phase correction for a given vx,vy
    def compute_S_mc_for_velocity(vx_try, vy_try, ref_idx):
        # target trajectory
        xt = x0 + vx_try * t_az
        yt = y0 + vy_try * t_az
        zt = np.ones_like(t_az) * z0

        # slant range vs slow-time
        R_t = np.sqrt((xp - xt)**2 + (yp - yt)**2 + (zp - zt)**2)    # (na,)
        # choose reference slant-range (single scalar) so we do phase-only correction
        R_ref_val = R_t[ref_idx]   # using chosen reference time; preserves range position

        # compensating phase per azimuth time (radians)
        # we want to remove the variation: phi_corr(t) = + (4*pi/lam) * (R_ref - R_t)
        phi_corr = (4.0 * np.pi / lam) * (R_ref_val - R_t)    # (na,)
        az_phase_factor = np.exp(1j * phi_corr)              # (na,)

        # apply the same per-azimuth scalar to all range bins (preserves range indices)
        S_mc_try = S * az_phase_factor[:, None]  # broadcasting
        return S_mc_try, R_t, R_ref_val

    # select reference time index
    if ref_time_index is None:
        ref_time_index = na // 2

    # optionally perform a small autofocus grid search to refine vx,vy
    vx_used = vx; vy_used = vy
    if autofocus_refine:
        # define grid around initial vx,vy
        vx_vals = np.linspace(vx - af_search_radius, vx + af_search_radius, af_search_steps)
        vy_vals = np.linspace(vy - af_search_radius, vy + af_search_radius, af_search_steps)
        best_metric = -np.inf
        best_pair = (vx, vy)
        # simple focus metric: summed absolute intensity in a small window around expected range/azimuth
        # Since we don't know exact pixel, use global sharpness: ratio of L4 to (L2^2) or inverse entropy
        def focus_metric(img):
            # normalized sharpness: sum(|img|^4) / (sum(|img|^2)^2)  (higher -> sharper)
            magsq = np.abs(img)**2
            s2 = magsq.sum()
            s4 = (magsq**2).sum()
            if s2 == 0: return -np.inf
            return s4 / (s2*s2 + 1e-20)

        for vx_try in vx_vals:
            for vy_try in vy_vals:
                S_mc_try, _, _ = compute_S_mc_for_velocity(vx_try, vy_try, ref_time_index)
                # quick azimuth refocus using FFT (simple) to get an image to evaluate
                # Note: this is not the production azimuth MF; but sufficient for autofocus metric
                img_try = ifft(fft(S_mc_try, axis=0), axis=0)   # az FFT then IFFT (simple demonstration)
                metric = focus_metric(img_try)
                if metric > best_metric:
                    best_metric = metric
                    best_pair = (vx_try, vy_try)
        vx_used, vy_used = best_pair

    # compute final corrected data using vx_used, vy_used
    S_mc, R_t_used, R_ref_val = compute_S_mc_for_velocity(vx_used, vy_used, ref_time_index)

    # optional simple azimuth refocus for display
    img_refocused = None
    if apply_azimuth_refocus:
        # simple method: apply FFT over azimuth (centered), then inverse to get focused result
        # Replace with real azimuth matched filter if available.
        img_refocused = ifft(fft(S_mc, axis=0), axis=0)

    return S_mc, img_refocused, vx_used, vy_used


#%%

# # ----------------------
# # Helpers
# # ----------------------
# def velocity_to_doppler(v_radial_m_s, wavelength_m):
#     return 2.0 * v_radial_m_s / wavelength_m

# def apply_phase_correction_time(tile, az_times, fd_hz, extra_phase_time=None):
#     """
#     Apply phase correction in slow-time (time-domain).
#     - tile: (N_az, N_rg)
#     - az_times: array length N_az
#     - fd_hz: linear Doppler freq to remove (Hz)
#     - extra_phase_time: optional func(az_times) -> phi(t) in radians to subtract
#     """
#     phase = -2.0 * np.pi * fd_hz * az_times   # negative because we "remove" fd
#     if extra_phase_time is not None:
#         phase = phase - extra_phase_time(az_times)
#     corr = np.exp(1j * phase)[:, None]
#     return tile * corr

# def apply_phase_correction_freq(tile, prf, fd_hz=0.0, phi_freq_func=None):
#     """
#     Apply correction in Doppler (frequency) domain:
#       1) FFT along azimuth (no shift or with shift, but handle consistently)
#       2) multiply each Doppler bin by exp(-j * Psi(f))
#       3) inverse FFT
#     - tile: (N_az, N_rg)
#     - prf: sampling rate in slow-time
#     - fd_hz: optional linear Doppler removal (applied in time domain BEFORE FFT for accuracy)
#     - phi_freq_func: function freq_axis -> Psi(f) [radians]. If None, no extra freq-phase is applied.
#     """
#     N_az = tile.shape[0]
#     # optional: remove linear Doppler first (do time-domain multiplication)
#     if fd_hz != 0.0:
#         az_times = (np.arange(N_az) - N_az/2.0) / prf
#         tile = apply_phase_correction_time(tile, az_times, fd_hz)

#     # FFT (we'll use fftshifted freq axis for easier reasoning)
#     S = np.fft.fftshift(np.fft.fft(tile, axis=0), axes=0)  # shape (N_az, N_rg)
#     # frequency axis (centered)
#     df = prf / N_az
#     freq_axis = (np.arange(N_az) - N_az//2) * df  # Hz

#     if phi_freq_func is not None:
#         # compute phase for each Doppler bin (array length N_az)
#         psi = phi_freq_func(freq_axis)  # radians, shape (N_az,)
#         # multiply each doppler row by exp(-j*psi)
#         S = S * np.exp(-1j * psi)[:, None]

#     # inverse FFT
#     corrected = np.fft.ifft(np.fft.ifftshift(S, axes=0), axis=0)
#     return corrected

# def entropy_metric(img, nbins=256, eps=1e-12):
#     intensity = np.abs(img)**2
#     flat = intensity.ravel()
#     if flat.max() == 0:
#         return np.inf
#     hist, _ = np.histogram(flat, bins=nbins, density=True)
#     p = hist + eps
#     p /= p.sum()
#     return -np.sum(p * np.log2(p))


# # Simple time-domain search (remove linear Doppler only)
# def search_velocity_time(tile, prf, wavelength, v_min, v_max, n_steps=61):
#     N_az = tile.shape[0]
#     az_times = (np.arange(N_az) - N_az/2.0) / prf
#     v_grid = np.linspace(v_min, v_max, n_steps)
#     ent = np.zeros_like(v_grid)
#     for i, v in enumerate(v_grid):
#         fd = velocity_to_doppler(v, wavelength)
#         corr = apply_phase_correction_time(tile, az_times, fd_hz=fd)
#         ent[i] = entropy_metric(corr)
#     idx = np.argmin(ent)
#     return v_grid[idx], v_grid, ent

# # Frequency-domain search where we allow a quadratic spectral-phase correction Psi(f) = alpha * f^2
# # We assume the dominant defocus is quadratic in time -> corresponds to quadratic in freq (up to constants).
# def search_velocity_freq(tile, prf, wavelength, v_min, v_max, n_steps=61, alpha_max=1e-6):
#     N_az = tile.shape[0]
#     df = prf / N_az
#     v_grid = np.linspace(v_min, v_max, n_steps)
#     ent = np.zeros_like(v_grid)
#     # freq axis for computing Psi(f) inside helper
#     freq_axis = (np.arange(N_az) - N_az//2) * df

#     for i, v in enumerate(v_grid):
#         fd = velocity_to_doppler(v, wavelength)
#         # choose a spectral-phase function Psi(f): try a quadratic with alpha chosen by scanning or a guess.
#         # Here we pick alpha proportional to fd to show interplay; in practice you'd estimate Psi(f).
#         def psi_of_f(f):
#             alpha = alpha_max  # tune or make it another search dimension
#             return alpha * (f**2)
#         corr = apply_phase_correction_freq(tile, prf, fd_hz=0.0, phi_freq_func=psi_of_f)
#         ent[i] = entropy_metric(corr)
#     idx = np.argmin(ent)
#     return v_grid[idx], v_grid, ent




# #%%#########  SIMULATIONS ############################
# # ----------------------
# # Params (example)
# # ----------------------
# N_az = 64
# N_rg = 128
# prf = 486.48631029955294
# c = 299792458.0
# center_freq = 5.405e9
# wavelength = c / center_freq
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

# # ----------------------
# # Search variants:
# #   - time domain correction for linear fd (works for linear)
# #   - freq domain correction for a quadratic spectral-phase (we'll craft Psi(f) corresponding to the time quadratic)
# # ----------------------

# # Run a couple of experiments:
# v_time, vgrid_time, ent_time = search_velocity_time(moved_noisy, prf, wavelength, 0.0, 20.0, n_steps=101)
# v_freq, vgrid_freq, ent_freq = search_velocity_freq(moved_noisy, prf, wavelength, 0.0, 20.0, n_steps=101, alpha_max=1e-6)

# print("True v:", true_velocity)
# print("Time-domain best v:", v_time)
# print("Freq-domain best v (with quadratic Psi):", v_freq)

# # Plot to inspect
# plt.figure(figsize=(10,4))
# plt.subplot(1,3,1)
# plt.title("Before (magnitude)")
# plt.imshow(np.abs(moved_noisy), origin='lower', aspect='auto')
# plt.colorbar()
# plt.subplot(1,3,2)
# plt.title("After time correction (v_time)")
# az_times = (np.arange(N_az) - N_az/2.0) / prf
# best_corr_time = apply_phase_correction_time(moved_noisy, az_times, velocity_to_doppler(v_time, wavelength))
# plt.imshow(np.abs(best_corr_time), origin='lower', aspect='auto')
# plt.colorbar()
# plt.subplot(1,3,3)
# plt.title("After freq correction (quadratic Psi)")
# best_corr_freq = apply_phase_correction_freq(moved_noisy, prf, fd_hz=0.0,
#                                             phi_freq_func=lambda f: 1e-6*(f**2))
# plt.imshow(np.abs(best_corr_freq), origin='lower', aspect='auto')
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.plot(vgrid_time, ent_time, label='time search')
# plt.plot(vgrid_freq, ent_freq, label='freq search')
# plt.axvline(true_velocity, color='k', linestyle='--', label='true v')
# plt.legend()
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Entropy')
# plt.grid(True)
# plt.show()



# #%%#########  REAL DATA ############################
# # ----------------------
# # Params (example)
# # ----------------------
# N_az = np.shape(imgVH_ship)[0]
# N_rg = np.shape(imgVH_ship)[1]
# # prf = 486.48631029955294
# # c = 299792458.0
# # center_freq = 5.405e9
# wavelength = c / center_freq
# # true_velocity = 8.5  # m/s

# # # ----------------------
# # # Simulation: multi-scatterer "ship" + quadratic defocus
# # # ----------------------
# # A = np.zeros((N_az, N_rg), dtype=np.complex64)
# # for offset in [-8, -3, 0, 3, 8]:
# #     A[N_az//2 + offset, N_rg//2] = 1.0 + 0j

# # az = np.arange(N_az)
# # rg = np.arange(N_rg)
# # az_win = np.exp(-((az - N_az//2)/(N_az*0.18))**2)
# # rg_win = np.exp(-((rg - N_rg//2)/(N_rg*0.06))**2)
# # psf = az_win[:, None] * rg_win[None, :]
# # clean = A * psf

# # az_times = (np.arange(N_az) - N_az/2.0) / prf
# # fd_true = velocity_to_doppler(true_velocity, wavelength)

# # # Create a QUADRATIC defocus in time (simulate motion error)
# # kappa = 400.0  # tune this to control blur strength; units such that pi*kappa*t^2 is radians
# # phase_error_time = (np.pi * kappa * az_times**2)  # radians
# # moved = clean * np.exp(1j * 2.0 * np.pi * fd_true * az_times)[:, None] * np.exp(1j * phase_error_time)[:, None]

# # rng = np.random.default_rng(1)
# # noise_level = 0.01
# # moved_noisy = moved + noise_level * (rng.standard_normal(moved.shape) + 1j*rng.standard_normal(moved.shape))

# # ----------------------
# # Search variants:
# #   - time domain correction for linear fd (works for linear)
# #   - freq domain correction for a quadratic spectral-phase (we'll craft Psi(f) corresponding to the time quadratic)
# # ----------------------

# # Run a couple of experiments:
# v_time, vgrid_time, ent_time = search_velocity_time(imgVH_ship, prf, wavelength, -20.0, 20.0, n_steps=101)
# v_freq, vgrid_freq, ent_freq = search_velocity_freq(imgVH_ship, prf, wavelength, -20.0, 20.0, n_steps=101, alpha_max=1e-6)

# print("True v:", true_velocity)
# print("Time-domain best v:", v_time)
# print("Freq-domain best v (with quadratic Psi):", v_freq)

# # Plot to inspect
# plt.figure(figsize=(10,4))
# plt.subplot(1,3,1)
# plt.title("Before (magnitude)")
# plt.imshow(np.abs(imgVH_ship), origin='lower', aspect='auto')
# plt.colorbar()
# plt.subplot(1,3,2)
# plt.title("After time correction (v_time)")
# az_times = (np.arange(N_az) - N_az/2.0) / prf

# best_corr_time = apply_phase_correction_time(imgVH_ship, az_times, velocity_to_doppler(v_time, wavelength))
# plt.imshow(np.abs(best_corr_time), origin='lower', aspect='auto')
# plt.colorbar()
# plt.subplot(1,3,3)
# plt.title("After freq correction (quadratic Psi)")


# best_corr_freq = apply_phase_correction_freq(imgVH_ship, prf, fd_hz=0.0,
#                                             phi_freq_func=lambda f: 1e-6*(f**2))
# # best_corr_freq = apply_phase_correction_freq(imgVH_ship, prf, fd_hz=0.0,
# #                                             phi_freq_func=lambda f: 1e6*(f**2))

# plt.imshow(np.abs(best_corr_freq), origin='lower', aspect='auto')
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.plot(vgrid_time, ent_time, label='time search')
# plt.plot(vgrid_freq, ent_freq, label='freq search')
# plt.axvline(true_velocity, color='k', linestyle='--', label='true v')
# plt.legend()
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Entropy')
# plt.grid(True)
# plt.show()
