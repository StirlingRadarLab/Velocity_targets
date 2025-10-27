# Armando Marino 09/10/2025


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

import pandas as pd
import tqdm

# for file copying
import shutil

import time


# Import the required libraries for GIS
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

# for exploring xml files
import xml.etree.ElementTree as ET
import glob

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# close all plots before running new ones
plt.close('all')


#%% FUNCTIONS TO READ SENTINEL PARAMETERS FROM THE MANIFEST 

# GENERAL ORBITAL PARAMETERS
def read_sentinel1_params(safe_folder, orbit_file=None, use_approx_velocity=True, nominal_altitude_m=693000.0):
    """
    Read Sentinel-1 parameters, including approximate or true satellite velocity.
    Also computes slant range times at start, center, and end of swath.

    Returns
    -------
    dict
        {
            "prf": float,
            "center_freq": float,
            "slant_range_time_center": float,
            "slant_range_time_start": float,
            "slant_range_time_end": float,
            "range_pixel_spacing": float,
            "R0": float,
            "R_start": float,
            "R_end": float,
            "velocity_vector": (vx, vy, vz) or None,
            "velocity_magnitude": float,
            "velocity_source": str
        }
    """
    c = 299792458.0  # m/s

    # --- locate annotation file ---
    ann_files = glob.glob(os.path.join(safe_folder, "annotation", "*.xml"))
    if not ann_files:
        raise FileNotFoundError("No annotation XMLs found in SAFE package")

    ann_tree = ET.parse(ann_files[0])
    ann_root = ann_tree.getroot()

    ns = {
        "s1": "http://www.esa.int/safe/sentinel-1.0",
        "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1",
    }

    # --- radar frequency ---
    center_freq = ann_root.findtext(".//s1sarl1:radarFrequency", namespaces=ns)
    if center_freq is None:
        center_freq = ann_root.findtext(".//radarFrequency")
    center_freq = float(center_freq)

    # --- PRF ---
    prfs = [x.text for x in ann_root.findall(".//s1sarl1:pulseRepetitionFrequencyList/s1sarl1:pulseRepetitionFrequency", ns)]
    if not prfs:
        prfs = [x.text for x in ann_root.findall(".//pulseRepetitionFrequencyList/pulseRepetitionFrequency")]
    prfs = [float(x) for x in prfs]
    if not prfs:
        ati = ann_root.findtext(".//s1sarl1:azimuthTimeInterval", namespaces=ns)
        if ati is None:
            ati = ann_root.findtext(".//azimuthTimeInterval")
        if ati:
            prfs = [1.0 / float(ati)]
    prf = prfs[0]

    # --- slant range time (center) ---
    slant_range_time_center = ann_root.findtext(".//s1sarl1:slantRangeTime", namespaces=ns)
    if slant_range_time_center is None:
        slant_range_time_center = ann_root.findtext(".//slantRangeTime")
    slant_range_time_center = float(slant_range_time_center)

    # --- range pixel spacing ---
    range_pixel_spacing = ann_root.findtext(".//s1sarl1:rangePixelSpacing", namespaces=ns)
    if range_pixel_spacing is None:
        range_pixel_spacing = ann_root.findtext(".//rangePixelSpacing")
    range_pixel_spacing = float(range_pixel_spacing)  # meters

    # --- number of range samples ---
    number_of_range_samples = ann_root.findtext(".//s1sarl1:numberOfSamples", namespaces=ns)
    if number_of_range_samples is None:
        number_of_range_samples = ann_root.findtext(".//numberOfSamples")
    number_of_range_samples = int(number_of_range_samples)

    # --- compute sample spacing in time ---
    delta_t = 2 * range_pixel_spacing / c  # slant range time per sample

    # --- compute start and end slant range times ---
    i_center = (number_of_range_samples - 1) / 2
    slant_range_time_start = slant_range_time_center - i_center * delta_t
    slant_range_time_end   = slant_range_time_center + i_center * delta_t

    # --- convert to slant range distances ---
    R0 = c * slant_range_time_center / 2.0
    R_start = c * slant_range_time_start / 2.0
    R_end   = c * slant_range_time_end / 2.0

    # --- initialize velocity lists ---
    vx_list, vy_list, vz_list = [], [], []

    # --- Case 1: external orbit file (.EOF) ---
    if orbit_file and os.path.exists(orbit_file):
        try:
            tree = ET.parse(orbit_file)
            root = tree.getroot()
            for osv in root.findall(".//OSV"):
                vx = osv.findtext("VX")
                vy = osv.findtext("VY")
                vz = osv.findtext("VZ")
                if vx and vy and vz:
                    vx_list.append(float(vx))
                    vy_list.append(float(vy))
                    vz_list.append(float(vz))
        except ET.ParseError:
            pass

    # --- Case 2: internal orbit/state vectors (older SAFE) ---
    if not vx_list:
        xml_candidates = glob.glob(os.path.join(safe_folder, "**", "*.xml"), recursive=True)
        for xml_file in xml_candidates:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except ET.ParseError:
                continue
            for sv in root.findall(".//stateVector"):
                vx = sv.findtext(".//velocity/x")
                vy = sv.findtext(".//velocity/y")
                vz = sv.findtext(".//velocity/z")
                if vx and vy and vz:
                    vx_list.append(float(vx))
                    vy_list.append(float(vy))
                    vz_list.append(float(vz))
            if vx_list:
                break

    # --- determine velocity ---
    if vx_list:
        velocity_vector = (np.mean(vx_list), np.mean(vy_list), np.mean(vz_list))
        velocity_magnitude = float(np.linalg.norm(velocity_vector))
        velocity_source = "orbit"
    else:
        if use_approx_velocity:
            mu = 3.986004418e14  # Earth's gravitational constant
            Re = 6371000.0       # Earth radius [m]
            r = Re + float(nominal_altitude_m)
            velocity_magnitude = float(np.sqrt(mu / r))
            velocity_vector = None
            velocity_source = f"approx(h={nominal_altitude_m:.0f}m)"
        else:
            raise ValueError(
                "No orbit state vectors found inside SAFE or in provided orbit file. "
                "Set use_approx_velocity=True to return an approximate speed."
            )

    return {
        "prf": prf,
        "center_freq": center_freq,
        "slant_range_time_center": slant_range_time_center,
        "slant_range_time_start": slant_range_time_start,
        "slant_range_time_end": slant_range_time_end,
        "range_pixel_spacing": range_pixel_spacing,
        "R0": R0,
        "R_start": R_start,
        "R_end": R_end,
        "velocity_vector": velocity_vector,
        "velocity_magnitude": velocity_magnitude,
        "velocity_source": velocity_source,
    }



# TIMING PARAMETERS
def get_sentinel1_acquisition_duration(safe_folder):
    """
    Robustly extract start/stop acquisition times for any Sentinel-1 SAFE (SLC or GRD).
    Works whether annotation XMLs exist or not.
    """
    # --- Case 1: annotation XMLs ---
    ann_files = glob.glob(os.path.join(safe_folder, "**", "*.xml"), recursive=True)
    ann_files = [f for f in ann_files if "annotation" in f.lower()]
    if ann_files:
        ann_file = ann_files[0]
        tree = ET.parse(ann_file)
        root = tree.getroot()
        ns = {
            "s1": "http://www.esa.int/safe/sentinel-1.0",
            "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1",
        }
        start_time = root.findtext(".//s1:adsHeader/s1:startTime", namespaces=ns) or root.findtext(".//startTime")
        stop_time  = root.findtext(".//s1:adsHeader/s1:stopTime", namespaces=ns) or root.findtext(".//stopTime")
        source = ann_file
    else:
        # --- Case 2: manifest.safe ---
        manifest_path = os.path.join(safe_folder, "manifest.safe")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No annotation or manifest.safe found under {safe_folder}")
        tree = ET.parse(manifest_path)
        root = tree.getroot()
        start_time = root.findtext(".//startTime")
        stop_time  = root.findtext(".//stopTime")
        source = manifest_path

    if not start_time or not stop_time:
        raise ValueError(f"Could not find start/stop time in {source}")

    start_dt = parse_sentinel1_time(start_time)
    stop_dt  = parse_sentinel1_time(stop_time)
    duration = (stop_dt - start_dt).total_seconds()

    return {
        "start_time": start_dt,
        "stop_time": stop_dt,
        "duration_seconds": duration,
        "source": source,
    }

def parse_sentinel1_time(timestr):
    """Parse Sentinel-1 time strings robustly (with/without 'Z' and fractional seconds)."""
    if timestr is None:
        return None
    timestr = timestr.strip()
    # Remove trailing 'Z' if present
    if timestr.endswith("Z"):
        timestr = timestr[:-1]
    # Try parsing with microseconds
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(timestr, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized Sentinel-1 time format: {timestr}")
    
    

#%% Function for retrivial using traditional processing of Doppler centroid

def velocity_estimation(img, lambda_c, prf):
    """
    Estimate the velocity of a ship in a small imaget, using the desplacement of the 
    the Doppler centroid 
    Parameters:
        img: Small image (usually 64x64) containiung the ship in the middle.
        lambda_c: wavelenght
        prf: PRF
        
    Returns:
        freq_axis: array that can be used to plot the frequancy variable of the strecturm, 
        doppler_centroid: location of centre of the Doppler spectrum 
        v_az: estimation of velocity in azimuth 
        power_spectrum: the magnitude of the Doppler spectrum
        
    """        
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
    # print(f"Estimated vessel across-track speed: {v_az:.2f} m/s")
    
    return freq_axis, doppler_centroid, v_az, power_spectrum


#%%
# Function to compute entropy-based sharpness
def image_entropy(img, bins=256):
    """
    Calculate the statistical entropy of the distribution obtained by the 
    intensity of the image. 
    Parameters:
        img: Small image (usually 64x64) containing the ship in the middle.
        bins: number of bins used to calculate the histogram (estimating pdf)
                
    Returns:
        H: information entropy 
    """  
    mag = np.abs(img)
    hist, _ = np.histogram(mag, bins=bins, range=(mag.min(), mag.max()), density=False)
    hist = hist.astype(float)
    hist /= hist.sum()           # normalize so sum = 1
    hist = hist[hist > 0]        # remove zeros to avoid log(0)
    H = -np.sum(hist * np.log2(hist))
    return H


# Function to refocus an image given parameters
def refocus_image(img_FFT, max_val, va):
    """
    Apply a transformation of the Doppler based on the low describing the 
    Doppler phase history. This provides re-focusing (it can blare or sharpen the image). 
    Parameters:
        img_FFT: The Fast Furier Transform of the imaget (usually 64x64)
        max_val: the adjustment parameter (connected to vx) to be tested
        va: the aximuth velocity to be tested
                
    Returns:
        H: information entropy 
    """  
    N_az = img_FFT.shape[0]
   
    # create the azimuth time axis
    t_az = (np.arange(N_az)) * duration

    # x = np.linspace(0, 1, dim[0])
    # quadratic = (max_val) * (1 - (x - centre)**2)
    # quadratic = (max_val) * ( (x - centre)**2 )

    quadratic = (max_val) * ( (vp - va)**2 * t_az**2 )
    phase_corr = np.exp(1j * (2*np.pi/lambda_c) * (R0_ship + 1/R0_ship * quadratic ) )
    # img_corr_FFT = img_FFT * phase_corr[:, np.newaxis]
    img_corr_FFT = img_FFT * phase_corr
    img_corr = ifft(ifftshift(img_corr_FFT, axes=1), axis=1)
    
    # plt.figure()
    # plt.plot(quadratic)
    # plt.title("Quadratic function")
    
    return img_corr


#%%


# defining paths where data are
path = Path("/home/am221/C/Data/S1/Velocity") 
path_save = Path("/home/am221/C/Funds/KIOST/2026/Images")


# filtering  windows
win = [7,7]     # averagiung window for boxcar and algorithms

# this following window is useful if one want to do some extra multilook 
# and reduce the number of pixels (subsample) to reduce the size of images
# if you have a powerful machine you can keep it [1,1]
# if you need to analyse SLC you MUST keep it to [1,1]
sub_win = [1,1]  # subsampling window for reducing size (partial multilook)

# reading the S1 parameters in manifest
# location of manifest
manifest_path = "S1A_IW_SLC__1SDV_20240102T092331_20240102T092350_051926_06460F_974E.SAFE" # path to Sentinel-1 SAFE manifest

# getting orbital parameters
params = read_sentinel1_params(path / manifest_path)

prf = params["prf"]
center_freq = params["center_freq"]
t_center = params['slant_range_time_center']
t_start  = params['slant_range_time_start']
t_end    = params['slant_range_time_end']
range_pixel_spacing = params["range_pixel_spacing"]
R0 = params["R0"]
R_start  = params["R_start"]
R_end  = params["R_end"]
        
# # Print in aligned format
# for name in params:
#     print(f"{name:<25} : {params[name]}")
    
    
if params["velocity_vector"] is not None:
    vx, vy, vz = params["velocity_vector"]
    vp = vx
vp = params["velocity_magnitude"]

# getrting timing parameters
timing = get_sentinel1_acquisition_duration(path / manifest_path)
start = timing['start_time']            
stop = timing['stop_time']
duration = timing['duration_seconds']

for name in timing:
    print(f"{name:<25} : {timing[name]}")
        
# from previous runs:
# center_freq = 5405000454.33435
# prf = 486.48631029955294
# slant_range_time = 5.830090502e-03  # seconds
# range_pixel_spacing = 2.329560e+01  # meters
# R0 = c * slant_range_time / 2       # slant range to first pixel

c = 299792458.0
lambda_c = c / center_freq



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
col_off = 0            # start column
row_off = 0            # satart raw
width = src.width      # size in columns
height = src.height    # size in raws


# The following is for focusing on smaller area, wchih is needed in order to 
# better select the ships 
# flag_ROI = "ROI1"
flag_ROI = "ROI2"
# flag_ROI = "ROI3"
# flag_ROI = "ROI4"

# select which ship to work on
flag_ship = "SHIP1"
# flag_ship = "SHIP2"
# flag_ship = "SHIP3"
# flag_ship = "SHIP4"


if flag_ROI == "ROI1": 
    # # part of the image      VIEW 1   
    col_off = 7700
    row_off = 0
    width = 1200
    height = 1200

elif flag_ROI == "ROI2": 
    # part of the image        VIEW 2
    col_off = 9700
    row_off = 0
    width = 2000
    height = 2000

elif flag_ROI == "ROI3": 
    # part of the image        VIEW 3
    col_off = 11700
    row_off = 0
    width = 2000
    height = 2000

elif flag_ROI == "ROI4": 
    # # part of the image        
    col_off = 20000
    row_off = 1000
    width = 2400
    height = 2400


# this is not relevant here, but if you are using a multilook, you wil need 
# to consider the following l·ines       
# col_off = 1800*sub_win[0]
# row_off = 450*sub_win[1]
# width = 200*sub_win[0]
# height = 200*sub_win[1]


# sizew of the imaget used to evaluate ships speed
size_imaget = 64
wd = int(size_imaget /2)
hg = int(size_imaget /2)



# selecting the small imagets around ships
if flag_ROI == "ROI1":
    
    if flag_ship == "SHIP1":
        # # SHIP 1       
        col_cr_ship = 725
        row_cr_ship =  364
    elif flag_ship == "SHIP2":
        # SHIP 2       
        col_cr_ship = 788
        row_cr_ship =  96
#################################

elif flag_ROI == "ROI2": 

    if flag_ship == "SHIP1":
        # SHIP 1       
        col_cr_ship = 134
        row_cr_ship = 1055
#################################

elif flag_ROI == "ROI3": 

    if flag_ship == "SHIP1":
        # SHIP 1       
        col_cr_ship = 547
        row_cr_ship = 907
    elif flag_ship == "SHIP2":
        # SHIP 2       
        col_cr_ship = 150
        row_cr_ship = 1802
    elif flag_ship == "SHIP3":
        # SHIP 3       
        col_cr_ship = 947
        row_cr_ship = 1829
################################



# R0 for imagette
R0_ship = R_start + (row_off + row_cr_ship)*range_pixel_spacing 


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


# VV intensity images for the full ROI
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('VV for region of interest ' + flag_ROI)
plt.imshow(np.abs(VV), cmap = 'gray', vmin = 0, vmax = 1.5*np.nanmean(np.abs(VV)))
fig_filename = "VV_" + flag_ROI
fig.savefig(path_save / fig_filename )   
   
# VH intensity images for the full ROI
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('VH for region of interest ' + flag_ROI)
plt.imshow(np.abs(VV), cmap = 'gray', vmin = 0, vmax = 1.5*np.nanmean(np.abs(VH)))
fig_filename = "VH_" + flag_ROI
fig.savefig(path_save / fig_filename )      
    


#%% SELECT SHIPS for analysis

# creating the small imagets containing the ship
imgVV_ship = VV[col_cr_ship-wd:col_cr_ship+wd, row_cr_ship-hg:row_cr_ship+hg]
imgVH_ship = VH[col_cr_ship-wd:col_cr_ship+wd, row_cr_ship-hg:row_cr_ship+hg]

# looking for a patch of clear sea as close as possible to the ship
# this is needed for corrections of the doppler centroid      
col_cr_sea = col_cr_ship + size_imaget
row_cr_sea =  row_cr_ship + size_imaget

# creating the small imagets contaiuning clear water
imgVV_sea = VV[col_cr_sea-wd:col_cr_sea+wd, row_cr_sea-hg:row_cr_sea+hg]
imgVH_sea = VH[col_cr_sea-wd:col_cr_sea+wd, row_cr_sea-hg:row_cr_sea+hg]


# VV intensity images for small ship
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('VV for ship of interest ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(imgVV_ship), cmap = 'gray', 
           vmin = 0, vmax = 2.5*np.nanmean(np.abs(imgVV_ship)))
fig_filename = "VV_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )   

# VH intensity images for small ship
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('VH for ship of interest ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(imgVH_ship), cmap = 'gray', 
           vmin = 0, vmax = 2.5*np.nanmean(np.abs(imgVH_ship)))
fig_filename = "VH_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )   

# VV intensity images for small ship
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('VV for nearby sea of interest ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(imgVV_ship), cmap = 'gray', 
           vmin = 0, vmax = 2.5*np.nanmean(np.abs(imgVV_ship)))
fig_filename = "VV_sea_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )   

# VH intensity images for small ship
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('VH for nearby sea of interest ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(imgVH_ship), cmap = 'gray', 
           vmin = 0, vmax = 2.5*np.nanmean(np.abs(imgVH_ship)))
fig_filename = "VH_sea_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  


# %% ######################################################
##########################################################
#                       Obtaining some images for spectrum
Win = 7

#producing the FFT of the images
imgVV_FFT = fftshift(fft(imgVV_ship, axis=1), axes=1)
imgVH_FFT = fftshift(fft(imgVH_ship, axis=1), axes=1)

# here we use VV as the image for refocusing, but VH could be used as well
img_FFT = imgVV_FFT
dim = np.shape(img_FFT)

#; Unhamming process
spectrum_medio  = (np.sum(abs(img_FFT), axis=0))/(dim[0]) 

# Smoothing the signal to get a good "unhamming" function
Kernel_Spectrum = np.ones((7,))/7
for kkk in range (0, 10): spectrum_medio = signal.fftconvolve(spectrum_medio, 
                                           Kernel_Spectrum, mode='same')
# To force that the maximum of my "unhamming" function to be equal 1
spectrum_medio  = spectrum_medio/max(spectrum_medio)
# finding what is the last pixel of noise outside the band
gf0 = np.where(spectrum_medio[0:int(dim[1]/2)] < 0.1)
ind_noise = 0
if len(gf0) > 0: 
    ind_noise = np.max(gf0)
# remove noise outside band in equalisation
spectrum_medio[spectrum_medio < 0.1] = 1  

N_az = spectrum_medio.shape[0]
df = prf / N_az # azimuth frequency resolution [Hz]

# create the frequancy axis
freq_axis = (np.arange(N_az) - N_az//2) * df

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('Magnitude spectrum for ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(img_FFT), cmap = 'gray', 
           vmin = 0, vmax = 6.5*np.nanmean(np.abs(img_FFT)))
fig_filename = "FFT_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('Smoothed mean spectrum for ' + flag_ship + ' in ' + flag_ROI)
plt.plot(freq_axis, spectrum_medio)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
fig_filename = "Mean_Spectrum_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  





#%% ##########################################################
##############################################################
####               SPEED FROM DOPPLER CENTROID 
#%% Retrieving the speed using the doppler centroid shift
# here we use VV but VH could be used too

[freq_axis, doppler_centroid_ship, v_az_ship, power_spectrum_ship] = velocity_estimation(imgVV_ship, lambda_c, prf)

[freq_axis, doppler_centroid_sea, v_az_sea, power_spectrum_sea] = velocity_estimation(imgVV_sea, lambda_c, prf)

vessel_velocity = -(lambda_c/2.0) * (doppler_centroid_ship - doppler_centroid_sea)

print(f"Estimated vessel across-track speed: {v_az_ship:.2f} m/s")
print(f"Estimated sea across-track speed: {v_az_sea:.2f} m/s")
print(f"Estimated vessel across-track speed: {vessel_velocity:.2f} m/s")



#%% #############################################################
#################################################################
#                  Autofocus

################################################################
#### THIS BIT TESTS IT WITH A DUMMY QUADRATIC TO SHOW PRINCIPLES
min_val = 0        # start value
max_val = 1.15       # end value
centre = 0
# Quadratic array (parabolic increase)
# normalized from 0 -> 1, then scaled to min_val -> max_val
x = np.linspace(0, 1, dim[0])
# quadratic = min_val + (max_val - min_val) * (1 - (x-centre)**2)
quadratic = (max_val) * ( (x - centre)**2 )

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('Example of quadratic function for autofocus')
plt.plot(quadratic)
fig_filename = "Quadratic" 
fig.savefig(path_save / fig_filename ) 

# imgVV_FFT_corr = imgVV_FFT*np.exp(1j*quadratic*(2*np.pi/lambda_c)/R0)
img_FFT_corr = img_FFT*np.exp(1j * (2*np.pi/lambda_c) * (R0_ship + 1/R0_ship*quadratic ) )
    
img_corr   = ifft(ifftshift(img_FFT_corr, axes=1), axis=1)
img_nocorr = ifft(ifftshift(img_FFT, axes=1), axis=1)

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('EXAMPLE: Image BEFORE focusing for ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(img_nocorr), cmap = 'gray', 
           vmin = 0, vmax = 2.5*np.nanmean(np.abs(img_nocorr)))
fig_filename = "EX_BEFORE_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('EXAMPLE: Image AFTER focusing for ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(img_corr), cmap = 'gray', 
           vmin = 0, vmax = 2.5*np.nanmean(np.abs(img_corr)))
fig_filename = "EX_AFTER_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  

H_org = image_entropy(img_nocorr, bins=256)
H_cor = image_entropy(img_corr, bins=256)
print(f"Entropy for original: {H_org:.3f}")
# print(f"Entropy for refocused: {H_cor:.3f}")





#%%  ##########################################################
##########  AUTOFOCUS ALGORITHM ###############################
# searcing parameters
grid_size = 100       # number of searching samples used
max_vals_FR = 1.15   # this is the value at far range 
min_value_NR = 0.9   # this is the value at near range
va_ext = 20          # maximum value for azimuith velocity

# max_vals_FR = 1000  # this is the value at far range 
# min_value_NR = 0.001   # this is the value at near range
# va_ext = 1000          # maximum value for azimuith velocity

# vectors to swip maximum and centre of the quadratic
adj_vals = np.linspace(min_value_NR, max_vals_FR, grid_size)
va_array = np.linspace(-va_ext, va_ext, grid_size)

best_entropy = np.inf
best_params = None

H = np.zeros([grid_size, grid_size])  

for i in range(len(adj_vals)):
    adj = adj_vals[i]
    for j in range(len(va_array)):
        va = va_array[j]
        img_corr = refocus_image(img_FFT, adj, va)
        H[i, j] = image_entropy(np.abs(img_corr))  
        if H[i, j] < best_entropy:
            best_entropy = H[i, j]
            best_params = (adj, va)

print(f"Best parameters: adj_val={best_params[0]:.3f}, va={best_params[1]:.3f}, entropy={best_entropy:.3f}")

# Generate best-focused image
img_best = refocus_image(img_FFT, *best_params)

# Show before/after
# plt.figure()
# plt.imshow(np.abs(imgVV_ship), cmap = 'gray', vmin = 0, vmax = 7.5*np.nanmean(np.abs(imgVV_ship)))
# plt.title("Before focusing")

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('Intensity BEFORE focusing for ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(imgVV_ship), cmap = 'gray', 
           vmin = 0, vmax = 7.5*np.nanmean(np.abs(imgVV_ship)))
fig_filename = "BEFORE_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('Intensity AFTER focusing for ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(np.abs(img_best), cmap = 'gray', 
           vmin = 0, vmax = 7.5*np.nanmean(np.abs(img_best)))
fig_filename = "AFTER_ship_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  

fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.title('Entropy map for ' + flag_ship + ' in ' + flag_ROI)
plt.imshow(H, 
           cmap='gray', 
           vmin=np.min(H), 
           vmax=np.max(H),
           extent=[va_array[0], va_array[-1], adj_vals[0], adj_vals[-1]],  # x_min, x_max, y_min, y_max
           origin='lower',   # ensures y-axis starts from min_val at bottom
           aspect='auto')    # avoids stretching
plt.colorbar(label='Entropy')
plt.xlabel('Azimuth velocity (m/s)')
plt.ylabel('R0 mismatch')
plt.show()
fig_filename = "EntopyMap_" + flag_ship + '_' + flag_ROI
fig.savefig(path_save / fig_filename )  

