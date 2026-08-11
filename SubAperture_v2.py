
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
def SubAperture_detectors(data, flag_axis, win, flag_vis, path_save):
    

    if flag_axis == 'Azimuth': data = np.transpose(data)
 
    dim = np.shape(data)     

    #; Obtain the sublooks (in range!)
    dataFFT = np.fft.fftn(data, axes=[1]) #columns: IDL-dim1 = Py-axis1
    spectrum = np.roll(dataFFT, int(dim[1]/2), axis=1) #colums: IDL-dim1 = Py-axis1 | colums: IDL-dim1 = Py-axis1
    # Unhamming process
    spectrum_medio  = (np.sum(abs(spectrum), axis=0))/(dim[0]) #row: IDL-dim2 = Py-axis0, row: IDL-dim2 = Py-axis0
    #Smoothing the signal to get a good "unhamming" function
    convFilter = np.ones((win,))/win
    for kkk in range (0, 10): spectrum_medio = signal.fftconvolve(spectrum_medio, convFilter, mode='same')
    #;To force that the maximum of my "unhamming" function to be equal 1
    spectrum_medio  = spectrum_medio/max(spectrum_medio)
 
    #; for Doppler analysis
    if flag_axis == 'Azimuth':

        gf0 = np.where(spectrum_medio>0.1*max(spectrum_medio),0,spectrum_medio) #set all pixels bigger than threshold to 0
        gf = gf0.nonzero() #eliminate all pixels = 0
        
        s = np.shape(gf)  
        n_smp_band = s[1] #nb of pixels in gf | columns: IDL-size[1] = Py-axis1 = Py-shape[1]
        n_smp_extra = dim[1]-n_smp_band #columns: IDL-size[1]=IDL-dim[0] = Py-axis1
    
        gf0 = np.where(spectrum_medio<0.1*max(spectrum_medio),0,spectrum_medio)
        gf = gf0.nonzero()
        spectrum_medio = np.where(spectrum_medio<0.1*max(spectrum_medio),spectrum_medio,1)
        plt.plot(spectrum_medio, 'b')
        plt.show()

    elif flag_axis == 'Range':
        gf0 = np.where(spectrum_medio[0:int(dim[1]/2)] < 0.1)
        ind_noise = 0
        if gf0[0].size > 0:
            ind_noise = int(np.max(gf0))
        spectrum_medio[spectrum_medio < 0.1] = 1
        n_smp_extra = 2 * ind_noise

    #;Line by line, I correct the spectrum to get the original one (without windowing!!!)    
    spectrum_corr = np.zeros(dim, dtype=np.complex64) 
    #for jjj=0, dim[1]-1 do spectrum_corr(*,jjj) = spectrum(*,jjj)/spectrum_medio(*)    
    for jjj in range (0, dim[0]-1):
        spectrum_corr[jjj:] = spectrum[jjj:]/spectrum_medio

            
    spectrum1 = np.zeros(dim, dtype=np.complex64)
    spectrum2 = np.zeros(dim, dtype=np.complex64)    
    spectrum1[:,int(n_smp_extra/2):int(dim[1]/2)-1] = spectrum_corr[:,int(n_smp_extra/2):int(dim[1]/2)-1]  # [i:j,*] -> [:,i:j]
    spectrum2[:,int(dim[1]/2):int(dim[1])-int(n_smp_extra/2)-1] = spectrum_corr[:,int(dim[1]/2):int(dim[1]-n_smp_extra/2)-1]
    
    data1 = np.fft.ifftn(np.roll(spectrum1, int(dim[1]/4-n_smp_extra/4), axis=1), axes=[1]) # ?? problem: ifft produces 1dim array
    data2 = np.fft.ifftn(np.roll(spectrum2, -int(dim[1]/4+n_smp_extra/4), axis=1), axes=[1])
 
    winFilterUngarded = np.ones((win,win),np.float32)/(win ** 2) #without guard windows

    SubNum = signal.convolve2d(data1*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
    SubDen1 = signal.convolve2d(data1*np.conj(data1), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
    SubDen2 = signal.convolve2d(data2*np.conj(data2), winFilterUngarded, mode='same', boundary='wrap', fillvalue=0)
    
    SubCorr = abs(SubNum/np.sqrt(abs(SubDen1)*abs(SubDen2)))
    SubCohe = abs(SubNum/np.sqrt(abs(SubDen1)*abs(SubDen2)))


    # visulisation part, only if the flag_vis is set to True
    if flag_vis == True:
        plt.figure()
        plt.imshow(np.abs(dataFFT), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
        plt.title("Magnitude spectrum of the data")
        plt.savefig(path_save / "spectrum_data.png", bbox_inches='tight')

        plt.figure()
        plt.plot(spectrum_medio)
        plt.title("Mean Spectrum of the entire image")
        plt.savefig(path_save / "spectrum_medio.png", bbox_inches='tight')

        plt.figure()
        plt.imshow(np.abs(spectrum_corr), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
        plt.title("Magnitude spectrum AFTER removing Hamming")
        plt.savefig(path_save / "spectrum_unhamming.png", bbox_inches='tight')

        plt.figure()
        plt.imshow(np.abs(spectrum1), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
        plt.title("Magnitude of FIRST portion of spectrum")
        plt.savefig(path_save / "spectrum1.png", bbox_inches='tight')

        plt.figure()
        plt.imshow(np.abs(spectrum2), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(dataFFT)))
        plt.title("Magnitude of SECOND portion of spectrum")
        plt.savefig(path_save / "spectrum2.png", bbox_inches='tight')

        plt.figure()
        plt.imshow(np.abs(data1), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(data1)))
        plt.title("Magnitude of FIRST subaperture")
        plt.savefig(path_save / "subaperture1.png", bbox_inches='tight')

        plt.figure()
        plt.imshow(np.abs(data2), cmap = 'gray', vmin = 0, vmax = 2.5*np.nanmean(np.abs(data2)))
        plt.title("Magnitude of SECOND subaperture")
        plt.savefig(path_save / "subaperture2.png", bbox_inches='tight')

        plt.figure()
        plt.imshow(np.abs(SubCohe), cmap = 'gray', vmin = 0.5, vmax = 1)
        plt.title("VV Sub Coherence image")
        plt.savefig(path_save / "SubCohe.png", bbox_inches='tight')
        plt.figure()
        plt.imshow(np.abs(SubCorr), cmap = 'gray', vmin = 1.5*np.nanmean(np.abs(SubCorr)), vmax = 2.5*np.nanmean(np.abs(SubCorr)))
        plt.title("VH Sub Correlation")
        plt.savefig(path_save / "SubCorr.png", bbox_inches='tight')


    return SubCohe, SubCorr



# defining paths where data are
path = Path("/home/am221/C/Data/S1/Velocity") 
path_save = Path("/home/am221/C/Data/S1/Velocity/Sub_detectors")
path_save_img = Path("/home/am221/C/Data/S1/Velocity/Sub_detectors/Images")


# filtering  windows
win = [7,7]     # averagiung window for boxcar and algorithms

# this following window is useful if one want to do some extra multilook 
# and reduce the number of pixels (subsample) to reduce the size of images
# if you have a powerful machine you can keep it [1,1]
sub_win = [3,3]  # subsampling window for reducing size (partial multilook)


# # In case this is needed
# manifest_path = "S1A_IW_1SLC__1SDV_20240102T092331_20240102T092350_051926_06460F_974E.SAFE" # path to Sentinel-1 SAFE manifest
# prf, center_freq = read_sentinel1_params(path / manifest_path)
## from previous runs
## center_freq = 5405000454.33435
## prf = 486.48631029955294


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
        metadata = src.meta
    
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

# select if you want to process rangfe or azimuth subapertures
flag_axis = 'Range'

# select if one wants to save images from detectors 
flag_vis = True

win = 5     # this is the window for the smoothing of the mean spectrum

# select what polarisation you want to analyse
data = VV

[SubCohe, SubCorr] = SubAperture_detectors(data, flag_axis, win, flag_vis, path_save_img)


#%% Saving output as geotiff

with rasterio.open(fullpath_img) as src:
    window = Window(col_off, row_off, width, height)
    cropped_transform = src.window_transform(window)
    crs = src.crs

out_meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'count': 1,
    'height': SubCohe.shape[0],
    'width': SubCohe.shape[1],
    'crs': crs,
    'transform': cropped_transform,
}

with rasterio.open(path_save / "SubCohe.tif", 'w', **out_meta) as dst:
    dst.write(SubCohe.astype('float32'), 1)

with rasterio.open(path_save / "SubCorr.tif", 'w', **out_meta) as dst:
    dst.write(SubCorr.astype('float32'), 1)





#%% Create PowerPoint presentation with output images

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

slides_content = [
    ("Magnitude spectrum of the data",            "spectrum_data.png"),
    ("Mean Spectrum of the entire image",          "spectrum_medio.png"),
    ("Magnitude spectrum AFTER removing Hamming",  "spectrum_unhamming.png"),
    ("Magnitude of FIRST portion of spectrum",     "spectrum1.png"),
    ("Magnitude of SECOND portion of spectrum",    "spectrum2.png"),
    ("Magnitude of FIRST subaperture",             "subaperture1.png"),
    ("Magnitude of SECOND subaperture",            "subaperture2.png"),
    ("VV Sub Coherence image",                     "SubCohe.png"),
    ("VH Sub Correlation",                         "SubCorr.png"),
]

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

blank_layout = prs.slide_layouts[6]  # fully blank layout

for title_text, img_file in slides_content:
    img_path = path_save_img / img_file
    if not img_path.exists():
        continue

    slide = prs.slides.add_slide(blank_layout)

    # title text box at the top
    txBox = slide.shapes.add_textbox(Inches(0.3), Inches(0.15), Inches(12.7), Inches(0.6))
    tf = txBox.text_frame
    tf.text = title_text
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].runs[0].font.size = Pt(24)
    tf.paragraphs[0].runs[0].font.bold = True

    # image centred on the slide
    slide.shapes.add_picture(str(img_path), Inches(1.5), Inches(0.85), Inches(10.3), Inches(6.3))

prs.save(path_save / "SubAperture_results.pptx")
print(f"PowerPoint saved to {path_save / 'SubAperture_results.pptx'}")

#%%