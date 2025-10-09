#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:51:12 2025

@author: am221
"""


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
from rasterio.plot import show

import xml.etree.ElementTree as ET
import glob

plt.close('all')


#%% Getting metadata and creating the cubes

# First we want to get the metadata
path = Path("/home/am221/C/Data/S1/Velocity") 
path_save = Path("/home/am221/C/Data/S1/Velocity/Save")

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



#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from pathlib import Path
import geopandas as gpd

# -----------------------
# AIS data
# -----------------------
path_AIS = Path("/home/am221/C/Data/S1/Velocity/AIS")
ais_csv = path_AIS / "AIS_2401020905_2401020940.csv"
ais_df = pd.read_csv(ais_csv, on_bad_lines="skip")

# Build GeoDataFrame in lon/lat WGS84
gdf = gpd.GeoDataFrame(
    ais_df,
    geometry=gpd.points_from_xy(ais_df['lon'], ais_df['lat']),
    crs="EPSG:4326"
)

# -----------------------
# Raster data
# -----------------------
fullpath_img = path / file_name  # <-- change to your raster path

with rasterio.open(fullpath_img) as src:
    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)
    print("Raster transform:", src.transform)
    
    # Read first band for display
    band1 = src.read(1)

    fig, ax = plt.subplots(figsize=(10, 10))

    if src.crs is not None:
        # Case 1: raster has CRS -> reproject AIS into raster CRS
        gdf_proj = gdf.to_crs(src.crs)

        show(band1, transform=src.transform, ax=ax, cmap="gray")
        gdf_proj.plot(ax=ax, color="red", markersize=30, alpha=0.7)

        for x, y, label in zip(gdf_proj.geometry.x, gdf_proj.geometry.y, gdf_proj.get("id", range(len(gdf_proj)))):
            ax.text(x, y, str(label), fontsize=8, color="blue", weight="bold",
                    ha="right", va="bottom")
        
        ax.set_title("AIS points projected to raster CRS")
    
    else:
        # Case 2: raster has no CRS -> just plot pixels as row/col index
        show(band1, ax=ax, cmap="gray")
        gdf.plot(ax=ax, color="red", markersize=30, alpha=0.7)

        ax.set_title("WARNING: raster has no CRS (AIS not aligned)")

    plt.show()



#%% Plotting points on raster


raster_path = "background_raster.tif"
with rasterio.open(raster_path) as src:
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Show raster
    show(src, ax=ax)

    # Plot AIS points
    ax.scatter(ais_df['lon'], ais_df['lat'], s=20, c='red', alpha=0.7)

    # Annotate points with IDss
    for idx, row in ais_df.iterrows():
        ax.text(row['lon'], row['lat'], str(row['id']),
                fontsize=8, color='blue', weight='bold',
                ha='right', va='bottom')  # adjust text position

    ax.set_title("AIS Points with IDs over Raster")
    plt.show()



#%%#############################################################
# import rasterio
# from rasterio.plot import show
# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. Load AIS CSV (assuming columns 'lon' and 'lat')
# ais_df = pd.read_csv("ais_data.csv")  # Replace with your file
# print(ais_df.head())

# # 2. Open GeoTIFF with rasterio
# raster_path = "background_raster.tif"  # Replace with your file
# with rasterio.open(raster_path) as src:
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Show raster
#     show(src, ax=ax)

#     # 3. Overlay AIS points
#     ax.scatter(
#         ais_df["lon"], 
#         ais_df["lat"], 
#         s=10,           # size of points
#         c="red",        # color
#         marker="o", 
#         alpha=0.7, 
#         transform=ax.transData
#     )

#     ax.set_title("AIS Points over Raster")
#     plt.show()

#%%
import rasterio
from rasterio.plot import show
import pandas as pd
import matplotlib.pyplot as plt

# Load AIS CSV (assuming columns: 'lon', 'lat', 'id')
ais_df = pd.read_csv("ais_data.csv")

raster_path = "background_raster.tif"
with rasterio.open(raster_path) as src:
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Show raster
    show(src, ax=ax)

    # Plot AIS points
    ax.scatter(ais_df['lon'], ais_df['lat'], s=20, c='red', alpha=0.7)

    # Annotate points with IDss
    for idx, row in ais_df.iterrows():
        ax.text(row['lon'], row['lat'], str(row['id']),
                fontsize=8, color='blue', weight='bold',
                ha='right', va='bottom')  # adjust text position

    ax.set_title("AIS Points with IDs over Raster")
    plt.show()




#%%

import geopandas as gpd
from shapely.geometry import Point

# Convert AIS dataframe to GeoDataFrame
gdf = gpd.GeoDataFrame(
    ais_df, 
    geometry=gpd.points_from_xy(ais_df.lon, ais_df.lat), 
    crs="EPSG:4326"
)

# Reproject to raster CRS
with rasterio.open(raster_path) as src:
    gdf = gdf.to_crs(src.crs)

    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax)

    gdf.plot(ax=ax, color="red", markersize=10, alpha=0.7)
    plt.show()
