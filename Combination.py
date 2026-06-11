"""
Open Cluster Binary Stars Detection Framework
Author: [Your Name/GitHub Username]
Description: A multi-method pipeline to identify binary star candidates in 
             open clusters using ESA Gaia space mission data.
"""

import os
import glob
import numpy as np
import pandas as pd
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

def mad_to_sigma(x):
    """
    Robust standard deviation estimation using Median Absolute Deviation (MAD).
    Standard standard deviation is sensitive to outliers (e.g., cluster non-members).
    MAD multiplied by 1.4826 provides a resilient and unbiased estimate of Sigma.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0: 
        return 0.02
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return 1.4826 * mad


def identify_binaries_pm_plx_fast(df, k_sigma_pm=3.0, k_sigma_plx=3.0, sep_thresh_pc=0.05):
    """
    Method 1: Proper Motion (PM) & Parallax (Plx) Coherence with Spatial Proximity
    --------------------------------------------------------------------------
    Astrophysical Logic:
        Bound binary systems must share nearly identical spatial velocities (PM)
        and distances (Parallax) due to their gravitational coupling.
    Implementation:
        1. Dynamically computes noise thresholds for PM and Plx using MAD.
        2. Converts celestial coordinates (RA/Dec/Plx) into 3D Cartesian coordinates (pc).
        3. Uses an ultra-fast cKDTree algorithm to find physically close pairs (O(N log N)).
        4. Filters those close pairs to ensure their velocity and distance differences 
           fall within the permitted statistical threshold.
    """
    cols = {"ra", "dec", "pmRA", "pmDE", "parallax"}
    if not cols.issubset(df.columns) or len(df) < 2: 
        return []
    
    # Calculate statistical thresholds using robust MAD
    pm_mag = np.hypot(df["pmRA"].values, df["pmDE"].values)
    pm_thresh = k_sigma_pm * mad_to_sigma(pm_mag)
    plx_thresh = k_sigma_plx * mad_to_sigma(df["parallax"].values)
    
    # Get cluster median parallax to estimate distance scale
    median_plx = np.nanmedian(df["parallax"].values)
    if median_plx <= 0: 
        return []
    dist_pc = 1000.0 / median_plx
    
    # Convert spherical coordinates to 3D Cartesian space (parsecs)
    ra_rad = np.deg2rad(df["ra"].values)
    dec_rad = np.deg2rad(df["dec"].values)
    x = dist_pc * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist_pc * np.cos(dec_rad) * np.sin(ra_rad)
    z = dist_pc * np.sin(dec_rad)
    coords_3d = np.vstack((x, y, z)).T
    
    # Build spatial tree to find close neighbors efficiently
    tree = cKDTree(coords_3d)
    pairs = tree.query_pairs(r=sep_thresh_pc)
    
    binary_indices = set()
    pmRA, pmDE, plx = df["pmRA"].values, df["pmDE"].values, df["parallax"].values
    
    # Evaluate kinematics only for spatially close candidates
    for i, j in pairs:
        delta_pm = np.hypot(pmRA[i] - pmRA[j], pmDE[i] - pmDE[j])
        if delta_pm <= pm_thresh:
            if abs(plx[i] - plx[j]) <= plx_thresh:
                binary_indices.update([df.index[i], df.index[j]])
                
    return list(binary_indices)


def identify_binaries_direct_fast(df, sep_thresh_au=1000):
    """
    Method 2: Resolved Wide Binaries via Direct Angular Separation
    --------------------------------------------------------------
    Astrophysical Logic:
        Targeting wide binary systems where Gaia resolves both components as distinct sources.
    Implementation:
        1. Projects 3D locations of all stars into parsecs.
        2. Converts the separation threshold from Astronomical Units (AU) to parsecs.
        3. Utilizes cKDTree to map and extract all pairs within the physical threshold.
    """
    cols = {"ra", "dec", "parallax"}
    if not cols.issubset(df.columns) or len(df) < 2: 
        return []
    
    median_plx = np.nanmedian(df["parallax"].values)
    if median_plx <= 0: 
        return []
    dist_pc = 1000.0 / median_plx
    
    # Convert AU threshold to parsecs (1 pc = 206265 AU)
    sep_thresh_pc = sep_thresh_au / 206265.0
    
    # Compute 3D Cartesian coordinates
    ra_rad = np.deg2rad(df["ra"].values)
    dec_rad = np.deg2rad(df["dec"].values)
    x = dist_pc * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist_pc * np.cos(dec_rad) * np.sin(ra_rad)
    z = dist_pc * np.sin(dec_rad)
    coords_3d = np.vstack((x, y, z)).T
    
    # Query spatial tree for resolved pairs
    tree = cKDTree(coords_3d)
    pairs = tree.query_pairs(r=sep_thresh_pc)
    
    binary_indices = set()
    for i, j in pairs:
        binary_indices.update([df.index[i], df.index[j]])
        
    return list(binary_indices)


def identify_binaries_cmd(df):
    """
    Method 3: Color-Magnitude Diagram (CMD) Main-Sequence Over-luminosity
    --------------------------------------------------------------------
    Astrophysical Logic:
        Unresolved binaries blend into a single point source. The combined flux 
        makes the system brighter than a single star of the same color, shifting it 
        up to 0.75 magnitudes above the single-star Main Sequence (MS).
    Implementation:
        1. Computes the color index (BP - RP) and the absolute G magnitude (M_G).
        2. Uses a running uniform filter to fit and trace the empirical Main Sequence trend line.
        3. Flags any star elevated by more than 0.3 magnitudes above the MS line as a binary.
    """
    cols = {"phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "parallax"}
    if not cols.issubset(df.columns): 
        return []
    
    sub = df.dropna(subset=list(cols)).copy()
    if sub.empty: 
        return []
    
    # Photometric calculations
    sub["bp_rp"] = sub["phot_bp_mean_mag"] - sub["phot_rp_mean_mag"]
    sub["distance_pc"] = 1000.0 / sub["parallax"]
    sub["abs_mag_g"] = sub["phot_g_mean_mag"] - 5 * np.log10(sub["distance_pc"] / 10.0)
    
    # Fit empirical Main Sequence using uniform filter
    sub_sorted = sub.sort_values("bp_rp")
    smooth_size = min(50, max(5, len(sub_sorted) // 5))
    main_seq = uniform_filter1d(sub_sorted["abs_mag_g"].values, size=smooth_size)
    
    # Delta magnitude (Positive means brighter than the MS line)
    sub_sorted["delta_mag"] = main_seq - sub_sorted["abs_mag_g"]
    
    # Threshold: stars elevated by > 0.3 mag are strong unresolved binary candidates
    binaries = sub_sorted[sub_sorted["delta_mag"] > 0.3]
    return list(binaries.index)


def identify_binaries_aen(df, col="astrometric_excess_noise", k_sigma=3):
    """
    Method 4: Astrometric Excess Noise (AEN) Anomalies
    --------------------------------------------------
    Astrophysical Logic:
        Close, unresolved binaries orbit a moving center of charge/mass. This orbital 
        wobble disrupts Gaia's standard single-star kinematic fit, producing a 
        statistically significant residual known as Astrometric Excess Noise.
    Implementation:
        1. Filters out NaN values from the Gaia AEN data column.
        2. Calculates the cluster's base astrometric noise level using MAD.
        3. Identifies stars with an AEN higher than the cluster median + K*Sigma.
    """
    if col not in df.columns: 
        return []
    
    aen = df[col].values
    finite_mask = np.isfinite(aen)
    if finite_mask.sum() == 0: 
        return []
    
    aen_nonan = aen[finite_mask]
    med = np.median(aen_nonan)
    sigma = 1.4826 * np.median(np.abs(aen_nonan - med))
    
    # Fallback to standard deviation if data lacks dispersion
    if sigma <= 0 or not np.isfinite(sigma): 
        sigma = np.nanstd(aen_nonan) if np.nanstd(aen_nonan) > 0 else 0.01
        
    threshold = med + k_sigma * sigma
    return list(df[df[col] > threshold].index)
