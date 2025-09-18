# PM-Plx
# اختلاف بین پارالکس و پراپرموشن کم باشه: احتمال باینری بودن
# محاسبهٔ Separation بین تمام ستاره‌ها به شکل دو به دو
# مشکل: احتمال حذف ستاره‌های کم نور


# با مقدارهای حدی پیش‌فرض
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd

def identify_binaries_pm_plx(df, pm_thresh=2.0, plx_thresh=0.2, sep_thresh_pc=0.5):
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg)
    distance_pc = 1000 / np.nanmedian(df["parallax"])  

    binaries = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            d_pm = np.sqrt((df["pmra"].iloc[i]-df["pmra"].iloc[j])**2 + 
                           (df["pmdec"].iloc[i]-df["pmdec"].iloc[j])**2)
            d_plx = abs(df["parallax"].iloc[i] - df["parallax"].iloc[j])
            d_sep = coords[i].separation(coords[j]).to(u.rad).value * distance_pc

            if (d_pm < pm_thresh) and (d_plx < plx_thresh) and (d_sep < sep_thresh_pc):
                binaries.append((i, j))
    return binaries




# بدون مقادیر پیش فرض / استخراج از خوشه
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
import astropy.units as u

# MAD 
def mad_to_sigma(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad


def build_cartesian_pc(ra_deg, dec_deg, parallax_mas):
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_pc = 1000.0 / parallax_mas
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = dist_pc * np.cos(dec) * np.cos(ra)
    y = dist_pc * np.cos(dec) * np.sin(ra)
    z = dist_pc * np.sin(dec)
    return np.vstack((x, y, z)).T, dist_pc

# ‌محاسبهٔ آستانه‌ها

def compute_adaptive_thresholds(df, k_pm=3.0, k_plx=3.0, k_sep=5.0, frac_rel_plx=0.05):
    # PM
    pm_mag = np.sqrt(df["pmRA"]**2 + df["pmDE"]**2)
    sigma_pm = mad_to_sigma(pm_mag)
    if np.isnan(sigma_pm) or sigma_pm < 0.05:
        sigma_pm = 0.2
    pm_thresh = k_pm * sigma_pm

    # Plx
    sigma_plx = mad_to_sigma(df["parallax"])
    median_plx = np.nanmedian(df["parallax"])
    if np.isnan(sigma_plx) or sigma_plx <= 0:
        sigma_plx = max(0.03, 0.01 * abs(median_plx))
    plx_thresh = max(k_plx * sigma_plx, frac_rel_plx * abs(median_plx))

    # 3D distance
    coords3d, dist_pc = build_cartesian_pc(df["RA_ICRS"], df["DE_ICRS"], df["parallax"])
    valid_idx = np.where(~np.isnan(dist_pc))[0]
    sep_thresh = 0.1
    if valid_idx.size >= 2:
        tree = cKDTree(coords3d[valid_idx])
        dists, _ = tree.query(coords3d[valid_idx], k=2, n_jobs=-1)
        nn_med = np.median(dists[:,1])
        if not np.isfinite(nn_med) or nn_med <= 0:
            nn_med = np.mean(dists[:,1][np.isfinite(dists[:,1])]) if np.any(np.isfinite(dists[:,1])) else 0.01
        sep_thresh = max(0.01, k_sep * nn_med)
    return pm_thresh, plx_thresh, sep_thresh


# شناسایی زوج‌ها
def identify_binaries_adaptive_pm_plx(df, k_pm=3.0, k_plx=3.0, k_sep=5.0, frac_rel_plx=0.05):
    pm_thresh, plx_thresh, sep_thresh = compute_adaptive_thresholds(df, k_pm, k_plx, k_sep, frac_rel_plx)
    coords3d, dist_pc = build_cartesian_pc(df["RA_ICRS"], df["DE_ICRS"], df["parallax"])
    valid_idx = np.where(~np.isnan(dist_pc))[0]
    if valid_idx.size < 2:
        return pd.DataFrame(), {"pm_thresh":pm_thresh, "plx_thresh":plx_thresh, "sep_thresh":sep_thresh}

    tree = cKDTree(coords3d[valid_idx])
    raw_pairs = tree.query_pairs(r=sep_thresh)
    results = []
    for a, b in raw_pairs:
        i, j = valid_idx[a], valid_idx[b]
        dpm = np.sqrt((df.iloc[i]["pmRA"] - df.iloc[j]["pmRA"])**2 +
                      (df.iloc[i]["pmDE"] - df.iloc[j]["pmDE"])**2)
        if dpm > pm_thresh:
            continue
        dplx = abs(df.iloc[i]["parallax"] - df.iloc[j]["parallax"])
        if dplx > plx_thresh:
            continue
        sep_pc = np.linalg.norm(coords3d[i] - coords3d[j])
        results.append({
            "i_index": df.index[i],
            "j_index": df.index[j],
            "ra_i": df.iloc[i]["RA_ICRS"], "dec_i": df.iloc[i]["DE_ICRS"],
            "ra_j": df.iloc[j]["RA_ICRS"], "dec_j": df.iloc[j]["DE_ICRS"],
            "sep_pc": sep_pc, "dPM_masyr": dpm, "dPlx_mas": dplx
        })
    return pd.DataFrame(results), {"pm_thresh":pm_thresh, "plx_thresh":plx_thresh, "sep_thresh":sep_thresh}
