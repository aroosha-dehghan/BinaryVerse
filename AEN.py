# AEN
# بر اساس اختلال ایجاد شده در فیت شدن داده‌ها با مدل اثر حرکت دوتایی‌ها حول مرکز جرم
# مشکل: حساس به نویز


import os
import glob
import re
import numpy as np
import pandas as pd
from astropy.io.votable import parse
import matplotlib.pyplot as plt

def identify_binaries_aen(df, col="astrometric_excess_noise", k_sigma=3):
    if col not in df.columns:
        return pd.DataFrame(), np.nan
    
    aen = df[col].values
    aen = aen[np.isfinite(aen)]
    if len(aen) == 0:
        return pd.DataFrame(), np.nan
    
    median_aen = np.median(aen)
    mad = np.median(np.abs(aen - median_aen))  
    sigma = 1.4826 * mad
    
    threshold = median_aen + k_sigma * sigma
    binaries = df[df[col] > threshold].copy()
    
    return binaries, threshold

vot_dir = "/Users/aroosha/Desktop/mine/uni/astro/AKU/Gaia/BinarySystems/Members-Fractionage/members"
out_dir = "/Users/aroosha/Desktop/aen_binaries_output"
os.makedirs(out_dir, exist_ok=True)

summaries = []
for f in glob.glob(os.path.join(vot_dir, "*.vot")):
    cluster_id = os.path.splitext(os.path.basename(f))[0]
    table = parse(f).get_first_table().to_table()
    df = table.to_pandas()
    
    binaries, threshold = identify_binaries_aen(df, col="inrt", k_sigma=3)  
    
    out_csv = os.path.join(out_dir, f"{cluster_id}_aen_binaries.csv")
    binaries.to_csv(out_csv, index=False)
    
    summaries.append({"cluster": cluster_id, "n_binaries": len(binaries), "threshold": threshold})

pd.DataFrame(summaries).to_csv(os.path.join(out_dir, "aen_summary.csv"), index=False)
