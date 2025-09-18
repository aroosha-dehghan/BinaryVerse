# Combination

import os
import glob
import numpy as np
import pandas as pd
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d



# CMD
def identify_binaries_cmd(df):
    cols = {"phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag","parallax"}
    if not cols.issubset(df.columns):
        return []
    sub = df.dropna(subset=list(cols)).copy()
    if sub.empty:
        return []
    sub["bp_rp"] = sub["phot_bp_mean_mag"] - sub["phot_rp_mean_mag"]
    sub["distance_pc"] = 1000.0 / sub["parallax"]
    sub["abs_mag_g"] = sub["phot_g_mean_mag"] - 5*np.log10(sub["distance_pc"]/10.0)
    sub_sorted = sub.sort_values("bp_rp")
    smooth_size = min(50, max(5, len(sub_sorted)//5))
    main_seq = uniform_filter1d(sub_sorted["abs_mag_g"].values, size=smooth_size)
    sub_sorted["delta_mag"] = main_seq - sub_sorted["abs_mag_g"]
    binaries = sub_sorted[sub_sorted["delta_mag"] > 0.3]
    return list(binaries.index)




# PM+Plx
def identify_binaries_pm_plx(df, k_sigma_pm=3.0, k_sigma_plx=3.0, sep_thresh_pc=0.05):
    cols = {"ra","dec","pmRA","pmDE","parallax"}
    if not cols.issubset(df.columns):
        return []
    n = len(df)
    if n<2: return []
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg)
    pmRA = df["pmRA"].values
    pmDE = df["pmDE"].values
    pm_mag = np.hypot(pmRA, pmDE)
    sigma_pm = 1.4826 * np.median(np.abs(pm_mag - np.nanmedian(pm_mag)))
    if sigma_pm <=0 or not np.isfinite(sigma_pm): sigma_pm = 0.2
    pm_thresh = k_sigma_pm * sigma_pm
    plx = df["parallax"].values
    sigma_plx = 1.4826 * np.median(np.abs(plx - np.nanmedian(plx)))
    if sigma_plx <=0 or not np.isfinite(sigma_plx): sigma_plx = 0.05
    plx_thresh = k_sigma_plx * sigma_plx
    median_plx = np.nanmedian(plx)
    if median_plx <=0 or not np.isfinite(median_plx): return []
    distance_pc = 1000.0 / median_plx
    pairs = []
    for i in range(n):
        for j in range(i+1,n):
            dpm = np.hypot(pmRA[i]-pmRA[j], pmDE[i]-pmDE[j])
            dplx = abs(plx[i]-plx[j])
            sep_pc = coords[i].separation(coords[j]).to(u.rad).value * distance_pc
            if dpm <= pm_thresh and dplx <= plx_thresh and sep_pc <= sep_thresh_pc:
                pairs.append(df.index[i])
                pairs.append(df.index[j])
    return list(set(pairs))




# AEN
def identify_binaries_aen(df, col="inrt", k_sigma=3):
    if col not in df.columns: return []
    aen = df[col].values
    finite_mask = np.isfinite(aen)
    if finite_mask.sum() == 0: return []
    aen_nonan = aen[finite_mask]
    med = np.median(aen_nonan)
    mad = np.median(np.abs(aen_nonan - med))
    sigma = 1.4826 * mad
    if sigma <=0 or not np.isfinite(sigma): sigma = np.nanstd(aen_nonan) if np.nanstd(aen_nonan)>0 else 0.01
    thr = med + k_sigma*sigma
    sel = df[df[col]>thr]
    return list(sel.index)


# Direct Imaging
def identify_binaries_direct(df, sep_thresh_au=1000):
    cols = {"ra","dec","parallax"}
    if not cols.issubset(df.columns): return []
    n = len(df)
    if n<2: return []
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg)
    median_plx = np.nanmedian(df["parallax"].values)
    if median_plx <=0 or not np.isfinite(median_plx): return []
    distance_pc = 1000.0 / median_plx
    sep_thresh_pc = sep_thresh_au / 206265.0
    pairs = []
    for i in range(n):
        for j in range(i+1,n):
            sep_pc = coords[i].separation(coords[j]).to(u.rad).value * distance_pc
            if sep_pc <= sep_thresh_pc:
                pairs.append(df.index[i])
                pairs.append(df.index[j])
    return list(set(pairs))




# cluster 
def process_cluster(vot_path, save_dir):
    cluster_id = os.path.splitext(os.path.basename(vot_path))[0]
    print(f"\n🔄 Processing {cluster_id} ...")
    try:
        table = parse(vot_path).get_first_table().to_table()
        df = table.to_pandas()
    except Exception as e:
        print(f"❌ Failed to parse {vot_path}: {e}")
        return None

    df = df.rename(columns={
        "RA_ICRS":"ra","DE_ICRS":"dec",
        "Plx":"parallax","Gmag":"phot_g_mean_mag",
        "BPmag":"phot_bp_mean_mag","RPmag":"phot_rp_mean_mag"
    })
    df = df.dropna(subset=["ra","dec","parallax"]).copy()
    if df.empty: return None
    df = df.reset_index(drop=True)

    cmd_ids = identify_binaries_cmd(df)
    pm_ids = identify_binaries_pm_plx(df)
    aen_ids = identify_binaries_aen(df)
    dir_ids = identify_binaries_direct(df)

    all_ids = set(cmd_ids) | set(pm_ids) | set(aen_ids) | set(dir_ids)
    sets = {"cmd": set(cmd_ids), "pm": set(pm_ids), "aen": set(aen_ids), "dir": set(dir_ids)}



    # Strong Candidates = detected by at least two methods
    strong_ids = set()
    keys = list(sets.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            strong_ids |= (sets[keys[i]] & sets[keys[j]])

    df["is_strong"] = df.index.isin(strong_ids)

    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"{cluster_id}_binaries.csv")
    if len(all_ids) > 0:
        df.loc[sorted(all_ids)].to_csv(out_csv, index=False)
        print("Saved candidates to:", out_csv)

    # Plot
    plt.figure(figsize=(7,7))
    plt.scatter(df["ra"], df["dec"], s=2, c="lightgray", alpha=0.6, label="All stars")
    if cmd_ids: plt.scatter(df.loc[cmd_ids,"ra"], df.loc[cmd_ids,"dec"], s=20, c="red", alpha=0.7, label="CMD")
    if pm_ids: plt.scatter(df.loc[pm_ids,"ra"], df.loc[pm_ids,"dec"], s=20, c="orange", alpha=0.7, label="PM+Plx")
    if aen_ids: plt.scatter(df.loc[aen_ids,"ra"], df.loc[aen_ids,"dec"], s=20, c="blue", alpha=0.7, label="AEN")
    if dir_ids: plt.scatter(df.loc[dir_ids,"ra"], df.loc[dir_ids,"dec"], s=20, c="green", alpha=0.7, label="Direct sep")
    for idx in strong_ids:
        plt.scatter(df.loc[idx,"ra"], df.loc[idx,"dec"], s=200, facecolors='none', edgecolors='black', linewidths=1.2)
        plt.text(df.loc[idx,"ra"], df.loc[idx,"dec"], str(idx), fontsize=8, color='black')

    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.title(f"{cluster_id}: total_cand={len(all_ids)}, strong={len(strong_ids)}")
    plt.legend(loc="best")
    plt.tight_layout()
    out_png = os.path.join(save_dir, f"{cluster_id}_binaries.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved plot to:", out_png)

    return {
        "cluster": cluster_id,
        "n_cmd": len(cmd_ids),
        "n_pm": len(pm_ids),
        "n_aen": len(aen_ids),
        "n_dir": len(dir_ids),
        "n_all": len(all_ids),
        "n_strong": len(strong_ids)
    }


if __name__=="__main__":
    vot_dir = "/Users/aroosha/Desktop/mine/uni/astro/AKU/Gaia/BinarySystems/Members-Fractionage/members"
    save_dir = "/Users/aroosha/Desktop/mine/uni/astro/AKU/Gaia/BinarySystems/Binaries/CombinedBinaries"
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for file in glob.glob(os.path.join(vot_dir,"*.vot")):
        res = process_cluster(file, save_dir)
        if res:
            results.append(res)

    if results:
        summary_csv = os.path.join(save_dir,"summary.csv")
        pd.DataFrame(results).to_csv(summary_csv, index=False)
        print("\n📊 Summary saved to:", summary_csv)

