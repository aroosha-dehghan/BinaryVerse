# Direct imaging/separation
# برای باینری‌های باز و نزدیک
# مشکل: باینری‌های خیلی باز قابل تفکیک نیستند چون اساس روش تشخیص باینری‌ها بر اساس جدایی‌زاویه‌ای هست




import numpy as np
import pandas as pd
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
import astropy.units as u

def identify_binaries_direct_imaging(df, sep_thresh_au=1000):
    if not {"ra","dec","parallax"}.issubset(df.columns):
        return []

    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg)
    parallax_median = np.nanmedian(df["parallax"].values)
    if parallax_median <= 0:
        return []

    distance_pc = 1000.0 / parallax_median
    sep_thresh_pc = sep_thresh_au / 206265.0  

    binaries = []
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            sep = coords[i].separation(coords[j]).to(u.rad).value * distance_pc
            if sep < sep_thresh_pc:
                binaries.append((i,j))
    return binaries


table = parse("my_cluster.vot").get_first_table().to_table()
df = table.to_pandas()
df = df.rename(columns={"RA_ICRS":"ra","DE_ICRS":"dec","Plx":"parallax"})

pairs = identify_binaries_direct_imaging(df, sep_thresh_au=1000)
print(f"🔍 Found {len(pairs)} candidate binary pairs")
