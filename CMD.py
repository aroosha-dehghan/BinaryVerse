# CMD
# مشکل: ممکنه ستارگان پراکنده رو تشخیص نده.
# بر اساس داده‌های فتومتری: BP, RP, G
# باینری‌های اسپکتروسکوپی باید به صورت دستی از داده‌های دیگه اضافه شوند.

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