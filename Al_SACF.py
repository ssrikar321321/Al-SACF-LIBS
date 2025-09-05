#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 13:55:57 2025

@author: saisrikar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aluminium Spectral Boltzmann + Self-Absorption Correction (SACF)
Focused tool: only Boltzmann plot (raw) and SACF-corrected Boltzmann plot.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

K_B = 8.617e-5  # Boltzmann constant (eV/K) -- not used directly but kept for reference

st.set_page_config(layout="wide")
st.title("Aluminium: Boltzmann plot & Self-Absorption Correction (SACF)")

# -------------------------
# --- Minimal NIST-like Al table (from user's provided data)
# -------------------------
# NOTE: g values estimated from J (2J+1) using the J values in your input.
# If you have exact statistical weights, replace the 'g' column accordingly.
nist_al = pd.DataFrame({
    "Wavelength_nm": [308.21529, 309.27084, 309.28386, 394.40058, 396.15200],
    "Aki_s-1":      [5.87e7,    7.29e7,    1.16e7,    4.99e7,    9.85e7],
    "Upper_State":  ["3s23d 2D 3/2", "3s23d 2D 5/2", "3s23d 2D 3/2", "3s24s 2S 1/2", "3s24s 2S 1/2"],
    "E_upper_eV":   [4.0214836, 4.0216502, 4.0214830, 3.1427212, 3.1427212],
    # approximate g = 2J+1: J=3/2 -> 4, J=5/2 -> 6, J=1/2 -> 2
    "g":            [4, 6, 4, 2, 2]
})

st.sidebar.header("Peak matching / plotting options")
peak_tol = st.sidebar.number_input("Peak-match tolerance (nm)", min_value=0.01, value=0.3, step=0.01)
peak_thresh = st.sidebar.slider("Peak detection threshold (relative)", 0.0, 1.0, 0.2)
signif_pct = st.sidebar.number_input("Flag change > (%)", min_value=1.0, value=10.0, step=1.0)

st.markdown("## Upload measurement spectrum (two-column whitespace or csv)")
measurement_file = st.file_uploader("Measurement Spectrum (.asc | .txt | .csv)", ["asc", "txt", "csv"])

if measurement_file is None:
    st.info("Upload a measurement file to continue. Format: two columns (wavelength [nm], intensity).")
    st.stop()

# -------------------------
# --- Read measurement file robustly
# -------------------------
def load_two_col(file):
    try:
        df = pd.read_csv(file, delim_whitespace=True, header=None, comment="#")
        if df.shape[1] >= 2:
            return df.iloc[:, 0].values, df.iloc[:, 1].values
    except Exception:
        pass
    # try comma
    file.seek(0)
    try:
        df = pd.read_csv(file, header=None)
        if df.shape[1] >= 2:
            return df.iloc[:, 0].values, df.iloc[:, 1].values
    except Exception:
        pass
    raise ValueError("Could not parse measurement file as two columns.")
    
wl_meas, inten_meas = load_two_col(measurement_file)

# optional background (still accept; though tool is focused on Boltzmann + SACF)
background_file = st.file_uploader("Background Spectrum (optional)", ["asc", "txt", "csv"])
if background_file:
    try:
        _, bg_int = load_two_col(background_file)
        # align lengths if same sampling assumed; safer to subtract via interpolation
        interp_bg = np.interp(wl_meas, np.linspace(wl_meas.min(), wl_meas.max(), len(bg_int)), bg_int)
        inten_meas = np.clip(inten_meas - interp_bg, 0, None)
    except Exception:
        st.warning("Could not parse background automatically; background ignored.")

# normalize for peak detection (relative)
norm_int = inten_meas / (inten_meas.max() if inten_meas.max() > 0 else 1.0)
peaks_idx, _ = find_peaks(norm_int, height=peak_thresh)
wl_peaks = wl_meas[peaks_idx]
inten_peaks = inten_meas[peaks_idx]

st.subheader("Detected peaks (within measurement)")
detected_df = pd.DataFrame({"Peak_wl_nm": wl_peaks, "Peak_intensity": inten_peaks})
st.dataframe(detected_df)

# --- match detected peaks to NIST Al lines (by nearest wl)
matched = []
for wl, inten in zip(wl_peaks, inten_peaks):
    diff = np.abs(nist_al["Wavelength_nm"] - wl)
    idx = int(diff.idxmin())
    if diff.iloc[idx] <= peak_tol:
        row = nist_al.loc[idx]
        matched.append({
            "Measured_peak_nm": float(wl),
            "Measured_intensity": float(inten),
            "NIST_wavelength_nm": float(row["Wavelength_nm"]),
            "Aki_s-1": float(row["Aki_s-1"]),
            "Upper_State": row["Upper_State"],
            "E_upper_eV": float(row["E_upper_eV"]),
            "g": int(row["g"])
        })

if len(matched) == 0:
    st.error("No measured peaks matched to the Al list. Try increasing tolerance or lowering threshold.")
    st.stop()

df_match = pd.DataFrame(matched).reset_index(drop=True)
st.subheader("Matched lines (to Al list)")
st.dataframe(df_match)

# --- select subset to use for Boltzmann (checkbox + default: all)
st.write("Select which matched lines to include in Boltzmann analysis")
use_mask = st.multiselect("Use rows (indices)", options=df_match.index.tolist(),
                          default=list(df_match.index.tolist()),
                          format_func=lambda i: f"{i}: {df_match.loc[i,'NIST_wavelength_nm']:.2f} nm")
if not use_mask:
    st.error("Select at least one matched line.")
    st.stop()

final = df_match.loc[use_mask].reset_index(drop=True)

# --- Compute normalized intensities (for Boltzmann we use relative units; use I/gA)
I_raw = final["Measured_intensity"].values.astype(float)
Aki = final["Aki_s-1"].values.astype(float)
E_up = final["E_upper_eV"].values.astype(float)
g = final["g"].values.astype(float)

# Prevent zeros / negative intensities for log
I_raw_clip = np.clip(I_raw, a_min=1e-12, a_max=None)

y_raw = np.log(I_raw_clip / (g * Aki))
lr_raw = LinearRegression().fit(E_up.reshape(-1,1), y_raw)
slope_raw = lr_raw.coef_[0]
Te_raw = -1.0 / slope_raw if slope_raw != 0 else np.nan

# Plot raw Boltzmann
fig_raw, ax_raw = plt.subplots(figsize=(6,4))
ax_raw.scatter(E_up, y_raw, label="raw data")
E_lin = np.linspace(E_up.min(), E_up.max(), 50)
ax_raw.plot(E_up, lr_raw.predict(E_up.reshape(-1,1)), "--", label=f"Linear fit (Te = {Te_raw:.2f} eV)")
ax_raw.set_xlabel("E_upper (eV)")
ax_raw.set_ylabel("ln(I / (g A))")
ax_raw.set_title("Boltzmann plot — RAW")
ax_raw.legend()
st.pyplot(fig_raw)

# --- SACF correction
st.markdown("## Self-absorption correction (SACF)")
ref_choice = st.selectbox("Choose reference line (for SACF)", options=final.index,
                          format_func=lambda i: f"{final.loc[i,'NIST_wavelength_nm']:.2f} nm (idx {i})")

def SACF(I, I1, A, A1, G, G1, E, E1, etemp):
    # original formula from your script; ensure etemp is >0
    if etemp <= 0:
        return 1.0
    fb = (I / I1) * (A1 / A) * (G1 / G) * np.exp((E - E1) / etemp)
    return fb if fb < 1.0 else 1.0

I_vec = I_raw_clip.copy()
I_ref = float(I_vec[ref_choice])
A_ref = float(Aki[ref_choice])
g_ref = float(g[ref_choice])
E_ref = float(E_up[ref_choice])

# Use Te from raw Boltzmann as etemp for SACF (as your original script did)
etemp_for_sacf = Te_raw if not np.isnan(Te_raw) and Te_raw > 0 else 1.0

z_fac = np.array([SACF(I_vec[i], I_ref, Aki[i], A_ref, g[i], g_ref, E_up[i], E_ref, etemp_for_sacf)
                  for i in range(len(I_vec))])
# Avoid division by zero:
z_fac = np.clip(z_fac, 1e-12, 1.0)

I_corr = I_vec / z_fac
I_corr_clip = np.clip(I_corr, a_min=1e-12, a_max=None)

# Recompute Boltzmann with corrected intensities
y_corr = np.log(I_corr_clip / (g * Aki))
lr_corr = LinearRegression().fit(E_up.reshape(-1,1), y_corr)
slope_corr = lr_corr.coef_[0]
Te_corr = -1.0 / slope_corr if slope_corr != 0 else np.nan

# Plot corrected Boltzmann
fig_corr, ax_corr = plt.subplots(figsize=(6,4))
ax_corr.scatter(E_up, y_corr, label="SACF-corrected data")
ax_corr.plot(E_up, lr_corr.predict(E_up.reshape(-1,1)), "-",
             label=f"Linear fit (Te = {Te_corr:.2f} eV)")
ax_corr.set_xlabel("E_upper (eV)")
ax_corr.set_ylabel("ln(I / (g A))")
ax_corr.set_title("Boltzmann plot — SACF corrected")
ax_corr.legend()
st.pyplot(fig_corr)

st.success(f"Inferred Te — RAW: {Te_raw:.3f} eV | SACF-corrected: {Te_corr:.3f} eV")

# -------------------------
# Table: show changes (wavelengths don't change under SACF; intensities do)
# -------------------------
result_tbl = final.copy()
result_tbl["I_raw"] = I_vec
result_tbl["I_corr"] = I_corr
result_tbl["z_factor"] = z_fac
result_tbl["pct_change_%"] = 100.0 * (result_tbl["I_corr"] - result_tbl["I_raw"]) / np.where(result_tbl["I_raw"]==0, 1e-12, result_tbl["I_raw"])
result_tbl["significant_change"] = np.abs(result_tbl["pct_change_%"]) > float(signif_pct)

st.subheader("Raw vs SACF-corrected results")
st.dataframe(result_tbl[["NIST_wavelength_nm","Measured_peak_nm","I_raw","I_corr","z_factor","pct_change_%","significant_change"]])

# CSV download
csv_bytes = result_tbl.to_csv(index=False).encode()
st.download_button("Download corrected results (CSV)", csv_bytes, file_name="al_sacf_corrected.csv", mime="text/csv")

st.markdown("""
**Notes & caveats**
- SACF modifies intensities (I) not wavelengths; therefore matched wavelengths remain the same.
- g values were inferred from the J values in your pasted data (use exact g if available).
""")
