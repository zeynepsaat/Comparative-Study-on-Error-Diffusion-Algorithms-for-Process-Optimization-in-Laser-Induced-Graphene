import os
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# GLOBAL FONT SETTINGS (Times New Roman + Bold)
# ----------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["xtick.labelcolor"] = "black"
plt.rcParams["ytick.labelcolor"] = "black"


# ----------------------------------------------------------
# SAFE FILENAME
# ----------------------------------------------------------
def clean_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip()
    name = name.replace(" ", "_")
    return name


# Lorentzian Model
def lorentzian(x, x0, a, gamma):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)


# Baseline correction (ALS)
def baseline_als(y, lam, p, niter=10):
    from scipy.sparse import diags
    from scipy.linalg import cho_solve, cho_factor

    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L)).toarray()
    w = np.ones(L)

    for _ in range(niter):
        W = np.diag(w)
        Z = W + lam * (D.T @ D)
        z = cho_solve(cho_factor(Z), w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# Peak analyzer
def analyze_band(x, y, band_range):
    mask = (x >= band_range[0]) & (x <= band_range[1])
    x_band = x[mask]
    y_band = y[mask]

    peaks, _ = find_peaks(y_band, height=0, distance=10)
    if len(peaks) == 0:
        return None, None, None

    peak_idx = peaks[np.argmax(y_band[peaks])]

    try:
        params, _ = curve_fit(
            lorentzian, x_band, y_band,
            p0=[x_band[peak_idx], y_band[peak_idx], 10],
            maxfev=5000
        )

        center = params[0]
        amplitude = params[1]
        fwhm = 2 * params[2]
        return center, amplitude, fwhm

    except:
        return None, None, None


# ----------------------------------------------------------
# MAIN PROCESSING
# ----------------------------------------------------------

folder_path = r"C:\Users\saatz\Desktop\ErrorDiff Raman\20-10"
output_folder = os.path.join(folder_path, "output_English")
os.makedirs(output_folder, exist_ok=True)

folder_name = os.path.basename(folder_path)
power, speed = folder_name.split("-")

# Peak windows
D_range = (1200, 1400)
G_range = (1500, 1650)
TwoD_range = (2500, 2900)

print("\n===== DEBUG RAW MAX VALUES =====")


# ----------------------------------------------------------
# PROCESS EACH FILE
# ----------------------------------------------------------
for file_name in os.listdir(folder_path):

    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(folder_path, file_name)

    # Load data
    data = pd.read_csv(
        file_path, sep=r"\s+", header=None,
        skiprows=1, usecols=[0, 1],
        names=["Wavenumber", "Intensity"]
    )

    x = data["Wavenumber"].values
    y = np.nan_to_num(data["Intensity"].values)

    # =============================
    # FIX FOR BROKEN JARVIS TAIL
    # =============================
    if "Jarvis" in file_name:
        mask_valid = x < 3500    # Beyond 2900 cm⁻¹ is corrupted, trim it
        x = x[mask_valid]
        y = y[mask_valid]

    # -------- REMOVE SPIKES (after trimming) --------
    y_mean = np.mean(y)
    y_std = np.std(y)
    z = (y - y_mean) / y_std
    y[z > 5] = y_mean      # Clean spikes above 5σ

    # Filtering & Baseline correction
    y_filtered = savgol_filter(y, 15, 3)
    y_corrected = y_filtered - baseline_als(y_filtered, lam=1e5, p=0.01)

    # Debug values
    raw_max = float(np.nanmax(y))
    corrected_max = float(np.nanmax(y_corrected))
    print(f"{file_name:<25} raw={raw_max:<10.1f} corrected={corrected_max:<10.1f}")

    # Peak analysis
    d_center, d_amp, d_fwhm = analyze_band(x, y_corrected, D_range)
    g_center, g_amp, g_fwhm = analyze_band(x, y_corrected, G_range)
    twod_center, twod_amp, twod_fwhm = analyze_band(x, y_corrected, TwoD_range)

    # Save Excel
    safe_name = clean_filename(os.path.splitext(file_name)[0])
    results = pd.DataFrame({
        "Metric": [
            "D Peak Center", "D Peak Intensity", "D FWHM",
            "G Peak Center", "G Peak Intensity", "G FWHM",
            "2D Peak Center", "2D Peak Intensity", "2D FWHM"
        ],
        "Value": [
            d_center, d_amp, d_fwhm,
            g_center, g_amp, g_fwhm,
            twod_center, twod_amp, twod_fwhm
        ]
    })

    results.to_excel(os.path.join(output_folder, f"{safe_name}_results.xlsx"), index=False)

    # Save corrected data for combined plot
    np.save(os.path.join(output_folder, f"{safe_name}_corrected.npy"), np.array([x, y_corrected]))

    # Individual plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Raw", alpha=0.6, linewidth=2)
    plt.plot(x, y_filtered, label="Filtered", alpha=0.6, linewidth=2)
    plt.plot(x, y_corrected, label="Baseline Corrected", alpha=0.9, linewidth=2.5)

    if d_center: plt.axvline(d_center, color="r", linestyle="--", linewidth=2)
    if g_center: plt.axvline(g_center, color="g", linestyle="--", linewidth=2)
    if twod_center: plt.axvline(twod_center, color="b", linestyle="--", linewidth=2)

    plt.title(f"Raman Spectrum - {safe_name}", fontsize=20)
    plt.xlabel("Wavenumber (cm⁻1)", fontsize=18)
    plt.ylabel("Intensity (a.u.)", fontsize=18)
    plt.xlim(1000, 3000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{safe_name}_plot.png"), dpi=1800)
    plt.close()


# ----------------------------------------------------------
# COMBINED PLOT
# ----------------------------------------------------------
Y_MIN = -10
Y_MAX = 15000

fig_combined, ax_combined = plt.subplots(figsize=(10, 6))

for file in os.listdir(output_folder):
    if file.endswith("_corrected.npy"):
        arr = np.load(os.path.join(output_folder, file))
        x, y_corrected = arr
        safe_name = file.replace("_corrected.npy", "")
        ax_combined.plot(x, y_corrected, label=safe_name, linewidth=2.2)

ax_combined.set_title(f"Baseline-Corrected Raman Spectra\n(Power={power}%, Speed={speed}%)",
                      fontsize=22)
ax_combined.set_xlabel("Wavenumber (cm⁻1)", fontsize=20)
ax_combined.set_ylabel("Intensity (a.u.)", fontsize=20)
ax_combined.set_xlim(1000, 3000)
ax_combined.set_ylim(Y_MIN, Y_MAX)
ax_combined.legend(fontsize=11)
fig_combined.tight_layout()
fig_combined.savefig(os.path.join(output_folder, "combined_spectra.png"), dpi=1800)
plt.close(fig_combined)


# ----------------------------------------------------------
# COMBINED PEAK-MARKED PLOT
# ----------------------------------------------------------
D_peaks, G_peaks, TwoD_peaks = [], [], []

for file in os.listdir(output_folder):
    if file.endswith("_results.xlsx"):
        df = pd.read_excel(os.path.join(output_folder, file))

        D_val = df.loc[df["Metric"] == "D Peak Center", "Value"].values[0]
        G_val = df.loc[df["Metric"] == "G Peak Center", "Value"].values[0]
        TwoD_val = df.loc[df["Metric"] == "2D Peak Center", "Value"].values[0]

        if not np.isnan(D_val): D_peaks.append(D_val)
        if not np.isnan(G_val): G_peaks.append(G_val)
        if not np.isnan(TwoD_val): TwoD_peaks.append(TwoD_val)

# Mean peak positions
D_mean = np.mean(D_peaks) if D_peaks else None
G_mean = np.mean(G_peaks) if G_peaks else None
TwoD_mean = np.mean(TwoD_peaks) if TwoD_peaks else None

fig_marked, ax_marked = plt.subplots(figsize=(10, 6))

for file in os.listdir(output_folder):
    if file.endswith("_corrected.npy"):
        arr = np.load(os.path.join(output_folder, file))
        x, y_corrected = arr
        safe_name = file.replace("_corrected.npy", "")
        ax_marked.plot(x, y_corrected, label=safe_name, linewidth=2.2)

if D_mean: ax_marked.axvline(D_mean, color="r", linestyle="--", linewidth=2)
if G_mean: ax_marked.axvline(G_mean, color="g", linestyle="--", linewidth=2)
if TwoD_mean: ax_marked.axvline(TwoD_mean, color="b", linestyle="--", linewidth=2)

ax_marked.set_title("Corrected Raman Spectra with Mean D/G/2D Peaks", fontsize=22)
ax_marked.set_xlabel("Wavenumber (cm⁻1)", fontsize=20)
ax_marked.set_ylabel("Intensity (a.u.)", fontsize=20)
ax_marked.set_xlim(1000, 3000)
ax_marked.set_ylim(Y_MIN, Y_MAX)
ax_marked.legend(fontsize=11)
fig_marked.tight_layout()
fig_marked.savefig(os.path.join(output_folder, "combined_spectra_marked.png"), dpi=1800)
plt.close(fig_marked)

print("\n✓ All processing completed successfully!")
