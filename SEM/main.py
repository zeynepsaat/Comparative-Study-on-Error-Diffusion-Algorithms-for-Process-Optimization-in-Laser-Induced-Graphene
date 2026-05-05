import os
import re

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm
import seaborn as sns

from skimage.morphology import skeletonize
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.interpolate import griddata

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# GLOBAL SETTINGS
# ============================================================

FIG_DIR = "SEM_figures"
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 1000


# ============================================================
# 0) Helper functions
# ============================================================

def preprocess_sem(img, crop_ratio=0.88):
    h, w = img.shape
    return img[: int(h * crop_ratio), :]


def parse_magnification(folder_name):
    m = re.match(r"^(\d+)x$", folder_name.lower())
    return int(m.group(1)) if m else None


def extract_algorithm(filename):
    return filename.split("_")[0]


def estimate_hfw_um(magnification):
    return 414000.0 / magnification


# ============================================================
# 1) Porosity
# ============================================================

def compute_porosity(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 2
    )
    return np.sum(binary == 255) / binary.size * 100.0


# ============================================================
# 2) Pore size
# ============================================================

def compute_pore_size(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 2
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return stats[1:, cv2.CC_STAT_AREA]


# ============================================================
# 3) Thickness
# ============================================================

def compute_thickness(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = binary == 255
    skeleton = skeletonize(mask).astype(np.uint8)
    dist = ndimage.distance_transform_edt(mask)
    thickness = dist[skeleton == 1] * 2
    return skeleton, thickness


# ============================================================
# 4) Connectivity
# ============================================================

def compute_connectivity(skeleton):
    length = np.sum(skeleton)
    if length == 0:
        return np.nan

    junctions = 0
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x]:
                n = np.sum(skeleton[y - 1:y + 2, x - 1:x + 2]) - 1
                if n >= 3:
                    junctions += 1

    return junctions / length


# ============================================================
# 5) Fractal Dimension (FIXED)
# ============================================================

def fractal_dimension(img):
    _, bw = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bw = bw == 0

    min_dim = min(bw.shape)
    sizes = 2 ** np.arange(int(np.log2(min_dim)), 1, -1)

    counts = []
    for s in sizes:
        H = (bw.shape[0] // s) * s
        W = (bw.shape[1] // s) * s
        S = bw[:H, :W]
        S = S.reshape(H // s, s, W // s, s)
        boxes = S.any(axis=(1, 3))
        counts.append(boxes.sum())

    counts = np.array(counts, dtype=float)
    sizes = np.array(sizes, dtype=float)

    if len(counts) < 2:
        return np.nan

    coeff = np.polyfit(
        np.log(1.0 / sizes),
        np.log(counts + 1e-9),
        1
    )
    return coeff[0]


# ============================================================
# 6) GLCM
# ============================================================

def compute_glcm_features(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
    return {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }


# ============================================================
# 7) MAIN ANALYSIS
# ============================================================

def analyze_folder(folder_path, output_csv):

    rows = []

    for root, _, files in os.walk(folder_path):
        mag = parse_magnification(os.path.basename(root))
        if mag is None:
            continue

        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                continue

            full_path = os.path.join(root, f)
            img_raw = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                continue

            micron_per_pixel = estimate_hfw_um(mag) / img_raw.shape[1]
            img = preprocess_sem(img_raw)
            alg = extract_algorithm(f)

            fract = fractal_dimension(img)
            glcm = compute_glcm_features(img)

            porosity = pore_area_um2 = pore_diam_um = np.nan
            thick_um = conn = np.nan

            if 2500 <= mag <= 10000:
                porosity = compute_porosity(img)
                pores = compute_pore_size(img)
                if len(pores):
                    pore_area_um2 = pores.mean() * micron_per_pixel ** 2
                    pore_diam_um = np.sqrt(4 * pore_area_um2 / np.pi)

            if mag >= 25000:
                sk, th = compute_thickness(img)
                if len(th):
                    thick_um = th.mean() * micron_per_pixel
                conn = compute_connectivity(sk)

            rows.append({
                "file": f,
                "algorithm": alg,
                "magnification": mag,
                "porosity_%": porosity,
                "avg_pore_area_um2": pore_area_um2,
                "avg_pore_equiv_diam_um": pore_diam_um,
                "avg_thickness_um": thick_um,
                "connectivity_index": conn,
                "fractal_dimension": fract,
                **glcm
            })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print("✔ Analysis completed →", output_csv)


# ============================================================
# 8) 3D BAR PLOTS
# ============================================================

def plot_3d_metric(csv_path, metric, mn, mx):
    df = pd.read_csv(csv_path)
    df = df[(df.magnification >= mn) & (df.magnification <= mx)]
    if df.empty:
        return

    algs = df.algorithm.unique()
    mags = sorted(df.magnification.unique())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = cm.get_cmap("Reds", len(mags))

    for j, m in enumerate(mags):
        for i, a in enumerate(algs):
            v = df[(df.algorithm == a) &
                   (df.magnification == m)][metric].mean()
            ax.bar3d(i, j, 0, 0.4, 0.4, 0 if np.isnan(v) else v,
                     color=cmap(j))

    ax.set_xticks(range(len(algs)))
    ax.set_xticklabels(algs, rotation=45)
    ax.set_yticks(range(len(mags)))
    ax.set_yticklabels([f"{m}x" for m in mags])
    ax.set_zlabel(metric)

    plt.savefig(
        os.path.join(FIG_DIR, f"3Dbar_{metric}_{mn}-{mx}.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 9) PCA + t-SNE
# ============================================================

def run_dimensionality_reduction(csv_path):
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(float).fillna(0).values
    X = StandardScaler().fit_transform(X)

    pca = PCA(2)
    Xp = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(Xp[:, 0], Xp[:, 1],
                c=df.magnification, cmap="Reds")
    plt.colorbar(label="Magnification")
    plt.savefig(os.path.join(FIG_DIR, "PCA.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()

    tsne = TSNE(2, perplexity=20, random_state=42)
    Xt = tsne.fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(Xt[:, 0], Xt[:, 1],
                c=df.magnification, cmap="Reds")
    plt.colorbar(label="Magnification")
    plt.savefig(os.path.join(FIG_DIR, "tSNE.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    folder = r"C:\Users\saatz\Desktop\Projects\SEM-Analysis\SEM_images"
    csv_out = "SEM_results_full.csv"

    analyze_folder(folder, csv_out)

    for metric in [
        "porosity_%", "avg_pore_area_um2",
        "avg_thickness_um", "connectivity_index",
        "fractal_dimension"
    ]:
        plot_3d_metric(csv_out, metric, 250, 1000000)

    run_dimensionality_reduction(csv_out)
