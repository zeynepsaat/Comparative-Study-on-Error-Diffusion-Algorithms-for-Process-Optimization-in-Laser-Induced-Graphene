import os
import cv2
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# 0) Helper directory
# ============================================================

OUTPUT_DIR = "SEM_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1) Parse magnification folder name (e.g.: 250x → 250)
# ============================================================

def parse_magnification(folder_name):
    folder_name = folder_name.lower()
    if folder_name.endswith("x") and folder_name[:-1].isdigit():
        return int(folder_name[:-1])
    return None


# ============================================================
# 2) SEM image bottom panel cropping
# ============================================================

def preprocess_sem(img, crop_ratio=0.88):
    h, w = img.shape
    return img[: int(h * crop_ratio), :]


# ============================================================
# 3) Pore segmentation (adaptive threshold)
# ============================================================

def get_pore_binary(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 2
    )
    return binary


# ============================================================
# 4) Save pore overlay visualization
# ============================================================

def save_pore_overlay(img, pore_binary, save_path):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = img_color.copy()

    # Pores highlighted in red
    overlay[pore_binary == 255] = (0, 0, 255)

    blended = cv2.addWeighted(overlay, 0.45, img_color, 0.55, 0)
    cv2.imwrite(save_path, blended)


# ============================================================
# 5) Thickness + Skeleton extraction
# ============================================================

def compute_skeleton(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = binary == 255
    skeleton = skeletonize(mask).astype(np.uint8)

    return skeleton


# ============================================================
# 6) Extract connectivity points (junctions)
# ============================================================

def compute_connectivity_points(skeleton):
    h, w = skeleton.shape
    points = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 1:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1
                if neighbors >= 3:
                    points.append((y, x))

    return points


# ============================================================
# 7) Save skeleton + connectivity overlay visualization
# ============================================================

def save_skeleton_overlay(img, skeleton, conn_points, save_path):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Skeleton = white
    img_color[skeleton == 1] = (255, 255, 255)

    # Junction points = blue
    for (y, x) in conn_points:
        cv2.circle(img_color, (x, y), 3, (255, 0, 0), -1)

    cv2.imwrite(save_path, img_color)


# ============================================================
# 8) Main processing function
# ============================================================

def generate_visual_outputs(folder):
    print(f"\n📁 Scanning folder: {folder}")

    for root, dirs, files in os.walk(folder):
        folder_name = os.path.basename(root)
        mag = parse_magnification(folder_name)

        # Process only magnification folders
        if mag is None:
            continue

        print(f"\n🔍 Magnification folder detected: {folder_name}")

        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                continue

            full_path = os.path.join(root, file)
            print(f"Processing: {full_path}")

            img_raw = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                print("⚠️ Error reading image")
                continue

            img = preprocess_sem(img_raw)

            base = os.path.splitext(file)[0]

            # ----------------------------------------------------
            # 1. Medium magnifications → PORE OVERLAY
            # ----------------------------------------------------
            if 2500 <= mag <= 10000:
                pore_binary = get_pore_binary(img)
                save_path = os.path.join(OUTPUT_DIR, f"{base}_pore_overlay.png")
                save_pore_overlay(img, pore_binary, save_path)
                print(f"   ✔ Saved pore overlay → {save_path}")

            # ----------------------------------------------------
            # 2. High magnifications → SKELETON + CONNECTIVITY
            # ----------------------------------------------------
            if mag >= 25000:
                skeleton = compute_skeleton(img)
                conn_points = compute_connectivity_points(skeleton)
                save_path = os.path.join(OUTPUT_DIR, f"{base}_connectivity_overlay.png")
                save_skeleton_overlay(img, skeleton, conn_points, save_path)
                print(f"   ✔ Saved skeleton/connectivity overlay → {save_path}")


# ============================================================
# 9) Main execution
# ============================================================

if __name__ == "__main__":
    folder = r"C:\Users\saatz\Desktop\Projects\SEM-Analysis\SEM_images"
    generate_visual_outputs(folder)
    print("\n🎉 All visual outputs saved in → SEM_outputs/")
