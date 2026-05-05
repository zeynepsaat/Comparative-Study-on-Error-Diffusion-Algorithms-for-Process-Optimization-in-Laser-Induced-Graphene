# Comparative Study on Error Diffusion Algorithms for Process Optimization in Laser-Induced Graphene Production

## Overview

This repository contains research code and analysis tools for investigating error diffusion algorithms used in the production of laser-induced graphene (LIG). The project focuses on comparing different error diffusion techniques and their impact on the structural and material properties of produced graphene samples.

## Project Structure

```
.
├── 4ProbeforSheetResistance.py    # 4-point probe resistance characterization script
├── SEM/                            # Scanning Electron Microscopy (SEM) analysis module
│   ├── main.py                    # Comprehensive SEM image analysis and metrics computation
│   └── detail.py                  # Detailed visualization and overlay generation for SEM images
├── Raman/                          # Raman spectroscopy analysis module
│   └── Compare.py                 # Raman spectrum processing and peak analysis
└── README.md                       # This file
```

## Modules

### 1. 4-Point Probe Sheet Resistance Characterization (`4ProbeforSheetResistance.py`)

Measures electrical properties of laser-induced graphene samples using a Keithley 2400 source meter.

**Features:**
- Automated current-voltage (I-V) characterization
- Sheet resistance calculation using 4-probe method
- Data export to Excel
- Visualization of I-V curves and resistance vs. current plots

**Dependencies:** `pyvisa`, `numpy`, `matplotlib`, `pymeasure`, `openpyxl`

### 2. SEM Image Analysis Module

#### main.py: Comprehensive Analysis
Analyzes SEM images at multiple magnifications and extracts quantitative metrics.

**Computed Metrics:**
- **Porosity (%)**: Pore area percentage in the material
- **Pore Size**: Average pore area (µm²) and equivalent diameter (µm)
- **Thickness**: Material thickness estimation (µm) at high magnification
- **Connectivity Index**: Network junction density (measure of pore interconnectivity)
- **Fractal Dimension**: Structural complexity estimate
- **GLCM Features**: Texture analysis (contrast, correlation, energy, homogeneity)

**Output:** `SEM_results_full.csv` with all metrics, plus 3D bar charts and dimensionality reduction plots (PCA, t-SNE)

**Dependencies:** `opencv-python`, `numpy`, `pandas`, `scikit-image`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

#### detail.py: Visual Analysis
Generates detailed overlay visualizations for pore and connectivity analysis.

**Outputs:**
- Pore overlay images (color-highlighted pores)
- Skeleton and connectivity overlay images (junction points marked)

**Processing by Magnification:**
- 2500x - 10000x: Pore structure visualization
- ≥ 25000x: Skeleton and junction analysis

### 3. Raman Spectroscopy Analysis (`Raman/Compare.py`)

Analyzes Raman spectroscopy data for graphene characterization.

**Features:**
- Spectral preprocessing (filtering, spike removal, baseline correction)
- Peak fitting using Lorentzian model
- D, G, and 2D peak analysis (center, intensity, FWHM)
- Individual and combined spectrum visualizations
- Automated Excel export of peak parameters

**Peak Ranges:**
- D band: 1200-1400 cm⁻¹
- G band: 1500-1650 cm⁻¹
- 2D band: 2500-2900 cm⁻¹

**Output:** Results Excel file, individual plots, combined spectra comparison, and peak-marked overlays

**Dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`

## Installation

### Requirements
- Python ≥ 3.12

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "Comparative Study on Error Diffusion Algorithms for Process Optimization in Laser-Induced Graphene Production"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install using pyproject.toml:
```bash
pip install .
```

## Usage

### 4-Point Probe Measurement
```bash
python 4ProbeforSheetResistance.py
```
Edit the script to configure:
- `sample_no`: Sample identifier
- `current_start`, `current_end`, `current_step`: Current sweep parameters
- VISA resource string for your Keithley device

### SEM Analysis
```bash
python SEM/main.py
```
Update the folder path in the script to point to your SEM image directory structure:
```
folder/
├── 250x/
│   ├── algorithm1_image1.png
│   └── algorithm2_image1.png
├── 500x/
│   ├── algorithm1_image1.png
│   └── ...
└── ...
```

### SEM Visualization
```bash
python SEM/detail.py
```
Generates visual overlays for pore and connectivity analysis in `SEM_outputs/`

### Raman Spectroscopy Analysis
```bash
python Raman/Compare.py
```
Update the script to point to your Raman data directory (should contain `.txt` files with wavenumber and intensity data)

## Data Formats

### SEM Images
- Supported formats: PNG, JPG, JPEG, TIF, BMP
- Directory structure must include magnification folders (e.g., `250x`, `500x`, `1000x`)
- Images are processed based on magnification level

### Raman Data
- Text format with space-separated columns: wavenumber and intensity
- Header row should be skipped or included as first line
- One measurement per file

## Output Files

### SEM Analysis Outputs
- `SEM_results_full.csv`: Quantitative metrics for all images
- `SEM_figures/3Dbar_*.png`: 3D visualizations of metrics vs. magnification
- `SEM_figures/PCA.png`: Principal component analysis plot
- `SEM_figures/tSNE.png`: t-SNE dimensionality reduction plot
- `SEM_outputs/`: Detailed overlay visualizations

### Raman Analysis Outputs
- `output_English/[sample]_results.xlsx`: Peak parameters (center, intensity, FWHM)
- `output_English/[sample]_plot.png`: Individual spectrum plots
- `output_English/combined_spectra.png`: All spectra overlay
- `output_English/combined_spectra_marked.png`: Spectra with mean peak positions marked

## Key Parameters

### SEM Analysis
- **Magnification ranges**:
  - Low (250-500x): General morphology
  - Medium (2500-10000x): Pore analysis
  - High (≥25000x): Thickness and connectivity
- **Image preprocessing**: Automatic bottom panel cropping (88% crop ratio)

### Raman Analysis
- **Baseline correction**: Asymmetric Least Squares (ALS) method
- **Peak fitting**: Lorentzian model
- **Savitzky-Golay filter**: Window size = 15, polynomial order = 3

## Troubleshooting

### SEM Analysis
- **No data extracted**: Check image directory structure matches expected format (magnification folders)
- **Low connectivity values**: May indicate insufficient skeleton pixels at lower magnifications

### Raman Analysis
- **Peak detection fails**: Verify input data format and remove corrupted files
- **Baseline correction issues**: Adjust `lam` and `p` parameters in `baseline_als()` function

## Contributing

This is an academic research project. For improvements, bug fixes, or questions, please contact the repository maintainer.

## License

Specify your project license here (e.g., MIT, GPL, etc.)

## Citation

If you use this code in your research, please cite:
```bibtex
@software{lid_graphene_analysis,
  title={Comparative Study on Error Diffusion Algorithms for Process Optimization in Laser-Induced Graphene Production},
  author={Saatci, Zeynep},
  year={2024},
  url={<repository-url>}
}
```

## Contact

For questions or inquiries, please contact: zeynepsaatci@example.com

---

**Last Updated:** 2024  
**Project Status:** Active Research