# Computer Vision Algorithms in Python

[![Python CI](https://github.com/mahdidou711/computer-vision-algorithms-python/actions/workflows/ci.yml/badge.svg)](https://github.com/mahdidou711/computer-vision-algorithms-python/actions/workflows/ci.yml)


This repository contains cleaned Python implementations of classical computer vision algorithms.

## Goals

The project is intended to show:

- circle detection
- Harris corner detection
- FAST corner detection
- ZMSSD-based matching
- homography estimation
- panorama generation
- visualization of intermediate image-processing results

## Repository structure

- src/ contains the Python source files
- results/figures/ contains generated visual results
- Original input images are not included; supply your own image files

## Algorithms and files

### Circle detection

- src/exo2_circle_detector_base.py

### Corner detection

- src/exo3_harris.py
- src/exo4_fast.py

### Feature matching

- src/exo5_matching_zmssd.py
- src/utils_matching.py

### Homography and panorama

- src/part2_exo1_timing.py
- src/part2_exo2_homography.py
- src/part2_exo3_panorama.py
- src/utils_homography.py
- src/utils_warp.py

### Utilities

- src/utils_image.py
- src/utils_visu.py

## Installation

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Example usage

Run one of the scripts from the repository root:

python src/exo3_harris.py /path/to/image.png --outdir figures

Provide image paths for the scripts you run. The matching, homography, and panorama scripts accept `--im1` and `--im2`; use these options instead of relying on machine-specific default paths.

Generated outputs normally go to `figures/`, while `results/figures/` contains committed reference results.

GitHub Actions currently checks Python syntax only; it does not run end-to-end image-processing tests.

## Notes

This repository contains cleaned source code and generated result figures. Course PDFs, ZIP archives, and temporary files were excluded.
