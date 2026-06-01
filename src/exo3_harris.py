import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from utils_image import read_gray, gaussian_blur, harris_response, harris_corners
from utils_visu import draw_corners_rgb, save_rgb


def save_gray(path: str, im: np.ndarray, title: str):
    plt.figure()
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_R_map(path: str, R: np.ndarray, title: str):
    # Visualisation simple: on garde R positif et on normalise en [0,255]
    Rp = np.maximum(R, 0.0)
    m = float(np.max(Rp))
    if m <= 0:
        vis = np.zeros_like(Rp, dtype=np.uint8)
    else:
        vis = (255.0 * (Rp / m)).astype(np.uint8)

    plt.figure()
    plt.imshow(vis, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--outdir", type=str, default="figures")

    # Harris
    parser.add_argument("--k", type=float, default=0.04)
    parser.add_argument("--gsize", type=int, default=5)
    parser.add_argument("--gsigma", type=float, default=1.2)

    # Seuil / sélection
    parser.add_argument("--mode", type=str, default="rel", choices=["rel", "top"])
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--topN", type=int, default=500)
    parser.add_argument("--nms_m", type=int, default=1)

    # pré-lissage optionnel (image bruitée)
    parser.add_argument("--pre_gauss", action="store_true")
    parser.add_argument("--pre_gsize", type=int, default=5)
    parser.add_argument("--pre_gsigma", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0, help="Facteur de réduction (ex: 0.15)")
    parser.add_argument("--mark", type=int, default=3, help="Rayon de la croix (3 -> 7x7)")
    args = parser.parse_args()

    img_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    I0 = read_gray(str(img_path))  # image originale (H0,W0)

    # Downscale pour accélérer le calcul Harris
    I = I0
    scale = float(args.scale)
    if scale != 1.0:
        if not (0.05 <= scale <= 1.0):
            raise ValueError("scale must be in [0.05, 1.0]")
        im = Image.fromarray(np.clip(I0, 0, 255).astype(np.uint8), mode="L")
        new_w = max(1, int(round(im.size[0] * scale)))
        new_h = max(1, int(round(im.size[1] * scale)))
        im = im.resize((new_w, new_h), resample=Image.BILINEAR)
        I = np.array(im, dtype=np.float32)

    # Pré-lissage (sur l’image utilisée pour Harris)
    I_work = I
    if args.pre_gauss:
        I_work = gaussian_blur(I, size=args.pre_gsize, sigma=args.pre_gsigma)

    # Harris sur l'image (potentiellement downscalée)
    R = harris_response(I_work, k=args.k, gsize=args.gsize, gsigma=args.gsigma)
    posr, posc, thr = harris_corners(R, mode=args.mode, alpha=args.alpha, topN=args.topN, nms_m=args.nms_m)

    # Remap des coins vers l’image originale si downscale
    posr_draw, posc_draw = posr, posc
    if scale != 1.0 and len(posr) > 0:
        y = (posr.astype(np.float32) - 1.0)
        x = (posc.astype(np.float32) - 1.0)
        y0 = np.round(y / scale).astype(np.int32)
        x0 = np.round(x / scale).astype(np.int32)

        # clamp dans [0..H0-1],[0..W0-1]
        H0, W0 = I0.shape
        y0 = np.clip(y0, 0, H0 - 1)
        x0 = np.clip(x0, 0, W0 - 1)

        posr_draw = (y0 + 1).astype(np.int32)
        posc_draw = (x0 + 1).astype(np.int32)

    print("I0:", I0.shape, "I_used:", I.shape)
    print("R :", R.shape, "min/max:", float(np.min(R)), float(np.max(R)))
    print("threshold:", float(thr), "nb_corners:", int(len(posr)))

    # Sauvegardes (R sur l’image utilisée; coins dessinés sur l’originale)
    save_gray(str(outdir / f"{img_path.stem}_gray.png"), I0, "Image (gris)")
    save_R_map(str(outdir / f"{img_path.stem}_R.png"), R, "Harris R (normalisé, R>0)")

    rgb = draw_corners_rgb(I0, posr_draw, posc_draw, radius=args.mark)
    save_rgb(str(outdir / f"{img_path.stem}_harris.png"), rgb)

# Debug lisible : coins sur l’image downscalée (sans remap)
    rgb_small = draw_corners_rgb(I, posr, posc, radius=max(1, args.mark // 2))
    save_rgb(str(outdir / f"{img_path.stem}_harris_small.png"), rgb_small)# Afficher quelques coins
    for i in range(min(10, len(posr))):
        print(f"corner[{i}] = (r,c)=({int(posr[i])},{int(posc[i])})")


if __name__ == "__main__":
    main()