import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from utils_warp import compute_panorama_canvas, apply_H, bilinear_sample


def read_rgb(path: str) -> np.ndarray:
    """Lit une image et renvoie (H,W,3) float32 dans [0,255]."""
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.float32)


def save_rgb(path: str, img: np.ndarray) -> None:
    """Sauvegarde (H,W,3) float32/uint8."""
    out = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--im1", type=str, default="/home/mahdidou711/linux_data/Projets/Py/tp1 453/Detection/set1-1.png")
    p.add_argument("--im2", type=str, default="/home/mahdidou711/linux_data/Projets/Py/tp1 453/Detection/set1-2.png")
    p.add_argument("--H", type=str, default="figures/H_best.txt")
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--outname", type=str, default="panorama.png")
    p.add_argument("--blend", type=str, default="avg", choices=["avg", "overwrite1", "overwrite2"])
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    I1 = read_rgb(args.im1)
    I2 = read_rgb(args.im2)
    H12 = np.loadtxt(args.H, dtype=np.float64)  # H: image1 -> image2

    H1, W1, _ = I1.shape
    H2, W2, _ = I2.shape

    # 1) canvas panorama
    offx, offy, Wp, Hp = compute_panorama_canvas(W1, H1, W2, H2, H12)

    # 2) grille panorama -> coordonnées repère image1
    U = np.arange(Wp, dtype=np.float64)
    V = np.arange(Hp, dtype=np.float64)
    UU, VV = np.meshgrid(U, V)  # (Hp,Wp)

    u1 = (UU - offx).reshape(-1)  # col dans image1
    v1 = (VV - offy).reshape(-1)  # row dans image1

    # 3) masque validité image1 (échantillonnage direct)
    valid1 = (u1 >= 0) & (u1 < W1) & (v1 >= 0) & (v1 < H1)

    # 4) coordonnées correspondantes dans image2 via H (image1 -> image2)
    u2, v2 = apply_H(H12, u1, v1)

    # 5) bilinéaire dans I2
    samp2, valid2 = bilinear_sample(I2, u2, v2)

    # 6) composer panorama
    pano = np.zeros((Hp * Wp, 3), dtype=np.float32)

    # pixels venant de I1 (copie)
    if np.any(valid1):
        u1i = u1[valid1].astype(np.int32)
        v1i = v1[valid1].astype(np.int32)
        pano[valid1, :] = I1[v1i, u1i, :]

    # fusion avec I2 reprojetée
    both = valid1 & valid2
    only2 = (~valid1) & valid2

    if args.blend == "overwrite2":
        pano[valid2, :] = samp2[valid2, :]
    elif args.blend == "overwrite1":
        pano[only2, :] = samp2[only2, :]
    else:  # avg
        pano[only2, :] = samp2[only2, :]
        pano[both, :] = 0.5 * pano[both, :] + 0.5 * samp2[both, :]

    pano = pano.reshape(Hp, Wp, 3)

    out_path = outdir / args.outname
    # --- Recadrage : enlever la bordure noire ---
    valid_pan = (valid1 | valid2).reshape(Hp, Wp)
    ys, xs = np.nonzero(valid_pan)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pano = pano[y0:y1+1, x0:x1+1, :]
    save_rgb(str(out_path), pano)
    print("canvas:", (Hp, Wp), "offset:", (offx, offy))
    print("saved:", out_path)


if __name__ == "__main__":
    main()