import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils_image import read_gray, gaussian_blur, fast_score_map, nms_2d_score, topN_from_score
from utils_visu import draw_corners_rgb, save_rgb


def save_gray(path: str, im: np.ndarray, title: str):
    plt.figure()
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_score_vis(path: str, S: np.ndarray, title: str):
    # normalisation en [0,255] pour visualisation
    m = float(np.max(S))
    if m <= 0:
        vis = np.zeros_like(S, dtype=np.uint8)
    else:
        vis = (255.0 * (S.astype(np.float32) / m)).astype(np.uint8)

    plt.figure()
    plt.imshow(vis, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def downscale(I: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return I
    im = Image.fromarray(np.clip(I, 0, 255).astype(np.uint8), mode="L")
    new_w = max(1, int(round(im.size[0] * scale)))
    new_h = max(1, int(round(im.size[1] * scale)))
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.array(im, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--outdir", type=str, default="figures")

    # FAST
    parser.add_argument("--t", type=int, default=20)
    parser.add_argument("--N", type=int, default=12)

    # NMS + sélection
    parser.add_argument("--nms_m", type=int, default=2)
    parser.add_argument("--topN", type=int, default=300)

    # prétraitement
    parser.add_argument("--pre_gauss", action="store_true")
    parser.add_argument("--gsize", type=int, default=5)
    parser.add_argument("--gsigma", type=float, default=1.0)

    # downscale (perf)
    parser.add_argument("--scale", type=float, default=1.0)

    # marqueurs
    parser.add_argument("--mark", type=int, default=3)

    args = parser.parse_args()

    img_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    I0 = read_gray(str(img_path))
    I = downscale(I0, float(args.scale))

    I_work = I
    if args.pre_gauss:
        I_work = gaussian_blur(I, size=args.gsize, sigma=args.gsigma)

    S = fast_score_map(I_work.astype(np.uint8), t=int(args.t), N=int(args.N))

    mask = nms_2d_score(S, m=int(args.nms_m), threshold=int(args.t))
    posr, posc = topN_from_score(S, mask, topN=int(args.topN))

    print("I_used:", I.shape, "min/max:", float(np.min(I)), float(np.max(I)))
    print("S max:", int(np.max(S)), "nb_points_after_nms:", int(np.sum(mask)), "topN:", int(len(posr)))

    # sauvegardes
    save_gray(str(outdir / f"{img_path.stem}_gray.png"), I, "Image (gris, utilisée)")
    save_score_vis(str(outdir / f"{img_path.stem}_S.png"), S, "FAST score (normalisé)")
    rgb = draw_corners_rgb(I, posr, posc, radius=int(args.mark))
    save_rgb(str(outdir / f"{img_path.stem}_fast.png"), rgb)


if __name__ == "__main__":
    main()