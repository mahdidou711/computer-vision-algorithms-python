import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils_image import (
    read_gray, gaussian_blur,
    sobel_gradients, gradient_magnitude, edge_mask_from_magnitude,
    hough_circles_accumulator, local_maxima_3d, top_n_from_mask, idx_to_params
)


def show_circles(im_gray: np.ndarray, circles, out_path: str, title: str):
    plt.figure()
    plt.imshow(im_gray, cmap="gray")
    plt.title(title)
    plt.axis("off")

    for (r, c, rad, val) in circles:
        x = c - 1
        y = r - 1

        plt.plot([x - 1, x + 1], [y, y], linewidth=1)
        plt.plot([x, x], [y - 1, y + 1], linewidth=1)

        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        xc = x + rad * np.cos(theta)
        yc = y + rad * np.sin(theta)
        plt.plot(xc, yc, linewidth=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def circ_score(val: float, rad: int) -> float:
    return float(val) / (2.0 * np.pi * float(rad) + 1e-6)


def too_close(r, c, rad, chosen, sep_rc, sep_rad):
    for (r2, c2, rad2, _, _) in chosen:
        d = ((r - r2) ** 2 + (c - c2) ** 2) ** 0.5
        if d < sep_rc and abs(rad - rad2) < sep_rad:
            return True
    return False


def pick_best_in_band(cands, rmin, rmax, key, chosen, sep_rc, sep_rad):
    """
    cands: liste (r,c,rad,val,score_circ)
    key: "raw" ou "circ"
    """
    best = None
    best_score = -1e30

    for (r, c, rad, val, sc) in cands:
        if not (rmin <= rad <= rmax):
            continue
        if too_close(r, c, rad, chosen, sep_rc, sep_rad):
            continue

        s = float(val) if key == "raw" else float(sc)
        if s > best_score:
            best_score = s
            best = (r, c, rad, val, sc)

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)

    # contours
    parser.add_argument("--t", type=float, default=0.18)

    # rayons
    parser.add_argument("--radmin", type=int, default=3)
    parser.add_argument("--radmax", type=int, default=26)
    parser.add_argument("--drad", type=int, default=1)

    # sélection
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--topK", type=int, default=5000)
    parser.add_argument("--min_votes", type=float, default=0.0)

    # gaussien optionnel
    parser.add_argument("--gauss", action="store_true")
    parser.add_argument("--gsize", type=int, default=5)
    parser.add_argument("--gsigma", type=float, default=1.2)

    # stratégie de sélection
    parser.add_argument("--strategy", type=str, default="bands", choices=["topN", "bands"])

    # anti-doublons
    parser.add_argument("--sep_rc", type=float, default=8.0)
    parser.add_argument("--sep_rad", type=float, default=3.0)

    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    img_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    I = read_gray(str(img_path))
    I_work = gaussian_blur(I, size=args.gsize, sigma=args.gsigma) if args.gauss else I

    Ix, Iy = sobel_gradients(I_work)
    mag = gradient_magnitude(Ix, Iy)
    edges = edge_mask_from_magnitude(mag, args.t)

    # diagnostics
    plt.figure(); plt.imshow(I, cmap="gray"); plt.axis("off"); plt.title("Image")
    plt.tight_layout(); plt.savefig(outdir / f"{img_path.stem}_gray.png", dpi=150); plt.close()

    plt.figure(); plt.imshow(mag, cmap="gray"); plt.axis("off"); plt.title(f"|∇I| (t={args.t})")
    plt.tight_layout(); plt.savefig(outdir / f"{img_path.stem}_mag.png", dpi=150); plt.close()

    plt.figure(); plt.imshow(edges, cmap="gray"); plt.axis("off"); plt.title(f"Contours (t={args.t})")
    plt.tight_layout(); plt.savefig(outdir / f"{img_path.stem}_edges.png", dpi=150); plt.close()

    acc = hough_circles_accumulator(edges, args.radmin, args.radmax, args.drad, weight_mag=None)

    # NMS 26 voisins : local_maxima_3d(M=1)
    mask = local_maxima_3d(acc, M=1, strict=True)

    # topK candidats (pics)
    peaks = top_n_from_mask(acc, mask, args.topK, min_val=args.min_votes)

    cands = []
    for (ir, ic, ik, val) in peaks:
        r, c, rad = idx_to_params(ir, ic, ik, args.radmin, args.drad)
        sc = circ_score(val, rad)
        cands.append((r, c, rad, float(val), float(sc)))

    circles = []
    if args.strategy == "topN":
        # simple: top-N raw avec filtre anti-doublons
        for (r, c, rad, val, sc) in cands:
            if not too_close(r, c, rad, circles, args.sep_rc, args.sep_rad):
                circles.append((r, c, rad, val, sc))
            if len(circles) >= args.N:
                break
    else:
        # stratégie bandes: 1 petit + 2 moyens + 1 grand (four.png)
        # petit (3..7) par score circonférence
        b = pick_best_in_band(cands, 3, 7, "circ", circles, args.sep_rc, args.sep_rad)
        if b: circles.append(b)

        b = pick_best_in_band(cands, 24, 40, "raw", circles, args.sep_rc, args.sep_rad)
        if b: circles.append(b)

        for _ in range(2):
            b = pick_best_in_band(cands, 8, 19, "raw", circles, args.sep_rc, args.sep_rad)
            if b: circles.append(b)

        # moyens (8..21) par votes bruts, en prend 2
        for _ in range(2):
            b = pick_best_in_band(cands, 8, 21, "raw", circles, args.sep_rc, args.sep_rad)
            if b: circles.append(b)

        circles = circles[:args.N]

    print("Cercles détectés (r,c,rad,val_acc,score_circ):")
    for (r, c, rad, val, sc) in circles:
        print(f"  (r,c,rad)=({r},{c},{rad})  val={val:.2f}  circ={sc:.4f}")

    circles_overlay = [(r, c, rad, val) for (r, c, rad, val, _) in circles]
    out_overlay = outdir / f"{img_path.stem}_circles_N{len(circles)}_t{args.t:.2f}_rad{args.radmin}-{args.radmax}_{args.strategy}.png"
    show_circles(
        I, circles_overlay, str(out_overlay),
        title=f"Cercles (N={len(circles)}, t={args.t}, rad=[{args.radmin},{args.radmax}], strat={args.strategy})"
    )
    print("saved:", out_overlay)


if __name__ == "__main__":
    main()