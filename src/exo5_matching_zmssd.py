import argparse
from pathlib import Path
import numpy as np

from utils_image import read_gray, gaussian_blur, harris_response, harris_corners
from utils_matching import filter_points_border, extract_zero_mean_patches, match_zmssd
from utils_visu import save_matches_png


def detect_harris_points(I: np.ndarray, topN: int, k: float, gsize: int, gsigma: float,
                         pre_gauss: bool, pre_gsize: int, pre_gsigma: float,
                         nms_m: int):
    Iw = gaussian_blur(I, size=pre_gsize, sigma=pre_gsigma) if pre_gauss else I
    R = harris_response(Iw, k=k, gsize=gsize, gsigma=gsigma)
    posr, posc, _ = harris_corners(R, mode="top", topN=topN, nms_m=nms_m)
    return posr, posc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--im1", type=str, default="/home/mahdidou711/linux_data/Projets/Py/tp1 453/Detection/set1-1.png")
    parser.add_argument("--im2", type=str, default="/home/mahdidou711/linux_data/Projets/Py/tp1 453/Detection/set1-2.png")
    parser.add_argument("--outdir", type=str, default="figures")

    # points (Harris)
    parser.add_argument("--top1", type=int, default=400)
    parser.add_argument("--top2", type=int, default=400)
    parser.add_argument("--k", type=float, default=0.04)
    parser.add_argument("--gsize", type=int, default=5)
    parser.add_argument("--gsigma", type=float, default=1.2)
    parser.add_argument("--nms_m", type=int, default=2)
    parser.add_argument("--pre_gauss", action="store_true")
    parser.add_argument("--pre_gsize", type=int, default=5)
    parser.add_argument("--pre_gsigma", type=float, default=1.0)

    # ZMSSD
    parser.add_argument("--w", type=int, default=7, help="demi-taille patch")
    parser.add_argument("--tau", type=float, default=-1.0, help="seuil ZMSSD (<=0: auto quantile)")
    parser.add_argument("--tau_q", type=float, default=0.80, help="quantile sur d1 si tau<=0")
    parser.add_argument("--ratio", type=float, default=-1.0, help="ratio test (<=0: off)")
    parser.add_argument("--mutual", action="store_true", help="married matching")
    parser.add_argument("--keepK", type=int, default=60, help="Garder seulement les K meilleurs matches (lisibilité)")
    parser.add_argument("--max_drow", type=int, default=25)
    parser.add_argument("--min_dcol", type=int, default=-200)
    parser.add_argument("--max_dcol", type=int, default=200)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    I1 = read_gray(args.im1)
    I2 = read_gray(args.im2)
    H1, W1 = I1.shape
    H2, W2 = I2.shape

    # 1) Points Harris
    posr1, posc1 = detect_harris_points(I1, args.top1, args.k, args.gsize, args.gsigma,
                                        args.pre_gauss, args.pre_gsize, args.pre_gsigma, args.nms_m)
    posr2, posc2 = detect_harris_points(I2, args.top2, args.k, args.gsize, args.gsigma,
                                        args.pre_gauss, args.pre_gsize, args.pre_gsigma, args.nms_m)

    # 2) Filtrage bords (patch complet)
    posr1, posc1 = filter_points_border(posr1, posc1, H1, W1, args.w)
    posr2, posc2 = filter_points_border(posr2, posc2, H2, W2, args.w)

    # 3) Patchs centrés, centrage moyenne
    P1 = extract_zero_mean_patches(I1, posr1, posc1, args.w)
    P2 = extract_zero_mean_patches(I2, posr2, posc2, args.w)

    # 4) Matching
    tau = None if args.tau <= 0 else float(args.tau)
    tau_q = float(args.tau_q) if args.tau <= 0 else None
    ratio = None if args.ratio <= 0 else float(args.ratio)

    bestj, bestd, tau_used = match_zmssd(P1, P2, tau=tau, tau_quantile=tau_q, ratio=ratio, mutual=args.mutual)

    # Construire listes (bestr,bestc) en 1-based (ou -1 si rejet)
    bestr = np.full(len(posr1), -1, dtype=np.int32)
    bestc = np.full(len(posr1), -1, dtype=np.int32)
    ok = bestj >= 0
    bestr[ok] = posr2[bestj[ok]]
    bestc[ok] = posc2[bestj[ok]]
    # Filtre déplacement (1-based -> diff en pixels)
    drow = (bestr - posr1)
    dcol = (bestc - posc1)

    ok &= (np.abs(drow) <= args.max_drow)
    ok &= (dcol >= args.min_dcol) & (dcol <= args.max_dcol)

    bestr[~ok] = -1
    bestc[~ok] = -1
# Option: ne garder que les K meilleurs matches (pour lisibilité)
    keepK = getattr(args, "keepK", 0)
    if keepK and keepK > 0:
        idx = np.where(ok)[0]
        if len(idx) > keepK:
            order = np.argsort(bestd[idx])  # bestd = distance, plus petit = meilleur
            keep_idx = idx[order[:keepK]]
            ok2 = np.zeros_like(ok)
            ok2[keep_idx] = True
            ok = ok2
            bestr[~ok] = -1
            bestc[~ok] = -1
    print("I1:", I1.shape, "points1:", int(len(posr1)))
    print("I2:", I2.shape, "points2:", int(len(posr2)))
    print("w:", int(args.w), "patch:", int(2*args.w+1), "x", int(2*args.w+1))
    print("tau_used:", None if tau_used is None else float(tau_used))
    print("mutual:", bool(args.mutual), "ratio:", None if ratio is None else float(ratio))
    print("matches_kept:", int(np.sum(ok)), "/", int(len(posr1)))

    out_path = outdir / "matches.png"
    save_matches_png(I1, I2, posr1, posc1, bestr, bestc, str(out_path), mark_radius=2, line_width=1)
    print("saved:", out_path)


if __name__ == "__main__":
    main()