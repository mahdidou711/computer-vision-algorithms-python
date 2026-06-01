import argparse
from pathlib import Path
import numpy as np

from utils_image import read_gray, gaussian_blur, harris_response, harris_corners
from utils_matching import filter_points_border, extract_zero_mean_patches, match_zmssd
from utils_homography import ransac_homography, reprojection_errors


def detect_harris_points(I, topN=400, k=0.04, gsize=5, gsigma=1.2, nms_m=2,
                         pre_gauss=True, pre_gsize=5, pre_gsigma=1.0):
    Iw = gaussian_blur(I, size=pre_gsize, sigma=pre_gsigma) if pre_gauss else I
    R = harris_response(Iw, k=k, gsize=gsize, gsigma=gsigma)
    posr, posc, _ = harris_corners(R, mode="top", topN=topN, nms_m=nms_m)
    return posr, posc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--im1", type=str, default="/home/mahdidou711/linux_data/Projets/Py/tp1 453/Detection/set1-1.png")
    p.add_argument("--im2", type=str, default="/home/mahdidou711/linux_data/Projets/Py/tp1 453/Detection/set1-2.png")
    p.add_argument("--outdir", type=str, default="figures")

    # matching (reprend ta config finale)
    p.add_argument("--w", type=int, default=7)
    p.add_argument("--tau_q", type=float, default=0.50)
    p.add_argument("--ratio", type=float, default=0.80)
    p.add_argument("--mutual", action="store_true", default=True)
    p.add_argument("--max_drow", type=int, default=20)
    p.add_argument("--min_dcol", type=int, default=-150)
    p.add_argument("--max_dcol", type=int, default=150)
    p.add_argument("--keepK", type=int, default=30)

    # RANSAC
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--eps", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    I1 = read_gray(args.im1)
    I2 = read_gray(args.im2)
    H1, W1 = I1.shape
    H2, W2 = I2.shape

    # 1) points Harris
    posr1, posc1 = detect_harris_points(I1, topN=400, pre_gauss=True)
    posr2, posc2 = detect_harris_points(I2, topN=400, pre_gauss=True)

    # 2) bords
    posr1, posc1 = filter_points_border(posr1, posc1, H1, W1, args.w)
    posr2, posc2 = filter_points_border(posr2, posc2, H2, W2, args.w)

    # 3) patchs zero-mean
    P1 = extract_zero_mean_patches(I1, posr1, posc1, args.w)
    P2 = extract_zero_mean_patches(I2, posr2, posc2, args.w)

    # 4) matching + filtres (quantile + ratio + mutual)
    bestj, bestd, tau_used = match_zmssd(P1, P2, tau=None, tau_quantile=args.tau_q,
                                         ratio=args.ratio, mutual=args.mutual)

    bestr = np.full(len(posr1), -1, dtype=np.int32)
    bestc = np.full(len(posr1), -1, dtype=np.int32)
    ok = bestj >= 0
    bestr[ok] = posr2[bestj[ok]]
    bestc[ok] = posc2[bestj[ok]]

    # 5) filtre déplacement
    drow = (bestr - posr1)
    dcol = (bestc - posc1)
    ok &= (np.abs(drow) <= args.max_drow)
    ok &= (dcol >= args.min_dcol) & (dcol <= args.max_dcol)
    bestr[~ok] = -1
    bestc[~ok] = -1

    # 6) keepK meilleurs (distance)
    idx = np.where(ok)[0]
    if len(idx) > args.keepK:
        order = np.argsort(bestd[idx])  # plus petit = meilleur
        keep_idx = idx[order[:args.keepK]]
        ok2 = np.zeros_like(ok)
        ok2[keep_idx] = True
        ok = ok2
        bestr[~ok] = -1
        bestc[~ok] = -1

    # 7) construire correspondances (0-based)
    idxm = np.where(ok)[0]
    u1 = (posc1[idxm] - 1).astype(np.float64)
    v1 = (posr1[idxm] - 1).astype(np.float64)
    u2 = (bestc[idxm] - 1).astype(np.float64)
    v2 = (bestr[idxm] - 1).astype(np.float64)

    print("matches_used_for_RANSAC:", len(idxm), "tau_used:", tau_used)

    # 8) RANSAC homographie
    H, inliers, sample_idx = ransac_homography(u1, v1, u2, v2, T=args.T, eps=args.eps, seed=args.seed)
    if H is None:
        print("RANSAC failed")
        return

    nin = int(np.sum(inliers))
    err_all = reprojection_errors(H, u1, v1, u2, v2)
    err_in = err_all[inliers] if nin > 0 else np.array([], dtype=np.float64)

    print("H =")
    print(H)
    print("inliers:", nin, "/", len(u1))
    if nin > 0:
        print("inlier error: mean =", float(np.mean(err_in)), "max =", float(np.max(err_in)))

    # 9) sauvegarde H
    np.savetxt(outdir / "H_best.txt", H, fmt="%.10e")
    print("saved:", outdir / "H_best.txt")


if __name__ == "__main__":
    main()