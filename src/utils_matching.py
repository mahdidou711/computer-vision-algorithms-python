import numpy as np


def filter_points_border(posr: np.ndarray, posc: np.ndarray, H: int, W: int, w: int):
    """
    Garde uniquement les points (1-based) dont le patch (2w+1)x(2w+1) est entièrement dans l'image.
    Retourne (posr_f, posc_f) en int32 (1-based).
    """
    r0 = posr.astype(np.int32) - 1
    c0 = posc.astype(np.int32) - 1
    ok = (r0 >= w) & (r0 < H - w) & (c0 >= w) & (c0 < W - w)
    return posr[ok].astype(np.int32), posc[ok].astype(np.int32)


def extract_zero_mean_patches(I: np.ndarray, posr: np.ndarray, posc: np.ndarray, w: int):
    """
    Extrait des patchs centrés sur (posr,posc) (1-based) et renvoie une matrice (K,P) float32 :
      - patchs vectorisés
      - centrés (moyenne soustraite)
    """
    if I.ndim != 2:
        raise ValueError("I must be 2D")
    H, W = I.shape
    K = len(posr)
    P = (2 * w + 1) * (2 * w + 1)

    patches = np.zeros((K, P), dtype=np.float32)

    for k in range(K):
        r = int(posr[k]) - 1
        c = int(posc[k]) - 1
        patch = I[r - w:r + w + 1, c - w:c + w + 1].astype(np.float32)
        patch = patch - float(np.mean(patch))
        patches[k, :] = patch.reshape(-1)

    return patches


def zmssd_matrix(P1: np.ndarray, P2: np.ndarray):
    """
    Calcule la matrice des distances ZMSSD entre patchs centrés :
      D[i,j] = ||P1[i] - P2[j]||^2
    P1: (N,P), P2: (M,P)
    """
    # (N,1,P) - (1,M,P) -> (N,M,P)
    diff = P1[:, None, :] - P2[None, :, :]
    D = np.sum(diff * diff, axis=2)
    return D.astype(np.float32)


def best2_from_dist(D: np.ndarray):
    """
    Pour chaque ligne i, retourne (j1,d1,j2,d2) : meilleurs et 2e meilleurs indices + distances.
    """
    N, M = D.shape
    j1 = np.argmin(D, axis=1)
    d1 = D[np.arange(N), j1]

    # 2e meilleur : on masque j1
    D2 = D.copy()
    D2[np.arange(N), j1] = np.inf
    j2 = np.argmin(D2, axis=1)
    d2 = D2[np.arange(N), j2]
    return j1, d1, j2, d2


def mutual_best_filter(j1_12: np.ndarray, d1_12: np.ndarray, j1_21: np.ndarray):
    """
    Garde (i -> j1_12[i]) si c'est aussi un meilleur match réciproque.
    Renvoie un masque bool (N,) de matches mutuels.
    """
    N = len(j1_12)
    ok = np.zeros(N, dtype=bool)
    for i in range(N):
        j = int(j1_12[i])
        if int(j1_21[j]) == i:
            ok[i] = True
    return ok


def match_zmssd(P1: np.ndarray, P2: np.ndarray,
                tau: float | None = None,
                tau_quantile: float | None = None,
                ratio: float | None = None,
                mutual: bool = False):
    """
    Matching ZMSSD :
      - calcule D
      - meilleur match par point de 1
      - filtres : seuil tau, seuil quantile, ratio test, mutual best
    Retourne:
      - bestj (N,) indices dans P2 (ou -1 si rejet)
      - bestd (N,) distances
      - tau_used
    """
    D = zmssd_matrix(P1, P2)
    j1, d1, j2, d2 = best2_from_dist(D)

    tau_used = None
    keep = np.ones(len(d1), dtype=bool)

    # seuil automatique par quantile
    if tau_quantile is not None:
        tau_used = float(np.quantile(d1, float(tau_quantile)))
        keep &= (d1 <= tau_used)

    # seuil explicite
    if tau is not None:
        tau_used = float(tau)
        keep &= (d1 <= tau_used)

    # ratio test (sur distances)
    if ratio is not None:
        rr = float(ratio)
        keep &= (d1 / (d2 + 1e-12) < rr)

    # mutual best
    if mutual:
        j1_21 = np.argmin(D, axis=0)  # meilleur i pour chaque j
        keep &= mutual_best_filter(j1, d1, j1_21)

    bestj = j1.astype(np.int32)
    bestd = d1.astype(np.float32)

    bestj_out = np.full_like(bestj, -1)
    bestj_out[keep] = bestj[keep]

    return bestj_out, bestd, tau_used