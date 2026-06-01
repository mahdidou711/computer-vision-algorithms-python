import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

# =========================
# I/O
# =========================
def read_gray(path: str) -> np.ndarray:
    """Lit une image et renvoie un tableau float32 en niveaux de gris dans [0,255]."""
    im = Image.open(path).convert("L")
    return np.array(im, dtype=np.float32)


# =========================
# Convolution 2D explicite
# =========================
def pad_image(im: np.ndarray, pad_h: int, pad_w: int, mode: str = "edge") -> np.ndarray:
    if im.ndim != 2:
        raise ValueError("pad_image: im must be 2D")
    if mode not in ("edge", "reflect"):
        raise ValueError("pad_image: mode must be 'edge' or 'reflect'")
    return np.pad(im, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


def convolve2d(im: np.ndarray, kernel: np.ndarray, pad_mode: str = "edge") -> np.ndarray:
    """Convolution 2D (vectorisée via sliding_window_view)."""
    if im.ndim != 2:
        raise ValueError("convolve2d: im must be 2D")
    if kernel.ndim != 2:
        raise ValueError("convolve2d: kernel must be 2D")

    H, W = im.shape
    Kh, Kw = kernel.shape
    if Kh % 2 == 0 or Kw % 2 == 0:
        raise ValueError("convolve2d: kernel must have odd dimensions")

    ph, pw = Kh // 2, Kw // 2
    pim = pad_image(im, ph, pw, mode=pad_mode).astype(np.float32, copy=False)

    # retournement pour convolution
    k = kernel[::-1, ::-1].astype(np.float32, copy=False)

    # fenêtres (H,W,Kh,Kw)
    win = sliding_window_view(pim, (Kh, Kw))
    out = np.tensordot(win, k, axes=((2, 3), (0, 1))).astype(np.float32, copy=False)
    return out


# =========================
# Sobel + magnitude
# =========================
def sobel_gradients(im: np.ndarray, pad_mode: str = "edge") -> tuple[np.ndarray, np.ndarray]:
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    Ix = convolve2d(im, Sx, pad_mode=pad_mode)
    Iy = convolve2d(im, Sy, pad_mode=pad_mode)
    return Ix, Iy


def gradient_magnitude(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
    return np.sqrt(Ix * Ix + Iy * Iy).astype(np.float32)


def edge_mask_from_magnitude(mag: np.ndarray, t: float) -> np.ndarray:
    """Masque binaire uint8 (0/255) avec seuil t*max(mag)."""
    if not (0.0 < t < 1.0):
        raise ValueError("t must be in (0,1)")
    thr = float(np.max(mag)) * t
    return (mag > thr).astype(np.uint8) * 255


# =========================
# Gaussien (optionnel)
# =========================
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Noyau gaussien 2D normalisé, size impair."""
    if size % 2 == 0 or size < 3:
        raise ValueError("size must be odd and >= 3")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= np.sum(k)
    return k


def gaussian_blur(im: np.ndarray, size: int = 5, sigma: float = 1.2, pad_mode: str = "edge") -> np.ndarray:
    k = gaussian_kernel(size, sigma)
    return convolve2d(im, k, pad_mode=pad_mode)


# =========================
# Hough cercles : accumulateur 3D
# Convention paramètres: r,c en {1..H},{1..W}
# Convention pixels numpy: y,x en {0..H-1},{0..W-1}
# =========================
def hough_circles_accumulator(edges: np.ndarray,
                              radmin: int, radmax: int, drad: int,
                              weight_mag: np.ndarray | None = None) -> np.ndarray:
    """
    edges: masque 0/255 (H,W)
    Accumulateur acc[ir,ic,ik] avec:
      r = 1 + ir
      c = 1 + ic
      rad = radmin + ik*drad
    """
    if edges.ndim != 2:
        raise ValueError("edges must be 2D")
    if drad <= 0:
        raise ValueError("drad must be > 0")
    if radmax < radmin:
        raise ValueError("radmax must be >= radmin")

    H, W = edges.shape
    Nr, Nc = H, W
    Nrad = int(np.floor((radmax - radmin) / drad) + 1)

    acc = np.zeros((Nr, Nc, Nrad), dtype=np.float32)

    ys, xs = np.nonzero(edges > 0)  # 0-based
    for (y, x) in zip(ys, xs):
        w = 1.0 if weight_mag is None else float(weight_mag[y, x])

        # pixel en convention 1-based (comme r,c)
        y1 = float(y + 1)
        x1 = float(x + 1)

        for ir in range(Nr):
            r = float(ir + 1)
            dy = y1 - r
            for ic in range(Nc):
                c = float(ic + 1)
                dx = x1 - c
                rad = (dx * dx + dy * dy) ** 0.5

                if rad < radmin or rad > radmax:
                    continue

                ik = int(round((rad - radmin) / drad))
                if 0 <= ik < Nrad:
                    acc[ir, ic, ik] += w

    return acc


# =========================
# Maxima locaux 3D (26 voisins) + top-N
# =========================
def local_maxima_3d(acc: np.ndarray, M: int = 1, strict: bool = True) -> np.ndarray:
    """
    Maxima locaux 3D dans un voisinage cube (2M+1)^3.
    M=1 -> 26 voisins (sujet).
    M=2 -> voisinage 5x5x5 (réduit les doublons).
    Bords = False.
    """
    if acc.ndim != 3:
        raise ValueError("acc must be 3D")
    if M < 1:
        raise ValueError("M must be >= 1")

    H, W, D = acc.shape
    if H < 2*M+1 or W < 2*M+1 or D < 2*M+1:
        return np.zeros_like(acc, dtype=bool)

    inner = acc[M:H-M, M:W-M, M:D-M]
    neigh_max = np.full_like(inner, -np.inf)

    for di in range(-M, M+1):
        for dj in range(-M, M+1):
            for dk in range(-M, M+1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                neigh = acc[(M+di):(H-M+di), (M+dj):(W-M+dj), (M+dk):(D-M+dk)]
                neigh_max = np.maximum(neigh_max, neigh)

    inner_mask = (inner > neigh_max) if strict else (inner >= neigh_max)

    mask = np.zeros_like(acc, dtype=bool)
    mask[M:H-M, M:W-M, M:D-M] = inner_mask
    return mask

def top_n_from_mask(acc: np.ndarray, mask: np.ndarray, N: int, min_val: float = 0.0):
    """Renvoie liste triée décroissante de (ir,ic,ik,val)."""
    if N <= 0:
        raise ValueError("N must be > 0")
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    vals = acc[coords[:, 0], coords[:, 1], coords[:, 2]]
    keep = vals > float(min_val)
    coords = coords[keep]
    vals = vals[keep]
    if vals.size == 0:
        return []
    order = np.argsort(vals)[::-1][:N]
    out = []
    for idx in order:
        ir, ic, ik = coords[idx]
        out.append((int(ir), int(ic), int(ik), float(vals[idx])))
    return out


def idx_to_params(ir: int, ic: int, ik: int, radmin: int, drad: int) -> tuple[int, int, int]:
    """Indices 0-based -> (r,c,rad) en convention 1-based pour r,c."""
    r = ir + 1
    c = ic + 1
    rad = radmin + ik * drad
    return int(r), int(c), int(rad)

def harris_response(I: np.ndarray,
                    k: float = 0.04,
                    gsize: int = 5,
                    gsigma: float = 1.2,
                    pad_mode: str = "edge") -> np.ndarray:
    """
    Calcule la réponse de Harris R (H,W) à partir d'une image I (H,W) float32.
    R = det(M) - k * tr(M)^2 avec M construit depuis Ix,Iy et lissage gaussien.
    """
    Ix, Iy = sobel_gradients(I, pad_mode=pad_mode)

    A = Ix * Ix
    B = Ix * Iy
    C = Iy * Iy

    Sxx = gaussian_blur(A, size=gsize, sigma=gsigma, pad_mode=pad_mode)
    Sxy = gaussian_blur(B, size=gsize, sigma=gsigma, pad_mode=pad_mode)
    Syy = gaussian_blur(C, size=gsize, sigma=gsigma, pad_mode=pad_mode)

    det = Sxx * Syy - Sxy * Sxy
    tr = Sxx + Syy
    R = det - k * (tr * tr)
    return R.astype(np.float32)


def nms_2d(R: np.ndarray, m: int = 1, threshold: float = 0.0) -> np.ndarray:
    """
    Non-maximum suppression 2D sur fenêtre (2m+1)x(2m+1).
    Renvoie un masque bool (H,W) des maxima locaux stricts au-dessus de threshold.
    """
    if R.ndim != 2:
        raise ValueError("R must be 2D")
    if m < 1:
        raise ValueError("m must be >= 1")

    H, W = R.shape
    mask = np.zeros((H, W), dtype=bool)

    for y in range(m, H - m):
        for x in range(m, W - m):
            v = float(R[y, x])
            if v <= threshold:
                continue

            win = R[y - m:y + m + 1, x - m:x + m + 1]
            vmax = float(np.max(win))
            if v != vmax:
                continue

            # maximum strict pour éviter plusieurs pixels identiques dans la même fenêtre
            if int(np.sum(win == vmax)) == 1:
                mask[y, x] = True

    return mask


def harris_corners(R: np.ndarray,
                   mode: str = "rel",
                   alpha: float = 0.01,
                   topN: int = 500,
                   nms_m: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extrait des coins à partir de R.
    Retourne (posr, posc, thr) avec posr,posc en 1-based (ligne,colonne).
    mode:
      - "rel": seuil thr = alpha * max(R), puis NMS
      - "top": NMS sans seuil, puis topN plus grands R
    """
    if mode not in ("rel", "top"):
        raise ValueError("mode must be 'rel' or 'top'")

    Rmax = float(np.max(R))
    if Rmax <= 0.0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0.0

    if mode == "rel":
        thr = alpha * Rmax
        mask = nms_2d(R, m=nms_m, threshold=thr)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), thr

        vals = R[coords[:, 0], coords[:, 1]]
        order = np.argsort(vals)[::-1]
        coords = coords[order]
    else:
        thr = 0.0
        mask = nms_2d(R, m=nms_m, threshold=thr)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), thr

        vals = R[coords[:, 0], coords[:, 1]]
        order = np.argsort(vals)[::-1]
        coords = coords[order][:topN]

    # conversion 0-based -> 1-based
    posr = (coords[:, 0] + 1).astype(np.int32)
    posc = (coords[:, 1] + 1).astype(np.int32)
    return posr, posc, thr
# Ordre standard FAST-16 (rayon 3) : (dr, dc) autour de (r,c)
FAST_CIRCLE_16 = [
    (-3,  0), (-3,  1), (-2,  2), (-1,  3),
    ( 0,  3), ( 1,  3), ( 2,  2), ( 3,  1),
    ( 3,  0), ( 3, -1), ( 2, -2), ( 1, -3),
    ( 0, -3), (-1, -3), (-2, -2), (-3, -1),
]


def fast_score_at(I: np.ndarray, r: int, c: int, t: int, N: int = 12) -> int:
    """
    Score FAST au pixel (r,c) (indices 0-based).
    Retourne S>=0 (int). Coin si S >= t.
    """
    Ip = int(I[r, c])
    vals = [int(I[r + dr, c + dc]) for (dr, dc) in FAST_CIRCLE_16]
    d = [v - Ip for v in vals]  # d_i

    # Test accéléré sur 4 points (indices 0,4,8,12)
    idx4 = [1, 5, 9, 13]
    bright4 = sum(1 for i in idx4 if d[i] >= t)
    dark4 = sum(1 for i in idx4 if d[i] <= -t)
    if bright4 < 3 and dark4 < 3:
        return 0

    # Circularité : duplique pour balayer des segments de longueur N
    dd = d + d[:N-1]

    best = 0
    for s in range(16):
        seg = dd[s:s+N]
        # segment "clair"
        if all(v >= t for v in seg):
            score = min(seg)  # min d_i sur le segment
            if score > best:
                best = score
        # segment "sombre"
        if all(v <= -t for v in seg):
            score = min(-v for v in seg)  # min(-d_i)
            if score > best:
                best = score

    return int(best)


def fast_score_map(I: np.ndarray, t: int, N: int = 12) -> np.ndarray:
    """
    Carte de score FAST S (H,W) uint16.
    Bord de 3 pixels mis à 0.
    """
    if I.ndim != 2:
        raise ValueError("I must be 2D")
    H, W = I.shape
    S = np.zeros((H, W), dtype=np.uint16)

    for r in range(3, H - 3):
        for c in range(3, W - 3):
            sc = fast_score_at(I, r, c, t=t, N=N)
            S[r, c] = sc

    return S


def nms_2d_score(S: np.ndarray, m: int = 2, threshold: int = 0) -> np.ndarray:
    """
    NMS 2D sur une carte de score entière S.
    Renvoie un masque bool des maxima locaux stricts au-dessus de threshold.
    """
    if S.ndim != 2:
        raise ValueError("S must be 2D")
    if m < 1:
        raise ValueError("m must be >= 1")

    H, W = S.shape
    mask = np.zeros((H, W), dtype=bool)

    for r in range(m, H - m):
        for c in range(m, W - m):
            v = int(S[r, c])
            if v <= threshold:
                continue

            win = S[r - m:r + m + 1, c - m:c + m + 1]
            vmax = int(np.max(win))
            if v != vmax:
                continue
            if int(np.sum(win == vmax)) == 1:
                mask[r, c] = True

    return mask


def topN_from_score(S: np.ndarray, mask: np.ndarray, topN: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait topN points (posr,posc) en 1-based à partir d'une carte score S et masque NMS.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    vals = S[coords[:, 0], coords[:, 1]].astype(np.int64)
    order = np.argsort(vals)[::-1][:topN]
    coords = coords[order]

    posr = (coords[:, 0] + 1).astype(np.int32)
    posc = (coords[:, 1] + 1).astype(np.int32)
    return posr, posc