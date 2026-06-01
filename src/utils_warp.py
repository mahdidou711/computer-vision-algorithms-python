import numpy as np


def to_homog(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(u,v) -> X homogène (3,N)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ones = np.ones_like(u)
    return np.stack([u, v, ones], axis=0)


def apply_H(H: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Applique H à (u,v) et renvoie (uh, vh) après normalisation."""
    X = to_homog(u, v)              # (3,N)
    Xp = H @ X                      # (3,N)
    w = Xp[2, :]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    uh = Xp[0, :] / w
    vh = Xp[1, :] / w
    return uh, vh


def bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    """
    Échantillonnage bilinéaire.
    img: (H,W,C) float32
    u,v: (N,) float64 en coordonnées pixel (col=x, row=y), 0-based
    Retourne:
      - out: (N,C) float32
      - valid: (N,) bool
    """
    if img.ndim != 3:
        raise ValueError("img must be (H,W,C)")
    H, W, C = img.shape

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # valid si les 4 voisins sont dans l'image
    valid = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)

    out = np.zeros((u.size, C), dtype=np.float32)
    if not np.any(valid):
        return out, valid

    uu = u[valid]
    vv = v[valid]
    u0v = u0[valid]; v0v = v0[valid]
    u1v = u1[valid]; v1v = v1[valid]

    a = (uu - u0v).astype(np.float32)   # frac x
    b = (vv - v0v).astype(np.float32)   # frac y

    Ia = img[v0v, u0v, :]   # (Nv,C)
    Ib = img[v0v, u1v, :]
    Ic = img[v1v, u0v, :]
    Id = img[v1v, u1v, :]

    wa = (1.0 - a) * (1.0 - b)
    wb = a * (1.0 - b)
    wc = (1.0 - a) * b
    wd = a * b

    out_valid = (Ia * wa[:, None] + Ib * wb[:, None] + Ic * wc[:, None] + Id * wd[:, None])
    out[valid, :] = out_valid.astype(np.float32)
    return out, valid


def compute_panorama_canvas(W1: int, H1: int, W2: int, H2: int, H12: np.ndarray):
    """
    Repère panorama = repère image1.
    On projette les coins de l'image2 vers image1 via H12^{-1}, puis bbox avec coins image1.
    Renvoie:
      - offset_x, offset_y (à ajouter aux coords image1 pour obtenir coords panorama)
      - Wp, Hp taille panorama
    """
    Hinv = np.linalg.inv(H12)

    # coins image1 (u,v) en 0-based
    c1 = np.array([0, W1 - 1, W1 - 1, 0], dtype=np.float64)
    r1 = np.array([0, 0, H1 - 1, H1 - 1], dtype=np.float64)

    # coins image2 -> repère image1
    c2 = np.array([0, W2 - 1, W2 - 1, 0], dtype=np.float64)
    r2 = np.array([0, 0, H2 - 1, H2 - 1], dtype=np.float64)
    c2_in_1, r2_in_1 = apply_H(Hinv, c2, r2)

    u_all = np.concatenate([c1, c2_in_1])
    v_all = np.concatenate([r1, r2_in_1])

    umin = float(np.floor(np.min(u_all)))
    vmin = float(np.floor(np.min(v_all)))
    umax = float(np.ceil(np.max(u_all)))
    vmax = float(np.ceil(np.max(v_all)))

    offset_x = -umin
    offset_y = -vmin

    Wp = int(umax - umin + 1)
    Hp = int(vmax - vmin + 1)
    return int(offset_x), int(offset_y), Wp, Hp