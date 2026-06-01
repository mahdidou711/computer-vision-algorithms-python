import numpy as np


def build_A_b(u1, v1, u2, v2):
    """
    Construit A (2m x 8) et b (2m,) pour le modèle avec h33=1.
    inconnues h = [h11 h12 h13 h21 h22 h23 h31 h32]^T
    """
    m = len(u1)
    A = np.zeros((2 * m, 8), dtype=np.float64)
    b = np.zeros((2 * m,), dtype=np.float64)

    for i in range(m):
        u, v = float(u1[i]), float(v1[i])
        up, vp = float(u2[i]), float(v2[i])

        # u' equation
        A[2 * i + 0, :] = [u, v, 1.0, 0.0, 0.0, 0.0, -u * up, -v * up]
        b[2 * i + 0] = up

        # v' equation
        A[2 * i + 1, :] = [0.0, 0.0, 0.0, u, v, 1.0, -u * vp, -v * vp]
        b[2 * i + 1] = vp

    return A, b


def h_from_4pts_solve(u1, v1, u2, v2, cond_max=1e12):
    """
    Estime H à partir de 4 correspondances (donc A est 8x8) via résolution Ah=b.
    Rejette si A est singulière / mal conditionnée.
    """
    A, b = build_A_b(u1, v1, u2, v2)
    # Conditionnement pour éviter les cas dégénérés
    c = np.linalg.cond(A)
    if not np.isfinite(c) or c > cond_max:
        return None

    try:
        h = np.linalg.solve(A, b)  # équivalent à inv(A)@b mais plus stable
    except np.linalg.LinAlgError:
        return None

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)
    return H


def project_points(H, u, v):
    """
    Projette (u,v) via H: retourne (uhat, vhat) après normalisation homogène.
    u,v: (m,)
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ones = np.ones_like(u)

    X = np.stack([u, v, ones], axis=0)         # (3,m)
    Xp = H @ X                                 # (3,m)
    w = Xp[2, :]
    # éviter division par 0
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    uh = Xp[0, :] / w
    vh = Xp[1, :] / w
    return uh, vh


def reprojection_errors(H, u1, v1, u2, v2):
    uh, vh = project_points(H, u1, v1)
    du = uh - np.asarray(u2, dtype=np.float64)
    dv = vh - np.asarray(v2, dtype=np.float64)
    return np.sqrt(du * du + dv * dv)


def refine_h_svd(u1, v1, u2, v2):
    """
    Raffinement sur n inliers (n>=4) selon la procédure SVD du sujet (pseudo-inverse).
    """
    A, b = build_A_b(u1, v1, u2, v2)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # b' = U^T b
    bp = U.T @ b
    # y_i = b'_i / d_i
    y = bp / (s + 1e-12)
    # h = V y, avec V = Vt^T
    h = (Vt.T) @ y

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)
    return H


def ransac_homography(u1, v1, u2, v2, T=500, eps=3.0, seed=0):
    """
    RANSAC homographie :
      - sample 4 matches
      - H via solve 8x8
      - inliers via erreur reprojection <= eps
      - raffinement final via SVD sur inliers
    Retourne (H_best, inlier_mask, sample_idx_best)
    """
    rng = np.random.default_rng(seed)

    u1 = np.asarray(u1, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    m = len(u1)
    if m < 4:
        return None, None, None

    best_inliers = None
    best_count = -1
    best_H = None
    best_sample = None

    for _ in range(int(T)):
        idx = rng.choice(m, size=4, replace=False)

        H = h_from_4pts_solve(u1[idx], v1[idx], u2[idx], v2[idx])
        if H is None:
            continue

        err = reprojection_errors(H, u1, v1, u2, v2)
        inliers = err <= float(eps)
        cnt = int(np.sum(inliers))

        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_H = H
            best_sample = idx.copy()

    if best_H is None:
        return None, None, None

    # Raffinement sur inliers
    if best_count >= 4:
        H_ref = refine_h_svd(u1[best_inliers], v1[best_inliers], u2[best_inliers], v2[best_inliers])
    else:
        H_ref = best_H

    return H_ref, best_inliers, best_sample