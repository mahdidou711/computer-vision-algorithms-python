import numpy as np
from PIL import Image, ImageDraw

def draw_corners_rgb(im_gray: np.ndarray,
                     posr: np.ndarray,
                     posc: np.ndarray,
                     radius: int = 3) -> np.ndarray:
    """
    Dessine des croix rouges sur une image grise.
    posr,posc en 1-based (ligne,colonne).
    radius : demi-taille de la croix (3 -> croix 7x7).
    Retourne une image RGB uint8.
    """
    if im_gray.ndim != 2:
        raise ValueError("im_gray must be 2D")
    if radius < 1:
        raise ValueError("radius must be >= 1")

    H, W = im_gray.shape
    base = np.clip(im_gray, 0, 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=2)

    for r1, c1 in zip(posr.tolist(), posc.tolist()):
        r = int(r1) - 1
        c = int(c1) - 1
        if r < 0 or r >= H or c < 0 or c >= W:
            continue

        # verticale
        for dr in range(-radius, radius + 1):
            rr = r + dr
            if 0 <= rr < H:
                rgb[rr, c, 0] = 255
                rgb[rr, c, 1] = 0
                rgb[rr, c, 2] = 0

        # horizontale
        for dc in range(-radius, radius + 1):
            cc = c + dc
            if 0 <= cc < W:
                rgb[r, cc, 0] = 255
                rgb[r, cc, 1] = 0
                rgb[r, cc, 2] = 0

    return rgb


def save_rgb(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)
    from PIL import ImageDraw

def save_matches_png(im1_gray: np.ndarray,
                     im2_gray: np.ndarray,
                     posr1: np.ndarray, posc1: np.ndarray,
                     bestr: np.ndarray, bestc: np.ndarray,
                     out_path: str,
                     mark_radius: int = 2,
                     line_width: int = 1):
    """
      - dessine coins sur im1 et im2
      - concatène horizontalement
      - trace les segments entre p1 et p2
    posr/posc en 1-based. Les matchs rejetés peuvent être -1.
    """
    rgb1 = draw_corners_rgb(im1_gray, posr1, posc1, radius=mark_radius)
    rgb2 = draw_corners_rgb(im2_gray, bestr[bestr > 0], bestc[bestc > 0], radius=mark_radius)

    H1, W1 = im1_gray.shape
    H2, W2 = im2_gray.shape
    if H1 != H2:
        # concat simple: on pad en hauteur si besoin
        H = max(H1, H2)
        tmp1 = np.zeros((H, W1, 3), dtype=np.uint8); tmp1[:H1, :, :] = rgb1
        tmp2 = np.zeros((H, W2, 3), dtype=np.uint8); tmp2[:H2, :, :] = rgb2
        comp = np.concatenate([tmp1, tmp2], axis=1)
    else:
        comp = np.concatenate([rgb1, rgb2], axis=1)

    im = Image.fromarray(comp, mode="RGB")
    drw = ImageDraw.Draw(im)

    for r1, c1, r2, c2 in zip(posr1.tolist(), posc1.tolist(), bestr.tolist(), bestc.tolist()):
        if r2 <= 0 or c2 <= 0:
            continue
        x1 = int(c1) - 1
        y1 = int(r1) - 1
        x2 = (int(c2) - 1) + W1
        y2 = int(r2) - 1
        drw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=line_width)

    im.save(out_path)