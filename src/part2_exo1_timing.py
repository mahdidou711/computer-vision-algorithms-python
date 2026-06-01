import time
import numpy as np

from utils_image import (
    read_gray, sobel_gradients, gradient_magnitude, edge_mask_from_magnitude,
    hough_circles_accumulator
)

def time_once(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0), out

def measure_accumulator_time(img_path: str,
                             t_edge: float,
                             radmin: int, radmax: int, drad: int,
                             warmup: int = 2,
                             repeats: int = 5):
    # 1) Prétraitement (hors chrono principal)
    I = read_gray(img_path)                    # float32 [0,255]
    Ix, Iy = sobel_gradients(I, pad_mode="edge")
    mag = gradient_magnitude(Ix, Iy)
    edges = edge_mask_from_magnitude(mag, t_edge)  # uint8 0/255
    ys, xs = np.nonzero(edges > 0)
    E = len(ys)
    print("E (#edge pixels) =", E, "density =", E/(I.shape[0]*I.shape[1]))
    # 2) Warm-up (cache / JIT numpy / OS)
    for _ in range(warmup):
        _ = hough_circles_accumulator(edges, radmin, radmax, drad, weight_mag=None)

    # 3) Mesures
    times = []
    for _ in range(repeats):
        dt, _ = time_once(hough_circles_accumulator, edges, radmin, radmax, drad, None)
        times.append(dt)

    times = np.array(times, dtype=np.float64)
    return {
        "N": int(I.shape[0]),
        "shape": I.shape,
        "t_edge": float(t_edge),
        "radmin": int(radmin),
        "radmax": int(radmax),
        "drad": int(drad),
        "repeats": int(repeats),
        "mean_s": float(times.mean()),
        "std_s": float(times.std(ddof=1)) if repeats >= 2 else 0.0,
        "min_s": float(times.min()),
        "max_s": float(times.max()),
    }

def extrapolate_T(T100: float, N: int, N0: int = 100) -> float:
    # Loi T(N) ≈ T(N0) * (N/N0)^4
    return float(T100) * (float(N) / float(N0))**4

if __name__ == "__main__":
    # four.png (N=100), paramètres cohérents avec ton exo2
    res = measure_accumulator_time(
        img_path="/home/mahdidou711/linux_data/Projets/Py/tp1 453/Hough/four.png",
        t_edge=0.15,
        radmin=3, radmax=26, drad=1,
        warmup=2, repeats=5
    )
    print(res)

    T100 = res["mean_s"]
    T600 = extrapolate_T(T100, N=600, N0=100)
    print(f"T100_mean = {T100:.6f} s")
    print(f"T600_est  = {T600:.2f} s  (≈ {T600/60:.2f} min)")