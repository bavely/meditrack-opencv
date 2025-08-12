# app.py
import os
import uuid
import tempfile
import traceback
from typing import Tuple, Optional, Any, List

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

APP_PORT = int(os.getenv("PORT", "5050"))
MEDIA_DIR = os.path.abspath("./media")
os.makedirs(MEDIA_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow all origins in dev


# =========================== Health & Media ===========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/media/<path:filename>")
def media(filename: str):
    # Serves files from MEDIA_DIR, including subfolders (e.g., debug/<id>/file.jpg)
    return send_from_directory(MEDIA_DIR, filename, as_attachment=False, max_age=3600)


# =========================== OpenCV helpers ===========================

def variance_of_laplacian(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def read_video_best_frame(video_path: str, sample_every: int = 2, max_frames: int = 600) -> np.ndarray:
    """Pick the sharpest frame (by Laplacian variance) sampling every N frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video (codec/ffmpeg missing?).")

    best_score, best_frame, count = -1.0, None, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = variance_of_laplacian(gray)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
        count += 1
        if count >= max_frames:
            break

    cap.release()
    if best_frame is None:
        raise RuntimeError("No frames read from video.")
    return best_frame


def _preprocess_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
    except Exception:
        pass
    return g


def find_label_full_bounds(frame: np.ndarray) -> tuple[int, int]:
    """
    Find FULL label height in a single frame.
    1) Use vertical text energy (|dx|) to locate label center.
    2) Use horizontal edge energy (|dy|) to expand up/down to label top/bottom.
    """
    h, w = frame.shape[:2]
    y_lo, y_hi = int(h * 0.25), int(h * 0.95)  # ignore cap + base
    g = _preprocess_gray(frame)

    # Step 1: find a text-dense center row with |dx|
    gradx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gx = cv2.convertScaleAbs(gradx)
    energy_x = gx[y_lo:y_hi].mean(axis=1)
    energy_x = cv2.GaussianBlur(energy_x.reshape(-1, 1), (1, 31), 0).ravel()
    y_center = int(np.argmax(energy_x)) + y_lo

    # Step 2: get top/bottom using |dy| row energy around that center
    grady = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gy = cv2.convertScaleAbs(grady)
    row_energy = gy.mean(axis=1)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 31), 0).ravel()

    # threshold relative to global and local center
    peak_center = row_energy[y_center]
    thresh = max(0.25 * float(row_energy.max()), 0.35 * float(peak_center))

    # scan upward for a run of low-energy rows
    y1 = y_center
    low = 0
    while y1 > 1:
        if row_energy[y1] < thresh:
            low += 1
            if low >= 10:
                break
        else:
            low = 0
        y1 -= 1
    y1 = max(0, y1 - 2)

    # scan downward
    y2 = y_center
    low = 0
    while y2 < h - 2:
        if row_energy[y2] < thresh:
            low += 1
            if low >= 10:
                break
        else:
            low = 0
        y2 += 1
    y2 = min(h, y2 + 2)

    # ensure minimum label height
    min_h = h // 12
    if (y2 - y1) < min_h:
        y1 = max(0, y_center - min_h // 2)
        y2 = min(h, y_center + min_h // 2)

    return y1, y2


def estimate_sides_at_y(frame: np.ndarray, y1: int, y2: int) -> tuple[int, int, int]:
    """Estimate left/right edges within the label band.

    The method emphasises vertical edges using a Scharr filter followed by
    Canny edge detection and morphological closing.  Detected edges are
    aggregated with a Hough line search for robust left/right borders.  The
    final borders are validated across multiple horizontal rows to avoid local
    artifacts.
    """

    g = _preprocess_gray(frame)
    roi = g[y1:y2, :]

    # --- Emphasise vertical edges -------------------------------------------------
    grad = cv2.Scharr(roi, cv2.CV_32F, 1, 0)
    grad_abs = cv2.convertScaleAbs(grad)
    edges = cv2.Canny(grad_abs, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    h_roi, w_roi = edges.shape

    # --- Detect dominant vertical borders via Hough transform --------------------
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(0.6 * h_roi),
        maxLineGap=10,
    )

    candidate_x: List[int] = []
    if lines is not None:
        for x1_l, y1_l, x2_l, y2_l in lines.reshape(-1, 4):
            if abs(x2_l - x1_l) <= max(3, int(0.01 * w_roi)):
                if abs(y2_l - y1_l) >= 0.6 * h_roi:
                    candidate_x.append(int((x1_l + x2_l) / 2))

    left: int
    right: int

    if candidate_x:
        left = min(candidate_x)
        right = max(candidate_x)
    else:
        # fallback: use tall contours if Hough fails
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xs: List[int] = []
        for c in contours:
            x, _, w_c, h_c = cv2.boundingRect(c)
            if h_c >= 0.5 * h_roi:
                xs.extend([x, x + w_c])

        if xs:
            left = min(xs)
            right = max(xs)
        else:
            # final fallback: gradient column energy as before
            col_energy = grad_abs.mean(axis=0)
            col_energy = cv2.GaussianBlur(
                col_energy.reshape(1, -1), (1, 31), 0
            ).ravel()
            thr = float(
                col_energy.mean()
                + 0.5 * (col_energy.max() - col_energy.mean())
            )

            def edge_from_left(arr: np.ndarray) -> int:
                run = 0
                for i, v in enumerate(arr):
                    run = run + 1 if v > thr else 0
                    if run >= 5:
                        return max(0, i - 4)
                return int(0.20 * len(arr))

            left = edge_from_left(col_energy)
            right = len(col_energy) - edge_from_left(col_energy[::-1]) - 1

    # --- Validate edges across multiple rows to avoid local artifacts ------------
    sample_rows = np.linspace(0, h_roi - 1, num=min(10, h_roi), dtype=int)
    left_samples: List[int] = []
    right_samples: List[int] = []
    for r in sample_rows:
        nz = np.where(edges[r] > 0)[0]
        if nz.size > 0:
            left_samples.append(int(nz[0]))
            right_samples.append(int(nz[-1]))

    if left_samples:
        left = int(np.median(left_samples + [left]))
    if right_samples:
        right = int(np.median(right_samples + [right]))

    # fallback to reasonable defaults if edge detection seems off
    if right - left < frame.shape[1] // 6:
        left = int(frame.shape[1] * 0.20)
        right = int(frame.shape[1] * 0.80)

    cx = (left + right) // 2
    return left, right, cx


def unwrap_cylindrical_band(
    frame: np.ndarray,
    y1: int,
    y2: int,
    cx: int,
    radius: int,
    fov: Optional[float] = None,
    out_w: int = 2048,
) -> np.ndarray:
    """Cylindrical inverse mapping for the ENTIRE label band.

    ``radius`` is the detected cylinder radius in pixels. ``fov`` is the horizontal
    field of view in radians.  If ``fov`` is ``None`` it is estimated from the
    frame width and radius.
    """
    band = frame[y1:y2, :, :]
    h, w = band.shape[:2]
    out_h = h

    radius = max(1.0, float(radius))
    f = radius  # focal length approximation in pixels
    if fov is None:
        fov = 2.0 * np.arctan((w / 2.0) / f)
    theta_min, theta_max = -fov / 2.0, fov / 2.0

    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    for x_out in range(out_w):
        theta = theta_min + (theta_max - theta_min) * (x_out / max(1, (out_w - 1)))
        x_in = cx + f * np.tan(theta)
        map_x[:, x_out] = np.clip(x_in, 0, w - 1)

    for y_out in range(out_h):
        map_y[y_out, :] = y_out

    return cv2.remap(
        band,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# -------- (Optional) polar helpers kept for experimentation -------- #

def detect_cylinder_center_radius(frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], int]]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, w // 12),
        param1=120,
        param2=40,
        minRadius=max(20, min(w, h) // 12),
        maxRadius=min(w, h) // 2,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        x, y, r = circles[0]
        return (int(x), int(y)), int(r)

    r = int(w * 1)
    if r > 0:
        return (w // 2, int(h * 0.45)), r
    return None


def detect_cylinder_center_radius_multi(
    frames: List[np.ndarray],
) -> Optional[Tuple[int, int, int]]:
    """Detect cylinder cross-section across several frames and return median center/radius.

    Returns (cx, cy, r) if any detections succeed, otherwise ``None``.
    """
    detections: List[Tuple[int, int, int]] = []
    for f in frames:
        det = detect_cylinder_center_radius(f)
        if det:
            (cx, cy), r = det
            detections.append((cx, cy, r))
    if not detections:
        return None
    cx = int(np.median([d[0] for d in detections]))
    cy = int(np.median([d[1] for d in detections]))
    r = int(np.median([d[2] for d in detections]))
    return cx, cy, r


def unwrap_with_warp_polar(frame: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    dst_width = 2048
    dst_height = max(256, int(radius * 1.2))
    return cv2.warpPolar(
        frame,
        (dst_width, dst_height),
        center,
        radius,
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
    )


def find_label_band(unwrapped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    row_energy = grad.mean(axis=1)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 51), 0).ravel()
    top_row = int(np.argmax(row_energy))
    h = unwrapped.shape[0]
    band_half = max(40, h // 6)
    y1 = max(0, top_row - band_half)
    y2 = min(h, top_row + band_half)
    return unwrapped[y1:y2, :, :]


# ==================== Multi-frame unwrap + mosaic ====================

def sample_sharp_frames(video_path: str, target_frames: int = 10, max_scan: int = 900) -> List[np.ndarray]:
    """Return up to target_frames sharp frames distributed across the clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_scan
    step = max(1, length // (target_frames * 2))  # oversample then keep best
    idx, pool = 0, []
    while idx < length and len(pool) < target_frames * 3:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = variance_of_laplacian(gray)
        pool.append((score, idx, frame))
        idx += step
    cap.release()
    if not pool:
        raise RuntimeError("No frames found.")
    # pick best by score but also spread in time
    pool.sort(key=lambda t: t[0], reverse=True)
    pool = pool[:target_frames * 2]
    pool.sort(key=lambda t: t[1])  # chronological
    # downsample to target_frames roughly evenly
    if len(pool) > target_frames:
        stride = len(pool) / float(target_frames)
        sel = [pool[int(i * stride)] for i in range(target_frames)]
    else:
        sel = pool
    return [f for _, _, f in sel]


def find_label_full_bounds_union(frames: List[np.ndarray]) -> tuple[int, int]:
    """Compute (y1,y2) union across frames so we never crop the label."""
    bands = [find_label_full_bounds(f) for f in frames]
    y1 = min(b[0] for b in bands)
    y2 = max(b[1] for b in bands)
    # pad a bit
    h = frames[0].shape[0]
    pad = max(6, int(0.02 * h))
    return max(0, y1 - pad), min(h, y2 + pad)


def unwrap_cylindrical_band_consistent(
    frame: np.ndarray,
    y1: int,
    y2: int,
    out_w: int,
    cx: int,
    radius: int,
    fov: Optional[float] = None,
) -> np.ndarray:
    """Same as :func:`unwrap_cylindrical_band` but parameterized for multiple strips."""
    band = frame[y1:y2, :, :]
    h, w = band.shape[:2]
    radius = max(1.0, float(radius))
    f = radius
    if fov is None:
        fov = 2.0 * np.arctan((w / 2.0) / f)
    theta_min, theta_max = -fov / 2.0, fov / 2.0

    map_x = np.zeros((h, out_w), dtype=np.float32)
    map_y = np.zeros((h, out_w), dtype=np.float32)
    thetas = theta_min + (theta_max - theta_min) * (
        np.arange(out_w) / max(1, (out_w - 1))
    )
    x_in = cx + f * np.tan(thetas)
    x_in = np.clip(x_in, 0, w - 1).astype(np.float32)
    map_x[:] = x_in
    map_y[:] = np.arange(h, dtype=np.float32)[:, None]
    return cv2.remap(
        band, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _phase_shift(a: np.ndarray, b: np.ndarray, search_y_frac: float = 0.5) -> float:
    """
    Estimate horizontal shift between two unwrapped strips using feature matching.
    ORB/SIFT keypoints are extracted on overlapping regions and a homography is
    computed with RANSAC to reject outliers.  The resulting translation provides
    sub-pixel alignment.
    """
    ha, wa = a.shape[:2]

    # central band to focus on label text
    band_frac = max(0.2, min(0.6, search_y_frac))
    y0 = int(ha * (0.5 - band_frac / 2.0))
    y1 = int(ha * (0.5 + band_frac / 2.0))
    y0 = max(0, min(ha - 8, y0))
    y1 = max(y0 + 8, min(ha, y1))

    ga = cv2.cvtColor(a[y0:y1], cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b[y0:y1], cv2.COLOR_BGR2GRAY)

    # fall back to ORB if SIFT is unavailable
    try:
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    except Exception:
        detector = cv2.ORB_create(800)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(ga, None)
    kp2, des2 = detector.detectAndCompute(gb, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0.0

    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 4:
        return 0.0
    matches = sorted(matches, key=lambda m: m.distance)[:200]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return 0.0

    # translation component gives horizontal shift (b -> a)
    shift_x = float(H[0, 2])
    return shift_x


def _blend_overwrite(canvas: np.ndarray, strip: np.ndarray, x: int, feather: int = 32):
    """
    Paste strip on canvas with feathering in the overlap zone.
    """
    h, w = strip.shape[:2]
    H, W = canvas.shape[:2]
    x0 = max(0, x)
    x1 = min(W, x + w)
    if x1 <= x0:
        return
    s0 = max(0, -x)
    s1 = s0 + (x1 - x0)

    # alpha feather on left overlap
    alpha = np.ones((h, x1 - x0), np.float32)
    if x0 > 0:
        left_overlap = min(feather, x1 - x0)
        ramp = np.linspace(0.0, 1.0, left_overlap, dtype=np.float32)
        alpha[:, :left_overlap] = ramp[None, :]

    # alpha feather on right border (anticipate future overlap)
    if x1 < W:
        right_overlap = min(feather, x1 - x0)
        ramp = np.linspace(1.0, 0.0, right_overlap, dtype=np.float32)
        alpha[:, -right_overlap:] = np.minimum(alpha[:, -right_overlap:], ramp[None, :])

    roi = canvas[:h, x0:x1, :].astype(np.float32)
    s = strip[:, s0:s1, :].astype(np.float32)
    blended = (alpha[..., None] * s + (1.0 - alpha[..., None]) * roi)
    canvas[:h, x0:x1, :] = np.clip(blended, 0, 255).astype(np.uint8)


def _multiband_blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, levels: int = 4) -> np.ndarray:
    """Blend ``overlay`` onto ``base`` using ``mask`` via Laplacian pyramids."""
    gp_base = [base.astype(np.float32)]
    gp_overlay = [overlay.astype(np.float32)]
    gp_mask = [mask.astype(np.float32)]
    for _ in range(levels):
        gp_base.append(cv2.pyrDown(gp_base[-1]))
        gp_overlay.append(cv2.pyrDown(gp_overlay[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))
    lp_base = [gp_base[-1]]
    lp_overlay = [gp_overlay[-1]]
    for i in range(levels, 0, -1):
        size = gp_base[i - 1].shape[1::-1]
        lap_b = gp_base[i - 1] - cv2.pyrUp(gp_base[i], dstsize=size)
        lap_o = gp_overlay[i - 1] - cv2.pyrUp(gp_overlay[i], dstsize=size)
        lp_base.append(lap_b)
        lp_overlay.append(lap_o)
    blended_pyr = []
    for lb, lo, gm in zip(lp_base, lp_overlay, gp_mask[::-1]):
        blended = gm[..., None] * lo + (1.0 - gm[..., None]) * lb
        blended_pyr.append(blended)
    img = blended_pyr[0]
    for b in blended_pyr[1:]:
        img = cv2.pyrUp(img, dstsize=b.shape[1::-1]) + b
    return np.clip(img, 0, 255).astype(np.uint8)


def estimate_frame_rotations(
    frames: List[np.ndarray], imu_angles: Optional[List[float]] = None
) -> List[float]:
    """Return cumulative yaw angles (radians) for each frame.

    If ``imu_angles`` is provided and matches the number of frames it is used
    directly (converted from degrees to radians). Otherwise, a simple optical
    flow based tracker is employed where the median horizontal motion of good
    feature points is converted to an approximate rotation assuming a 60°
    field of view.
    """

    if imu_angles and len(imu_angles) == len(frames):
        base = imu_angles[0]
        return [np.deg2rad(a - base) for a in imu_angles]

    angles = [0.0]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev = np.asarray(prev, dtype=np.uint8)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    cum = 0.0
    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = np.asarray(gray, dtype=prev.dtype)
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if p0 is None or len(p0) < 4:
            angles.append(cum)
            prev = gray
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk_params)
        if p1 is None:
            angles.append(cum)
            prev = gray
            continue
        valid = st.reshape(-1) == 1
        if np.count_nonzero(valid) < 4:
            angles.append(cum)
            prev = gray
            continue
        flow = (p1 - p0)[valid]
        shift_x = float(np.median(flow[:, 0]))
        theta = (shift_x / frames[0].shape[1]) * np.deg2rad(60.0)
        cum += theta
        angles.append(cum)
        prev = gray
    return angles


def map_frame_to_cylinder(frame: np.ndarray, angle: float) -> np.ndarray:
    """Map ``frame`` onto a cylindrical surface and rotate by ``angle``.

    The frame is unwrapped using ``cv2.warpPolar`` and then horizontally
    shifted based on the provided rotation angle (in radians).
    """

    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center)
    polar = cv2.warpPolar(
        frame,
        (w * 2, h),
        center,
        radius,
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    shift = int((angle / (2 * np.pi)) * polar.shape[1])
    return np.roll(polar, shift, axis=1)


def build_panoramic_texture(frames: List[np.ndarray], angles: List[float]) -> np.ndarray:
    """Accumulate cylindrical mappings of ``frames`` into a panorama."""

    pano: Optional[np.ndarray] = None
    for f, a in zip(frames, angles):
        cyl = map_frame_to_cylinder(f, a)
        if pano is None:
            pano = np.zeros_like(cyl)
        mask = np.any(cyl > 0, axis=2)
        pano[mask] = cyl[mask]
    if pano is None:
        raise RuntimeError("No frames to build panorama.")
    return pano


def build_label_mosaic(strips: List[np.ndarray], base_width: int = 9000) -> np.ndarray:
    """
    Register and stitch strips horizontally. base_width is a safety canvas.
    """
    if not strips:
        raise RuntimeError("No strips to stitch.")
    h = max(s.shape[0] for s in strips)
    canvas = np.zeros((h, base_width, 3), np.uint8)
    x_positions = [base_width // 2 - strips[0].shape[1] // 2]
    canvas[:, x_positions[0] : x_positions[0] + strips[0].shape[1]] = strips[0]

    left_side = True
    for i in range(1, len(strips)):
        ref_idx = 0
        shift = _phase_shift(strips[ref_idx], strips[i])
        int_shift = int(np.floor(shift))
        frac_shift = shift - int_shift
        M = np.float32([[1, 0, frac_shift], [0, 1, 0]])
        aligned = cv2.warpAffine(
            strips[i], M, (strips[i].shape[1], strips[i].shape[0]), flags=cv2.INTER_LINEAR
        )
        if left_side:
            x_new = x_positions[ref_idx] - strips[i].shape[1] + int_shift
        else:
            x_new = x_positions[ref_idx] + strips[ref_idx].shape[1] + int_shift
        left_side = not left_side
        x_positions.append(x_new)

        h_s, w_s = aligned.shape[:2]
        x0 = max(0, x_new)
        x1 = min(base_width, x_new + w_s)
        if x1 <= x0:
            continue
        s0 = max(0, -x_new)
        s1 = s0 + (x1 - x0)

        roi_base = canvas[:h_s, x0:x1]
        roi_overlay = aligned[:, s0:s1]
        mask = np.ones((h_s, x1 - x0), np.float32)
        feather = min(32, x1 - x0)
        if x0 > 0:
            ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
            mask[:, :feather] = ramp[None, :]
        if x1 < base_width:
            ramp = np.linspace(1.0, 0.0, feather, dtype=np.float32)
            mask[:, -feather:] = np.minimum(mask[:, -feather:], ramp[None, :])
        blended = _multiband_blend(roi_base, roi_overlay, mask)
        canvas[:h_s, x0:x1] = blended

    # crop to content (remove empty margins)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    xs = np.where(thr.sum(axis=0) > 0)[0]
    if xs.size > 0:
        x0, x1 = xs[0], xs[-1] + 1
        canvas = canvas[:, x0:x1]
    return canvas


# ============================ Core pipeline ============================

def extract_flat_label_image(
    video_path: str,
    mode: str = "mosaic",
    debug_dir: Optional[str] = None,
    imu_angles: Optional[List[float]] = None,
) -> np.ndarray:
    """Extract an unwrapped label or panorama from ``video_path``.

    ``mode`` can be ``"mosaic"`` (recommended), ``"cyl"`` (single best frame),
    ``"polar"`` or ``"panorama"`` which uses rotation tracking and cylindrical
    mapping across multiple frames.
    """
    if mode not in ("mosaic", "cyl", "polar", "panorama"):
        mode = "mosaic"

    if mode == "cyl":
        frame = read_video_best_frame(video_path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "01_best_frame.jpg"), frame)

        y1, y2 = find_label_full_bounds(frame)
        circle = detect_cylinder_center_radius(frame)
        if circle:
            (cx, _cy), radius = circle
        else:
            left, right, cx = estimate_sides_at_y(frame, y1, y2)
            radius = max(1, (right - left) // 2)

        if debug_dir:
            dbg = frame.copy()
            cv2.line(dbg, (0, y1), (dbg.shape[1], y1), (0, 255, 0), 2)
            cv2.line(dbg, (0, y2), (dbg.shape[1], y2), (0, 255, 0), 2)
            if circle:
                cv2.circle(dbg, (cx, _cy), radius, (255, 0, 0), 2)
            else:
                cv2.line(dbg, (left, y1), (left, y2), (255, 0, 0), 2)
                cv2.line(dbg, (right, y1), (right, y2), (255, 0, 0), 2)
                cv2.line(
                    dbg,
                    (((left + right) // 2), y1),
                    (((left + right) // 2), y2),
                    (0, 0, 255),
                    2,
                )
            cv2.imwrite(os.path.join(debug_dir, "02_bounds_overlay.jpg"), dbg)

        fov = 2.0 * np.arctan((frame.shape[1] / 2.0) / float(radius))
        strip = unwrap_cylindrical_band(frame, y1, y2, cx, radius, fov)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "03_cyl_strip.jpg"), strip)
        return strip

    if mode == "polar":
        frame = read_video_best_frame(video_path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "01_best_frame.jpg"), frame)
        detection = detect_cylinder_center_radius(frame)
        if detection:
            (pcx, pcy), r = detection
            polar = unwrap_with_warp_polar(frame, (pcx, pcy), r)
            cropped = find_label_band(polar)
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "03_polar.jpg"), polar)
                cv2.imwrite(os.path.join(debug_dir, "04_polar_cropped.jpg"), cropped)
            return cropped
        # fall through to mosaic if polar fails

    if mode == "panorama":
        frames = sample_sharp_frames(video_path, target_frames=13)
        angles = estimate_frame_rotations(frames, imu_angles)
        pano = build_panoramic_texture(frames, angles)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "01_panorama.jpg"), pano)
        return pano

    # --------- MOSAIC (multi-frame) ----------
    frames = sample_sharp_frames(video_path, target_frames=13)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_best_frame.jpg"), frames[0])

    y1, y2 = find_label_full_bounds_union(frames)

    circle_params = detect_cylinder_center_radius_multi(frames)
    if circle_params:
        CX0, CY0, RAD0 = circle_params
        FOV0 = 2.0 * np.arctan((frames[0].shape[1] / 2.0) / float(RAD0))
    else:
        CX0 = CY0 = RAD0 = FOV0 = None

    out_w = 1024  # per-strip width; ~30–40% overlap appears naturally from geometry
    strips: List[np.ndarray] = []
    for i, f in enumerate(frames):
        L, R, CX_est = estimate_sides_at_y(f, y1, y2)
        if circle_params:
            cx = CX0
            radius = RAD0
            fov = FOV0
        else:
            cx = CX_est
            radius = max(1, (R - L) // 2)
            fov = 2.0 * np.arctan((f.shape[1] / 2.0) / float(radius))
        strip = unwrap_cylindrical_band_consistent(f, y1, y2, out_w, cx, radius, fov)
        strips.append(strip)
        if debug_dir and i == 0:
            dbg = f.copy()
            cv2.line(dbg, (0, y1), (dbg.shape[1], y1), (0, 255, 0), 2)
            cv2.line(dbg, (0, y2), (dbg.shape[1], y2), (0, 255, 0), 2)
            if circle_params:
                cv2.circle(dbg, (CX0, CY0), RAD0, (255, 0, 0), 2)
            else:
                cv2.line(dbg, (L, y1), (L, y2), (255, 0, 0), 2)
                cv2.line(dbg, (R, y1), (R, y2), (255, 0, 0), 2)
                cv2.line(dbg, (CX_est, y1), (CX_est, y2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, "02_bounds_overlay.jpg"), dbg)

    mosaic = build_label_mosaic(strips, base_width=9000)
    # ---- Prepare mosaic for OCR ---------------------------------------------
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    angle = 0.0
    if lines is not None:
        angs: List[float] = []
        for rho, theta in lines[:, 0]:
            deg = theta * 180.0 / np.pi - 90.0
            if -45.0 < deg < 45.0:
                angs.append(deg)
        if angs:
            angle = float(np.median(angs))
    if abs(angle) > 0.1:
        h_m, w_m = gray.shape
        rot = cv2.getRotationMatrix2D((w_m / 2.0, h_m / 2.0), angle, 1.0)
        gray = cv2.warpAffine(
            gray,
            rot,
            (w_m, h_m),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    try:
        denoised = cv2.fastNlMeansDenoising(
            gray, h=10, templateWindowSize=7, searchWindowSize=21
        )
    except Exception:
        denoised = gray

    def sauvola(img: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128) -> np.ndarray:
        mean = cv2.boxFilter(img, cv2.CV_32F, (window, window))
        sqmean = cv2.sqrBoxFilter(img, cv2.CV_32F, (window, window))
        std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))
        thresh = mean * (1 + k * (std / R - 1))
        out = (img.astype(np.float32) > thresh).astype(np.uint8) * 255
        return out

    try:
        mosaic = sauvola(denoised)
    except Exception:
        _, mosaic = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if debug_dir:
        # still drop a representative strip for continuity with your debug viewer
        cv2.imwrite(os.path.join(debug_dir, "03_cyl_strip.jpg"), strips[0])
        cv2.imwrite(os.path.join(debug_dir, "04_ocr_ready.jpg"), mosaic)
    return mosaic


def save_image(img: np.ndarray) -> str:
    filename = f"{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(MEDIA_DIR, filename)
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return filename


# =============================== Routes ===============================

@app.post("/unwrap")
def unwrap():
    if "file" not in request.files:
        abort(400, description="No file part")

    file = request.files["file"]
    if not file.filename:
        abort(400, description="No selected file")
    lower = file.filename.lower()
    if not lower.endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv")):
        abort(400, description="Please upload a video file.")

    suffix = os.path.splitext(lower)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    mode = request.args.get(
        "mode", "mosaic"
    )  # "mosaic" (default), "cyl", "polar", or "panorama"
    debug = request.args.get("debug") == "1"
    debug_dir = os.path.join(MEDIA_DIR, "debug", uuid.uuid4().hex) if debug else None

    imu_raw = request.form.get("imu") or request.args.get("imu")
    imu_angles: Optional[List[float]] = None
    if imu_raw:
        try:
            imu_angles = [float(v) for v in imu_raw.split(",") if v.strip()]
        except Exception:
            imu_angles = None

    try:
        flat = extract_flat_label_image(
            tmp_path, mode=mode, debug_dir=debug_dir, imu_angles=imu_angles
        )
        filename = save_image(flat)

        base = request.url_root.rstrip("/")
        image_url = f"{base}/media/{filename}"

        resp: dict[str, Any] = {"imageUrl": image_url}

        # Optional: attach debug artifacts if available
        if debug and debug_dir:
            files: List[str] = []
            for f in [
                "01_best_frame.jpg",
                "02_bounds_overlay.jpg",
                "03_cyl_strip.jpg",     # representative strip
                "03_polar.jpg",
                "04_polar_cropped.jpg",
                "01_panorama.jpg",
            ]:
                p = os.path.join(debug_dir, f)
                if os.path.exists(p):
                    rel = os.path.relpath(p, MEDIA_DIR).replace("\\", "/")
                    files.append(f"{base}/media/{rel}")
            if files:
                resp["debug"] = files

    except Exception as e:
        traceback.print_exc()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        abort(500, description=f"Processing failed: {e}")

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return jsonify(resp)


# =============================== Main ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
