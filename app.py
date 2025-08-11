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
    """
    Estimate left/right edges within the label band using |dx| column energy.
    More stable than raw Canny peaks.
    """
    g = _preprocess_gray(frame)
    roi = g[y1:y2, :]
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gx = cv2.convertScaleAbs(gx)
    col_energy = gx.mean(axis=0)
    col_energy = cv2.GaussianBlur(col_energy.reshape(1, -1), (1, 31), 0).ravel()

    thr = float(col_energy.mean() + 0.5 * (col_energy.max() - col_energy.mean()))

    def edge_from_left(arr: np.ndarray) -> int:
        run = 0
        for i, v in enumerate(arr):
            run = run + 1 if v > thr else 0
            if run >= 5:
                return max(0, i - 4)
        return int(0.20 * len(arr))

    left = edge_from_left(col_energy)
    right = len(col_energy) - edge_from_left(col_energy[::-1]) - 1

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
    left: int,
    right: int,
    out_w: int = 2048,
) -> np.ndarray:
    """
    Cylindrical inverse mapping for the ENTIRE label band.
    x_in = cx + f * tan(theta), theta in [-FOV, +FOV].
    """
    band = frame[y1:y2, :, :]
    h, w = band.shape[:2]
    out_h = h

    radius = max(1.0, (right - left) / 2.0)  # visible half-width
    f = radius  # focal length approx

    # Wider FOV to cover more of the visible curvature
    theta_min, theta_max = np.deg2rad(-88), np.deg2rad(88)

    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    for x_out in range(out_w):
        theta = theta_min + (theta_max - theta_min) * (x_out / max(1, (out_w - 1)))
        x_in = cx + f * np.tan(theta)
        map_x[:, x_out] = np.clip(x_in, 0, w - 1)

    for y_out in range(out_h):
        map_y[y_out, :] = y_out

    return cv2.remap(band, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


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
    frame: np.ndarray, y1: int, y2: int, out_w: int, cx: int, left: int, right: int
) -> np.ndarray:
    """Same as unwrap_cylindrical_band, but parameterized so all strips share geometry."""
    band = frame[y1:y2, :, :]
    h, w = band.shape[:2]
    radius = max(1.0, (right - left) / 2.0)
    f = radius
    theta_min, theta_max = np.deg2rad(-88), np.deg2rad(88)

    map_x = np.zeros((h, out_w), dtype=np.float32)
    map_y = np.zeros((h, out_w), dtype=np.float32)
    thetas = theta_min + (theta_max - theta_min) * (np.arange(out_w) / max(1, (out_w - 1)))
    x_in = cx + f * np.tan(thetas)
    x_in = np.clip(x_in, 0, w - 1).astype(np.float32)
    map_x[:] = x_in
    map_y[:] = np.arange(h, dtype=np.float32)[:, None]
    return cv2.remap(band, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _phase_shift(a: np.ndarray, b: np.ndarray, search_y_frac: float = 0.5) -> int:
    """
    Estimate horizontal shift between two unwrapped strips using phase correlation.
    We take a horizontal band around the center (text is strongest there).
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]

    # central band (guarantee at least 8 rows)
    band_frac = max(0.2, min(0.6, search_y_frac))
    y0 = int(ha * (0.5 - band_frac / 2.0))
    y1 = int(ha * (0.5 + band_frac / 2.0))
    y0 = max(0, min(ha - 8, y0))
    y1 = max(y0 + 8, min(ha, y1))

    ga = cv2.cvtColor(a[y0:y1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b[y0:y1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1D Hanning window across width, broadcast per row (avoids shape mismatch)
    win = np.hanning(ga.shape[1]).astype(np.float32)[None, :]
    ga = ga * win
    gb = gb * win

    (shift_x, _), _ = cv2.phaseCorrelate(ga, gb)
    return int(round(shift_x))


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


def build_label_mosaic(strips: List[np.ndarray], base_width: int = 9000) -> np.ndarray:
    """
    Register and stitch strips horizontally. base_width is a safety canvas.
    """
    if not strips:
        raise RuntimeError("No strips to stitch.")
    h = max(s.shape[0] for s in strips)
    canvas = np.zeros((h, base_width, 3), np.uint8)
    # place first strip in the middle
    x_positions = [base_width // 2 - strips[0].shape[1] // 2]
    _blend_overwrite(canvas, strips[0], x_positions[0])

    # stitch others alternating left/right using phase correlation to the first strip
    left_side = True
    for i in range(1, len(strips)):
        ref_idx = 0
        shift = _phase_shift(strips[ref_idx], strips[i])
        if left_side:
            x_new = x_positions[ref_idx] - strips[i].shape[1] + shift
        else:
            x_new = x_positions[ref_idx] + strips[ref_idx].shape[1] + shift
        left_side = not left_side
        x_positions.append(x_new)
        _blend_overwrite(canvas, strips[i], x_new)

    # crop to content (remove empty margins)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    xs = np.where(thr.sum(axis=0) > 0)[0]
    if xs.size > 0:
        x0, x1 = xs[0], xs[-1] + 1
        canvas = canvas[:, x0:x1]
    return canvas


# ============================ Core pipeline ============================

def extract_flat_label_image(video_path: str, mode: str = "mosaic", debug_dir: Optional[str] = None) -> np.ndarray:
    """
    mode: "mosaic" (recommended), "cyl" (single best frame), or "polar".
    """
    if mode not in ("mosaic", "cyl", "polar"):
        mode = "mosaic"

    if mode == "cyl":
        frame = read_video_best_frame(video_path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "01_best_frame.jpg"), frame)

        y1, y2 = find_label_full_bounds(frame)
        left, right, cx = estimate_sides_at_y(frame, y1, y2)

        if debug_dir:
            dbg = frame.copy()
            cv2.line(dbg, (0, y1), (dbg.shape[1], y1), (0, 255, 0), 2)
            cv2.line(dbg, (0, y2), (dbg.shape[1], y2), (0, 255, 0), 2)
            cv2.line(dbg, (left, y1), (left, y2), (255, 0, 0), 2)
            cv2.line(dbg, (right, y1), (right, y2), (255, 0, 0), 2)
            cv2.line(dbg, (((left + right) // 2), y1), (((left + right) // 2), y2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, "02_bounds_overlay.jpg"), dbg)

        strip = unwrap_cylindrical_band(frame, y1, y2, (left + right) // 2, left, right)
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

    # --------- MOSAIC (multi-frame) ----------
    frames = sample_sharp_frames(video_path, target_frames=13)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_best_frame.jpg"), frames[0])

    y1, y2 = find_label_full_bounds_union(frames)

    out_w = 1024  # per-strip width; ~30–40% overlap appears naturally from geometry
    strips: List[np.ndarray] = []
    for i, f in enumerate(frames):
        L, R, CX = estimate_sides_at_y(f, y1, y2)
        strip = unwrap_cylindrical_band_consistent(f, y1, y2, out_w, CX, L, R)
        strips.append(strip)
        if debug_dir and i == 0:
            dbg = f.copy()
            cv2.line(dbg, (0, y1), (dbg.shape[1], y1), (0, 255, 0), 2)
            cv2.line(dbg, (0, y2), (dbg.shape[1], y2), (0, 255, 0), 2)
            cv2.line(dbg, (L, y1), (L, y2), (255, 0, 0), 2)
            cv2.line(dbg, (R, y1), (R, y2), (255, 0, 0), 2)
            cv2.line(dbg, (CX, y1), (CX, y2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, "02_bounds_overlay.jpg"), dbg)

    mosaic = build_label_mosaic(strips, base_width=9000)
    if debug_dir:
        # still drop a representative strip for continuity with your debug viewer
        cv2.imwrite(os.path.join(debug_dir, "03_cyl_strip.jpg"), strips[0])
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

    mode = request.args.get("mode", "mosaic")  # "mosaic" (default), "cyl", or "polar"
    debug = request.args.get("debug") == "1"
    debug_dir = os.path.join(MEDIA_DIR, "debug", uuid.uuid4().hex) if debug else None

    try:
        flat = extract_flat_label_image(tmp_path, mode=mode, debug_dir=debug_dir)
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
