# app.py
import os
import uuid
import time
import tempfile
import traceback
import subprocess
import shutil
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

# =========================== Config ===========================
APP_PORT = int(os.getenv("PORT", "5050"))
MEDIA_DIR = os.path.abspath("./media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# How many frames to sample from the whole video (keep this small)
MAX_FRAMES = 10
# Contact-sheet shape and cell size (keeps image reasonable for OCR)
GRID_COLS = 4
CELL_HEIGHT = 340  # per-frame height inside the grid
GRID_PAD = 6      # pixels between cells

# =============================================================

app = Flask(__name__)
CORS(app)  # allow all origins in dev


# =========================== Health & Media ===========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/media/<path:filename>")
def media(filename: str):
    return send_from_directory(MEDIA_DIR, filename, as_attachment=False, max_age=3600)


# =========================== Utilities ================================

def resolve_ffmpeg() -> str:
    """
    Find ffmpeg via FFMPEG_PATH or PATH.
    (No extra deps; if you installed ffmpeg, this will find it.)
    """
    cand = os.getenv("FFMPEG_PATH")
    if cand:
        cand = cand.strip().strip('"').strip("'")
        if os.path.isfile(cand):
            return cand
        if os.path.isdir(cand):
            exe = os.path.join(cand, "ffmpeg.exe")
            if os.path.isfile(exe):
                return exe
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return ""


def variance_of_laplacian(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _autocrop_black_borders(img: np.ndarray, thresh: int = 8) -> np.ndarray:
    """Remove constant black borders (pillarbox/letterbox)."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    ys = np.where(m.sum(axis=1) > 0)[0]
    xs = np.where(m.sum(axis=0) > 0)[0]
    if ys.size < 5 or xs.size < 5:
        return img
    y1, y2 = int(ys[0]), int(ys[-1]) + 1
    x1, x2 = int(xs[0]), int(xs[-1]) + 1
    cropped = img[y1:y2, x1:x2]
    if min(cropped.shape[:2]) < 20:
        return img
    return cropped


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(1, int(round(w * (target_h / float(h)))))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def pick_best_frame(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    """Pick the sharpest frame by Laplacian variance."""
    if not frames:
        return None
    scores = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        scores.append((variance_of_laplacian(gray), f))
    scores.sort(key=lambda t: t[0], reverse=True)
    return scores[0][1]


def build_contact_sheet(frames: List[np.ndarray],
                        cols: int = GRID_COLS,
                        cell_h: int = CELL_HEIGHT,
                        pad: int = GRID_PAD,
                        bg: int = 255) -> np.ndarray:
    """
    Compact grid of frames (color). Predictable size and fast.
    """
    imgs = []
    for f in frames:
        if f is None:
            continue
        f = _autocrop_black_borders(f)
        f = _resize_to_height(f, cell_h)
        if min(f.shape[:2]) >= 20:
            imgs.append(f)

    if not imgs:
        raise RuntimeError("No valid frames for contact sheet.")

    # Normalize width so OCR text size is consistent
    widths = [i.shape[1] for i in imgs]
    target_w = int(np.median(widths))
    norm = [cv2.resize(i, (target_w, cell_h), interpolation=cv2.INTER_AREA) for i in imgs]

    rows = (len(norm) + cols - 1) // cols
    sheet_w = cols * target_w + (cols + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad
    canvas = np.full((sheet_h, sheet_w, 3), bg, np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(norm):
                break
            y = pad + r * (cell_h + pad)
            x = pad + c * (target_w + pad)
            canvas[y:y + cell_h, x:x + target_w] = norm[idx]
            idx += 1
    return canvas


def postprocess_for_ocr_fast(img: np.ndarray) -> np.ndarray:
    """
    Faster OCR pipeline than non-local means:
      grayscale -> light Gaussian -> CLAHE -> adaptive threshold -> small morph.
    Returns a clean binary image (PNG-safe).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Light blur for noise without killing edges (much faster than NLM)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # Adaptive threshold: slightly larger block for uneven lighting
    h, w = gray.shape[:2]
    block = 41 if min(h, w) > 500 else 31
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, 6
    )

    # Tiny morphology: clean specks, close micro gaps
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k_open, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k_close, iterations=1)
    return bin_img


def save_image(img: np.ndarray, ext: str = ".png", prefix: str = "") -> str:
    filename = f"{prefix}{uuid.uuid4().hex}{ext}"
    out_path = os.path.join(MEDIA_DIR, filename)
    params = []
    if ext.lower() == ".jpg": #was originally .jpg
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    ok = cv2.imwrite(out_path, img, params)
    if not ok:
        raise RuntimeError(f"Failed to save image to {out_path}")
    return filename


# ============================ Frame extraction ============================

def extract_representative_frames_ffmpeg(ffmpeg_cmd: str, video_path: str, out_dir: Path,
                                         max_frames: int = MAX_FRAMES) -> List[Path]:
    """
    Use ffmpeg's 'thumbnail' filter to pick representative frames and limit count.
    This is far faster than writing dozens of frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 'thumbnail=n' picks one "best" frame from each batch of n frames.
    # We don't know the total frames here; pick a reasonable cycle.
    cycle = 100
    proc = subprocess.run(
        [
            ffmpeg_cmd,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", video_path,
            "-vf", f"thumbnail={cycle},scale=iw:-2",  # keep width, even height
            "-frames:v", str(max_frames),
            "-vsync", "vfr",
            str(out_dir / "frame_%03d.png"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed: {proc.stderr.strip() or 'unknown error'}")

    return sorted(out_dir.glob("frame_*.png"))


# =============================== Routes ===============================

@app.post("/unwrap")
def unwrap():
    if "file" not in request.files:
        abort(400, description="No file part")

    t0 = time.perf_counter()
    file = request.files["file"]
    if not file or not file.filename:
        abort(400, description="No selected file")
    lower = file.filename.lower()
    if not lower.endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv")):
        abort(400, description="Please upload a video file.")

    suffix = os.path.splitext(lower)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    job_id = uuid.uuid4().hex[:8]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_dir = tmpdir / "frames"

            # --- Extract a small, representative set of frames ---
            t1 = time.perf_counter()
            ffmpeg_cmd = resolve_ffmpeg() or "./ffmpeg/bin/ffmpeg"
            if not shutil.which(ffmpeg_cmd) and not os.path.isfile(ffmpeg_cmd):
                abort(500, description="FFmpeg not found. Add to PATH or set FFMPEG_PATH.")

            frame_paths = extract_representative_frames_ffmpeg(ffmpeg_cmd, tmp_path, frame_dir, MAX_FRAMES)
            t2 = time.perf_counter()

            # Persist the selected frames (small count) under media/job_id/
            job_media_dir = Path(MEDIA_DIR) / f"frames_{job_id}"
            job_media_dir.mkdir(parents=True, exist_ok=True)
            stored_frame_names: List[str] = []
            for p in frame_paths:
                dest = job_media_dir / p.name
                shutil.copy2(p, dest)
                stored_frame_names.append(str(Path("frames_" + job_id) / p.name))

            # Load frames into memory
            frames = [cv2.imread(str(p)) for p in frame_paths if p.is_file()]
            frames = [f for f in frames if f is not None]
            if not frames:
                raise RuntimeError("No frames extracted from video.")

            # Pick a single sharp frame for a "best-frame OCR" (often enough)
            best = pick_best_frame(frames)

            # Build a compact contact sheet (fast + predictable)
            t3 = time.perf_counter()
            sheet_color = build_contact_sheet(frames, cols=GRID_COLS, cell_h=CELL_HEIGHT, pad=GRID_PAD)
            t4 = time.perf_counter()

            # OCR-friendly versions
            best_ocr = postprocess_for_ocr_fast(best) if best is not None else None
            sheet_ocr = postprocess_for_ocr_fast(sheet_color)
            t5 = time.perf_counter()

            # Save outputs (PNG for lossless OCR)
            sheet_color_name = save_image(sheet_color, ext=".png", prefix=f"sheet_{job_id}_")
            sheet_ocr_name = save_image(sheet_ocr, ext=".png", prefix=f"ocrsheet_{job_id}_")
            best_ocr_name = save_image(best_ocr, ext=".png", prefix=f"ocrbest_{job_id}_") if best_ocr is not None else None
            t6 = time.perf_counter()

        base = request.url_root.rstrip("/")
        resp: dict[str, Any] = {
            "frames": [f"{base}/media/{name}" for name in stored_frame_names],  # selected frames (few, persisted)
            "sheetUrl": f"{base}/media/{sheet_color_name}",   # color contact sheet for eyeballing
            "ocrSheetUrl": f"{base}/media/{sheet_ocr_name}",  # OCR-friendly grid (binary)
            "ocrBestFrameUrl": f"{base}/media/{best_ocr_name}" if best_ocr_name else None,
            "timing": {
                "total_s": round(t6 - t0, 3),
                "ffmpeg_extract_s": round(t2 - t1, 3),
                "sheet_build_s": round(t4 - t3, 3),
                "ocr_post_s": round(t5 - t4, 3),
                "save_s": round(t6 - t5, 3),
            }
        }
    except Exception as e:
        traceback.print_exc()
        abort(500, description=f"Processing failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Server-side log for profiling
    print(f"[unwrap] timing: {resp['timing']}")
    return jsonify(resp)


# =============================== Main ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
