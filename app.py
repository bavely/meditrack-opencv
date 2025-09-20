# app.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import uuid
import time
import tempfile
import traceback
import subprocess
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# =========================== Config ===========================
APP_PORT = int(os.getenv("PORT", "5050"))

# S3 / AWS - All from environment variables
AWS_REGION = os.getenv("AWS_REGION")
if not AWS_REGION:
    raise RuntimeError("AWS_REGION environment variable is required in .env file")

S3_BUCKET = os.getenv("S3_BUCKET")
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is required in .env file")

# Optional: logical prefix inside your bucket
S3_PREFIX = os.getenv("S3_PREFIX", "meditrack")

# Optional: pre-signed URL TTL (seconds)
S3_URL_TTL = int(os.getenv("S3_URL_TTL", "300"))

# Optional: use a CMK for server-side encryption (RECOMMENDED for HIPAA)
KMS_KEY_ID = os.getenv("KMS_KEY_ID")

# Processing knobs
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "10"))
GRID_COLS = int(os.getenv("GRID_COLS", "4"))
CELL_HEIGHT = int(os.getenv("CELL_HEIGHT", "340"))
GRID_PAD = int(os.getenv("GRID_PAD", "6"))

# =============================================================
app = Flask(__name__)
CORS(app)  # allow all origins in dev

# ---------- S3 helpers ----------
def create_s3_client():
    """Create S3 client with proper error handling"""
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")

        _boto_cfg = BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            region_name=AWS_REGION
        )

        client_kwargs = {
            "region_name": AWS_REGION,
            "config": _boto_cfg,
        }

        if aws_access_key and aws_secret_key:
            client_kwargs["aws_access_key_id"] = aws_access_key
            client_kwargs["aws_secret_access_key"] = aws_secret_key
            if aws_session_token:
                client_kwargs["aws_session_token"] = aws_session_token
        elif aws_access_key or aws_secret_key:
            print(
                "Incomplete AWS credentials found in environment; "
                "falling back to boto3 default credential chain."
            )

        # Create S3 client, relying on boto3's default credential provider chain
        client = boto3.client("s3", **client_kwargs)
        
        # Test the connection by checking if bucket exists and is accessible
        try:
            client.head_bucket(Bucket=S3_BUCKET)
            print(f"✓ Successfully connected to S3 bucket: {S3_BUCKET} in region: {AWS_REGION}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise RuntimeError(f"S3 bucket '{S3_BUCKET}' does not exist in region '{AWS_REGION}'")
            elif error_code == '403':
                raise RuntimeError(f"Access denied to S3 bucket '{S3_BUCKET}'. Check your AWS credentials and permissions.")
            else:
                raise RuntimeError(f"S3 bucket access error: {e}")
        
        return client
    except NoCredentialsError:
        raise RuntimeError(
            "Unable to locate AWS credentials. Configure environment variables, an IAM role, or a shared credentials file."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create S3 client: {e}")

# Initialize S3 client
s3 = create_s3_client()

def s3_key(*parts: str) -> str:
    # Build a safe key under the chosen prefix; never include PHI in keys.
    safe = [p.strip("/ ").replace("\\", "/") for p in parts if p and p.strip()]
    return "/".join([S3_PREFIX] + safe)

def s3_put_bytes(data: bytes, key: str, content_type: str = "application/octet-stream", metadata: Optional[dict] = None) -> None:
    """Upload bytes to S3 with proper error handling"""
    try:
        extra = {
            "Bucket": S3_BUCKET,
            "Key": key,
            "Body": data,
            "ContentType": content_type,
            "Metadata": metadata or {},
        }
        
        # Enforce server-side encryption (SSE-KMS if provided, else AES256)
        if KMS_KEY_ID and KMS_KEY_ID.strip():
            extra["ServerSideEncryption"] = "aws:kms"
            extra["SSEKMSKeyId"] = KMS_KEY_ID
            print(f"Using KMS encryption with key: {KMS_KEY_ID}")
        else:
            extra["ServerSideEncryption"] = "AES256"
            print("Using AES256 encryption")

        s3.put_object(**extra)
        print(f"✓ Successfully uploaded to S3: {key}")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            raise RuntimeError(f"Access denied when uploading to S3 key '{key}'. Check your AWS permissions.")
        elif error_code == 'InvalidBucketName':
            raise RuntimeError(f"Invalid S3 bucket name: {S3_BUCKET}")
        else:
            raise RuntimeError(f"S3 upload error for key '{key}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload to S3 key '{key}': {e}")

def s3_presign_get(key: str, expires: int = S3_URL_TTL) -> str:
    """Generate pre-signed URL with error handling"""
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate pre-signed URL for key '{key}': {e}")

def cv2_to_bytes(img: np.ndarray, ext: str) -> Tuple[bytes, str]:
    """
    Encode a numpy image to bytes for S3 and return (bytes, mime_type).
    """
    ext = ext.lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        ext = ".png"  # default to PNG for OCR friendliness
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return (buf.tobytes(), mime)

# =========================== Health ===========================
@app.get("/health")
def health():
    """Health check endpoint that also verifies S3 connectivity"""
    try:
        # Test S3 connectivity
        s3.head_bucket(Bucket=S3_BUCKET)
        s3_status = "connected"
    except Exception as e:
        s3_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "s3_connection": s3_status,
        "bucket": S3_BUCKET,
        "region": AWS_REGION,
        "prefix": S3_PREFIX,
        "kms_key": KMS_KEY_ID if KMS_KEY_ID else "AES256 (default)",
        "config": {
            "max_frames": MAX_FRAMES,
            "grid_cols": GRID_COLS,
            "cell_height": CELL_HEIGHT,
            "grid_pad": GRID_PAD,
            "url_ttl": S3_URL_TTL
        }
    }

# =========================== Utilities ========================
def resolve_ffmpeg() -> str:
    """Resolve FFmpeg path from environment or system PATH"""
    # First check environment variable
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path:
        ffmpeg_path = ffmpeg_path.strip().strip('"').strip("'")
        if os.path.isfile(ffmpeg_path):
            return ffmpeg_path
        if os.path.isdir(ffmpeg_path):
            exe = os.path.join(ffmpeg_path, "ffmpeg.exe")
            if os.path.isfile(exe):
                return exe
            exe = os.path.join(ffmpeg_path, "ffmpeg")
            if os.path.isfile(exe):
                return exe
    
    # Check system PATH
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    
    # Check common installation paths
    common_paths = [
        "./ffmpeg/bin/ffmpeg",
        "./ffmpeg/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg"
    ]
    
    for path in common_paths:
        if os.path.isfile(path):
            return path
    
    return ""

def variance_of_laplacian(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())

def _autocrop_black_borders(img: np.ndarray, thresh: int = 8) -> np.ndarray:
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    h, w = gray.shape[:2]
    block = 41 if min(h, w) > 500 else 31
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, 6
    )
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k_open, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k_close, iterations=1)
    return bin_img

# ============================ Frame extraction ============================
def extract_representative_frames_ffmpeg(ffmpeg_cmd: str, video_path: str, out_dir: Path,
                                         max_frames: int = MAX_FRAMES) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cycle = 100
    proc = subprocess.run(
        [
            ffmpeg_cmd,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", video_path,
            "-vf", f"thumbnail={cycle},scale=iw:-2",
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
    job_prefix = s3_key("jobs", job_id)  # e.g., "meditrack/jobs/abcd1234"
    print(f"Processing job {job_id} - uploading to S3 prefix: {job_prefix}")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_dir = tmpdir / "frames"

            # --- Extract a small, representative set of frames ---
            t1 = time.perf_counter()
            ffmpeg_cmd = resolve_ffmpeg()
            if not ffmpeg_cmd:
                abort(500, description="FFmpeg not found. Install FFmpeg and add to PATH or set FFMPEG_PATH in .env file.")

            print(f"Using FFmpeg: {ffmpeg_cmd}")
            frame_paths = extract_representative_frames_ffmpeg(ffmpeg_cmd, tmp_path, frame_dir, MAX_FRAMES)
            print(f"Extracted {len(frame_paths)} frames")
            t2 = time.perf_counter()

            # Upload selected frames to S3 (private) and collect pre-signed GET URLs
            stored_frame_urls: List[str] = []
            print("Uploading individual frames to S3...")
            for i, p in enumerate(frame_paths):
                with open(p, "rb") as f:
                    data = f.read()
                key = f"{job_prefix}/frames/{p.name}"  # PNGs from ffmpeg
                s3_put_bytes(
                    data,
                    key,
                    content_type="image/png",
                    metadata={"job": job_id, "type": "frame", "frame_index": str(i)}
                )
                stored_frame_urls.append(s3_presign_get(key))

            # Load frames into memory for processing
            frames = [cv2.imread(str(p)) for p in frame_paths if p.is_file()]
            frames = [f for f in frames if f is not None]
            if not frames:
                raise RuntimeError("No frames extracted from video.")

            # Pick a single sharp frame for a "best-frame OCR"
            best = pick_best_frame(frames)

            # Build a compact contact sheet
            t3 = time.perf_counter()
            print("Building contact sheet...")
            sheet_color = build_contact_sheet(frames, cols=GRID_COLS, cell_h=CELL_HEIGHT, pad=GRID_PAD)
            t4 = time.perf_counter()

            # OCR-friendly versions
            print("Preparing OCR-friendly versions...")
            best_ocr = postprocess_for_ocr_fast(best) if best is not None else None
            sheet_ocr = postprocess_for_ocr_fast(sheet_color)
            t5 = time.perf_counter()

            # Save outputs to S3 (PNG for lossless OCR)
            print("Uploading processed images to S3...")
            
            # Sheet (color)
            sheet_color_bytes, sheet_color_mime = cv2_to_bytes(sheet_color, ".png")
            sheet_color_key = f"{job_prefix}/outputs/sheet_{job_id}_{uuid.uuid4().hex[:8]}.png"
            s3_put_bytes(sheet_color_bytes, sheet_color_key, content_type=sheet_color_mime,
                         metadata={"job": job_id, "type": "sheet_color"})
            sheet_color_url = s3_presign_get(sheet_color_key)

            # OCR sheet (binary)
            sheet_ocr_bytes, sheet_ocr_mime = cv2_to_bytes(sheet_ocr, ".png")
            sheet_ocr_key = f"{job_prefix}/outputs/ocrsheet_{job_id}_{uuid.uuid4().hex[:8]}.png"
            s3_put_bytes(sheet_ocr_bytes, sheet_ocr_key, content_type=sheet_ocr_mime,
                         metadata={"job": job_id, "type": "sheet_ocr"})
            sheet_ocr_url = s3_presign_get(sheet_ocr_key)

            # Best frame OCR (optional)
            best_ocr_url = None
            if best_ocr is not None:
                best_ocr_bytes, best_ocr_mime = cv2_to_bytes(best_ocr, ".png")
                best_ocr_key = f"{job_prefix}/outputs/ocrbest_{job_id}_{uuid.uuid4().hex[:8]}.png"
                s3_put_bytes(best_ocr_bytes, best_ocr_key, content_type=best_ocr_mime,
                             metadata={"job": job_id, "type": "best_ocr"})
                best_ocr_url = s3_presign_get(best_ocr_key)

            t6 = time.perf_counter()

        resp: dict[str, Any] = {
            "jobId": job_id,
            "frames": stored_frame_urls,               # pre-signed GET URLs for individual frames
            "sheetUrl": sheet_color_url,               # color contact sheet (pre-signed)
            "ocrSheetUrl": sheet_ocr_url,              # OCR-friendly grid (pre-signed)
            "ocrBestFrameUrl": best_ocr_url,           # pre-signed or None
            "s3": {
                "bucket": S3_BUCKET,
                "region": AWS_REGION,
                "prefix": job_prefix,
            },
            "timing": {
                "total_s": round(t6 - t0, 3),
                "ffmpeg_extract_s": round(t2 - t1, 3),
                "sheet_build_s": round(t4 - t3, 3),
                "ocr_post_s": round(t5 - t4, 3),
                "save_s": round(t6 - t5, 3),
            }
        }
        
        print(f"✓ Job {job_id} completed successfully")
        
    except Exception as e:
        print(f"✗ Job {job_id} failed: {e}")
        traceback.print_exc()
        abort(500, description=f"Processing failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    print(f"[unwrap] Job {job_id} timing: {resp['timing']}")
    return jsonify(resp)

# =============================== Main ===============================
if __name__ == "__main__":
    print("=" * 50)
    print("MEDITRACK VIDEO PROCESSOR")
    print("=" * 50)
    print(f"Port: {APP_PORT}")
    print(f"AWS Region: {AWS_REGION}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"S3 Prefix: {S3_PREFIX}")
    print(f"Max Frames: {MAX_FRAMES}")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)