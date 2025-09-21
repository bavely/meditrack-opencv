# meditrack-opencv

Processes short medicine-bottle videos into a small set of sharp frames and
OCR-friendly contact sheets.

## Installation

```bash
pip install -r requirements.txt
```

## Runtime requirements

- [FFmpeg](https://ffmpeg.org/) must be installed and discoverable on `PATH`.
  Set `FFMPEG_PATH` if the executable lives elsewhere.
- The service listens on port `5050` by default; override with the `PORT`
  environment variable.

## Configuration

The service reads its configuration from environment variables. You can either
create a local `.env` file (loaded automatically on startup) or set the
variables in the process environment before launching the app.

### Required

- `AWS_REGION` – AWS region that hosts the target S3 bucket.
- `S3_BUCKET` – S3 bucket name where frames and contact sheets will be stored.

### Optional tuning flags

- `S3_PREFIX` – Logical prefix inside the bucket that keeps job artifacts
  grouped together (defaults to `meditrack`).
- `S3_URL_TTL` – Number of seconds that generated pre-signed download URLs stay
  valid (defaults to `300`).
- `KMS_KEY_ID` – AWS KMS key ID or ARN to enforce SSE-KMS encryption. When not
  provided the app falls back to S3-managed `AES256` encryption.
- `MAX_FRAMES` – Maximum number of representative frames FFmpeg extracts from
  each upload.
- `GRID_COLS` – Number of columns used when laying out the contact sheet.
- `CELL_HEIGHT` – Pixel height of each frame in the contact sheet grid.
- `GRID_PAD` – Padding in pixels around each cell within the contact sheet.

### AWS credentials & permissions

The application uses the standard boto3 credential chain. Provide credentials
through one of the following:

- Environment variables such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  and optionally `AWS_SESSION_TOKEN`.
- An IAM role attached to the compute environment (EC2 instance profile,
  ECS/EKS task role, Lambda execution role, etc.).
- A shared credentials/config file under `~/.aws`.

The IAM principal must be allowed to `s3:ListBucket` (for the bucket
availability check), `s3:PutObject`, and `s3:GetObject` on the configured
bucket/prefix. Uploads always request server-side encryption: either SSE-KMS
with your supplied `KMS_KEY_ID` or S3-managed `AES256` when no key is provided.
Ensure your bucket policy and (when applicable) KMS key policy permit this
behavior so that uploads are accepted and the generated pre-signed URLs remain
usable.

## Docker

```bash
docker build -t meditrack-opencv .
docker run -p 5050:5050 meditrack-opencv
```

Add `-e FFMPEG_PATH=/usr/bin/ffmpeg` when the binary is not on `PATH`.

## API

### `GET /health`
Performs a readiness probe that also confirms the app can reach the configured
S3 bucket and echoes the active runtime configuration. The JSON payload looks
like:

```json
{
  "status": "ok",
  "s3_connection": "connected",
  "bucket": "<bucket-name>",
  "region": "<aws-region>",
  "prefix": "<s3-prefix>",
  "kms_key": "<kms-key-or-AES256>",
  "config": {
    "max_frames": 10,
    "grid_cols": 4,
    "cell_height": 340,
    "grid_pad": 6,
    "url_ttl": 300
  }
}
```

`s3_connection` reports the outcome of the bucket probe (for example,
`"connected"` or an error string). Because the endpoint verifies S3 access and
surfaces the active configuration values, operators can quickly interpret any
non-OK responses and diagnose misconfigurations.

### `POST /unwrap`
Accepts a form field named `file` containing a video. The response is JSON with:

- `jobId` – short identifier you can persist in your system to correlate
  follow-up processing or log lines with the underlying unwrap attempt.
- `frames` – array of URLs for the sampled frames.
- `sheetUrl` – color contact sheet for quick review.
- `ocrSheetUrl` – binarized contact sheet optimized for OCR.
- `ocrBestFrameUrl` – OCR-friendly version of the single sharpest frame.
- `s3` – metadata describing where assets were stored. It includes
  `bucket`, `region`, and `prefix`, which mirror the AWS settings used for the
  job so downstream tooling can build direct S3 references if needed.
- `timing` – performance metrics for the request. `total_s` captures the full
  wall-clock duration, while `ffmpeg_extract_s`, `sheet_build_s`,
  `ocr_post_s`, and `save_s` break down the major processing phases to help
  operators monitor throughput or spot bottlenecks.

All of the frame and sheet URLs are pre-signed links generated during the
request. Download the assets directly from these URLs; they expire after
`S3_URL_TTL` seconds (default `300`).

## Usage

```bash
curl -F "file=@video.mp4" http://localhost:5050/unwrap
```

Persist the `jobId` alongside your own records, and use the `s3` and `timing`
metadata to confirm where artifacts live and how long each stage took.

