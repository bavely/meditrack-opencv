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
Performs a readiness probe and validates connectivity to the configured S3
bucket. A successful response returns `200 OK` with a JSON body structured as
follows:

- `status` (`string`) – High-level readiness indicator (`"ok"` when the service
  is healthy).
- `s3_connection` (`string`) – Result of the S3 access check (for example,
  `"connected"` or an error message).
- `bucket` (`string`) – Name of the S3 bucket the service uses for storage.
- `region` (`string`) – AWS region tied to the bucket and active configuration.
- `prefix` (`string`) – Logical bucket prefix where job artifacts are written.
- `kms_key` (`string`) – Encryption configuration reported as the chosen KMS
  key ID/ARN or `AES256` when falling back to S3-managed keys.
- `config` (`object`) – Snapshot of runtime tuning parameters containing:
  - `max_frames` (`integer`) – Maximum number of frames FFmpeg extracts per
    upload.
  - `grid_cols` (`integer`) – Number of columns used when assembling contact
    sheets.
  - `cell_height` (`integer`) – Pixel height allocated to each frame in the
    contact sheet grid.
  - `grid_pad` (`integer`) – Padding in pixels between cells within the grid.
  - `url_ttl` (`integer`) – Lifetime in seconds for generated pre-signed URLs.

Because the endpoint verifies S3 access and surfaces the active configuration
values, operators can quickly interpret non-OK responses and diagnose
misconfigurations.

### `POST /unwrap`
Accepts a form field named `file` containing a video. The response is JSON with:

- `jobId` – short identifier you can persist to correlate follow-up processing,
  callbacks, or log entries with the originating unwrap attempt.
- `frames` – array of URLs for the sampled frames.
- `sheetUrl` – color contact sheet for quick review.
- `ocrSheetUrl` – binarized contact sheet optimized for OCR.
- `ocrBestFrameUrl` – OCR-friendly version of the single sharpest frame.
- `s3` – metadata describing where artifacts were stored so downstream systems
  can build direct S3 references or audit storage:
  - `bucket` – target S3 bucket name.
  - `region` – AWS region that owns the bucket.
  - `prefix` – logical key prefix under which the job's assets were written.
- `timing` – performance metrics for the request to help monitor throughput and
  spot bottlenecks:
  - `total_s` – overall wall-clock processing duration.
  - `ffmpeg_extract_s` – time spent extracting representative frames.
  - `sheet_build_s` – time required to assemble the contact sheet.
  - `ocr_post_s` – time consumed by OCR-oriented post-processing steps.
  - `save_s` – time needed to upload artifacts to S3.

All of the frame and sheet URLs are pre-signed links generated during the
request. Download the assets directly from these URLs; they expire after
`S3_URL_TTL` seconds (default `300`). Use `jobId` for bookkeeping,
consult the `s3` metadata when you need raw S3 paths, and review `timing` to
track performance trends.

## Usage

```bash
curl -F "file=@video.mp4" http://localhost:5050/unwrap
```

