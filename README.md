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

## Docker

```bash
docker build -t meditrack-opencv .
docker run -p 5050:5050 meditrack-opencv
```

Add `-e FFMPEG_PATH=/usr/bin/ffmpeg` when the binary is not on `PATH`.

## API

### `GET /health`
Simple readiness probe. Returns `{"status": "ok"}`.

### `POST /unwrap`
Accepts a form field named `file` containing a video. The response is JSON with:

- `frames` – array of URLs for the sampled frames.
- `sheetUrl` – color contact sheet for quick review.
- `ocrSheetUrl` – binarized contact sheet optimized for OCR.
- `ocrBestFrameUrl` – OCR-friendly version of the single sharpest frame.

### `GET /media/<path>`
Serves any generated artifact such as frames and sheets.

## Usage

```bash
curl -F "file=@video.mp4" http://localhost:5050/unwrap
```

The response includes the fields listed above along with a `timing` object for
basic profiling.

