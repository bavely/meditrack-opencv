# meditrack-opencv
This service unwraps medicine bottle labels from short videos. It now supports
building a full panoramic texture using either device IMU data or optical flow
to track camera rotation between frames. Frames are sampled with
[FFmpeg](https://ffmpeg.org/) instead of OpenCV, producing a composite image
for OCR.

System-level `ffmpeg` must be available; the supplied `Dockerfile` installs it.
Set the optional `FFMPEG_PATH` environment variable if the binary is not on
`PATH`. The `/unwrap` endpoint checks for FFmpeg and returns a descriptive
error if it cannot be found.

## Debugging
Include `debug=1` as a query parameter on `/unwrap` requests to save
intermediate images. Artifacts are written under `media/debug/<uuid>/`
where `<uuid>` is a unique id for each request. Files may include:

- `01_best_frame.jpg` – sharpest frame selected from the input video
- `02_bounds_overlay.jpg` – label bounds drawn on the chosen frame
- `03_cyl_strip.jpg` – cylindrical strip used for mosaicing
- `03_polar.jpg` – polar transform of the input frame
- `04_polar_cropped.jpg` – cropped polar image

## Example
Send a video and enable debug output:

```
curl -F "video=@sample.mp4" "http://localhost:5050/unwrap?mode=mosaic&debug=1"
```

Then download a debug artifact by id:

```
curl -O http://localhost:5050/media/debug/<uuid>/01_best_frame.jpg
curl -O http://localhost:5050/media/debug/<uuid>/02_bounds_overlay.jpg
```

The response from `/unwrap` will also include direct URLs to these files.
