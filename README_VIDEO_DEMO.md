# SAM 3D Body - Video Processing Demo

Process videos to estimate 3D human pose and render mesh visualizations with multiple camera views.

## Quick Start

### Basic Usage
```bash
python demo_video.py \
    --video_path video.mp4 \
    --checkpoint_path checkpoints/model.ckpt
```

### Fast Processing (Tennis/Sports Videos)
```bash
python demo_video.py \
    --video_path tennis_serve.mp4 \
    --checkpoint_path checkpoints/model.ckpt \
    --skip_detection \
    --static_camera \
    --render_floor \
    --resolution 1080p \
    --cleanup
```

## Command-Line Arguments

### Input/Output Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video_path` | str | None | Path to input video file (use this OR --video_folder) |
| `--video_folder` | str | None | Path to folder with multiple videos (batch processing) |
| `--output_fps` | int | 30 | Frame rate for output video |
| `--resolution` | str | None | Output resolution: `1080p`, `4k`, or `None` for original |
| `--cleanup` | flag | False | Remove intermediate frame folders after processing |

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint_path` | str | **required** | Path to SAM 3D Body model checkpoint |
| `--detector_name` | str | vitdet | Human detection model (use `vitdet` or empty string) |
| `--segmentor_name` | str | sam2 | Human segmentation model |
| `--fov_name` | str | moge2 | FOV estimation model |
| `--detector_path` | str | "" | Path to detector model folder (or set `SAM3D_DETECTOR_PATH`) |
| `--segmentor_path` | str | "" | Path to segmentor model folder (or set `SAM3D_SEGMENTOR_PATH`) |
| `--fov_path` | str | "" | Path to FOV model folder (or set `SAM3D_FOV_PATH`) |
| `--mhr_path` | str | "" | Path to MoHR/assets folder (or set `SAM3D_MHR_PATH`) |

### Detection & Processing Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--skip_detection` | flag | False | Skip human detection, use full frame (10-20x faster) |
| `--bbox_thresh` | float | 0.8 | Bounding box detection confidence threshold |
| `--use_mask` | flag | False | Use mask-conditioned prediction (auto-generated from bbox) |
| `--largest_body_only` | flag | False | Process only the largest detected body |

### Performance Optimizations

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--static_camera` | flag | False | Cache FOV from first 5 frames (faster for fixed camera) |

### Visualization Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--render_floor` | flag | True | Render floor plane in side, back, and top views |

## Usage Examples

### 1. Single Video - Maximum Quality
```bash
python demo_video.py \
    --video_path input.mp4 \
    --checkpoint_path checkpoints/model.ckpt \
    --resolution 4k \
    --use_mask
```

### 2. Tennis Video - Maximum Speed
For videos where the player is always fully visible:
```bash
python demo_video.py \
    --video_path tennis.mp4 \
    --checkpoint_path checkpoints/model.ckpt \
    --skip_detection \
    --static_camera \
    --render_floor \
    --cleanup
```

**Performance**: ~10-20x faster (2-5 minutes for 1000 frames vs 30-60 minutes)

### 3. Multiple Videos - Batch Processing
```bash
python demo_video.py \
    --video_folder videos/ \
    --checkpoint_path checkpoints/model.ckpt \
    --skip_detection \
    --static_camera \
    --resolution 1080p
```

### 4. Complex Scene - Multiple People
When there are multiple people in frame:
```bash
python demo_video.py \
    --video_path crowd.mp4 \
    --checkpoint_path checkpoints/model.ckpt \
    --largest_body_only \
    --bbox_thresh 0.9
```

### 5. Dynamic Camera - Full Detection
For videos with moving/zooming camera:
```bash
python demo_video.py \
    --video_path action.mp4 \
    --checkpoint_path checkpoints/model.ckpt \
    --use_mask
```

## Environment Variables

You can set paths via environment variables instead of command-line arguments:

```bash
export SAM3D_DETECTOR_PATH="/path/to/detector"
export SAM3D_SEGMENTOR_PATH="/path/to/segmentor"
export SAM3D_FOV_PATH="/path/to/fov"
export SAM3D_MHR_PATH="/path/to/mhr"

python demo_video.py --video_path video.mp4 --checkpoint_path model.ckpt
```

## Output Structure

For input video `tennis_serve.mp4`:
```
video_directory/
├── tennis_serve.mp4              # Original video
├── tennis_serve/                 # Extracted frames (720p)
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── tennis_serve_output/          # Processed frames with visualizations
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
└── tennis_serve_output.mp4       # Final output video
```

Use `--cleanup` to automatically remove the frame folders after processing.

## Performance Tips

### For Maximum Speed
1. Use `--skip_detection` when subject is always fully visible
2. Use `--static_camera` for tripod/fixed camera videos
3. Use `--cleanup` to save disk space
4. Process at lower resolution first, then upscale with `--resolution`

### When to Use Each Flag

| Scenario | Recommended Flags |
|----------|-------------------|
| Tennis serve (behind player) | `--skip_detection --static_camera --render_floor` |
| Sports training (full body visible) | `--skip_detection --static_camera` |
| Crowd scene (multiple people) | `--largest_body_only --bbox_thresh 0.9` |
| Handheld/moving camera | (no special flags, full detection) |
| Close-up with zoom changes | (no special flags, full detection) |

### Speed Comparison (1000-frame video, 720p)

| Configuration | Processing Time | Speedup |
|---------------|-----------------|---------|
| Baseline (all features) | 30-60 minutes | 1x |
| `--static_camera` | 20-40 minutes | 1.5x |
| `--skip_detection --static_camera` | 2-5 minutes | 10-20x |

## Troubleshooting

### No detections / Empty output
- Try lowering `--bbox_thresh` (default: 0.8)
- Use `--skip_detection` if subject is fully visible
- Check that input video is valid and not corrupted

### Multiple people detected
- Use `--largest_body_only` to track only the main subject
- Increase `--bbox_thresh` for stricter detection

### Slow processing
- Use `--skip_detection` for ~10x speedup (if subject always visible)
- Use `--static_camera` for fixed camera videos
- Process at original resolution, don't upscale with `--resolution`

### Out of memory
- Reduce input video resolution before processing
- Use `--cleanup` to free up disk space during processing
- Close other GPU-intensive applications

## Output Visualizations

Each processed frame contains 6 views:
1. **2D Keypoints** - Detected skeleton overlay on original image
2. **3D Mesh** - Rendered 3D body mesh on original image
3. **Side Right View** - Profile view from the right
4. **Side Left View** - Profile view from the left
5. **Back View** - View from behind
6. **Top View** - Bird's eye view from above

Optional floor plane is rendered in views 3-6 when `--render_floor` is enabled (tennis court green).

## Additional Resources

- Full performance optimization guide: `PERFORMANCE_OPTIMIZATIONS.md`
- Model checkpoints: Download from Hugging Face
- Issue tracker: GitHub repository

---

**Note**: First run will download model weights automatically. Ensure you have sufficient disk space (~10GB for all models).
