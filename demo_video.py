'''
Process single video:
python demo_video.py \
    --video_path /path/to/video.mp4 \
    --checkpoint_path /path/to/dinov3.ckpt \
    --mhr_path /path/to/mhr_model.pt \
    --output_fps 30 \
    --resolution 4k \
    --skip_detection \
    --cleanup

Process all videos in a folder:
python demo_video.py \
    --video_folder /path/to/videos \
    --checkpoint_path /path/to/dinov3.ckpt \
    --mhr_path /path/to/mhr_model.pt \
    --output_fps 30 \
    --resolution 4k \
    --skip_detection \
    --cleanup
'''

import argparse
import os
from glob import glob
import shutil

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def extract_frames_from_video(video_path, output_folder):
    """Extract frames from video and save to output folder, resizing smallest dimension to 720p"""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Calculate dimensions: smallest dimension becomes 720p, maintaining aspect ratio
    min_dimension = min(orig_height, orig_width)
    
    if min_dimension <= 720:
        # Skip resize if already 720p or smaller
        target_height = orig_height
        target_width = orig_width
        should_resize = False
        print(f"Extracting {frame_count} frames from video (FPS: {fps})...")
        print(f"Frame size {orig_width}x{orig_height} is already 720p or smaller, skipping resize")
    else:
        scale_factor = 720 / min_dimension
        target_height = int(orig_height * scale_factor)
        target_width = int(orig_width * scale_factor)
        should_resize = True
        print(f"Extracting {frame_count} frames from video (FPS: {fps})...")
        print(f"Resizing from {orig_width}x{orig_height} to {target_width}x{target_height} (smallest dimension: 720p)")
    
    frame_idx = 0
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame to 720p if needed
            if should_resize:
                frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame
            
            frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_filename, frame_resized)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_folder}")
    return frame_idx, fps


def create_video_from_frames(frames_folder, output_video_path, fps=30, resolution=None):
    """Create video from processed frames
    
    Args:
        frames_folder: Folder containing processed frame images
        output_video_path: Path where output video will be saved
        fps: Frame rate for output video
        resolution: Target resolution as string ('1080p', '4k') or None for original
    """
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(frames_folder, ext))
        ]
    )
    
    if not images_list:
        raise ValueError(f"No images found in {frames_folder}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(images_list[0])
    orig_height, orig_width = first_frame.shape[:2]
    
    # Determine output dimensions
    if resolution == "1080p":
        target_height = 1080
        target_width = int(orig_width * (1080 / orig_height))
        print(f"Resizing to 1080p: {target_width}x{target_height}")
    elif resolution == "4k":
        target_height = 2160
        target_width = int(orig_width * (2160 / orig_height))
        print(f"Resizing to 4K: {target_width}x{target_height}")
    else:
        target_height = orig_height
        target_width = orig_width
        print(f"Using original resolution: {target_width}x{target_height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))
    
    print(f"Creating video at {fps} FPS from {len(images_list)} frames...")
    
    for img_path in tqdm(images_list):
        frame = cv2.imread(img_path)
        if frame is not None:
            # Resize if needed
            if resolution in ["1080p", "4k"]:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")


def process_single_video(video_path, estimator, args):
    """Process a single video file
    
    Args:
        video_path: Path to the video file
        estimator: SAM3DBodyEstimator instance
        args: Command line arguments
    """
    # Get video path and create folders
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(os.path.abspath(video_path))
    
    # Create frames folder (video_name)
    frames_folder = os.path.join(video_dir, video_name)
    
    # Create output folder (video_name_output)
    output_folder = os.path.join(video_dir, f"{video_name}_output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Checkpoint file to track progress
    checkpoint_file = os.path.join(output_folder, "progress.txt")
    
    # Load previously processed frames
    processed_frames = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_frames = set(line.strip() for line in f if line.strip())
        print(f"Found checkpoint: {len(processed_frames)} frames already processed")
    
    # Step 1: Extract frames from video
    print(f"\n=== Processing: {os.path.basename(video_path)} ===")
    print("Step 1: Extracting frames from video")
    frame_count, original_fps = extract_frames_from_video(video_path, frames_folder)
    
    # Step 2: Process frames
    print("Step 2: Processing frames")
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(frames_folder, ext))
        ]
    )
    
    # Filter out already processed frames
    remaining_images = [img for img in images_list if os.path.basename(img) not in processed_frames]
    
    if len(remaining_images) < len(images_list):
        print(f"Resuming from checkpoint: {len(remaining_images)} frames remaining")
    
    if len(remaining_images) == 0:
        print("All frames already processed, skipping to video creation")
    else:
        # Create mesh output folder if saving meshes
        if args.save_mesh:
            mesh_folder = os.path.join(video_dir, f"{video_name}_meshes")
            os.makedirs(mesh_folder, exist_ok=True)
            print(f"Mesh files will be saved to: {mesh_folder}")
        
        for image_path in tqdm(remaining_images, desc="Processing frames"):
            outputs = estimator.process_one_image(
                image_path,
                bbox_thr=args.bbox_thresh,
                use_mask=args.use_mask,
            )

            img = cv2.imread(image_path)
            rend_img = visualize_sample_together(img, outputs, estimator.faces, render_floor=args.render_floor, largest_body_only=args.largest_body_only)
            
            # Save with same filename as input frame
            output_filename = os.path.basename(image_path)
            cv2.imwrite(
                os.path.join(output_folder, output_filename),
                rend_img.astype(np.uint8),
            )
            
            # Save mesh files if requested
            if args.save_mesh and outputs:
                frame_name = os.path.splitext(output_filename)[0]
                for pid, person_output in enumerate(outputs):
                    renderer = Renderer(focal_length=person_output["focal_length"], faces=estimator.faces)
                    tmesh = renderer.vertices_to_trimesh(
                        person_output["pred_vertices"],
                        person_output["pred_cam_t"],
                        LIGHT_BLUE
                    )
                    mesh_filename = f"{frame_name}_person_{pid:03d}.ply"
                    mesh_path = os.path.join(mesh_folder, mesh_filename)
                    tmesh.export(mesh_path)
            
            # Update checkpoint file
            with open(checkpoint_file, 'a') as f:
                f.write(output_filename + '\n')
    
    # Step 3: Create output video
    print("Step 3: Creating output video")
    output_video_path = os.path.join(video_dir, f"{video_name}_output.mp4")
    create_video_from_frames(output_folder, output_video_path, fps=args.output_fps, resolution=args.resolution)
    
    # Optional: Clean up intermediate frames if requested
    if args.cleanup:
        print("Cleaning up intermediate files")
        # Remove checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        shutil.rmtree(frames_folder)
        print(f"Removed extracted frames folder: {frames_folder}")
        shutil.rmtree(output_folder)
        print(f"Removed processed frames folder: {output_folder}")
    else:
        # Keep checkpoint file but mark as complete
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'a') as f:
                f.write('\n# Processing completed successfully\n')
    
    print(f"✓ Completed: {os.path.basename(video_path)}")
    print(f"  Output video: {output_video_path}")
    if not args.cleanup:
        print(f"  Extracted frames: {frames_folder}")
        print(f"  Processed frames: {output_folder}")
    
    return output_video_path


def main(args):
    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("\n=== Loading SAM 3D Body model ===")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name and not args.skip_detection:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    elif args.skip_detection:
        print("\n=== Skipping human detection (using full frame) ===")
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
        static_camera=args.static_camera,
    )
    
    # Get list of videos to process
    video_list = []
    if args.video_folder:
        # Process all videos in folder
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]
        for ext in video_extensions:
            video_list.extend(glob(os.path.join(args.video_folder, ext)))
        video_list = sorted(video_list)
        
        if not video_list:
            print(f"No video files found in {args.video_folder}")
            return
        
        print(f"\nFound {len(video_list)} video(s) to process:")
        for i, video in enumerate(video_list, 1):
            print(f"  {i}. {os.path.basename(video)}")
    elif args.video_path:
        # Process single video
        video_list = [args.video_path]
    else:
        print("Error: Either --video_path or --video_folder must be specified")
        return
    
    # Process each video
    print(f"\n=== Processing {len(video_list)} video(s) ===")
    output_videos = []
    for i, video_path in enumerate(video_list, 1):
        print(f"\n[{i}/{len(video_list)}] {os.path.basename(video_path)}")
        
        # Reset cache between videos for static_camera mode
        if args.static_camera and i > 1:
            estimator.reset_cache()
        
        try:
            output_video = process_single_video(video_path, estimator, args)
            output_videos.append(output_video)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    # Summary
    print(f"\n=== All Done! ===")
    print(f"Successfully processed {len(output_videos)}/{len(video_list)} video(s)")
    if output_videos:
        print("\nOutput videos:")
        for output_video in output_videos:
            print(f"  - {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo_video.py --video_path ./video.mp4 --checkpoint_path ./checkpoints/model.ckpt --mhr_path ./mhr_model.pt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--video_path",
        default=None,
        type=str,
        help="Path to input video file (mutually exclusive with --video_folder)",
    )
    parser.add_argument(
        "--video_folder",
        default=None,
        type=str,
        help="Path to folder containing multiple video files (mutually exclusive with --video_path)",
    )
    parser.add_argument(
        "--output_fps",
        default=30,
        type=int,
        help="Frame rate for output video (default: 30)",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        type=str,
        choices=[None, "1080p", "4k"],
        help="Output video resolution: '1080p' (1920x1080), '4k' (3840x2160), or None for original resolution (default: None)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Remove intermediate extracted frames after processing",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed). Set to empty string '' to skip detection and use full frame (faster).",
    )
    parser.add_argument(
        "--skip_detection",
        action="store_true",
        default=False,
        help="Skip human detection and use full frame (much faster, use when player is always fully visible in frame)",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--render_floor",
        action="store_true",
        default=False,
        help="Render floor plane in side, back, and top views (default: True)",
    )
    parser.add_argument(
        "--largest_body_only",
        action="store_true",
        default=False,
        help="When multiple bodies detected, only process and render the largest one (default: False)",
    )
    parser.add_argument(
        "--static_camera",
        action="store_true",
        default=False,
        help="Enable static camera mode: FOV is estimated from first 5 frames and cached (faster for videos with fixed camera)",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        default=False,
        help="Save 3D mesh as PLY files for each frame and person detected (creates video_name_meshes folder)",
    )
    args = parser.parse_args()

    main(args)
