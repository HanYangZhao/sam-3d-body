'''
python demo_video.py \
    --video_path /path/to/video.mp4 \
    --checkpoint_path /path/to/dinov3.ckpt \
    --mhr_path /path/to/mhr_model.pt \
    --output_fps 30 \
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
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm


def extract_frames_from_video(video_path, output_folder):
    """Extract frames from video and save to output folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Extracting {frame_count} frames from video (FPS: {fps})...")
    
    frame_idx = 0
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_folder}")
    return frame_idx, fps


def create_video_from_frames(frames_folder, output_video_path, fps=30):
    """Create video from processed frames"""
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
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video at {fps} FPS from {len(images_list)} frames...")
    
    for img_path in tqdm(images_list):
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")


def main(args):
    # Get video path and create folders
    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(os.path.abspath(video_path))
    
    # Create frames folder (video_name)
    frames_folder = os.path.join(video_dir, video_name)
    
    # Create output folder (video_name_output)
    output_folder = os.path.join(video_dir, f"{video_name}_output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Extract frames from video
    print("\n=== Step 1: Extracting frames from video ===")
    frame_count, original_fps = extract_frames_from_video(video_path, frames_folder)
    
    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("\n=== Step 2: Loading SAM 3D Body model ===")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
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
    )

    # Step 3: Process frames
    print("\n=== Step 3: Processing frames ===")
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(frames_folder, ext))
        ]
    )

    for image_path in tqdm(images_list, desc="Processing frames"):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        
        # Save with same filename as input frame
        output_filename = os.path.basename(image_path)
        cv2.imwrite(
            os.path.join(output_folder, output_filename),
            rend_img.astype(np.uint8),
        )
    
    # Step 4: Create output video
    print("\n=== Step 4: Creating output video ===")
    output_video_path = os.path.join(video_dir, f"{video_name}_output.mp4")
    create_video_from_frames(output_folder, output_video_path, fps=args.output_fps)
    
    # Optional: Clean up intermediate frames if requested
    if args.cleanup:
        print("\n=== Cleaning up intermediate files ===")
        shutil.rmtree(frames_folder)
        print(f"Removed frames folder: {frames_folder}")
    
    print(f"\n=== Done! ===")
    print(f"Output video: {output_video_path}")
    print(f"Processed frames: {output_folder}")


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
        required=True,
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output_fps",
        default=30,
        type=int,
        help="Frame rate for output video (default: 30)",
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
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
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
    args = parser.parse_args()

    main(args)
