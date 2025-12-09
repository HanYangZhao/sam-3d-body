# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified version for macOS without visualization
import argparse
import os
from glob import glob
import json

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
from tqdm import tqdm


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Load model and assets
    print("Loading SAM 3D Body model...")
    
    # Determine device (MPS for Mac, CUDA for others, fallback to CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=args.checkpoint_path,
        mhr_path=mhr_path,
        device=device,
    )
    model = model.eval()

    # Initialize optional modules
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

    # Setup estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    # Get all images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(glob(os.path.join(args.image_folder, ext)))
    image_files = sorted(image_files)

    print(f"Processing {len(image_files)} images...")

    for img_path in tqdm(image_files):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to load {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Process image
        try:
            outputs = estimator.process_one_image(
                img_path,
                bbox_thr=args.bbox_thresh,
                use_mask=args.use_mask,
            )

            # Save mesh outputs
            if outputs and len(outputs) > 0:
                for idx, output in enumerate(outputs):
                    suffix = f"_{idx}" if len(outputs) > 1 else ""
                    
                    # Save vertices as numpy file
                    if 'vertices' in output:
                        vertices_path = os.path.join(output_folder, f"{base_name}{suffix}_vertices.npy")
                        np.save(vertices_path, output['vertices'])
                    
                    # Save camera parameters
                    if 'camera' in output:
                        camera_path = os.path.join(output_folder, f"{base_name}{suffix}_camera.json")
                        camera_data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                      for k, v in output['camera'].items()}
                        with open(camera_path, 'w') as f:
                            json.dump(camera_data, f, indent=2)
                    
                    # Save all output data
                    output_path = os.path.join(output_folder, f"{base_name}{suffix}_output.npz")
                    save_dict = {}
                    for k, v in output.items():
                        if isinstance(v, (np.ndarray, torch.Tensor)):
                            if isinstance(v, torch.Tensor):
                                v = v.cpu().numpy()
                            save_dict[k] = v
                    np.savez(output_path, **save_dict)
                    
                print(f"Saved outputs for {img_name} ({len(outputs)} detections)")
            else:
                print(f"No detections for {img_name}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nProcessing complete! Outputs saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - No Visualization (macOS compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo_no_viz.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
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
