# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def visualize_sample_together(img_cv2, outputs, faces, render_floor=True, largest_body_only=False):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()
    
    # If largest_body_only is True, filter to keep only the largest detected body
    if largest_body_only and len(outputs) > 1:
        # Calculate bbox area for each detected body
        bbox_areas = []
        for output in outputs:
            bbox = output['bbox']  # [x1, y1, x2, y2]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            bbox_areas.append(area)
        # Keep only the output with the largest bbox
        largest_idx = np.argmax(bbox_areas)
        outputs = [outputs[largest_idx]]
        print(f"Filtered to largest body (bbox area: {bbox_areas[largest_idx]:.0f} pixels)")

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Calculate mesh bounding box to ensure full body is in frame
    mesh_min = np.min(all_pred_vertices, axis=0)
    mesh_max = np.max(all_pred_vertices, axis=0)
    mesh_size = np.linalg.norm(mesh_max - mesh_min)
    
    # Adjust camera distance to fit the full body in frame
    # Use a conservative multiplier to ensure nothing is cut off
    img_height = img_cv2.shape[0]
    focal_length = person_output["focal_length"]
    
    # Calculate required distance based on mesh size and image dimensions
    # Distance should be: mesh_size * focal_length / (image_size * safety_margin)
    safety_margin = 0.7  # Use 70% of available space to ensure full body fits
    required_z_distance = mesh_size * focal_length / (img_height * safety_margin)
    
    # Make sure camera is far enough back to see the whole body
    if fake_pred_cam_t[2] < required_z_distance:
        fake_pred_cam_t[2] = required_z_distance
    
    # Render front view
    renderer = Renderer(focal_length=focal_length, faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    # Render side view
    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side_right = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
            side_view_direction='right',
            render_floor=render_floor
        )
        * 255
    )
    img_mesh_side_left = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
            side_view_direction='left',
            render_floor=render_floor
        )
        * 255
    )
    img_mesh_top = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            top_view=True,
            side_view=False,
            render_floor=render_floor
        )
        * 255
    )
    
    img_mesh_back = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
            side_view_direction='left',
            rot_angle=180,
            render_floor=render_floor
        )
        * 255
    )

    # Arrange images in multiple rows to keep aspect ratio < 16:9
    all_images = [img_keypoints, img_mesh, img_mesh_side_right, img_mesh_side_left, img_mesh_back, img_mesh_top]
    
    # Get dimensions
    img_height, img_width = all_images[0].shape[:2]
    num_images = len(all_images)
    
    # Calculate total width if all in one row
    total_width_single_row = img_width * num_images
    target_ratio = 16.0 / 9.0
    
    # Determine optimal layout
    if total_width_single_row / img_height <= target_ratio:
        # Single row is fine
        cur_img = np.concatenate(all_images, axis=1)
    else:
        # Need multiple rows - find maximum images per row that keeps aspect ratio <= 16:9
        # We want: (img_width * imgs_per_row) / (img_height * num_rows) <= 16/9
        # where num_rows = ceil(num_images / imgs_per_row)
        best_imgs_per_row = 1  # Worst case fallback
        
        # Try from most images per row down to 1, take the first that works
        for imgs_per_row in range(num_images, 0, -1):
            num_rows = (num_images + imgs_per_row - 1) // imgs_per_row  # Ceiling division
            row_width = img_width * imgs_per_row
            total_height = img_height * num_rows
            aspect_ratio = row_width / total_height
            
            if aspect_ratio <= target_ratio:
                best_imgs_per_row = imgs_per_row
                break
        
        # Arrange images in rows
        rows = []
        for i in range(0, num_images, best_imgs_per_row):
            row_images = all_images[i:i + best_imgs_per_row]
            
            # Pad last row if needed to match width
            if len(row_images) < best_imgs_per_row:
                # Create white padding images
                padding_needed = best_imgs_per_row - len(row_images)
                for _ in range(padding_needed):
                    row_images.append(np.ones_like(white_img) * 255)
            
            row = np.concatenate(row_images, axis=1)
            rows.append(row)
        
        cur_img = np.concatenate(rows, axis=0)

    return cur_img
