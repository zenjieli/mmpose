"""Visualize each step of the car keypoint inference pipeline.

Saves intermediate images to visualize:
  1. Original image with tight bbox + GT keypoints
  2. 30% margin crop with tight bbox drawn
  3. Tight bbox + 1.25× padding region drawn on the margin crop
  4. Aspect-ratio-adjusted region drawn, then the warped 192×256 result
  5. Final 192×256 with predicted keypoints

Usage:
    python custom/tools/debug_pipeline_viz.py
"""

import cv2
import numpy as np
import json
import math

# ── constants ──────────────────────────────────────────────────────────────────
BBOX_MARGIN = 0.30
BBOX_PADDING = 1.25
INPUT_W, INPUT_H = 192, 256
ASPECT_RATIO = INPUT_W / INPUT_H  # 0.75

# Colors (BGR)
COLOR_TIGHT  = (0, 255, 0)      # green - tight bbox
COLOR_MARGIN = (0, 255, 255)    # yellow - margin crop
COLOR_PADDED = (255, 0, 0)      # blue - 1.25× padded
COLOR_ASPECT = (255, 0, 255)    # magenta - aspect-ratio adjusted
COLOR_KP     = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
KP_LABELS    = ['RF', 'LF', 'LB', 'RB']

OUT_DIR = 'work_dirs/pipeline_debug'


def draw_bbox(img, xyxy, color, label='', thickness=2):
    """Draw a bbox with optional label."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_keypoints(img, kps_xy, color_list=COLOR_KP, labels=KP_LABELS,
                   offset=(0, 0)):
    """Draw keypoints as colored circles with labels."""
    for i, (label, kp) in enumerate(zip(labels, kps_xy)):
        pt = (int(kp[0] + offset[0]), int(kp[1] + offset[1]))
        cv2.circle(img, pt, 4, color_list[i], -1)
        cv2.putText(img, label, (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_list[i], 1)
    # Draw quad edges
    kp_int = [(int(kp[0] + offset[0]), int(kp[1] + offset[1])) for kp in kps_xy]
    for a, b in [[0, 1], [1, 2], [2, 3], [3, 0]]:
        cv2.line(img, kp_int[a], kp_int[b], (0, 255, 255), 1)


def _rotate_point(pt, angle_rad):
    sn, cs = math.sin(angle_rad), math.cos(angle_rad)
    return np.array([cs * pt[0] - sn * pt[1],
                     sn * pt[0] + cs * pt[1]], dtype=np.float32)


def _get_3rd_point(a, b):
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _get_warp_matrix(center, scale, output_size):
    src_w = scale[0]
    dst_w, dst_h = output_size
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.], dtype=np.float32), 0.0)
    dst_dir = np.array([dst_w * -0.5, 0.], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    return cv2.getAffineTransform(src, dst)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load annotation
    with open('data/Pascal3D+_car_pose/car_keypoints.json') as f:
        data = json.load(f)
    ann = data['annotations'][0]
    img_info = data['images'][ann['image_id'] - 1]

    img_path = f"data/Pascal3D+_car_pose/{img_info['file_name']}"
    img = cv2.imread(img_path)
    assert img is not None, f'Cannot read {img_path}'
    ih, iw = img.shape[:2]
    print(f'Image: {img_path}  ({iw}×{ih})')

    # Tight bbox (convert xywh → xyxy)
    bx, by, bw, bh = ann['bbox']
    tight = [bx, by, bx + bw, by + bh]
    print(f'Tight bbox (xyxy): {tight}')

    # GT keypoints
    kp_flat = ann['keypoints']
    gt_kps = np.array([[kp_flat[i*3], kp_flat[i*3+1]]
                        for i in range(4)], dtype=np.float32)
    print(f'GT keypoints:\n{gt_kps}')

    # ── Step 1: Original image ──────────────────────────────────────────────
    vis1 = img.copy()
    draw_bbox(vis1, tight, COLOR_TIGHT, 'tight bbox')
    draw_keypoints(vis1, gt_kps)
    out1 = f'{OUT_DIR}/1_original.jpg'
    cv2.imwrite(out1, vis1)
    print(f'\nStep 1 → {out1}')

    # ── Step 2: 30% margin crop ─────────────────────────────────────────────
    cx1 = max(0, int(tight[0] - bw * BBOX_MARGIN))
    cy1 = max(0, int(tight[1] - bh * BBOX_MARGIN))
    cx2 = min(iw, int(tight[2] + bw * BBOX_MARGIN))
    cy2 = min(ih, int(tight[3] + bh * BBOX_MARGIN))
    margin_crop_xyxy = [cx1, cy1, cx2, cy2]

    crop = img[cy1:cy2, cx1:cx2].copy()
    # Tight bbox in crop coords
    adjusted = [tight[0] - cx1, tight[1] - cy1, tight[2] - cx1, tight[3] - cy1]
    print(f'\nMargin crop (xyxy): {margin_crop_xyxy}  → crop size: {crop.shape[1]}×{crop.shape[0]}')
    print(f'Adjusted tight bbox in crop: {adjusted}')

    vis2 = crop.copy()
    draw_bbox(vis2, [0, 0, crop.shape[1], crop.shape[0]], COLOR_MARGIN,
              'margin crop', 1)
    draw_bbox(vis2, adjusted, COLOR_TIGHT, 'tight bbox')
    draw_keypoints(vis2, gt_kps, offset=(-cx1, -cy1))
    out2 = f'{OUT_DIR}/2_margin_crop.jpg'
    cv2.imwrite(out2, vis2)
    print(f'Step 2 → {out2}')

    # ── Step 3: 1.25× padding on tight bbox ─────────────────────────────────
    ax1, ay1, ax2, ay2 = adjusted
    aw, ah = ax2 - ax1, ay2 - ay1
    cx, cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    pad_w, pad_h = aw * BBOX_PADDING, ah * BBOX_PADDING

    padded_xyxy = [cx - pad_w / 2, cy - pad_h / 2,
                   cx + pad_w / 2, cy + pad_h / 2]
    print(f'\n1.25× padded scale: ({pad_w:.1f}, {pad_h:.1f})')
    print(f'Padded region (xyxy): [{padded_xyxy[0]:.1f}, {padded_xyxy[1]:.1f}, '
          f'{padded_xyxy[2]:.1f}, {padded_xyxy[3]:.1f}]')

    vis3 = crop.copy()
    draw_bbox(vis3, [0, 0, crop.shape[1], crop.shape[0]], COLOR_MARGIN,
              'margin crop', 1)
    draw_bbox(vis3, adjusted, COLOR_TIGHT, 'tight bbox', 1)
    draw_bbox(vis3, padded_xyxy, COLOR_PADDED, '1.25x padded')
    draw_keypoints(vis3, gt_kps, offset=(-cx1, -cy1))
    out3 = f'{OUT_DIR}/3_padded.jpg'
    cv2.imwrite(out3, vis3)
    print(f'Step 3 → {out3}')

    # ── Step 4: Aspect ratio adjustment ─────────────────────────────────────
    scale = np.array([pad_w, pad_h], dtype=np.float32)
    w, h = scale
    if w > h * ASPECT_RATIO:
        ar_scale = np.array([w, w / ASPECT_RATIO], dtype=np.float32)
    else:
        ar_scale = np.array([h * ASPECT_RATIO, h], dtype=np.float32)

    ar_xyxy = [cx - ar_scale[0] / 2, cy - ar_scale[1] / 2,
               cx + ar_scale[0] / 2, cy + ar_scale[1] / 2]
    print(f'\nAspect-ratio adjusted scale: ({ar_scale[0]:.1f}, {ar_scale[1]:.1f})')
    print(f'AR region (xyxy): [{ar_xyxy[0]:.1f}, {ar_xyxy[1]:.1f}, '
          f'{ar_xyxy[2]:.1f}, {ar_xyxy[3]:.1f}]')
    print(f'AR of adjusted: {ar_scale[0]/ar_scale[1]:.3f} (target: {ASPECT_RATIO:.3f})')

    vis4 = crop.copy()
    draw_bbox(vis4, [0, 0, crop.shape[1], crop.shape[0]], COLOR_MARGIN,
              'margin crop', 1)
    draw_bbox(vis4, adjusted, COLOR_TIGHT, 'tight bbox', 1)
    draw_bbox(vis4, padded_xyxy, COLOR_PADDED, '1.25x padded', 1)
    draw_bbox(vis4, ar_xyxy, COLOR_ASPECT, 'aspect adjusted')
    draw_keypoints(vis4, gt_kps, offset=(-cx1, -cy1))

    # Also warp to 192×256 and show
    center = np.array([cx, cy], dtype=np.float32)
    warp_mat = _get_warp_matrix(center, ar_scale, (INPUT_W, INPUT_H))
    warped = cv2.warpAffine(crop, warp_mat, (INPUT_W, INPUT_H),
                            flags=cv2.INTER_LINEAR)

    # Draw all bboxes on warped too
    vis4_warped = warped.copy()
    cv2.putText(vis4_warped, 'Warped 192x256', (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out4 = f'{OUT_DIR}/4_aspect_ratio.jpg'
    cv2.imwrite(out4, vis4)
    out4w = f'{OUT_DIR}/4_aspect_ratio_warped.jpg'
    cv2.imwrite(out4w, vis4_warped)
    print(f'Step 4 → {out4}, {out4w}')

    # ── Step 5: Run model and draw keypoints on warped ──────────────────────
    # Load model and run inference
    from mmpose.apis import init_model
    from mmengine.dataset import default_collate
    import torch
    from mmpose.datasets.transforms import (
        GetBBoxCenterScale, LoadImage, PackPoseInputs, TopdownAffine,
    )

    config = 'custom/configs/rtmpose-t_pascal3d_car-256x192.py'
    checkpoint = 'work_dirs/rtmpose-t_pascal3d_car/best_coco_AP_epoch_380.pth'
    print(f'\nLoading model: {checkpoint}')
    model = init_model(config, checkpoint, device='cuda:0')
    model.eval()
    model.test_cfg = dict(flip_test=False)

    # Run through pipeline on the margin crop with tight bbox
    data_info = dict(
        img_path='',
        img=crop.copy(),
        bbox=np.array(adjusted, dtype=np.float32).reshape(1, 4),
        bbox_score=np.array([1.0], dtype=np.float32).reshape(1, 1),
        keypoints=np.zeros((1, 4, 2), dtype=np.float32),
        keypoints_visible=np.zeros((1, 4), dtype=np.float32),
    )

    data = LoadImage(backend_args=dict(backend='local'))(data_info)
    data = GetBBoxCenterScale()(data)
    data = TopdownAffine(input_size=(INPUT_W, INPUT_H))(data)
    data = PackPoseInputs()(data)

    with torch.no_grad():
        results = model.test_step(default_collate([data]))

    pred_kps_crop = np.array(results[0].pred_instances.keypoints)[0]  # in crop coords

    # Map to original image coords
    pred_kps_orig = pred_kps_crop.copy()
    pred_kps_orig[:, 0] += cx1
    pred_kps_orig[:, 1] += cy1

    print(f'\nPredicted keypoints (crop coords):\n{pred_kps_crop}')
    print(f'Predicted keypoints (orig coords):\n{pred_kps_orig}')
    print(f'GT keypoints (orig coords):\n{gt_kps}')

    # Draw on the warped 192×256 image
    vis5 = warped.copy()

    # Inverse-map GT keypoints to warped space for comparison
    # kp_warped = (kp_crop - center + scale/2) / scale * model_size
    model_size = np.array([INPUT_W, INPUT_H], dtype=np.float32)
    gt_kps_crop = gt_kps.copy()
    gt_kps_crop[:, 0] -= cx1
    gt_kps_crop[:, 1] -= cy1
    gt_warped = (gt_kps_crop - center + ar_scale / 2) / ar_scale * model_size
    pred_warped = (pred_kps_crop - center + ar_scale / 2) / ar_scale * model_size

    # Draw GT (thin, for reference)
    for i, (label, kp) in enumerate(zip(KP_LABELS, gt_warped)):
        pt = (int(kp[0]), int(kp[1]))
        cv2.circle(vis5, pt, 3, (128, 128, 128), 1)
        cv2.putText(vis5, f'GT_{label}', (pt[0] + 4, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (128, 128, 128), 1)

    # Draw predictions
    draw_keypoints(vis5, pred_warped)

    out5 = f'{OUT_DIR}/5_keypoints_warped.jpg'
    cv2.imwrite(out5, vis5)
    print(f'\nStep 5 → {out5}')

    # Also draw predictions on the original image
    vis5_orig = img.copy()
    draw_bbox(vis5_orig, tight, COLOR_TIGHT, 'tight bbox')
    draw_keypoints(vis5_orig, gt_kps, labels=[f'GT_{l}' for l in KP_LABELS])
    # Override label drawing for predictions
    for i, (label, kp) in enumerate(zip(KP_LABELS, pred_kps_orig)):
        pt = (int(kp[0]), int(kp[1]))
        cv2.circle(vis5_orig, pt, 5, COLOR_KP[i], -1)
        cv2.putText(vis5_orig, f'P_{label}', (pt[0] + 6, pt[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_KP[i], 1)
    out5_orig = f'{OUT_DIR}/5_keypoints_original.jpg'
    cv2.imwrite(out5_orig, vis5_orig)
    print(f'Step 5 (orig) → {out5_orig}')

    print(f'\nAll images saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
