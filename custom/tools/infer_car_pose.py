"""Run car keypoint inference on a single image with a given bounding box.

Usage:
    python custom/tools/infer_car_pose.py \
        --img path/to/image.jpg \
        --bbox xmin ymin xmax ymax \
        [--checkpoint work_dirs/rtmpose-t_pascal3d_car/best_coco_AP_epoch_380.pth] \
        [--out output.png]

Inference pipeline
------------------
The model was trained on Pascal3D+ images (~500px) where cars fill a large
fraction of the frame.  On high-resolution video frames (e.g. 1920x1080),
passing the tight detection bbox directly produces poor results because the
car occupies only a tiny fraction of the warped 192x256 input.

To match the training distribution we:

  1. Crop the image around the detection bbox with 30% margin on each side.
     This produces a ~500px crop where the car fills most of the area,
     similar to Pascal3D+ training images, and provides real image context
     around the car.
  2. Pass the *tight detection bbox* (adjusted to crop coordinates) to the
     model.  The model's GetBBoxCenterScale adds 1.25× padding which extends
     into the real image context from the margin crop — exactly matching the
     training schema where the pipeline receives a tight bbox and adds 1.25×
     padding into surrounding image content.

Predicted keypoints are offset by the crop origin to return coordinates in the
original image frame.
"""

import argparse
import os

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fraction of bbox width/height to add as margin on each side.
BBOX_MARGIN = 0.3


def _crop_with_margin(img, bbox_xyxy):
    """Expand bbox by BBOX_MARGIN, clamp to image bounds, return crop + origin."""
    xmin, ymin, xmax, ymax = bbox_xyxy
    w, h = xmax - xmin, ymax - ymin
    ih, iw = img.shape[:2]

    cx1 = max(0, int(xmin - w * BBOX_MARGIN))
    cy1 = max(0, int(ymin - h * BBOX_MARGIN))
    cx2 = min(iw, int(xmax + w * BBOX_MARGIN))
    cy2 = min(ih, int(ymax + h * BBOX_MARGIN))

    crop = img[cy1:cy2, cx1:cx2].copy()
    adjusted = [xmin - cx1, ymin - cy1, xmax - cx1, ymax - cy1]
    return crop, adjusted, (cx1, cy1)


def _run_model(model, img_bgr, bbox_xyxy):
    """Run MMPose model on a single image + bbox. Returns (4, 2) keypoints."""
    from mmengine.dataset import default_collate
    import torch

    xmin, ymin, xmax, ymax = bbox_xyxy
    bbox_xywh = [xmin, ymin, xmax - xmin, ymax - ymin]

    from mmpose.datasets.transforms import (
        GetBBoxCenterScale,
        LoadImage,
        PackPoseInputs,
        TopdownAffine,
    )

    data_info = dict(
        img_path='',
        img=img_bgr.copy(),
        bbox=np.array(bbox_xywh, dtype=np.float32).reshape(1, 4),
        bbox_score=np.array([1.0], dtype=np.float32).reshape(1, 1),
        keypoints=np.zeros((1, 4, 2), dtype=np.float32),
        keypoints_visible=np.zeros((1, 4), dtype=np.float32),
    )

    data = LoadImage(backend_args=dict(backend='local'))(data_info)
    data = GetBBoxCenterScale()(data)
    data = TopdownAffine(input_size=(192, 256))(data)
    data = PackPoseInputs()(data)

    with torch.no_grad():
        results = model.test_step(default_collate([data]))

    return np.array(results[0].pred_instances.keypoints)[0]


def infer(img, bbox, model, out_path=None, show=False):
    """Infer car keypoints on an image.

    Args:
        img: BGR ndarray or path to image file.
        bbox: [xmin, ymin, xmax, ymax] in original image coordinates.
        model: loaded MMPose model.
        out_path: optional path to save visualization.
        show: whether to display the result.

    Returns:
        keypoints: (4, 2) ndarray in original image coordinates.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(f'Cannot read {img}')

    xmin, ymin, xmax, ymax = [int(v) for v in bbox]

    # Crop around bbox with margin to provide context, then pass the tight
    # bbox (in crop coordinates) so GetBBoxCenterScale's 1.25× padding extends
    # into real image content from the margin — matching training exactly.
    crop, adjusted, (ox, oy) = _crop_with_margin(img, [xmin, ymin, xmax, ymax])
    keypoints = _run_model(model, crop, adjusted)
    keypoints[:, 0] += ox
    keypoints[:, 1] += oy

    labels = ['RF', 'LF', 'LB', 'RB']
    for label, kp in zip(labels, keypoints):
        print(f'{label}: ({kp[0]:.1f}, {kp[1]:.1f})')

    if out_path or show:
        vis = img.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
        kp_int = keypoints.astype(int)
        for a, b in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            cv2.line(vis, tuple(kp_int[a]), tuple(kp_int[b]), (0, 255, 255), 2)
        for i, (label, kp) in enumerate(zip(labels, kp_int)):
            cv2.circle(vis, tuple(kp), 5, colors[i], -1)
            cv2.putText(vis, label, (kp[0] + 6, kp[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (128, 128, 128), 1)

        if out_path:
            cv2.imwrite(out_path, vis)
            print(f'Saved to {out_path}')
        if show:
            cv2.imshow('car pose', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return keypoints


def main():
    parser = argparse.ArgumentParser(description='Car keypoint inference')
    parser.add_argument('--img', required=True, help='Path to input image')
    parser.add_argument('--bbox', required=True, nargs=4, type=int,
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                        help='Bounding box: xmin ymin xmax ymax')
    parser.add_argument('--checkpoint',
                        default='work_dirs/rtmpose-t_pascal3d_car/'
                                'best_coco_AP_epoch_380.pth',
                        help='Model checkpoint')
    parser.add_argument('--config',
                        default='custom/configs/'
                                'rtmpose-t_pascal3d_car-256x192.py',
                        help='Model config')
    parser.add_argument('--out', default=None, help='Output visualization path')
    parser.add_argument('--show', action='store_true', help='Show result window')
    args = parser.parse_args()

    from mmpose.apis import init_model
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    model.eval()
    model.test_cfg = dict(flip_test=False)

    infer(args.img, args.bbox, model, args.out, args.show)


if __name__ == '__main__':
    main()
