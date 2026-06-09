"""Standalone car-keypoint inference via ONNX Runtime.

Zero mmpose / mmdet / mmcv dependencies at runtime.
Requirements: onnxruntime, opencv-python, numpy  (all pip-installable).

The preprocessing and postprocessing exactly mirror the mmpose pipeline used
during training and in infer_car_pose.py so results are numerically equivalent.

Preprocessing pipeline
----------------------
1. Crop the image around the detection bbox with 30 % margin (same as
   infer_car_pose.py) to match the Pascal3D+ training distribution.
2. Pass the *tight detection bbox* (adjusted to crop coordinates) so that
   GetBBoxCenterScale's 1.25× padding extends into real image context from
   the margin crop — matching the training schema exactly.
3. Compute bbox center / scale (with 1.25× padding, matching GetBBoxCenterScale).
4. Fix aspect ratio to 192 / 256, compute affine matrix, warp to 192×256.
5. BGR → RGB, subtract ImageNet mean, divide by std → float32 NCHW blob.

Postprocessing pipeline
-----------------------
1. argmax over SimCC x / y distributions → keypoint indices in model space.
2. Divide by simcc_split_ratio (2.0) → pixel coordinates in 192×256 input.
3. Inverse linear map back to crop space.
4. Add crop origin → final coordinates in original image frame.

Usage
-----
    python custom/tools/infer_car_pose_onnx.py \\
        --img   path/to/image.jpg \\
        --bbox  XMIN YMIN XMAX YMAX \\
        [--model work_dirs/car_pose.onnx] \\
        [--device cpu|cuda] \\
        [--out   output.png] \\
        [--show]

To export the ONNX model first (requires the mmpose venv):
    python custom/tools/export_car_pose_onnx.py
"""

import argparse
import math

import cv2
import numpy as np
import onnxruntime as ort

# ── constants (must match the training config) ────────────────────────────────
_INPUT_W = 192
_INPUT_H = 256
_SIMCC_SPLIT_RATIO = 2.0
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
_BBOX_PADDING = 1.25   # GetBBoxCenterScale default
_BBOX_MARGIN  = 0.30   # extra crop margin (mirrors infer_car_pose.py)

# ── geometry helpers (ported from projects/rtmpose/examples/onnxruntime/main.py)

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = math.sin(angle_rad), math.cos(angle_rad)
    return np.array([cs * pt[0] - sn * pt[1],
                     sn * pt[0] + cs * pt[1]], dtype=np.float32)


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return (b + np.array([-direction[1], direction[0]], dtype=np.float32))


def _get_warp_matrix(center: np.ndarray,
                     scale: np.ndarray,
                     rot_deg: float,
                     output_size) -> np.ndarray:
    """Affine matrix mapping bbox region → output_size (src→dst, rot=0)."""
    rot_rad = math.radians(rot_deg)
    src_w   = scale[0]
    dst_w, dst_h = output_size

    src_dir = _rotate_point(np.array([src_w * -0.5, 0.], dtype=np.float32),
                            rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    return cv2.getAffineTransform(src, dst)


def _bbox_xyxy2cs(bbox_xyxy, padding: float):
    """Convert [x1,y1,x2,y2] to (center, scale) with padding."""
    x1, y1, x2, y2 = bbox_xyxy
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale  = np.array([(x2 - x1) * padding,
                       (y2 - y1) * padding], dtype=np.float32)
    return center, scale


def _fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = scale
    if w > h * aspect_ratio:
        return np.array([w, w / aspect_ratio], dtype=np.float32)
    return np.array([h * aspect_ratio, h], dtype=np.float32)


# ── preprocessing / postprocessing ───────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray, bbox_xyxy):
    """Prepare a 192×256 ONNX input blob from a BGR image (the margin crop).

    Args:
        img_bgr  : the margin-cropped BGR image.
        bbox_xyxy: tight detection bbox [x1, y1, x2, y2] in crop coordinates.

    Returns:
        blob   : float32 [1, 3, 256, 192] ready for the ONNX session
        center : (2,) bbox center used later for coordinate remapping
        scale  : (2,) aspect-ratio-corrected scale for coordinate remapping
    """
    center, scale = _bbox_xyxy2cs(bbox_xyxy, padding=_BBOX_PADDING)

    aspect_ratio = _INPUT_W / _INPUT_H
    scale = _fix_aspect_ratio(scale, aspect_ratio)

    warp_mat   = _get_warp_matrix(center, scale, 0.0, (_INPUT_W, _INPUT_H))
    img_warped = cv2.warpAffine(img_bgr, warp_mat, (_INPUT_W, _INPUT_H),
                                flags=cv2.INTER_LINEAR)

    # BGR → RGB, float32 normalise, NCHW
    img_rgb  = img_warped[:, :, ::-1].astype(np.float32)
    img_norm = (img_rgb - _MEAN) / _STD
    blob     = img_norm.transpose(2, 0, 1)[np.newaxis]   # [1, 3, H, W]

    return blob, center, scale


def _decode_simcc(simcc_x: np.ndarray,
                  simcc_y: np.ndarray,
                  center:  np.ndarray,
                  scale:   np.ndarray):
    """Decode SimCC outputs to keypoints in the space of the input image.

    Args:
        simcc_x : float32 [1, K, W*ratio]
        simcc_y : float32 [1, K, H*ratio]
        center  : (2,) bbox center
        scale   : (2,) aspect-ratio-corrected scale

    Returns:
        keypoints : (K, 2) float32
        scores    : (K,)   float32
    """
    N, K, _ = simcc_x.shape
    sx = simcc_x.reshape(N * K, -1)
    sy = simcc_y.reshape(N * K, -1)

    x_idx = np.argmax(sx, axis=1)
    y_idx = np.argmax(sy, axis=1)
    x_val = sx[np.arange(N * K), x_idx]
    y_val = sy[np.arange(N * K), y_idx]

    # keypoints in model-input space (0..W) × (0..H)
    kps = np.stack([x_idx, y_idx], axis=-1).astype(np.float32)
    kps /= _SIMCC_SPLIT_RATIO   # → pixel coords in 192 × 256 input

    # score = min(x_confidence, y_confidence); zero out invalid predictions
    scores = np.minimum(x_val, y_val).reshape(K).astype(np.float32)
    scores[scores <= 0.0] = 0.0

    kps = kps.reshape(K, 2)

    # inverse linear map: model input space → image space
    # kp_orig = kp_model / model_size * scale + center - scale / 2
    model_size = np.array([_INPUT_W, _INPUT_H], dtype=np.float32)
    kps = kps / model_size * scale + center - scale * 0.5

    return kps, scores


# ── crop helper ───────────────────────────────────────────────────────────────

def _crop_with_margin(img: np.ndarray, bbox_xyxy):
    """Expand bbox by _BBOX_MARGIN, clamp to image, return crop + adjusted bbox + origin."""
    xmin, ymin, xmax, ymax = bbox_xyxy
    bw, bh = xmax - xmin, ymax - ymin
    ih, iw = img.shape[:2]
    cx1 = max(0, int(xmin - bw * _BBOX_MARGIN))
    cy1 = max(0, int(ymin - bh * _BBOX_MARGIN))
    cx2 = min(iw, int(xmax + bw * _BBOX_MARGIN))
    cy2 = min(ih, int(ymax + bh * _BBOX_MARGIN))
    adjusted = [xmin - cx1, ymin - cy1, xmax - cx1, ymax - cy1]
    return img[cy1:cy2, cx1:cx2].copy(), adjusted, (cx1, cy1)


# ── public API ────────────────────────────────────────────────────────────────

def load_model(onnx_path: str, device: str = 'cpu') -> ort.InferenceSession:
    """Load the ONNX model into an ONNXRuntime session."""
    providers = (['CUDAExecutionProvider'] if device.startswith('cuda')
                 else ['CPUExecutionProvider'])
    return ort.InferenceSession(onnx_path, providers=providers)


def infer(img, bbox, sess: ort.InferenceSession,
          out_path: str = None, show: bool = False) -> np.ndarray:
    """Run car-keypoint inference on a single image + bounding box.

    Args:
        img      : BGR ndarray or path to an image file.
        bbox     : [xmin, ymin, xmax, ymax] in original image coordinates.
        sess     : ONNXRuntime InferenceSession (from load_model).
        out_path : optional path to write the visualisation image.
        show     : whether to display the result in a window.

    Returns:
        keypoints : (4, 2) float32 ndarray in original image coordinates,
                    ordered [RF, LF, LB, RB].
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(f'Cannot read {img!r}')

    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    crop, adjusted, (ox, oy) = _crop_with_margin(img, [xmin, ymin, xmax, ymax])

    blob, center, scale = _preprocess(crop, adjusted)

    input_name = sess.get_inputs()[0].name
    simcc_x, simcc_y = sess.run(['simcc_x', 'simcc_y'], {input_name: blob})

    keypoints, scores = _decode_simcc(simcc_x, simcc_y, center, scale)

    # map from crop space back to original image space
    keypoints[:, 0] += ox
    keypoints[:, 1] += oy

    labels = ['RF', 'LF', 'LB', 'RB']
    for label, kp in zip(labels, keypoints):
        print(f'{label}: ({kp[0]:.1f}, {kp[1]:.1f})')

    if out_path or show:
        vis    = img.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
        kp_int = keypoints.astype(int)
        for a, b in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            cv2.line(vis, tuple(kp_int[a]), tuple(kp_int[b]),
                     (0, 255, 255), 2)
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Standalone car-keypoint inference (ONNX, no mmpose)')
    parser.add_argument('--img',   required=True, help='Path to input image')
    parser.add_argument('--bbox',  required=True, nargs=4, type=int,
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                        help='Detection bounding box')
    parser.add_argument('--model', default='work_dirs/car_pose.onnx',
                        help='ONNX model file (from export_car_pose_onnx.py)')
    parser.add_argument('--device', default='cpu',
                        help='cpu or cuda (requires onnxruntime-gpu)')
    parser.add_argument('--out',   default=None,
                        help='Path to save visualisation image')
    parser.add_argument('--show',  action='store_true',
                        help='Display result in a window')
    args = parser.parse_args()

    sess = load_model(args.model, args.device)
    infer(args.img, args.bbox, sess, args.out, args.show)


if __name__ == '__main__':
    main()
