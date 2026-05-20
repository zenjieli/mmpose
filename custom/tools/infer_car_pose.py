"""Run car keypoint inference on a single image with a given bounding box.

Usage:
    python custom/tools/infer_car_pose.py \
        --img path/to/image.jpg \
        --bbox xmin ymin xmax ymax \
        [--checkpoint work_dirs/rtmpose-t_pascal3d_car/best_coco_AP_epoch_380.pth] \
        [--out output.png]
"""

import argparse
import os

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def infer(img_path, bbox, checkpoint, config, out_path=None, show=False):
    import torch
    from mmpose.apis import init_model
    from mmpose.datasets.transforms import (
        GetBBoxCenterScale,
        LoadImage,
        PackPoseInputs,
        TopdownAffine,
    )
    from mmengine.dataset import default_collate

    model = init_model(config, checkpoint, device='cuda:0')
    model.eval()
    model.test_cfg = dict(flip_test=False)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {img_path}')

    xmin, ymin, xmax, ymax = bbox
    bbox_xywh = [xmin, ymin, xmax - xmin, ymax - ymin]

    data_info = dict(
        img_path=img_path,
        img=img.copy(),
        bbox=np.array(bbox_xywh, dtype=np.float32).reshape(1, 4),
        bbox_score=np.array([1.0], dtype=np.float32).reshape(1, 1),
        keypoints=np.zeros((1, 4, 2), dtype=np.float32),
        keypoints_visible=np.zeros((1, 4), dtype=np.float32),
    )

    data = LoadImage(backend_args=dict(backend='local'))(data_info)
    data = GetBBoxCenterScale()(data)
    data = TopdownAffine(input_size=(192, 256))(data)
    data = PackPoseInputs()(data)
    data_batch = default_collate([data])

    with torch.no_grad():
        results = model.test_step(data_batch)

    keypoints = np.array(results[0].pred_instances.keypoints)[0]

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
    infer(args.img, args.bbox, args.checkpoint, args.config, args.out, args.show)


if __name__ == '__main__':
    main()
