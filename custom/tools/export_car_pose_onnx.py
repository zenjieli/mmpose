"""Export the car-pose RTMPose-t model to ONNX.

The exported model is a self-contained neural network with no mmpose runtime
dependency.  All preprocessing (crop, affine warp, normalisation) and
postprocessing (SimCC argmax, coordinate rescaling) are done in plain
numpy/OpenCV by the companion standalone script infer_car_pose_onnx.py.

Exported interface
------------------
  input   : float32  [1, 3, 256, 192]  – RGB, normalised with ImageNet stats
  simcc_x : float32  [1, 4, 384]       – SimCC x-axis logits (split_ratio=2)
  simcc_y : float32  [1, 4, 512]       – SimCC y-axis logits (split_ratio=2)

Usage
-----
    # from the repo root, with the mmpose venv active:
    python custom/tools/export_car_pose_onnx.py \\
        [--checkpoint work_dirs/rtmpose-t_pascal3d_car/best_coco_AP_epoch_380.pth] \\
        [--config    custom/configs/rtmpose-t_pascal3d_car-256x192.py] \\
        [--out       work_dirs/car_pose.onnx] \\
        [--opset     17]
"""

import argparse

import torch
import torch.nn as nn


class _PoseWrapper(nn.Module):
    """Backbone + (optional neck) + head, stripped of the data preprocessor.

    The data preprocessor (bgr→rgb + ImageNet normalisation) is excluded so
    that the ONNX model accepts a plain normalised RGB float tensor.
    This keeps the model independent of mmcv/mmdet at inference time.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = getattr(model, 'neck', None)
        self.head = model.head

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: float32 [N, 3, H, W] – RGB, ImageNet-normalised.

        Returns:
            simcc_x: float32 [N, K, W*split_ratio]
            simcc_y: float32 [N, K, H*split_ratio]
        """
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        pred_x, pred_y = self.head.forward(feats)
        return pred_x, pred_y


def main():
    parser = argparse.ArgumentParser(
        description='Export car-pose RTMPose-t to ONNX')
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/rtmpose-t_pascal3d_car/best_coco_AP_epoch_380.pth',
        help='Path to .pth checkpoint')
    parser.add_argument(
        '--config',
        default='custom/configs/rtmpose-t_pascal3d_car-256x192.py',
        help='MMPose config file')
    parser.add_argument(
        '--out',
        default='work_dirs/car_pose.onnx',
        help='Output ONNX file path')
    parser.add_argument(
        '--opset', type=int, default=17,
        help='ONNX opset version (>=11 required; 17 recommended)')
    parser.add_argument(
        '--device', default='cpu',
        help='Device for loading the model (cpu or cuda:N)')
    args = parser.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    from mmpose.apis import init_model
    print(f'Loading model: {args.config}')
    print(f'Checkpoint:    {args.checkpoint}')
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()

    # ── wrap and export ───────────────────────────────────────────────────────
    wrapper = _PoseWrapper(model)
    wrapper.eval()

    # input: [N, 3, H, W] = [1, 3, 256, 192]
    dummy_input = torch.zeros(1, 3, 256, 192, device=args.device)

    print(f'Exporting to:  {args.out}  (opset {args.opset})')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            args.out,
            input_names=['input'],
            output_names=['simcc_x', 'simcc_y'],
            opset_version=args.opset,
            dynamic_axes={
                'input':   {0: 'batch'},
                'simcc_x': {0: 'batch'},
                'simcc_y': {0: 'batch'},
            },
        )

    print('Done.')
    print()
    print('Run standalone inference (no mmpose required) with:')
    print('  python custom/tools/infer_car_pose_onnx.py \\')
    print(f'      --model {args.out} \\')
    print('      --img path/to/image.jpg \\')
    print('      --bbox XMIN YMIN XMAX YMAX')


if __name__ == '__main__':
    main()
