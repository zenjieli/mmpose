"""Convert Pascal3D+ car annotations to COCO-style 4-keypoint pose dataset.

Keypoints (from Pascal3D+ anchors):
  0: left_front_light
  1: right_front_light
  2: left_back_trunk
  3: right_back_trunk

Visible keypoints (status=1) use their annotated 2D locations.
Occluded keypoints are projected from the CAD model using the annotated
viewpoint parameters (azimuth, elevation, distance, focal, theta, etc.).

Usage:
    python custom/tools/pascal3d_car_to_coco.py \
        --cad data/Pascal3D+/CAD/car.mat \
        --ann-dir data/Pascal3D+/Annotations/car_pascal \
        --img-dir data/Pascal3D+/Images/car_pascal \
        --out data/Pascal3D+_car_pose/car_keypoints.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pascal3d.explore_car_annotations import parse_annotation

KEYPOINT_NAMES = [
    "right_front_light",
    "left_front_light",
    "left_back_trunk",
    "right_back_trunk",
]
KEYPOINT_LABELS = ["RF", "LF", "LB", "RB"]
SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]

# Pascal3D+ has front lights swapped (left_front_light is actually the
# right front light from the car's perspective). This mapping corrects it.
PASCAL3D_ANCHOR = {
    "right_front_light": "left_front_light",
    "left_front_light": "right_front_light",
    "left_back_trunk": "left_back_trunk",
    "right_back_trunk": "right_back_trunk",
}

COCO_CATEGORY = {
    "id": 1,
    "name": "car",
    "supercategory": "vehicle",
    "keypoints": KEYPOINT_LABELS,
    "skeleton": SKELETON,
}


def load_cad_models(cad_path):
    """Load CAD models and return list of dicts with 3D keypoint positions."""
    import scipy.io as sio

    cad = sio.loadmat(cad_path, squeeze_me=True)
    models = []
    for i in range(cad["car"].shape[0]):
        c = cad["car"][i]
        model = {}
        for name in KEYPOINT_NAMES:
            model[name] = c[PASCAL3D_ANCHOR[name]].astype(float)
        models.append(model)
    return models


def project_3d(points_3d, azimuth, elevation, distance, focal, px, py, theta, viewport):
    """Project 3D CAD points to 2D image coordinates.

    Port of Pascal3D+ Annotation_tool/project_3d.m
    """
    a = np.radians(azimuth)
    e = np.radians(elevation)
    d = distance
    f = focal

    # Camera center
    C = np.array([d * np.cos(e) * np.sin(a),
                   -d * np.cos(e) * np.cos(a),
                   d * np.sin(e)])

    # Rotate coordinate system by theta == rotate model by -theta
    a = -a
    e = -(np.pi / 2 - e)

    Rz = np.array([[np.cos(a), -np.sin(a), 0],
                    [np.sin(a),  np.cos(a), 0],
                    [0,          0,          1]])
    Rx = np.array([[1,  0,          0],
                    [0,  np.cos(e), -np.sin(e)],
                    [0,  np.sin(e),  np.cos(e)]])
    R = Rx @ Rz

    # Projection matrix
    M = viewport
    K = np.array([[M * f, 0, 0],
                   [0, M * f, 0],
                   [0, 0, -1]])
    P = K @ np.hstack([R, -R @ C.reshape(3, 1)])

    # Project
    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    x = P @ pts_h.T
    x[0, :] /= x[2, :]
    x[1, :] /= x[2, :]
    x = x[:2, :]

    # 2D rotation
    R2d = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    x = (R2d @ x).T

    # Flip y, add principal point
    x[:, 1] *= -1
    x[:, 0] += px
    x[:, 1] += py

    return x


def convert(cad_path, ann_dir, img_dir, out_path, min_bbox_area=32 * 32):
    import glob

    cad_models = load_cad_models(cad_path)
    mat_files = sorted(glob.glob(os.path.join(ann_dir, "*.mat")))
    print(f"Found {len(mat_files)} annotation files")

    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    skipped_no_viewpoint = 0
    projected_count = 0

    for mat_path in mat_files:
        cars = parse_annotation(mat_path)
        if not cars:
            continue

        filename = cars[0]["image"]
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            continue

        img_id += 1
        imgsize = cars[0]["imgsize"]
        images.append({
            "id": img_id,
            "file_name": f"images/{filename}",
            "width": imgsize[0],
            "height": imgsize[1],
        })

        for car in cars:
            bbox = car["bbox"]
            if bbox is None:
                continue
            xmin, ymin, xmax, ymax = bbox
            bw = xmax - xmin
            bh = ymax - ymin
            if bw * bh < min_bbox_area:
                continue

            vp = car.get("viewpoint", {})
            cad_idx = car.get("cad_index", 0)
            if isinstance(cad_idx, str):
                continue

            # Need viewpoint + valid cad_index for projection
            if not vp or vp.get("azimuth") is None or vp.get("distance", 0) == 0:
                skipped_no_viewpoint += 1
                continue
            if cad_idx < 0 or cad_idx >= len(cad_models):
                skipped_no_viewpoint += 1
                continue

            cad_model = cad_models[cad_idx]

            # Gather 3D points for all 4 keypoints
            pts3d = np.array([cad_model[name] for name in KEYPOINT_NAMES])
            pts2d = project_3d(
                pts3d,
                azimuth=vp["azimuth"],
                elevation=vp["elevation"],
                distance=vp["distance"],
                focal=vp.get("focal", 1.0),
                px=vp.get("px", imgsize[0] / 2),
                py=vp.get("py", imgsize[1] / 2),
                theta=np.radians(vp.get("theta", 0.0)),
                viewport=3000,
            )

            # Build keypoints: use annotated location if visible, else use projected
            kps = []
            n_vis = 0
            for i, kp_name in enumerate(KEYPOINT_NAMES):
                anc = car["anchors"].get(PASCAL3D_ANCHOR[kp_name], {})
                status = anc.get("status", -1)
                loc = anc.get("location")

                if status == 1 and loc is not None:
                    kps.extend([loc[0], loc[1], 2])
                    n_vis += 1
                else:
                    kps.extend([float(pts2d[i, 0]), float(pts2d[i, 1]), 1])
                    projected_count += 1

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "keypoints": kps,
                "num_keypoints": n_vis,
            })

    output = {
        "info": {"description": "Pascal3D+ Car 4-Keypoint Dataset. "
                  "Keypoints: RF=Right Front Light, LF=Left Front Light, "
                  "LB=Left Back Trunk, RB=Right Back Trunk"},
        "images": images,
        "annotations": annotations,
        "categories": [COCO_CATEGORY],
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Projected keypoints: {projected_count}")
    print(f"Skipped (no viewpoint): {skipped_no_viewpoint}")

    vis_counts = {}
    for a in annotations:
        v = a["num_keypoints"]
        vis_counts[v] = vis_counts.get(v, 0) + 1
    print(f"Visibility breakdown: {vis_counts}")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pascal3D+ car annotations to COCO keypoint format")
    parser.add_argument("--cad", default="data/Pascal3D+/CAD/car.mat",
                        help="Path to car CAD models .mat file")
    parser.add_argument("--ann-dir", default="data/Pascal3D+/Annotations/car_pascal",
                        help="Directory with .mat annotation files")
    parser.add_argument("--img-dir", default="data/Pascal3D+/Images/car_pascal",
                        help="Directory with .jpg image files")
    parser.add_argument("--out", default="data/Pascal3D+_car_pose/car_keypoints.json",
                        help="Output COCO-format JSON path (inside dataset root)")
    parser.add_argument("--min-bbox-area", type=float, default=1024,
                        help="Minimum bbox area in pixels (default: 1024)")
    args = parser.parse_args()
    convert(args.cad, args.ann_dir, args.img_dir, args.out,
            min_bbox_area=args.min_bbox_area)


if __name__ == "__main__":
    main()
