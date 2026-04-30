"""Convert COCO car segmentation masks to a 4-keypoint pose dataset.

Keypoints are the extreme points of the car mask contour:
  0: left   — point on the mask with the smallest x
  1: bottom — point on the mask with the largest  y
  2: right  — point on the mask with the largest  x
  3: top    — point on the mask with the smallest y (highest in image)

These points lie on the actual car surface and vary with car shape/viewpoint,
making them meaningful learning targets for a pose model.

Crowd annotations (iscrowd=1) are skipped because their masks are RLE-encoded
and extreme points are less meaningful for occluded instances.

Usage:
    python custom/tools/coco_car_bbox_to_keypoints.py \\
        --ann data/coco/annotations/instances_train2017.json \\
        --out data/coco/car_annotations/car_kps_train2017.json

    python custom/tools/coco_car_bbox_to_keypoints.py \\
        --ann data/coco/annotations/instances_val2017.json \\
        --out data/coco/car_annotations/car_kps_val2017.json
"""

import argparse
import json
import numpy as np


KEYPOINT_NAMES = ['left', 'bottom', 'right', 'top']

# Skeleton edges (0-indexed): connect adjacent extreme points
SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]

CAR_CATEGORY_NAME = 'car'


def seg_to_points(segmentation):
    """Extract all (x, y) contour points from a COCO polygon segmentation.

    Args:
        segmentation: list of polygons, each a flat [x0,y0,x1,y1,...] list.

    Returns:
        np.ndarray of shape (N, 2) with all contour points, or None if empty.
    """
    pts = []
    for poly in segmentation:
        coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts.append(coords)
    if not pts:
        return None
    return np.concatenate(pts, axis=0)


def extreme_keypoints(points):
    """Return flat keypoint list for the 4 extreme contour points.

    Order: left, bottom, right, top.
    Returns [x0,y0,v0, x1,y1,v1, x2,y2,v2, x3,y3,v3] with v=2.
    """
    left   = points[np.argmin(points[:, 0])]   # smallest x
    bottom = points[np.argmax(points[:, 1])]   # largest  y
    right  = points[np.argmax(points[:, 0])]   # largest  x
    top    = points[np.argmin(points[:, 1])]   # smallest y

    kps = []
    for pt in [left, bottom, right, top]:
        kps.extend([float(pt[0]), float(pt[1]), 2])
    return kps


def convert(ann_path, out_path, min_area=3200, max_area_ratio=0.4):
    """Convert car annotations.

    Args:
        ann_path: input COCO instances JSON.
        out_path: output keypoint JSON.
        min_area: minimum bbox pixel area (default 3200 ≈ 56×56).
        max_area_ratio: max ratio of bbox area to image area (default 0.4);
            filters out interior/dashboard shots.
    """
    with open(ann_path) as f:
        coco = json.load(f)

    # Find the car category id
    car_cat_id = None
    for cat in coco['categories']:
        if cat['name'] == CAR_CATEGORY_NAME:
            car_cat_id = cat['id']
            break
    if car_cat_id is None:
        raise ValueError(f'Category "{CAR_CATEGORY_NAME}" not found in {ann_path}')

    print(f'Found car category id: {car_cat_id}')

    # Build image-id → image-info lookup for size filtering
    img_info = {img['id']: img for img in coco['images']}

    # Filter to non-crowd car annotations that have polygon segmentations
    car_anns = [
        a for a in coco['annotations']
        if a['category_id'] == car_cat_id
        and a.get('iscrowd', 0) == 0
        and isinstance(a.get('segmentation'), list)
        and len(a['segmentation']) > 0
    ]
    print(f'Total usable car annotations: {len(car_anns)}')

    skipped = 0
    skipped_small = 0
    skipped_large = 0
    new_anns = []
    for ann in car_anns:
        bw, bh = ann['bbox'][2], ann['bbox'][3]
        bbox_area = bw * bh

        # Skip tiny cars
        if bbox_area < min_area:
            skipped_small += 1
            continue

        # Skip cars that dominate the image (interior/dashboard shots)
        iw = img_info[ann['image_id']]['width']
        ih = img_info[ann['image_id']]['height']
        if bbox_area / (iw * ih) > max_area_ratio:
            skipped_large += 1
            continue

        points = seg_to_points(ann['segmentation'])
        if points is None or len(points) < 4:
            skipped += 1
            continue

        new_ann = {
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': 1,  # remapped to 1 in the new dataset
            'bbox': ann['bbox'],
            'area': ann['area'],
            'iscrowd': 0,
            'keypoints': extreme_keypoints(points),
            'num_keypoints': 4,
        }
        new_anns.append(new_ann)

    if skipped_small:
        print(f'Skipped {skipped_small} tiny cars (bbox area < {min_area})')
    if skipped_large:
        print(f'Skipped {skipped_large} oversized cars (bbox/image area > {max_area_ratio})')
    if skipped:
        print(f'Skipped {skipped} annotations with too few contour points')

    # Keep only images that have at least one annotation
    used_image_ids = {a['image_id'] for a in new_anns}
    images = [img for img in coco['images'] if img['id'] in used_image_ids]
    print(f'Images with cars: {len(images)}')
    print(f'Final annotations: {len(new_anns)}')

    new_category = {
        'id': 1,
        'name': 'car',
        'supercategory': 'vehicle',
        'keypoints': KEYPOINT_NAMES,
        'skeleton': SKELETON,
    }

    output = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': images,
        'annotations': new_anns,
        'categories': [new_category],
    }

    with open(out_path, 'w') as f:
        json.dump(output, f)
    print(f'Saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO car masks to 4-extreme-point keypoint annotations')
    parser.add_argument('--ann', required=True,
                        help='Input COCO instances annotation JSON')
    parser.add_argument('--out', required=True,
                        help='Output keypoint annotation JSON')
    parser.add_argument('--min-area', type=float, default=3200,
                        help='Min bbox pixel area (default: 3200)')
    parser.add_argument('--max-area-ratio', type=float, default=0.4,
                        help='Max bbox/image area ratio (default: 0.4)')
    args = parser.parse_args()
    convert(args.ann, args.out,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio)


if __name__ == '__main__':
    main()
