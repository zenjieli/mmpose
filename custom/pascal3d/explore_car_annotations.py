"""Explore Pascal3D+ car annotations and extract anchor keypoints.

Pascal3D+ annotation format (.mat files):
  record.objects -> list of annotated objects per image
  Each object has:
    - class: e.g. 'car', 'person', 'bus'
    - bbox: [xmin, ymin, xmax, ymax]
    - anchors: 12 semantic keypoints per car (see below)
    - viewpoint: azimuth, elevation, distance, focal, theta
    - cad_index: which CAD model was used for annotation

12 anchor keypoints on each car:
  left_front_wheel, right_front_wheel,
  left_back_wheel, right_back_wheel,
  upper_left_windshield, upper_right_windshield,
  upper_left_rearwindow, upper_right_rearwindow,
  left_front_light, right_front_light,
  left_back_trunk, right_back_trunk

Each anchor has:
  - location: 2D pixel coords [x, y] (empty if not visible)
  - status code:
      1 = visible (location provided)
      2 = occluded/truncated (no location)
      3, 4 = other visibility issues (no location)
      5 = not visible from this viewpoint (no location)
      0 = rare/unknown

Key findings for car objects:
  - 1229 annotation files, 2364 car objects (many files contain mixed classes)
  - Front lights visible: ~29% of car objects
  - Rear trunk visible: ~31% of car objects
  - All 4 corners (front lights + rear trunk) simultaneously visible: 0%
  - At least 2 of 4 corners visible: ~43%
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _get_val(x):
    """Unwrap 0-dim numpy arrays."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def parse_anchors(obj):
    """Extract anchor keypoints from a single car object.

    Args:
        obj: numpy void/structured array for one object.

    Returns:
        dict mapping anchor name -> {'location': [x,y] or None, 'status': int}
        or None if anchors are missing.
    """
    anc = obj["anchors"]
    if anc.ndim != 0:
        return None
    inner = anc.item()
    if inner is None:
        return None

    # Single-object files: inner is a 0-dim structured array with named fields.
    # Multi-object files: inner is a tuple.
    if isinstance(inner, np.ndarray) and inner.dtype.names:
        names = inner.dtype.names
        vals = [_get_val(inner[n]) for n in names]
    elif isinstance(inner, tuple):
        names = anc.dtype.names
        vals = list(inner)
    else:
        return None

    result = {}
    for name, val in zip(names, vals):
        if isinstance(val, np.ndarray) and val.dtype.names:
            loc_raw = _get_val(val["location"])
            status = int(_get_val(val["status"]))
            if isinstance(loc_raw, np.ndarray) and loc_raw.size > 0:
                result[name] = {"location": loc_raw.astype(float).tolist(), "status": status}
            else:
                result[name] = {"location": None, "status": status}
        else:
            result[name] = {"location": None, "status": -1}
    return result


def parse_annotation(mat_path):
    """Parse a single Pascal3D+ .mat annotation file.

    Returns a list of dicts, one per car object:
      {
        "image": str,          # image filename
        "imgname": str,        # image path within dataset
        "imgsize": [w, h, c],
        "bbox": [xmin, ymin, xmax, ymax],
        "anchors": {name: {"location": [x,y]|None, "status": int}},
        "viewpoint": {"azimuth": float, "elevation": float, ...},
        "cad_index": int,
        "subtype": str,
      }
    """
    import scipy.io as sio

    data = sio.loadmat(str(mat_path), squeeze_me=True)
    rec = data["record"]
    obj = rec["objects"].item()

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            objs = [obj]
        else:
            objs = [obj[i] for i in range(len(obj))]
    else:
        return []

    imgname = str(_get_val(rec["imgname"]))
    imgsize = _get_val(rec["imgsize"])
    if isinstance(imgsize, np.ndarray):
        imgsize = imgsize.astype(int).tolist()

    cars = []
    for o in objs:
        cls = str(_get_val(o["class"]))
        if cls != "car":
            continue

        bbox_raw = _get_val(o["bbox"])
        bbox = bbox_raw.astype(int).tolist() if isinstance(bbox_raw, np.ndarray) else None

        anchors = parse_anchors(o)
        if anchors is None:
            continue

        vp_raw = _get_val(o["viewpoint"])
        viewpoint = {}
        vp_wanted = {"azimuth", "elevation", "distance", "focal", "px", "py", "theta", "viewport"}
        if isinstance(vp_raw, np.ndarray) and vp_raw.dtype.names:
            for fn in vp_wanted:
                if fn in vp_raw.dtype.names:
                    try:
                        viewpoint[fn] = float(_get_val(vp_raw[fn]))
                    except (ValueError, TypeError):
                        pass
        elif isinstance(vp_raw, tuple):
            vp_dtype = o["viewpoint"].dtype
            if vp_dtype.names:
                for fn, val in zip(vp_dtype.names, vp_raw):
                    if fn in vp_wanted:
                        try:
                            viewpoint[fn] = float(val)
                        except (ValueError, TypeError):
                            pass

        cad_index = _get_val(o["cad_index"])
        subtype = _get_val(o["subtype"])
        if isinstance(cad_index, np.ndarray):
            cad_index = int(cad_index)
        if isinstance(subtype, np.ndarray):
            subtype = str(subtype)

        cars.append({
            "image": Path(mat_path).stem + ".jpg",
            "imgname": imgname,
            "imgsize": imgsize,
            "bbox": bbox,
            "anchors": anchors,
            "viewpoint": viewpoint,
            "cad_index": cad_index,
            "subtype": str(subtype) if subtype else "",
        })
    return cars


def print_stats(annotation_dir):
    """Print dataset statistics for car annotations."""
    import glob

    anchor_names = [
        "left_front_wheel", "left_back_wheel", "right_front_wheel", "right_back_wheel",
        "upper_left_windshield", "upper_right_windshield",
        "upper_left_rearwindow", "upper_right_rearwindow",
        "left_front_light", "right_front_light",
        "left_back_trunk", "right_back_trunk",
    ]

    files = sorted(glob.glob(str(annotation_dir)))
    if not files:
        print(f"No .mat files found at {annotation_dir}")
        return

    n_objects = 0
    n_car_objects = 0
    status_counts = {name: {} for name in anchor_names}
    vis_both_front = 0
    vis_both_rear = 0
    vis_all_four = 0
    vis_at_least_2 = 0
    vis_at_least_1 = 0

    for f in files:
        cars = parse_annotation(f)
        for car in cars:
            n_car_objects += 1
            for an in anchor_names:
                s = car["anchors"].get(an, {}).get("status", -1)
                status_counts[an][s] = status_counts[an].get(s, 0) + 1

            fl = car["anchors"].get("left_front_light", {}).get("status", -1)
            fr = car["anchors"].get("right_front_light", {}).get("status", -1)
            bl = car["anchors"].get("left_back_trunk", {}).get("status", -1)
            br = car["anchors"].get("right_back_trunk", {}).get("status", -1)
            n_vis = sum(1 for s in [fl, fr, bl, br] if s == 1)
            if n_vis >= 1:
                vis_at_least_1 += 1
            if n_vis >= 2:
                vis_at_least_2 += 1
            if fl == 1 and fr == 1:
                vis_both_front += 1
            if bl == 1 and br == 1:
                vis_both_rear += 1
            if n_vis == 4:
                vis_all_four += 1

    n = n_car_objects
    print(f"Files: {len(files)} | Car objects: {n}")
    print()
    print("Per-anchor visibility (status=1):")
    for an in anchor_names:
        total = sum(status_counts[an].values())
        vis = status_counts[an].get(1, 0)
        print(f"  {an:30s}: {vis:4d}/{total} ({100*vis/total:.1f}%)")
    print()
    print("4-corner visibility (front lights + rear trunk):")
    print(f"  Both front lights:    {vis_both_front:4d}/{n} ({100*vis_both_front/n:.1f}%)")
    print(f"  Both rear trunk:      {vis_both_rear:4d}/{n} ({100*vis_both_rear/n:.1f}%)")
    print(f"  All four visible:     {vis_all_four:4d}/{n} ({100*vis_all_four/n:.1f}%)")
    print(f"  >= 2 corners visible: {vis_at_least_2:4d}/{n} ({100*vis_at_least_2/n:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Pascal3D+ car annotations")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/Pascal3D+/Annotations/car_pascal/*.mat",
        help="Glob pattern for .mat annotation files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output one parsed car annotation per line as JSON",
    )
    args = parser.parse_args()

    if args.json:
        import glob
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

        for f in sorted(glob.glob(args.path)):
            for car in parse_annotation(f):
                print(json.dumps(car))
    else:
        print_stats(args.path)
