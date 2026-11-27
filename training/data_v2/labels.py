"""Automatic labels for training data v2: objects, which way people face, and depth.

Everything a v2 question is built from comes from three off-the-shelf models, run once per image:

- RT-DETRv2 (COCO, 80 classes) finds objects. It runs twice, at 640 and 1280 px input, so small objects in
  3-4 MP images are found too; the two passes are merged with per-class non-maximum suppression.
- ViTPose (COCO keypoints) gives each large-enough person a facing angle, with the same rule the agent's
  `orient` tool uses (`vlmbench.tools.pose.facing_from_keypoints`).
- Depth Anything V2 (Base) gives relative depth. Each object gets a rough 3D point with the same
  back-projection as the agent's `distance` / `viewpoint` tools (60 degree field of view, z = 1 / (nearness
  + 0.1)): only the ordering of distances is meaningful.

Boxes are pixel [x1, y1, x2, y2] on the original image, like TreeBench's `target_boxes`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from vlmbench.tools.base import ToolError
from vlmbench.tools.pose import L_EYE, L_HIP, L_SH, NOSE, R_EYE, R_HIP, R_SH, facing_from_keypoints

DETECTOR = "PekingU/rtdetr_v2_r101vd"
POSE = "usyd-community/vitpose-base-simple"
DEPTH = "depth-anything/Depth-Anything-V2-Base-hf"
DET_SIZES = (640, 1280)  # detector input sizes (square), merged
DET_MIN_SCORE = 0.15  # kept, so later steps can see near-misses (e.g. an uncertain fourth car)
POSE_MIN_HEIGHT = 96  # px: smaller people get no facing estimate
DEPTH_DOWNSAMPLE = 8  # the saved depth map is 1/8 of the image side
FOV_DEG, DEPTH_OFFSET = 60.0, 0.1  # as in vlmbench.tools.depth.Distance
# the detector's config uses some PASCAL VOC spellings
PLAIN_NAMES = {"aeroplane": "airplane", "diningtable": "dining table", "motorbike": "motorcycle",
               "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv"}


@dataclass
class Models:
    det_proc: Any
    det: Any
    pose_proc: Any
    pose: Any
    depth_proc: Any
    depth: Any
    device: str


def load_models(device: str | None = None) -> Models:
    from transformers import (AutoImageProcessor, AutoModelForDepthEstimation, AutoProcessor,
                              RTDetrV2ForObjectDetection, VitPoseForPoseEstimation)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    det_proc = AutoImageProcessor.from_pretrained(DETECTOR)
    det = RTDetrV2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
    pose_proc = AutoProcessor.from_pretrained(POSE)
    pose = VitPoseForPoseEstimation.from_pretrained(POSE).to(device).eval()
    depth_proc = AutoImageProcessor.from_pretrained(DEPTH)
    depth = AutoModelForDepthEstimation.from_pretrained(DEPTH).to(device).eval()
    return Models(det_proc, det, pose_proc, pose, depth_proc, depth, device)


@torch.inference_mode()
def detect(m: Models, img: Image.Image) -> list[dict]:
    """Objects as {"label", "score", "box"}; two input sizes, merged by per-class NMS (IoU 0.6)."""
    from torchvision.ops import batched_nms

    boxes, scores, labels = [], [], []
    for side in DET_SIZES:
        inputs = m.det_proc(images=img, size={"height": side, "width": side}, return_tensors="pt").to(m.device)
        out = m.det(**inputs)
        res = m.det_proc.post_process_object_detection(out, target_sizes=[(img.height, img.width)],
                                                       threshold=DET_MIN_SCORE)[0]
        boxes.append(res["boxes"].float().cpu())
        scores.append(res["scores"].float().cpu())
        labels.append(res["labels"].cpu())
    b, s, lab = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
    keep = batched_nms(b, s, lab, 0.6)
    names = m.det.config.id2label
    objs = []
    for i in keep.tolist():
        x1, y1, x2, y2 = (float(v) for v in b[i])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img.width), x2), min(float(img.height), y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        name = names[int(lab[i])]
        objs.append({"label": PLAIN_NAMES.get(name, name), "score": round(float(s[i]), 3),
                     "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    objs.sort(key=lambda o: -o["score"])
    return objs


@torch.inference_mode()
def depth_map(m: Models, img: Image.Image) -> np.ndarray:
    """Relative nearness in [0, 1] at full image size (1 = nearest), as the agent's depth tool computes it."""
    inputs = m.depth_proc(images=img, return_tensors="pt").to(m.device)
    pred = m.depth(**inputs).predicted_depth  # (1, h, w); Depth Anything predicts disparity (larger = nearer)
    pred = torch.nn.functional.interpolate(pred[:, None], size=(img.height, img.width), mode="bicubic",
                                           align_corners=False)[0, 0].float().cpu().numpy()
    lo, hi = float(pred.min()), float(pred.max())
    return (pred - lo) / (hi - lo) if hi > lo else pred * 0


def box_nearness(dmap: np.ndarray, box) -> dict[str, float]:
    """Median nearness of the central half of a box (less background), plus its spread."""
    h, w = dmap.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
    cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
    cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
    patch = dmap[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)]
    return {"near": float(np.median(patch)), "near_p25": float(np.percentile(patch, 25)),
            "near_p75": float(np.percentile(patch, 75))}


def point_xz(near: float, box, size: tuple[int, int]) -> tuple[float, float]:
    """Top-down map position (x to the image right, z away from the camera), as Distance._point."""
    w, _ = size
    z = 1.0 / (near + DEPTH_OFFSET)
    f = w / (2 * math.tan(math.radians(FOV_DEG) / 2))
    u = (box[0] + box[2]) / 2
    return ((u - w / 2) / f * z, z)


@torch.inference_mode()
def facings(m: Models, img: Image.Image, person_boxes: list[list[float]]) -> list[dict | None]:
    """Facing estimate per person box (None when the pose is unreadable). Adds the cues used to filter:
    `signed` (shoulder spread / torso, > 0 = their left on the image right), `face` (mean nose/eye score)
    and `kp_min` (lowest of the shoulder and hip scores)."""
    if not person_boxes:
        return []
    out_list: list[dict | None] = []
    # crop each person with a little context, as the orient tool does
    crops, offsets = [], []
    for b in person_boxes:
        x1, y1, x2, y2 = b
        mx, my = (x2 - x1) / 10, (y2 - y1) / 20
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(img.width, x2 + mx), min(img.height, y2 + my)
        crops.append(img.crop((int(x1), int(y1), int(x2), int(y2))))
        offsets.append((x1, y1))
    for crop in crops:
        box = [[[0.0, 0.0, float(crop.width), float(crop.height)]]]
        inputs = m.pose_proc(crop, boxes=box, return_tensors="pt").to(m.device)
        res = m.pose_proc.post_process_pose_estimation(m.pose(**inputs), boxes=box)[0][0]
        kp, sc = res["keypoints"].tolist(), res["scores"].tolist()
        try:
            est = facing_from_keypoints(kp, sc)
        except ToolError:
            out_list.append(None)
            continue
        sh_mid = ((kp[L_SH][0] + kp[R_SH][0]) / 2, (kp[L_SH][1] + kp[R_SH][1]) / 2)
        if sc[L_HIP] >= 0.3 and sc[R_HIP] >= 0.3:
            hip_mid = ((kp[L_HIP][0] + kp[R_HIP][0]) / 2, (kp[L_HIP][1] + kp[R_HIP][1]) / 2)
            torso = math.dist(sh_mid, hip_mid)
        else:
            torso = 2.5 * abs(sh_mid[1] - kp[NOSE][1]) if sc[NOSE] >= 0.3 else 0
        signed = (kp[L_SH][0] - kp[R_SH][0]) / max(torso, 1.0)
        est.update({"signed": round(signed, 3), "face": round(sum(sc[i] for i in (NOSE, L_EYE, R_EYE)) / 3, 3),
                    "kp_min": round(min(sc[i] for i in (L_SH, R_SH, L_HIP, R_HIP)), 3)})
        out_list.append(est)
    return out_list


def annotate(m: Models, img: Image.Image) -> tuple[dict, np.ndarray]:
    """One image's labels (objects with nearness, 3D point and, for people, facing) and its depth map."""
    img = img.convert("RGB")
    objs = detect(m, img)
    dmap = depth_map(m, img)
    for i, o in enumerate(objs):
        o["id"] = i
        o.update({k: round(v, 4) for k, v in box_nearness(dmap, o["box"]).items()})
        x, z = point_xz(o["near"], o["box"], img.size)
        o["xz"] = [round(x, 4), round(z, 4)]
    people = [o for o in objs if o["label"] == "person" and o["score"] >= 0.45
              and o["box"][3] - o["box"][1] >= POSE_MIN_HEIGHT]
    for o, est in zip(people, facings(m, img, [o["box"] for o in people])):
        o["facing"] = est
    small = dmap[::DEPTH_DOWNSAMPLE, ::DEPTH_DOWNSAMPLE].astype(np.float16)
    return {"width": img.width, "height": img.height, "objects": objs}, small


# --------------------------------------------------------------------------------------------- perspective
QUADRANTS = ("Front left", "Front right", "Left rear", "Right rear")  # TreeBench's four options, its order


def quadrant(ref_xz, facing_deg: float, target_xz, margin_deg: float) -> tuple[str | None, float]:
    """TreeBench option for the target as seen by the reference, or None when the bearing is within
    `margin_deg` of an axis (left/right or front/rear would be a guess)."""
    from vlmbench.tools.viewpoint import relative_position

    bearing = relative_position(ref_xz, facing_deg, target_xz)["bearing"]
    a = abs(bearing)
    if a < margin_deg or a > 180 - margin_deg or abs(a - 90) < margin_deg:
        return None, bearing
    front = a < 90
    right = bearing > 0
    return {(True, False): "Front left", (True, True): "Front right", (False, False): "Left rear",
            (False, True): "Right rear"}[(front, right)], bearing


# Bearing margin around every axis. The facing angle comes from shoulder width, which is rough (tens of
# degrees), so a target nearly straight ahead/behind (or level) flips side on a small error. First set to 20;
# raised to 30 after the TreeBench check (check_perspective_labels.py) showed every left/right mistake had
# |bearing| < 27. That check is therefore optimistic; the hand check on generated items is the real one.
PERSPECTIVE_MARGIN_DEG = 30.0


def confident_facing(est: dict | None) -> bool:
    """Facing estimates trusted for labels: both shoulders and hips clearly found, and the pose model's
    overall confidence not low."""
    return bool(est) and est["kp_min"] >= 0.4 and est["confidence"] != "low"


def snapped_facing(est: dict | None) -> float | None:
    """Toward (180) or away from (0) the camera, or None. The turn angle the orient rule estimates is rough,
    and from behind it rests on a nose the model cannot see (the hand check found a woman walking straight
    away labelled as turned 30 degrees). So only the front/back call is used, and only when two cues agree:
    the shoulder order (their left shoulder on the image right = facing the camera) and the face (visible
    when facing the camera, hidden when facing away)."""
    if not confident_facing(est) or abs(est["signed"]) < 0.3:
        return None
    if est["signed"] > 0 and est["face"] >= 0.6:
        return 180.0
    if est["signed"] < 0 and est["face"] < 0.4:
        return 0.0
    return None


TURN_SLACK_DEG = 15.0  # error allowed on the measured turn
AXIS_MARGIN_DEG = 10.0  # and no candidate bearing this close to an axis


def turn_deg(est: dict) -> float:
    """How far the body is turned from square-on, from shoulder width (as facing_from_keypoints)."""
    return math.degrees(math.acos(min(abs(est["signed"]) / 0.8, 1.0)))


MIN_HEIGHT_FOR_TURN_SIDE = 200  # px: below this the side a person is turned to is not trusted (see below)


def robust_quadrant(ref_xz, est: dict, target_xz, ref_height: float = 1e9) -> tuple[str | None, float]:
    """The quadrant, if it holds for every facing the pose leaves open; else None.

    Facing the camera, the face is visible, so the side the person is turned to (nose left or right of the
    shoulder centre) is reliable: the measured angle is used, +- 20 degrees. Facing away, the nose is not
    visible and the side is a guess (the hand check's error), so only how far they are turned (t, from
    shoulder width) is used: the facing may be base - t or base + t, each +- TURN_SLACK_DEG. Every candidate
    facing must give the same quadrant, none within AXIS_MARGIN_DEG of an axis."""
    base = snapped_facing(est)
    if base is None:
        return None, 0.0
    # A person under MIN_HEIGHT_FOR_TURN_SIDE px is treated like one facing away: the hand check of generated
    # items found both perspective errors on small, distant people (a ~150 px runner whose measured 37-degree
    # turn flipped left and right).
    if base == 180.0 and ref_height >= MIN_HEIGHT_FOR_TURN_SIDE:
        centres, slack = [float(est["angle"])], 20.0
    else:
        t = turn_deg(est)
        centres, slack = [base - t, base + t], TURN_SLACK_DEG
    labels, bearing0 = set(), quadrant(ref_xz, centres[0], target_xz, 0.0)[1]
    for c in centres:
        for s in (-slack, 0.0, slack):
            labels.add(quadrant(ref_xz, c + s, target_xz, AXIS_MARGIN_DEG)[0])
    if len(labels) != 1 or None in labels:
        return None, bearing0
    return labels.pop(), bearing0
