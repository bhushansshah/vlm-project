"""Which way a person faces, from body keypoints (ViTPose): input for the viewpoint tool."""

from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw

from .base import ModelTool, ToolContext, ToolError, ToolResult, fmt_box, parse_box, to_pixels
from .image_ops import BOX_SCHEMA, resize_for_view

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP = 0, 1, 2, 3, 4, 5, 6, 11, 12
SKELETON = [(L_SH, R_SH), (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP), (L_SH, 7), (7, 9), (R_SH, 8), (8, 10),
            (L_HIP, 13), (13, 15), (R_HIP, 14), (14, 16), (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR)]
# shoulder width / torso length for a person seen straight on (rough adult proportion)
FRONTAL_RATIO = 0.8


def describe_angle(angle: float) -> str:
    """Facing angle in words. Convention: 0 = away from the camera, 90 = toward image-right,
    180 = toward the camera, 270 = toward image-left."""
    names = ["away from the camera", "away from the camera, turned toward image-right", "toward image-right",
             "toward the camera, turned toward image-right", "toward the camera",
             "toward the camera, turned toward image-left", "toward image-left",
             "away from the camera, turned toward image-left"]
    return names[round((angle % 360) / 45) % 8]


def facing_from_keypoints(kp, scores, min_score: float = 0.3) -> dict[str, Any]:
    """Estimate the facing angle from 17 COCO keypoints (pixel x, y) and their scores.

    A person facing the camera shows their left shoulder on the image's right (and their face);
    facing away, the order flips and the face is hidden. How wide the shoulders look relative to
    the torso says how far the body is turned; where the nose sits between the shoulders says to
    which side. Returns {"angle", "facing", "confidence", "why"}."""
    ok = [s >= min_score for s in scores]
    if not (ok[L_SH] and ok[R_SH]):
        raise ToolError("could not find both shoulders of a person in this box; give a tighter box around one "
                        "person, or judge the facing direction yourself (zoom in)")
    sh_mid = ((kp[L_SH][0] + kp[R_SH][0]) / 2, (kp[L_SH][1] + kp[R_SH][1]) / 2)
    if ok[L_HIP] and ok[R_HIP]:
        hip_mid = ((kp[L_HIP][0] + kp[R_HIP][0]) / 2, (kp[L_HIP][1] + kp[R_HIP][1]) / 2)
        torso = math.dist(sh_mid, hip_mid)
    else:  # seated / cut off: fall back to the head-to-shoulder distance
        torso = 2.5 * abs(sh_mid[1] - kp[NOSE][1]) if ok[NOSE] else 0
    signed = (kp[L_SH][0] - kp[R_SH][0]) / max(torso, 1.0)  # > 0: the person's left is on the image right
    face = sum(scores[i] for i in (NOSE, L_EYE, R_EYE)) / 3
    toward_camera = signed > 0 if abs(signed) > 0.15 else face >= 0.5
    frontal = min(abs(signed) / FRONTAL_RATIO, 1.0)  # 1 = square to the camera, 0 = profile
    turn = math.degrees(math.acos(frontal))  # 0 = square on, 90 = side-on

    side, why_side = 0, "the side it is turned to is unclear"
    if ok[NOSE] and turn > 10:
        offset = kp[NOSE][0] - sh_mid[0]
        side = 1 if offset > 0 else -1
        why_side = f"the nose is on the image-{'right' if side > 0 else 'left'} side of the shoulders"
    if toward_camera:
        angle = 180 - side * turn
    else:
        angle = side * turn
    angle %= 360
    conf_scores = [scores[i] for i in (L_SH, R_SH, L_HIP, R_HIP, NOSE)]
    conf = sum(conf_scores) / len(conf_scores)
    why = (f"{'face visible, ' if face >= 0.5 else 'face hidden, '}"
           f"the person's left shoulder is on the image-{'right' if signed > 0 else 'left'}; "
           f"shoulders look {round(100 * frontal)}% as wide as square-on; {why_side}")
    return {"angle": round(angle), "facing": describe_angle(angle),
            "confidence": "high" if conf > 0.7 else "medium" if conf > 0.45 else "low", "why": why}


class Orient(ModelTool):
    name = "orient"
    description = ("Estimate which way ONE person is facing, from body keypoints (shoulders, hips, nose, eyes, "
                   "ears). Give a box around that person (their whole body if visible). Returns the facing "
                   "direction in image terms and as an angle for the viewpoint tool (0 = away from the camera, "
                   "90 = toward image-right, 180 = toward the camera, 270 = toward image-left), plus an image "
                   "with the keypoints. Use it before viewpoint for 'from his/her perspective' questions. Can be "
                   "wrong for small, hidden or unusual poses: check the keypoint image.")
    parameters = {"box": {**BOX_SCHEMA, "description": "box around one person"}}
    required = ("box",)
    example = {"box": {"left": 420, "top": 300, "right": 520, "bottom": 900}}

    def load(self):
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        model_id = self.options.get("model", "usyd-community/vitpose-base-simple")
        proc = AutoProcessor.from_pretrained(model_id)
        model = VitPoseForPoseEstimation.from_pretrained(model_id).to(self.device()).eval()
        return proc, model

    def keypoints(self, person: Image.Image):
        """17 COCO keypoints (pixel coordinates in ``person``) and their scores."""
        import torch

        proc, model = self.model()
        box = [[[0.0, 0.0, float(person.width), float(person.height)]]]  # COCO x, y, w, h
        with self._lock, torch.inference_mode():
            inputs = proc(person, boxes=box, return_tensors="pt").to(model.device)
            out = model(**inputs)
            res = proc.post_process_pose_estimation(out, boxes=box)[0][0]
        return res["keypoints"].tolist(), res["scores"].tolist()

    def run(self, ctx: ToolContext, box: Any = None) -> ToolResult:
        b = parse_box(box, "box")
        x1, y1, x2, y2 = to_pixels(b, ctx.size)
        mx, my = (x2 - x1) // 10, (y2 - y1) // 20  # a little context, like a person detector's box
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(ctx.size[0], x2 + mx), min(ctx.size[1], y2 + my)
        person = ctx.image.crop((x1, y1, x2, y2))
        if min(person.size) < 24:
            raise ToolError(f"the box is only {person.width}x{person.height} px; too small to read a pose. "
                            "Judge the facing direction yourself (zoom in).")
        kp, scores = self.keypoints(person)
        est = facing_from_keypoints(kp, scores)
        view = resize_for_view(person, 256 * 256, 640 * 640).convert("RGB")
        s = view.width / person.width
        d = ImageDraw.Draw(view)
        for a, c in SKELETON:
            if scores[a] >= 0.3 and scores[c] >= 0.3:
                d.line([(kp[a][0] * s, kp[a][1] * s), (kp[c][0] * s, kp[c][1] * s)], fill=(0, 255, 255), width=3)
        for i, (x, y) in enumerate(kp):
            if scores[i] >= 0.3:
                col = (255, 0, 0) if i in (L_EYE, L_EAR, L_SH, L_HIP, 7, 9, 13, 15) else (0, 80, 255)
                d.ellipse([x * s - 4, y * s - 4, x * s + 4, y * s + 4], fill=col)
        return ToolResult(
            text=(f"Person in {fmt_box([round(v) for v in b])} faces {est['facing']} (angle {est['angle']}; "
                  f"confidence {est['confidence']}): {est['why']}. Keypoint image: red = the person's LEFT side, "
                  "blue = their RIGHT side."),
            images=[view], data={**est, "box": [round(v) for v in b]})
