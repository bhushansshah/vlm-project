import numpy as np
import pytest
from PIL import Image

from vlmbench.datasets.base import Sample
from vlmbench.datasets.treebench import evidence_scores
from vlmbench.tools import ToolContext, ToolError, build_tools
from vlmbench.tools.colour import colour_name, dominant_colours
from vlmbench.tools.pose import L_SH, NOSE, R_SH, facing_from_keypoints
from vlmbench.tools.viewpoint import direction_words, facing_angle, relative_position


@pytest.fixture
def ctx():
    img = Image.new("RGB", (2000, 1000), "white")
    img.paste((220, 20, 20), (200, 200, 400, 400))  # red block, 0-1000 box [100, 200, 200, 400]
    img.paste((20, 40, 200), (1400, 500, 1600, 800))  # blue block, [700, 500, 800, 800]
    c = ToolContext(image=img)
    c.cache["depth"] = np.full((1000, 2000), 0.5, dtype=np.float32)  # flat depth: no depth model needed
    return c


def _keypoints(l_sh_x, r_sh_x, nose_x=None, face=0.9):
    kp = [[100.0, 100.0] for _ in range(17)]
    scores = [0.1] * 17
    kp[L_SH], kp[R_SH] = [l_sh_x, 100.0], [r_sh_x, 100.0]
    kp[11], kp[12] = [l_sh_x, 200.0], [r_sh_x, 200.0]  # hips under the shoulders: torso 100 px
    for i in (L_SH, R_SH, 11, 12):
        scores[i] = 0.9
    if nose_x is not None:
        kp[NOSE] = [nose_x, 60.0]
        scores[NOSE] = scores[1] = scores[2] = face
    return kp, scores


def test_facing_from_keypoints():
    # square to the camera: the person's left shoulder is on the image's right, face visible
    assert facing_from_keypoints(*_keypoints(140, 60, nose_x=100))["angle"] == 180
    # back to the camera: left shoulder on the image's left, face hidden
    assert facing_from_keypoints(*_keypoints(60, 140, face=0.0))["angle"] == 0
    # facing the camera but turned toward image-right (narrow shoulders, nose to the right)
    turned = facing_from_keypoints(*_keypoints(110, 90, nose_x=125))
    assert 90 < turned["angle"] < 180 and "image-right" in turned["facing"]
    with pytest.raises(ToolError):
        facing_from_keypoints([[0, 0]] * 17, [0.0] * 17)


def test_viewpoint_geometry():
    assert facing_angle("toward_camera") == 180 and facing_angle("Image-Left") == 270 and facing_angle(-90) == 270
    with pytest.raises(ToolError):
        facing_angle("north")
    # a reference facing the camera: an object on the image's right at the same depth is on ITS left
    rel = relative_position((0.0, 5.0), 180.0, (1.0, 5.0))
    assert rel["right"] < 0 and direction_words(rel["bearing"])[0] == "rear left"
    # ... and something nearer the camera is in front of it
    rel = relative_position((0.0, 5.0), 180.0, (1.0, 3.0))
    assert direction_words(rel["bearing"])[0] == "front left"
    # facing away from the camera, left and right agree with the image
    rel = relative_position((0.0, 5.0), 0.0, (1.0, 7.0))
    assert direction_words(rel["bearing"])[0] == "front right"
    assert "uncertain" in direction_words(3.0)[1]


def test_viewpoint_tool_flips_left_right_for_a_reference_facing_the_camera(ctx):
    vp = build_tools(["viewpoint"])[0]
    ref = {"label": "woman", "box": [100, 200, 200, 400]}
    target = [{"label": "blue box", "box": [700, 500, 800, 800]}]
    toward = vp(ctx, {"reference": {**ref, "facing": "toward_camera"}, "targets": target})
    away = vp(ctx, {"reference": {**ref, "facing": "away_from_camera"}, "targets": target})
    assert toward.data["targets"][0]["direction"].endswith("left")
    assert away.data["targets"][0]["direction"].endswith("right")
    assert toward.images and "point of view" in toward.text
    with pytest.raises(ToolError):
        vp(ctx, {"reference": ref, "targets": target})  # facing is required


def test_order_sorts_in_the_stated_direction(ctx, monkeypatch):
    order = build_tools(["order"])[0]
    dets = [{"px": [0, 0, 1, 1], "score": 0.9, "label": "post", "box": b}
            for b in ([500, 0, 550, 100], [100, 0, 150, 100], [800, 0, 850, 100])]
    monkeypatch.setattr(order.detect, "find", lambda *a, **k: [dict(d) for d in dets])
    res = order(ctx, {"query": "post", "order": "right_to_left"})
    assert [it["box"][0] for it in res.data["items"]] == [800, 500, 100]
    assert res.text.splitlines()[1].startswith("1st") and res.images
    with pytest.raises(ToolError):
        order(ctx, {"query": "post", "order": "clockwise"})


def test_colour_names_blocks(ctx):
    assert colour_name((220, 20, 20)) == "red" and colour_name((20, 40, 200)) == "blue"
    assert colour_name((10, 10, 10)) == "black" and colour_name((245, 245, 245)) == "white"
    assert dominant_colours(np.array([[255, 0, 0]] * 30 + [[0, 0, 255]] * 10))[0][1] == pytest.approx(0.75)
    res = build_tools(["colour"])[0](ctx, {"objects": [{"label": "block", "box": [100, 200, 200, 400]}]})
    assert res.data["objects"][0]["colours"][0]["name"] == "red"


def test_enhance_limits_and_output(ctx, monkeypatch):
    enh = build_tools(["enhance"])[0]
    monkeypatch.setattr(enh, "upscale", lambda crop: crop.resize((crop.width * 4, crop.height * 4)))
    res = enh(ctx, {"box": [100, 200, 150, 260]})  # 100 x 60 px
    assert res.data["crop_px"] == [100, 60] and res.images[0].width >= 400
    with pytest.raises(ToolError):
        enh(ctx, {"box": [0, 0, 500, 500]})  # 1000 x 500 px: use zoom


def test_evidence_scores(image_path):
    s = Sample("tb1", image_path, "q", "Perception/Attributes", "A",
               meta={"target_boxes": ((40, 20, 120, 100),), "options": ("x", "y")})  # image is 400 x 200
    final = {"final": True, "model_output": '<evidence>[{"label": "cup", "box": {"left": 100, "top": 100, '
                                            '"right": 300, "bottom": 500}}]</evidence><answer>A</answer>'}
    zoom = {"tool": "zoom", "arguments": {"box": {"left": 50, "top": 50, "right": 400, "bottom": 600}}}
    sc = evidence_scores(s, {"trace": [zoom, final]}, (400, 200))
    assert sc["cited"] == 1 and sc["miou"] == pytest.approx(1.0) and sc["inspected_target"]
    none = evidence_scores(s, {"trace": [{"final": True, "model_output": "<answer>A</answer>"}]}, (400, 200))
    assert none["miou"] == 0 and not none["inspected_target"]
    assert evidence_scores(s, {"response": "A"}, (400, 200)) is None
