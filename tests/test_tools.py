import types

import numpy as np
import pytest
from PIL import Image

from vlmbench.tools import TOOLS, ToolContext, ToolError, build_tools
from vlmbench.tools.base import parse_box, sub_to_orig, to_pixels


@pytest.fixture
def ctx():
    img = Image.new("RGB", (2000, 1000), "white")
    img.paste((255, 0, 0), (1000, 500, 1100, 600))  # a red square at 0-1000 box [500, 500, 550, 600]
    return ToolContext(image=img)


def test_box_helpers():
    assert parse_box([600, 700, 100, 200]) == (100, 200, 600, 700)  # corners reordered
    assert parse_box([-5, 0, 2000, 10]) == (0, 0, 1000, 10)  # clipped
    with pytest.raises(ToolError):
        parse_box([1, 2, 3])
    with pytest.raises(ToolError):
        parse_box([5, 5, 5, 9])  # empty
    assert to_pixels((500, 500, 550, 600), (2000, 1000)) == (1000, 500, 1100, 600)
    assert sub_to_orig([0, 0, 50, 50], (100, 100), (500, 500, 600, 600)) == [500, 500, 550, 550]


def test_zoom_enlarges_small_regions(ctx):
    res = build_tools(["zoom"])[0](ctx, {"box": [500, 500, 550, 600]})
    view = res.images[0]
    assert max(view.size) == 300 and res.data["crop_px"] == [100, 100]  # enlarged at most 3x
    assert view.getpixel((150, 150)) == (255, 0, 0)  # it is the red square
    assert "holds little detail" in res.text  # a 100 px crop gets the small-region note
    big = build_tools(["zoom"])[0](ctx, {"box": [0, 0, 500, 500]})
    assert "little detail" not in big.text
    with pytest.raises(ToolError):
        build_tools(["zoom"])[0](ctx, {"box": [500, 500, 550, 600], "colour": "x"})  # unknown argument
    with pytest.raises(ToolError):
        build_tools(["zoom"])[0](ctx, {})  # missing box


def test_zoom_keeps_thin_strips_readable():
    from vlmbench.tools.image_ops import resize_for_view
    strip = Image.new("RGB", (2264, 195))
    assert resize_for_view(strip, 512 * 512, 1024 * 1024).size == (2048, 176)  # area fits; only the 2048 px side cap applies
    assert resize_for_view(Image.new("RGB", (4000, 3000)), 512 * 512, 1024 * 1024).size == (1182, 887)


def test_grid_returns_image(ctx):
    res = build_tools(["grid"])[0](ctx, {"region": [0, 0, 500, 500]})
    assert res.images and res.data["region"] == [0, 0, 500, 500]


def test_registry():
    assert set(TOOLS) == {"zoom", "grid", "detect", "ocr", "depth", "count", "mark", "distance",
                          "enhance", "order", "orient", "viewpoint", "colour"}
    with pytest.raises(ValueError):
        build_tools(["teleport"])


class FakeDino:
    """Grounding DINO stand-in: returns ``boxes(w, h)`` with scores, keeping those above the threshold."""

    def __init__(self, boxes, scores):
        self.boxes, self.scores, self.thresholds = boxes, scores, []

    def __call__(self, images, text, return_tensors):
        import torch
        self.size = images.size
        return types.SimpleNamespace(to=lambda d: {"input_ids": torch.zeros(1, 3)})

    def post_process_grounded_object_detection(self, out, ids, threshold, text_threshold, target_sizes):
        import torch
        self.thresholds.append(threshold)
        keep = [(b, s) for b, s in zip(self.boxes(*self.size), self.scores) if s >= threshold]
        return [{"boxes": torch.tensor([b for b, _ in keep]).reshape(-1, 4),
                 "scores": torch.tensor([s for _, s in keep]), "text_labels": ["cup"] * len(keep)}]


class Net:
    device = "cpu"

    def __call__(self, **inputs):
        return None


def test_detect_maps_boxes_to_original_coordinates(ctx, monkeypatch):
    tool = build_tools(["detect"])[0]
    # two overlapping boxes (one duplicate) + one box covering the whole crop
    proc = FakeDino(lambda w, h: [[0, 0, w / 2, h / 2], [1, 1, w / 2, h / 2], [0, 0, w, h]], [0.9, 0.5, 0.8])
    monkeypatch.setattr(type(tool), "model", lambda self: (proc, Net()))
    res = tool(ctx, {"query": "cup", "region": [500, 0, 1000, 1000]})
    assert res.data["count"] == 1  # duplicate and whole-region box removed
    assert res.data["detections"][0]["box"] == [500, 0, 750, 500]
    assert res.images


def test_ocr_maps_boxes(ctx, monkeypatch):
    tool = build_tools(["ocr"])[0]

    class Reader:
        def readtext(self, arr):
            h, w = arr.shape[:2]
            return [([[0, 0], [w / 2, 0], [w / 2, h / 2], [0, h / 2]], "STOP", 0.97)]

    monkeypatch.setattr(type(tool), "model", lambda self: Reader())
    res = tool(ctx, {"region": [0, 0, 500, 500]})
    assert res.data["items"][0]["text"] == "STOP" and res.data["items"][0]["box"] == [0, 0, 250, 250]
    assert '"STOP"' in res.text


def test_depth_orders_nearest_first(ctx, monkeypatch):
    tool = build_tools(["depth"])[0]
    dmap = np.zeros((1000, 2000), dtype=np.uint8)
    dmap[:, 1000:] = 200  # right half is nearer
    monkeypatch.setattr(type(tool), "model", lambda self: (lambda img: {"depth": Image.fromarray(dmap)}))
    res = tool(ctx, {"objects": [{"label": "left", "box": [0, 0, 400, 1000]},
                                 {"label": "right", "box": [600, 0, 1000, 1000]}]})
    assert [o["label"] for o in res.data["objects"]] == ["right", "left"]
    assert "depth" in ctx.cache  # depth map computed once per question
    with pytest.raises(ToolError):
        tool(ctx, {"objects": []})


def test_box_is_accepted_for_region(ctx):
    res = build_tools(["grid"])[0](ctx, {"box": [0, 0, 500, 500]})
    assert res.data["region"] == [0, 0, 500, 500]


def test_named_box_fields():
    from vlmbench.tools.base import fmt_box
    assert parse_box({"left": 100, "top": 200, "right": 300, "bottom": 400}) == (100, 200, 300, 400)
    with pytest.raises(ToolError):
        parse_box({"x": 1, "y": 2})
    assert fmt_box([1, 2, 3, 4]) == '{"left": 1, "top": 2, "right": 3, "bottom": 4}'


def test_detect_shows_low_confidence_candidates_instead_of_none(ctx, monkeypatch):
    from vlmbench.tools.detect import Detect
    proc = FakeDino(lambda w, h: [[0, 0, w / 4, h / 4]], [0.2])  # only a weak candidate
    monkeypatch.setattr(Detect, "model", lambda self: (proc, Net()))
    res = build_tools(["detect"])[0](ctx, {"query": "cup"})
    assert res.data["count"] == 0 and len(res.data["low_confidence"]) == 1
    assert "does NOT mean the object is absent" in res.text and res.images
    assert proc.thresholds == [0.3, 0.15]


def test_count_numbers_candidates_and_mark_draws_points(ctx, monkeypatch):
    from vlmbench.tools.detect import Detect
    proc = FakeDino(lambda w, h: [[0, 0, w / 4, h / 4], [w / 2, h / 2, w * 3 / 4, h * 3 / 4]], [0.25, 0.6])
    monkeypatch.setattr(Detect, "model", lambda self: (proc, Net()))
    res = build_tools(["count"])[0](ctx, {"query": "cup"})
    assert len(res.data["candidates"]) == 2 and proc.thresholds == [0.2]  # low threshold: fewer misses
    assert "NOT a final count" in res.text and res.images

    mark = build_tools(["mark"])[0]
    res = mark(ctx, {"points": [{"x": 525, "y": 550}, {"x": 900, "y": 900}], "region": [500, 500, 600, 600]})
    assert res.data["points"] == [(525, 550), (900, 900)] and "Points [2] are outside" in res.text
    with pytest.raises(ToolError):
        mark(ctx, {"points": [{"x": 1}]})


def test_distance_ranks_by_3d_distance_not_image_distance(monkeypatch):
    from vlmbench.tools.depth import Depth
    img = Image.new("RGB", (1000, 1000))
    dmap = np.zeros((1000, 1000), dtype=np.uint8)
    dmap[:, :] = 50
    dmap[0:200, 0:200] = 250  # reference: near the camera, top-left
    dmap[0:200, 300:500] = 30  # "far": close on the picture, but far behind
    dmap[600:800, 0:200] = 240  # "near": further on the picture, at the same depth
    monkeypatch.setattr(Depth, "model", lambda self: (lambda im: {"depth": Image.fromarray(dmap)}))
    res = build_tools(["distance"])[0](ToolContext(image=img), {
        "reference": {"label": "gripper", "box": [0, 0, 200, 200]},
        "objects": [{"label": "far", "box": [300, 0, 500, 200]}, {"label": "near", "box": [0, 600, 200, 800]}]})
    assert [o["label"] for o in res.data["objects"]] == ["near", "far"]
    assert res.text.startswith("Closest to farthest from gripper")
