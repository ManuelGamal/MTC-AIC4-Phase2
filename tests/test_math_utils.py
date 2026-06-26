# We extract the pure math functions from inference.py to avoid top-level execution errors during import
def compute_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area_a, area_b = max(0, aw)*max(0, ah), max(0, bw)*max(0, bh)
    return inter / (area_a + area_b - inter + 1e-6)

def clip_bbox(bbox, W, H):
    x = max(0.0, min(float(bbox[0]), W - 1.0))
    y = max(0.0, min(float(bbox[1]), H - 1.0))
    w = max(0.0, min(float(bbox[2]), W - x))
    h = max(0.0, min(float(bbox[3]), H - y))
    return [x, y, w, h]

def test_compute_iou():
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 10, 10]
    iou = compute_iou(box1, box2)
    # intersection is 5x5 = 25. union is 100 + 100 - 25 = 175. iou = 25/175 = 1/7 ~= 0.1428
    assert abs(iou - 0.142857) < 1e-4

def test_clip_bbox():
    W, H = 100, 100
    # Normal
    assert clip_bbox([10, 10, 20, 20], W, H) == [10.0, 10.0, 20.0, 20.0]
    # Out of bounds negative
    assert clip_bbox([-10, -10, 20, 20], W, H) == [0.0, 0.0, 20.0, 20.0]
    # Out of bounds positive
    assert clip_bbox([90, 90, 20, 20], W, H) == [90.0, 90.0, 10.0, 10.0]
