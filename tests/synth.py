"""Build synthetic 21-point hands for tests (upright hand, palm facing camera)."""

# knuckle (MCP) x-positions relative to wrist, hand height ~0.3 of frame
MCP_X = {"index": -0.06, "middle": -0.02, "ring": 0.02, "pinky": 0.06}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(index=False, middle=False, ring=False, pinky=False, thumb=False,
              thumb_dir="side", pinch=None, origin=(0.5, 0.8), scale=1.0):
    """pinch: None | 'index' | 'middle' -> thumb tip placed on that fingertip."""
    ox, oy = origin
    s = scale
    pts = [None] * 21
    pts[0] = (ox, oy)
    ext = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, b in BASE.items():
        mx, my = ox + MCP_X[name] * s, oy - 0.15 * s
        pts[b] = (mx, my)
        if ext[name]:
            pts[b + 1] = (mx, my - 0.05 * s)
            pts[b + 2] = (mx, my - 0.09 * s)
            pts[b + 3] = (mx, my - 0.12 * s)
        else:  # curled: tip folds back towards the palm
            pts[b + 1] = (mx, my - 0.04 * s)
            pts[b + 2] = (mx, my - 0.01 * s)
            pts[b + 3] = (mx, my + 0.02 * s)

    # thumb (on the left side of the image for this hand)
    pts[1] = (ox - 0.04 * s, oy - 0.03 * s)
    pts[2] = (ox - 0.07 * s, oy - 0.06 * s)
    if thumb and thumb_dir == "up":
        pts[2] = (ox - 0.06 * s, oy - 0.12 * s)
        pts[3] = (ox - 0.06 * s, oy - 0.18 * s)
        pts[4] = (ox - 0.06 * s, oy - 0.24 * s)
    elif thumb and thumb_dir == "down":
        pts[2] = (ox - 0.06 * s, oy - 0.10 * s)
        pts[3] = (ox - 0.06 * s, oy - 0.03 * s)
        pts[4] = (ox - 0.06 * s, oy + 0.03 * s)
    elif thumb:
        pts[3] = (ox - 0.11 * s, oy - 0.09 * s)
        pts[4] = (ox - 0.15 * s, oy - 0.12 * s)
    else:  # tucked across the palm
        pts[3] = (ox - 0.05 * s, oy - 0.10 * s)
        pts[4] = (ox - 0.01 * s, oy - 0.13 * s)

    if pinch:
        tip = pts[BASE[pinch] + 3]
        pts[4] = (tip[0] - 0.005, tip[1] + 0.005)
        pts[3] = (tip[0] - 0.04 * s, tip[1] + 0.03 * s)
        # a pinching finger is bent forward, which shortens it in 2D
        b = BASE[pinch]
        mx, my = pts[b]
        pts[b + 3] = (mx - 0.01 * s, my - 0.07 * s)
        pts[4] = (pts[b + 3][0] - 0.005, pts[b + 3][1] + 0.005)
    return pts


def move(pts, dx, dy):
    return [(x + dx, y + dy) for x, y in pts]
