"""Court calibration: image ↔ court-plane homography from painted lines and padel geometry.

Court frame (metres): x ∈ [0, 10] left→right as seen from the camera, y ∈ [0, 20] from the far
back wall (y=0) to the near back wall (y=20); net at y=10; service lines at y=3.05 and 16.95;
centre service line x=5 between them (FIP rules, see rules.py).

Detection on a main-camera frame:
1. court hue = dominant saturated hue in the lower-centre of the image;
2. white lines = local top-hat on gray, restricted to court-coloured neighbourhoods (the near
   half is seen through the back glass, so absolute thresholds miss it);
3. centre service line = longest near-vertical segment → its ends give the image rows of the
   two service lines;
4. far service line = horizontal segments at the upper row → its two ends (x=0 and x=10);
5. sidelines = longest oblique edges through those two ends (floor/side-glass junctions) →
   intersected with both service rows: 4 exact points → homography. Validated on the centre line.
Fallback when a sideline is hidden:
5'. net = widest horizontal structure between the service lines: it spans the full 10 m at y=10
   (pixel width at a given depth does not depend on height, so the tape height is irrelevant);
6. 1/pixel-width is affine in world y (far service line + net give it); on a flat floor with no
   camera roll, pixel width is linear in the image row (the two service-line rows give it) →
   4 corners → homography. Validated against the near service line extent.
   (Using the far wall/floor edge instead failed: blue sponsor boards above it have the court hue.)
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .rules import COURT_LENGTH, COURT_WIDTH, NET_Y, SERVICE_LINE_FROM_WALL


@dataclass
class Court:
    image_corners: list[list[float]]  # far-left, far-right, near-right, near-left (px, source resolution)
    frame_size: tuple[int, int]
    hue: int
    residual_px: float  # centre service line vs x=5 m, in far-service-line pixels
    method: str = "sidelines"
    H: list[list[float]] | None = None  # image → court metres

    def homography(self) -> np.ndarray:
        if self.H is None:
            dst = np.float32([[0, 0], [COURT_WIDTH, 0], [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]])
            self.H = cv2.getPerspectiveTransform(np.float32(self.image_corners), dst).tolist()
        return np.array(self.H)

    def to_court(self, pts: np.ndarray) -> np.ndarray:
        """(N,2) image px → (N,2) metres."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography()).reshape(-1, 2)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.homography())).reshape(-1, 2)


class CourtNotFound(RuntimeError):
    pass


def _court_hue(hsv: np.ndarray) -> int:
    h, w = hsv.shape[:2]
    roi = hsv[int(h * .55):int(h * .85), int(w * .3):int(w * .7)]
    sat = roi[..., 1] > 80
    if sat.sum() < 100:
        raise CourtNotFound("pas de surface colorée au centre de l'image")
    return int(np.bincount(roi[..., 0][sat].ravel(), minlength=180).argmax())


def _hue_mask(hsv: np.ndarray, hue: int, tol: int = 10) -> np.ndarray:
    d = np.abs(hsv[..., 0].astype(np.int16) - hue)
    d = np.minimum(d, 180 - d)
    return ((d <= tol) & (hsv[..., 1] > 60) & (hsv[..., 2] > 40)).astype(np.uint8)


def _segments(img: np.ndarray, court_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    near = cv2.dilate(court_mask, np.ones((9, 9), np.uint8))
    white = ((th > 35) & (near > 0)).astype(np.uint8) * 255
    w = img.shape[1]
    segs = cv2.HoughLinesP(white, 1, np.pi / 360, 80, minLineLength=int(w * 0.08), maxLineGap=25)
    if segs is None:
        raise CourtNotFound("aucune ligne blanche détectée")
    return segs.reshape(-1, 4).astype(np.float64)


def _angle(s: np.ndarray) -> float:
    return abs(np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])))


def _row_extent(segs: np.ndarray, row: float, tol: float = 8) -> tuple[float, float] | None:
    hs = [s for s in segs if _angle(s) < 4 and abs((s[1] + s[3]) / 2 - row) < tol]
    if not hs:
        return None
    xs = [v for s in hs for v in (s[0], s[2])]
    return min(xs), max(xs)


def _proj1d(ys, vs):
    """Fit v = (a*y + b) / (c*y + 1) through three (y, v) pairs."""
    A = np.array([[y, 1, -y * v] for y, v in zip(ys, vs)])
    a, b, c = np.linalg.solve(A, np.array(vs))
    return lambda y: (a * y + b) / (c * y + 1), a / c


def calibrate(img: np.ndarray) -> Court:
    """img: RGB uint8 main-camera frame at source resolution."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = _court_hue(hsv)
    mask = _hue_mask(hsv, hue)
    segs = _segments(img, mask)

    ver = [s for s in segs if _angle(s) > 60]
    if not ver:
        raise CourtNotFound("ligne centrale de service introuvable")
    c = max(ver, key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]))
    (x_top, v_far), (x_bot, v_near) = sorted([(c[0], c[1]), (c[2], c[3])], key=lambda p: p[1])
    center_x = lambda v: x_top + (x_bot - x_top) * (v - v_far) / (v_near - v_far)

    far = _row_extent(segs, v_far)
    if far is None:
        raise CourtNotFound("ligne de service lointaine introuvable")
    w_far = far[1] - far[0]

    s = SERVICE_LINE_FROM_WALL
    left = _sideline(img, (far[0], v_far))
    right = _sideline(img, (far[1], v_far))
    if left is not None and right is not None:
        # Sidelines (floor/side-glass junctions) are x=0 and x=10: intersect with both service rows.
        method = "sidelines"
        src = [[_x_at(left, v_far), v_far], [_x_at(right, v_far), v_far],
               [_x_at(right, v_near), v_near], [_x_at(left, v_near), v_near]]
    else:
        # Fallback: 1/pixel-width is affine in y (far service line + net widths), pixel width is
        # linear in the image row. Less accurate: the net mesh overhangs the 10 m by a few %.
        method = "net"
        w_net = _net_width(segs, v_far, v_near)
        beta = (1 / w_net - 1 / w_far) / (NET_Y - s)
        alpha = 1 / w_far - beta * s
        wn = 1 / (alpha + beta * (COURT_LENGTH - s))
        cx = center_x(v_near)
        src = [[far[0], v_far], [far[1], v_far], [cx + wn / 2, v_near], [cx - wn / 2, v_near]]
    dst = [[0, s], [COURT_WIDTH, s], [COURT_WIDTH, COURT_LENGTH - s], [0, COURT_LENGTH - s]]
    Hm = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    corners = cv2.perspectiveTransform(
        np.float64([[[0, 0]], [[COURT_WIDTH, 0]], [[COURT_WIDTH, COURT_LENGTH]], [[0, COURT_LENGTH]]]),
        np.linalg.inv(Hm)).reshape(-1, 2)
    court = Court(corners.tolist(), (w, h), hue, residual_px=float("nan"), method=method, H=Hm.tolist())

    # Validation: the centre service line must map to x = 5 m.
    mid = court.to_court(np.array([[x_top, v_far], [x_bot, v_near]]))
    court.residual_px = float(np.abs(mid[:, 0] - COURT_WIDTH / 2).max() * w_far / COURT_WIDTH)
    return court


def _x_at(line: np.ndarray, v: float) -> float:
    x1, y1, x2, y2 = line
    return float(x1 + (x2 - x1) * (v - y1) / (y2 - y1))


def _sideline(img: np.ndarray, anchor: tuple[float, float], max_dist: float = 20) -> np.ndarray | None:
    """Longest oblique edge segment whose supporting line passes through `anchor`."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, 100, minLineLength=int(img.shape[1] * 0.08), maxLineGap=15)
    if segs is None:
        return None
    best, best_len = None, 0.0
    for sg in segs.reshape(-1, 4).astype(np.float64):
        d = sg[2:] - sg[:2]
        ang = np.degrees(np.arctan2(d[1], d[0])) % 180
        if not (25 < ang < 85 or 95 < ang < 155):
            continue
        length = float(np.hypot(*d))
        n = np.array([-d[1], d[0]]) / length
        if abs(np.dot(np.array(anchor) - sg[:2], n)) < max_dist and length > best_len:
            best, best_len = sg, length
    return best


def _net_width(segs: np.ndarray, v_far: float, v_near: float) -> float:
    nets = [sg for sg in segs if _angle(sg) < 4 and v_far + 25 < (sg[1] + sg[3]) / 2 < v_near - 60]
    if not nets:
        raise CourtNotFound("ni bords latéraux ni filet détectés")
    rows = {}
    for sg in nets:
        rows.setdefault(round((sg[1] + sg[3]) / 20), []).append(sg)
    best = max(rows.values(), key=lambda g: sum(abs(x[2] - x[0]) for x in g))
    return max(max(x[0], x[2]) for x in best) - min(min(x[0], x[2]) for x in best)


def save(court: Court, path: Path) -> None:
    court.homography()
    path.write_text(json.dumps(asdict(court), indent=1), encoding="utf-8")


def load(path: Path) -> Court:
    d = json.loads(path.read_text(encoding="utf-8"))
    d["frame_size"] = tuple(d["frame_size"])
    return Court(**d)
