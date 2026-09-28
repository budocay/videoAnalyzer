"""Hit detection and stroke typing.

Hits
- Candidates: ball-impact sounds (band-pass 1.5–6 kHz onset peaks). They include bounces and
  glass hits, so each one must be confirmed by a racket-arm wrist-speed peak of a player.
- Rules: consecutive hits alternate between the two teams (the ball must cross the net).
  A Viterbi pass over the candidates picks, for each one, {far team, near team, not a hit}
  maximising wrist evidence under that constraint.
- Without usable audio (music, commentary), wrist-speed peaks become the candidates.

Stroke type (pose at impact, see rules.STROKES)
- racket arm = the wrist that moves fastest around impact; YOLO labels anatomical left/right, so
  forehand vs backhand = side of the impact relative to the hip axis, independent of camera
  orientation and handedness.
- overhead = wrist above the head; volley = not overhead and < 4.5 m from the net;
  service = first hit of a shot starting at serve, hitter behind the service line, wrist low.
- overhead subtype (bandeja / víbora / smash): heuristic on wrist speed and distance to the
  net, optionally refined by the VLM on crops (vision is better at spin/shape than our 17 points).
"""
import subprocess
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, find_peaks, medfilt, sosfiltfilt

from ..extract import VideoInfo
from .players import ShotTracks
from .rules import NET_Y, SERVICE_LINE_FROM_NET

AUDIO_SR = 22050
HOP_S = 0.01
WRIST_WINDOW = (-0.20, 0.10)  # seconds around the sound where the swing peak may sit
NOT_HIT_SCORE = 3.0           # wrist speed / player's own median below which a sound is a bounce/glass
MIN_GAP_S = 0.45              # two hits can't be closer than this
VOLLEY_MAX_DIST = 4.5         # metres from the net


@dataclass
class Hit:
    shot_id: int
    t: float          # absolute video time (s)
    frame: int        # frame index inside the shot
    slot: int         # 0,1 far side · 2,3 near side
    side: str         # "fond" (far) / "proche" (near), as seen from the camera
    x: float
    y: float          # court metres
    dist_net: float
    speed: float      # wrist speed / body height, per frame
    evidence: float   # speed / the player's median speed in the shot (noise-floor normalised)
    arm: str          # "droit" / "gauche" (anatomical racket arm)
    stroke: str = "?"
    confidence: float = 0.5
    source: str = "audio+pose"
    height: str = ""  # contact height from pose: "bas" (< shoulder), "haut" (shoulder..head), "tete" (above head)
    stroke_pose: str = ""  # label of each method, kept separately for evaluation
    stroke_vlm: str = ""
    stroke_clf: str = ""
    clf_prob: float = 0.0

    @property
    def key(self) -> str:
        """Stable id of a hit inside a video: shot, frame in shot, player slot."""
        return f"{self.shot_id}:{self.frame}:{self.slot}"


def audio_onsets(info: VideoInfo, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                          "-i", str(info.path), "-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    if len(a) < AUDIO_SR // 2:
        return np.array([]), np.array([])
    sos = butter(4, [1500, 6000], btype="band", fs=AUDIO_SR, output="sos")
    b = sosfiltfilt(sos, a)
    hop = int(AUDIO_SR * HOP_S)
    n = len(b) // hop
    env = np.sqrt((b[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    on = np.maximum(0, np.diff(env, prepend=env[0]))
    on = on / (np.median(on) + 1e-9)
    peaks, props = find_peaks(on, height=max(np.percentile(on, 95), 8.0), distance=int(0.25 / HOP_S))
    return peaks * HOP_S, props["peak_heights"]


def wrist_speed(tr: ShotTracks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, 4) speed of the fastest wrist (body heights per frame), the same divided by each
    player's median (far players' keypoints jitter more, which inflates raw speed), and (T, 4)
    index of that wrist (0=left, 1=right)."""
    k = tr.kpts
    T = len(k)
    speed = np.zeros((T, 4))
    arm = np.zeros((T, 4), dtype=int)
    if T < 3:
        return speed, speed, arm
    for s in range(4):
        w, c = k[:, s, [9, 10], :2], k[:, s, [9, 10], 2]
        height = np.linalg.norm(k[:, s, 0, :2] - k[:, s, [15, 16], :2].mean(axis=1), axis=1)
        height[height < 20] = np.nan
        v = np.linalg.norm(np.diff(w, axis=0), axis=2)
        v[(c[1:] < 0.3) | (c[:-1] < 0.3)] = 0
        v = v / height[1:, None]
        v = np.nan_to_num(v)
        arm[1:, s] = v.argmax(axis=1)
        speed[1:, s] = medfilt(v.max(axis=1), 3)
    med = np.array([np.median(speed[speed[:, s] > 0, s]) if (speed[:, s] > 0).any() else 1.0
                    for s in range(4)])
    return speed, speed / np.maximum(med, 1e-3), arm


def _viterbi(cands: list[tuple[float, np.ndarray]]) -> list[int | None]:
    """cands: (time, score per team [far, near]). Returns chosen team per candidate or None.
    Constraint: chosen hits alternate teams and are >= MIN_GAP_S apart."""
    n = len(cands)
    if n == 0:
        return []
    # state: last team that hit (0/1) or 2 = nobody yet; value = best score; keep last hit time
    best = {2: (0.0, [], -1e9)}
    for t, sc in cands:
        nxt = {}
        for last, (val, path, t_last) in best.items():
            opts = [(None, NOT_HIT_SCORE, last, t_last)]
            for team in (0, 1):
                if team != last and t - t_last >= MIN_GAP_S:
                    opts.append((team, sc[team], team, t))
            for choice, gain, new_last, new_t in opts:
                v = val + gain
                if new_last not in nxt or v > nxt[new_last][0]:
                    nxt[new_last] = (v, path + [choice], new_t)
        best = nxt
    return max(best.values(), key=lambda x: x[0])[1]


def detect_hits(info: VideoInfo, tr: ShotTracks, duration: float) -> list[Hit]:
    raw, speed, arm = wrist_speed(tr)
    T = len(speed)
    times, strength = audio_onsets(info, tr.t0, tr.t0 + duration)
    source = "audio+pose"
    if len(times) < 2:
        source = "pose"
        team_speed = np.maximum(speed[:, :2].max(axis=1), speed[:, 2:].max(axis=1))
        pk, _ = find_peaks(team_speed, height=NOT_HIT_SCORE, distance=int(MIN_GAP_S * tr.fps))
        times = pk / tr.fps
    cands, detail = [], []
    for t in times:
        a = max(0, int((t + WRIST_WINDOW[0]) * tr.fps))
        b = min(T, int((t + WRIST_WINDOW[1]) * tr.fps) + 1)
        if b <= a:
            continue
        win = speed[a:b]
        per_slot = win.max(axis=0)
        sc = np.array([per_slot[:2].max(), per_slot[2:].max()])
        cands.append((t, sc))
        detail.append((t, a, win, per_slot))
    choice = _viterbi(cands)
    hits = []
    for (t, a, win, per_slot), team in zip(detail, choice):
        if team is None:
            continue
        slots = (0, 1) if team == 0 else (2, 3)
        slot = max(slots, key=lambda s: per_slot[s])
        f = a + int(win[:, slot].argmax())
        p = tr.pos[f, slot]
        if np.isnan(p).any():
            ok = np.where(np.isfinite(tr.pos[:, slot, 0]))[0]
            if len(ok) == 0:
                continue
            p = tr.pos[ok[np.abs(ok - f).argmin()], slot]
        hits.append(Hit(tr.shot_id, round(tr.t0 + t, 3), int(f), int(slot), "fond" if team == 0 else "proche",
                        round(float(p[0]), 2), round(float(p[1]), 2), round(abs(float(p[1]) - NET_Y), 2),
                        round(float(raw[f, slot]), 3), round(float(per_slot[slot]), 2),
                        "droit" if arm[f, slot] == 1 else "gauche",
                        source=source))
    return hits


def classify_stroke(hit: Hit, tr: ShotTracks, first_in_shot: bool, shot_start: float) -> None:
    k = tr.kpts[hit.frame, hit.slot]
    wrist = k[10] if hit.arm == "droit" else k[9]
    nose, lsh, rsh, lhip, rhip = k[0], k[5], k[6], k[11], k[12]
    head_y = nose[1] if nose[2] > 0.3 else min(lsh[1], rsh[1]) - 0.25 * abs(lhip[1] - lsh[1])
    body_h = max(abs(((lhip[1] + rhip[1]) / 2) - head_y), 1.0)
    # min wrist height in ±2 frames (±0.07 s) of the speed peak: a wider window catches the racket
    # preparation (armed high) and labelled 43 % of hits as overheads on the Paris final vs 36 %
    a, b = max(0, hit.frame - 2), min(len(tr.kpts), hit.frame + 3)
    wi = 10 if hit.arm == "droit" else 9
    top = tr.kpts[a:b, hit.slot, wi]
    top = top[top[:, 2] > 0.3]
    wrist_top_y = top[:, 1].min() if len(top) else wrist[1]
    overhead = wrist_top_y < head_y - 0.05 * body_h
    shoulder_y = min(lsh[1], rsh[1]) if min(lsh[2], rsh[2]) > 0.3 else head_y + 0.2 * body_h
    hit.height = "tete" if overhead else ("haut" if wrist_top_y < shoulder_y else "bas")

    hip_c = (lhip[:2] + rhip[:2]) / 2
    axis = rhip[:2] - lhip[:2]  # anatomical left → right
    side = float(np.dot(wrist[:2] - hip_c, axis))
    forehand = (side > 0) == (hit.arm == "droit")

    service_line = hit.dist_net >= SERVICE_LINE_FROM_NET - 0.5
    if first_in_shot and hit.t - shot_start < 2.5 and service_line and not overhead \
            and wrist[1] > (lhip[1] + rhip[1]) / 2 - 0.3 * body_h:
        hit.stroke, hit.confidence = "service", 0.6
    elif overhead:
        if hit.dist_net < 5.0 and hit.speed > 0.45:
            hit.stroke, hit.confidence = "smash", 0.5
        elif hit.dist_net >= 5.0 and hit.speed < 0.35:
            hit.stroke, hit.confidence = "bandeja", 0.45
        else:
            hit.stroke, hit.confidence = "vibora", 0.35
    elif hit.dist_net < VOLLEY_MAX_DIST:
        hit.stroke = "volee_cd" if forehand else "volee_revers"
        hit.confidence = 0.6
    else:
        hit.stroke = "coup_droit" if forehand else "revers"
        hit.confidence = 0.6
    hit.stroke_pose = hit.stroke


def hits_to_dicts(hits: list[Hit]) -> list[dict]:
    return [asdict(h) for h in hits]


# --- VLM verification / classification ----------------------------------------------------------

STROKE_PROMPT = (
    "Tu es analyste de padel. Voici 3 images recadrées sur un joueur, juste avant, pendant et juste "
    "après un instant où l'on entend l'impact de la balle (match professionnel, caméra de retransmission).\n"
    "Contexte mesuré : le joueur est à {dist:.1f} m du filet ({zone}), bras raquette présumé {arm}. "
    "Hauteur de frappe mesurée : {height}. Proposition automatique : {guess}.\n"
    "Types de coups compatibles avec la position et la hauteur mesurées (choisis parmi eux) :\n{defs}\n"
    "1) Ce joueur frappe-t-il la balle sur ces images ? 2) Si oui, quel coup ?\n"
    "Réponds uniquement avec un objet JSON sur une ligne :\n"
    '{{"frappe": true|false, "coup": "<clé du type ou null>", "confiance": <0 à 1>}}'
)


def _zone(dist: float) -> str:
    from .rules import NET_ZONE_MAX
    if dist < NET_ZONE_MAX:
        return "zone de volée, près du filet"
    if dist < SERVICE_LINE_FROM_NET:
        return "zone de transition"
    return "fond de court, derrière la ligne de service"


def hit_crops(info: VideoInfo, hit: Hit, tr: ShotTracks, size: int = 336,
              offsets: tuple[float, ...] = (-0.2, 0.0, 0.2), context: bool = False) -> list:
    """Square crops around the hitter at `offsets` seconds from the impact (full-resolution frames).
    With context=True, a downscaled full frame with the hitter boxed is prepended."""
    from PIL import Image, ImageDraw

    w, h = info.width, info.height
    lead = max(0.0, -min(offsets)) + 0.05
    span = lead + max(0.0, max(offsets)) + 0.1
    start = max(0.0, hit.t - lead)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
                          "-i", str(info.path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    frames = np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    k = tr.kpts[hit.frame, hit.slot]
    ok = k[:, 2] > 0.3
    if ok.sum() < 3 or n == 0:
        return []
    (x0, y0), (x1, y1) = k[ok, :2].min(axis=0), k[ok, :2].max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(y1 - y0, x1 - x0, 60) * 1.1  # room for the racket above the head
    box = (int(max(0, cx - half)), int(max(0, cy - half * 1.1)), int(min(w, cx + half)), int(min(h, cy + half * 0.9)))
    fps = tr.fps
    picks = [round((d + hit.t - start) * fps) for d in offsets]
    out = []
    if context:
        f = frames[min(max(round((hit.t - start) * fps), 0), n - 1)]
        im = Image.fromarray(f)
        ImageDraw.Draw(im).rectangle(box, outline=(255, 230, 0), width=6)
        out.append(im.resize((round(size * w / h), size)))
    for i in picks:
        f = frames[min(max(i, 0), n - 1)]
        out.append(Image.fromarray(f[box[1]:box[3], box[0]:box[2]]).resize((size, size)))
    return out


OVERHEAD = {"bandeja", "vibora", "smash"}
HEIGHT_TEXT = {"bas": "sous l'épaule", "haut": "entre l'épaule et la tête", "tete": "au-dessus de la tête"}


def allowed_strokes(hit: Hit) -> list[str]:
    """Strokes compatible with measured contact height and distance to the net (padel technique)."""
    from .rules import STROKES

    ok = set(STROKES)
    if hit.height == "bas":
        ok -= OVERHEAD
    elif hit.height == "tete":
        ok &= OVERHEAD
    if hit.height != "bas":
        ok.discard("service")  # padel serve is hit at or below the waist
    if hit.dist_net > 7.0:
        ok -= {"volee_cd", "volee_revers"}
    if hit.dist_net < 3.0:
        ok -= {"coup_droit", "revers", "lob", "sortie_vitre", "service"}
    return [k for k in STROKES if k in ok] or list(STROKES)


def refine_with_vlm(vlm, info: VideoInfo, hits: list[Hit], tr: ShotTracks, tmpdir,
                    refs: dict | None = None) -> list[Hit]:
    """Let the VLM confirm each hit and choose the stroke. Rejected hits are dropped."""
    import json as _json
    import re
    from pathlib import Path

    from .rules import STROKES

    kept = []
    for i, h in enumerate(hits):
        crops = hit_crops(info, h, tr)
        if not crops:
            kept.append(h)
            continue
        paths = []
        for j, im in enumerate(crops):
            p = Path(tmpdir) / f"hit_{h.shot_id}_{i}_{j}.jpg"
            im.save(p, quality=92)
            paths.append(p)
        allowed = allowed_strokes(h)
        defs = "\n".join(f"- {k} : {STROKES[k]}" for k in allowed)
        prompt = STROKE_PROMPT.format(dist=h.dist_net, zone=_zone(h.dist_net), arm=h.arm,
                                      height=HEIGHT_TEXT.get(h.height, "inconnue"), guess=h.stroke, defs=defs)
        ref_keys = [k for k in allowed if refs and k in refs][:4]
        if ref_keys:  # few-shot: validated examples first, then the 3 crops to classify
            prompt = (f"Les {len(ref_keys)} premières images sont des exemples validés, dans l'ordre : "
                      + ", ".join(ref_keys) + ". Les 3 dernières images sont le coup à analyser.\n" + prompt)
            paths = [refs[k] for k in ref_keys] + paths
        a = vlm.ask(prompt, paths, max_tokens=60)
        m = re.search(r"\{.*\}", a.text, flags=re.S)
        try:
            r = _json.loads(m.group(0)) if m else {}
        except _json.JSONDecodeError:
            r = {}
        if r.get("frappe") is False and h.evidence < 2 * NOT_HIT_SCORE:
            continue  # VLM says no swing and pose evidence is not overwhelming
        if r.get("coup") in STROKES:
            h.stroke_vlm = r["coup"]
        if r.get("coup") in allowed:
            # the VLM's own confidence is always ~0.95: use agreement with the pose heuristic instead
            h.confidence = 0.8 if r["coup"] == h.stroke else 0.6
            h.stroke = r["coup"]
            h.source += "+vlm"
        elif h.stroke not in allowed:
            h.stroke = allowed[0]
        kept.append(h)
    return kept


def reconcile(hit: Hit, tr: ShotTracks, first_in_shot: bool, shot_start: float) -> None:
    """Recompute pose height; keep the VLM label only if still compatible with it."""
    label, conf, source = hit.stroke, hit.confidence, hit.source
    if "vlm" in source and not hit.stroke_vlm:
        hit.stroke_vlm = label
    classify_stroke(hit, tr, first_in_shot, shot_start)  # sets height + heuristic label
    if "vlm" in source and label in allowed_strokes(hit):
        hit.stroke, hit.confidence = label, conf
    hit.source = source
