"""Shot segmentation + VLM shot classification.

1. Cuts: one low-res full-rate decode, 512-bin RGB histogram per frame, total-variation distance
   between consecutive frames. On the Paris final the distribution is bimodal (p99 = 0.12,
   p99.5 = 0.50), so a fixed threshold of 0.3 is safe; cuts closer than MIN_SHOT_S are merged.
   Plus jump cuts inside a fixed camera (condensed matches), from isolated motion spikes: see jump_cuts.
2. Classification: 3 keyframes per shot go to the local VLM, which answers in JSON (shot type,
   live play vs replay/slow motion, scoreboard visible). Only `principale` + live shots are
   analysed as rallies; everything else (close-ups, crowd, replays, graphics…) is "parasite".
"""
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..extract import VideoInfo
from .video_io import stream

CUT_THRESHOLD = 0.3
MIN_SHOT_S = 0.6
LOW_W, LOW_H = 96, 54
MIN_REF_CORR = 0.6  # Pearson correlation of 64x36 gray frames with the match-camera reference
# (Paris final: match camera 0.73–0.92 — shadows move during the match —, everything else ≤ 0.45)
SCOREBOARD_MAJORITY = 0.7
JUMP_RATIO, JUMP_FLOOR, JUMP_ISOLATION, MIN_JUMP_SHOT_S = 6.0, 1.5, 2.5, 2.0

SHOT_TYPES = {
    "principale": ("caméra de match : plan large fixe filmé en hauteur derrière un fond de court, tout le "
                   "terrain visible en perspective, le filet horizontal au milieu de l'image, 4 joueurs en jeu"),
    "autre_angle": ("terrain ou joueurs vus depuis un autre point de vue : vue aérienne ou drone, caméra au "
                    "niveau du filet ou du sol, vue de côté, caméra mobile, présentation des joueurs"),
    "gros_plan": "gros plan ou plan moyen sur un ou deux joueurs",
    "public": "public, tribunes, banc, staff",
    "graphique": "écran graphique, titre, carton, publicité, logo",
    "autre": "autre chose",
}

CLASSIFY_PROMPT = (
    "Voici 3 images d'un même plan d'une retransmission de padel (début, milieu, fin du plan).\n"
    "Classe ce plan. Types possibles :\n{types}\n"
    "Réponds uniquement avec un objet JSON sur une ligne, sans texte autour :\n"
    '{{"type": "<un des types>", "ralenti_ou_replay": true|false, "score_affiche": true|false}}\n'
    "ralenti_ou_replay = true si c'est une rediffusion ou un ralenti (logo ou bandeau REPLAY, flou de "
    "mouvement exagéré, transition de replay). score_affiche = true si un tableau de score est incrusté."
)


@dataclass
class Shot:
    id: int
    start_f: int
    end_f: int  # exclusive
    start: float
    end: float
    kind: str = "?"
    replay: bool = False
    scoreboard: bool = False
    motion: float = 0.0  # mean abs gray diff per frame (0..255), low in slow motion
    keyframes: list[str] = field(default_factory=list)

    @property
    def rally(self) -> bool:
        return self.kind == "principale" and not self.replay

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_signals(info: VideoInfo) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame histogram distance to the previous frame, and gray motion energy."""
    dist, motion = [0.0], [0.0]
    prev_h = prev_g = None
    for f in stream(info, LOW_W, LOW_H):
        q = (f >> 5).astype(np.int32)
        h = np.bincount((q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]).ravel(), minlength=512)
        h = h / (LOW_W * LOW_H)
        g = f.mean(axis=2)
        if prev_h is not None:
            dist.append(float(np.abs(h - prev_h).sum() / 2))
            motion.append(float(np.abs(g - prev_g).mean()))
        prev_h, prev_g = h, g
    return np.array(dist), np.array(motion)


def jump_cuts(motion: np.ndarray, fps: float) -> list[int]:
    """Cuts inside a fixed camera (condensed matches: dead time removed, same framing on both sides). The colour
    histogram barely moves (0.03–0.09 vs the 0.3 threshold) but the players teleport: a one-frame spike of motion
    energy, ≥ JUMP_RATIO × the local median and ≥ JUMP_ISOLATION × its neighbours (a pan or zoom lasts several
    frames). Lyon stream: 77 cuts, one per point, stable for ratios 6–8."""
    from scipy.ndimage import median_filter

    base = median_filter(motion, size=31)
    out: list[int] = []
    for i in range(2, len(motion) - 2):
        m = motion[i]
        nb = max(motion[i - 2], motion[i - 1], motion[i + 1], motion[i + 2])
        if m > JUMP_RATIO * max(base[i], 0.3) and m > JUMP_FLOOR and m > JUMP_ISOLATION * nb:
            if not out or i - out[-1] >= MIN_JUMP_SHOT_S * fps:  # flashes / graphic transitions come in bursts
                out.append(i)
    return out


def split_shots(dist: np.ndarray, fps: float, threshold: float = CUT_THRESHOLD,
                min_shot_s: float = MIN_SHOT_S, extra_cuts=()) -> list[tuple[int, int]]:
    """Frame ranges [start, end) between cuts; very short fragments are merged into the next shot."""
    n = len(dist)
    cuts = sorted({int(i) for i in np.where(dist > threshold)[0]} | {int(i) for i in extra_cuts})
    bounds = [0]
    for c in cuts:
        if c - bounds[-1] >= min_shot_s * fps:
            bounds.append(c)
    if n - bounds[-1] < min_shot_s * fps and len(bounds) > 1:
        bounds.pop()
    bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))


def _extract_keyframes(info: VideoInfo, shots: list[Shot], outdir: Path, width: int = 640) -> None:
    """One decode pass, grabbing frames at 20/50/80 % of each shot."""
    outdir.mkdir(parents=True, exist_ok=True)
    wanted: dict[int, list[tuple[Shot, int]]] = {}
    for s in shots:
        span = s.end_f - s.start_f
        for k, r in enumerate((0.2, 0.5, 0.8)):
            wanted.setdefault(s.start_f + int(span * r), []).append((s, k))
    height = round(info.height * width / info.width / 2) * 2
    from PIL import Image

    last = max(wanted)
    for n, f in enumerate(stream(info, width, height)):
        for s, k in wanted.get(n, []):
            p = outdir / f"shot{s.id:04d}_{k}.jpg"
            Image.fromarray(f).save(p, quality=90)
            s.keyframes.append(str(p))
        if n >= last:
            break


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _signature(path: str) -> np.ndarray:
    from PIL import Image

    g = np.asarray(Image.open(path).convert("L").resize((64, 36)), dtype=np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def refine_with_reference(shots: list[Shot], min_corr: float = MIN_REF_CORR) -> np.ndarray | None:
    """The match camera never moves and players are small, so its frames are near-identical.
    Build a reference from the longest VLM 'principale' shots and demote look-alikes that differ."""
    cands = sorted((s for s in shots if s.kind == "principale" and s.keyframes),
                   key=lambda s: s.duration, reverse=True)[:8]
    if len(cands) < 2:
        return None
    ref = np.median([_signature(s.keyframes[1]) for s in cands], axis=0)
    for s in shots:
        if s.kind != "principale" or not s.keyframes:
            continue
        corr = max(float((_signature(k) * ref).mean()) for k in s.keyframes)
        if corr < min_corr:
            s.kind = "autre_angle"
    # Broadcasts show the score overlay during live play and hide it in replays: if most
    # match-camera shots carry it, a match-camera shot without it is a replay.
    main = [s for s in shots if s.kind == "principale"]
    if main and sum(s.scoreboard for s in main) / len(main) >= SCOREBOARD_MAJORITY:
        for s in main:
            if not s.scoreboard:
                s.replay = True
    return ref


def detect(info: VideoInfo, cache: Path, log=print) -> list[Shot]:
    """Cuts + keyframes (no model). Cached in cache/shots.json."""
    path = cache / "shots_raw.json"
    if path.exists():
        return [Shot(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
    if (cache / "frame_dist.npy").exists() and (cache / "frame_motion.npy").exists():
        dist, motion = np.load(cache / "frame_dist.npy"), np.load(cache / "frame_motion.npy")
    else:
        dist, motion = frame_signals(info)
        np.save(cache / "frame_dist.npy", dist)
        np.save(cache / "frame_motion.npy", motion)
    fps = info.native_fps
    jumps = jump_cuts(motion, fps)
    shots = [Shot(i, a, b, round(a / fps, 3), round(b / fps, 3), motion=round(float(motion[a + 1:b].mean()) if b - a > 1 else 0.0, 3))
             for i, (a, b) in enumerate(split_shots(dist, fps, extra_cuts=jumps))]
    _extract_keyframes(info, shots, cache / "keyframes")
    path.write_text(json.dumps([asdict(s) for s in shots], indent=1), encoding="utf-8")
    log(f"  {len(shots)} plans détectés (dont {len(jumps)} coupes franches à caméra fixe)")
    return shots


def classify(vlm, shots: list[Shot], cache: Path, log=print) -> list[Shot]:
    """VLM classification of every shot. Cached per shot in cache/shots.json."""
    path = cache / "shots.json"
    done = {d["id"]: d for d in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    types = "\n".join(f"- {k} : {v}" for k, v in SHOT_TYPES.items())
    for s in shots:
        if s.id in done:
            s.kind, s.replay, s.scoreboard = done[s.id]["kind"], done[s.id]["replay"], done[s.id]["scoreboard"]
            continue
        a = vlm.ask(CLASSIFY_PROMPT.format(types=types), [Path(p) for p in s.keyframes], max_tokens=60)
        r = _parse(a.text)
        kind = r.get("type") if r.get("type") in SHOT_TYPES else "autre"
        s.kind, s.replay, s.scoreboard = kind, bool(r.get("ralenti_ou_replay")), bool(r.get("score_affiche"))
        log(f"  plan {s.id:3d} {s.start:7.1f}-{s.end:7.1f}s {kind:11} replay={s.replay!s:5} "
            f"score={s.scoreboard!s:5} mvt={s.motion:5.2f} ({a.seconds:.1f} s)")
        done[s.id] = asdict(s)
        path.write_text(json.dumps(list(done.values()), indent=1), encoding="utf-8")
    refine_with_reference(shots)
    return shots
