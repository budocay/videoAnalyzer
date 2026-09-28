"""Scoreboard reading (VLM) → point winners, and team identification (names ↔ shirt colours).

Scoreboard
- Located once: the VLM names the corner holding the score overlay on a match-camera frame.
- Read at the start of every rally shot on a full-resolution crop of that corner, as JSON rows
  {team, serving, values}. Last value = points (0/15/30/40/A or tie-break count), the others =
  games per set.
- Point winner of rally i = the team whose score advanced by exactly one point between the
  start of rally i and the start of rally i+1 (padel scoring, see rules.py). In a highlights
  montage many points are skipped: then the winner stays unknown rather than guessed.

Teams
- Close-up shots show names printed on shirts; the VLM returns {name, shirt colour}.
- Names are matched to scoreboard teams; colours to the torso-colour clusters of the tracks.
"""
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..extract import VideoInfo

POINT_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4, "AV": 4}

LOCATE_PROMPT = (
    "Sur cette image de retransmission de padel, où est incrusté le tableau de score (noms des "
    "équipes et points) ? Réponds uniquement par un de ces mots : haut_gauche, haut_droite, "
    "bas_gauche, bas_droite, aucun."
)
READ_PROMPT = (
    "Lis ce tableau de score de padel. Pour chaque ligne (une par équipe, de haut en bas), donne le "
    "nom affiché, si l'équipe est au service (icône/étoile/balle à côté du nom), et toutes les valeurs "
    "numériques de gauche à droite (jeux de chaque set, puis points du jeu en cours : 0, 15, 30, 40 ou A).\n"
    "Réponds uniquement avec un JSON sur une ligne :\n"
    '{"lignes": [{"equipe": "...", "service": true|false, "valeurs": ["...", "..."]}, '
    '{"equipe": "...", "service": true|false, "valeurs": ["...", "..."]}]}\n'
    'Si aucun tableau de score n\'est lisible : {"lignes": []}'
)
SHIRT_PROMPT = (
    "Regarde uniquement le joueur de padel au premier plan (le plus grand dans l'image). Son nom "
    "est-il lisible sur son maillot ou dans un bandeau à l'écran qui le désigne ? Si oui, donne ce nom et "
    "la couleur dominante de SON maillot. Réponds uniquement avec un JSON sur une ligne :\n"
    '{"nom": "NOM EN MAJUSCULES ou null", "couleur_maillot": "rouge|orange|noir|blanc|bleu|vert|'
    'jaune|rose|gris|violet|autre"}'
)

PALETTE = {
    "rouge": (200, 50, 45), "orange": (230, 110, 50), "noir": (45, 45, 50), "blanc": (225, 225, 225),
    "bleu": (40, 70, 190), "vert": (50, 150, 70), "jaune": (230, 210, 60), "rose": (230, 120, 170),
    "gris": (130, 130, 130), "violet": (120, 60, 160),
}
CORNERS = {"haut_gauche": (0, 0), "haut_droite": (1, 0), "bas_gauche": (0, 1), "bas_droite": (1, 1)}


@dataclass
class ScoreState:
    teams: list[str]
    serving: list[bool]
    games: list[list[int]]  # per team, games per set (last = current set)
    points: list[str]       # per team, current game points

    def key(self):
        return (tuple(tuple(g) for g in self.games), tuple(self.points))

    def tiebreak(self) -> bool:
        return self.games[0][-1] == self.games[1][-1] == 6

    def valid(self) -> bool:
        if len(self.games[0]) == 0 or len(self.games[0]) > 3:
            return False
        if self.tiebreak():
            return all(p.isdigit() for p in self.points)
        return all(p in POINT_ORDER for p in self.points)

    def progress(self) -> tuple:
        """Monotonic key: (sets played, total games in current set)."""
        return (len(self.games[0]), self.games[0][-1] + self.games[1][-1])


def clean_sequence(states: list["ScoreState | None"]) -> list["ScoreState | None"]:
    """Drop invalid reads and reads that go backwards compared with both neighbours."""
    out = [s if s is not None and s.valid() else None for s in states]
    for i, s in enumerate(out):
        if s is None:
            continue
        prev = next((out[j] for j in range(i - 1, -1, -1) if out[j] is not None), None)
        nxt = next((out[j] for j in range(i + 1, len(out)) if out[j] is not None), None)
        if (prev and s.progress() < prev.progress()) or (nxt and s.progress() > nxt.progress()):
            if prev and nxt and prev.progress() <= nxt.progress():
                out[i] = None
    return out


def score_line(states: list["ScoreState | None"]) -> str:
    """Set-by-set story from the last valid read, e.g. 'set 1 : 6-3 · set 2 : 4-6 · set 3 : 6-6 (tie-break 6-1)'."""
    last = next((s for s in reversed(states) if s is not None), None)
    if last is None:
        return ""
    parts = [f"set {i + 1} : {a}-{b}" for i, (a, b) in enumerate(zip(*last.games))]
    tail = f"tie-break {last.points[0]}-{last.points[1]}" if last.tiebreak() else f"jeu en cours {last.points[0]}-{last.points[1]}"
    return f"{last.teams[0]} vs {last.teams[1]} — " + " · ".join(parts) + f" ({tail}, dernier score lu)"


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _frame(info: VideoInfo, t: float) -> np.ndarray | None:
    w, h = info.width, info.height
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(info.path), "-map", "0:v:0",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    if len(raw) < w * h * 3:
        return None
    return np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3)


def _save(img: np.ndarray, path: Path, width: int | None = None) -> Path:
    from PIL import Image

    im = Image.fromarray(img)
    if width:
        im = im.resize((width, round(im.height * width / im.width)))
    im.save(path, quality=92)
    return path


def locate(vlm, info: VideoInfo, t: float, tmp: Path) -> str | None:
    f = _frame(info, t)
    if f is None:
        return None
    a = vlm.ask(LOCATE_PROMPT, [_save(f, tmp / "locate.jpg", 1280)], max_tokens=10)
    word = a.text.strip().lower().split()[0] if a.text.strip() else ""
    return word if word in CORNERS else None


def _crop(f: np.ndarray, corner: str) -> np.ndarray:
    h, w = f.shape[:2]
    cx, cy = CORNERS[corner]
    x0, y0 = int(cx * w * 0.55), int(cy * h * 0.75)
    return f[y0:y0 + int(h * 0.25), x0:x0 + int(w * 0.45)]


def parse_rows(rows: list[dict]) -> ScoreState | None:
    if len(rows) != 2:
        return None
    teams, serving, games, points = [], [], [], []
    for r in rows:
        vals = [str(v).strip().upper() for v in r.get("valeurs", []) if str(v).strip()]
        if not vals:
            return None
        teams.append(str(r.get("equipe", "")).strip().upper())
        serving.append(bool(r.get("service")))
        points.append(vals[-1])
        try:
            games.append([int(v) for v in vals[:-1]])
        except ValueError:
            return None
    if len(games[0]) != len(games[1]):
        return None
    return ScoreState(teams, serving, games, points)


def read(vlm, info: VideoInfo, t: float, corner: str, tmp: Path) -> ScoreState | None:
    f = _frame(info, t)
    if f is None:
        return None
    a = vlm.ask(READ_PROMPT, [_save(_crop(f, corner), tmp / "score.jpg")], max_tokens=150)
    return parse_rows(_json(a.text).get("lignes", []))


def point_winner(a: ScoreState, b: ScoreState) -> int | None:
    """Index of the team that won the single point separating a from b, else None."""
    if a is None or b is None:
        return None
    ga, gb = a.games, b.games
    if ga == gb:  # same game
        pa = [POINT_ORDER.get(p) for p in a.points]
        pb = [POINT_ORDER.get(p) for p in b.points]
        if None in pa or None in pb:  # tie-break: plain integers
            try:
                pa, pb = [int(p) for p in a.points], [int(p) for p in b.points]
            except ValueError:
                return None
            d = [pb[0] - pa[0], pb[1] - pa[1]]
            return 0 if d == [1, 0] else 1 if d == [0, 1] else None
        d = [pb[0] - pa[0], pb[1] - pa[1]]
        if d == [1, 0]:
            return 0
        if d == [0, 1]:
            return 1
        # advantage lost: A-40 → 40-40
        if d == [-1, 0] and pa[0] == 4:
            return 1
        if d == [0, -1] and pa[1] == 4:
            return 0
        return None
    # a game was won: points reset, one team's games +1 in the same set (or a new set started)
    if all(p in ("0",) for p in b.points):
        if len(ga[0]) == len(gb[0]):
            d = [gb[0][-1] - ga[0][-1], gb[1][-1] - ga[1][-1]]
            return 0 if d == [1, 0] else 1 if d == [0, 1] else None
        if len(gb[0]) == len(ga[0]) + 1:  # set finished: last game of previous set decides
            d = [gb[0][-2] - ga[0][-1], gb[1][-2] - ga[1][-1]]
            return 0 if d == [1, 0] else 1 if d == [0, 1] else None
    return None


def _norm(name: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", name.upper()) if unicodedata.category(c) != "Mn")


def identify_players(vlm, shots, tmp: Path, max_shots: int = 30) -> list[dict]:
    """Raw observations {nom, couleur} of the foreground player in close-ups (noisy: kept all)."""
    obs = []
    for s in [s for s in shots if s.kind == "gros_plan" and s.keyframes][:max_shots]:
        a = vlm.ask(SHIRT_PROMPT, [Path(s.keyframes[1])], max_tokens=40)
        j = _json(a.text)
        name, col = _norm(str(j.get("nom") or "")).strip(), str(j.get("couleur_maillot", "")).strip().lower()
        if len(name) >= 3 and name != "NULL" and col in PALETTE:
            obs.append({"nom": name, "couleur": col, "plan": s.id})
    return obs


def team_votes(team: str, obs: list[dict]) -> list[str]:
    """Colour words observed for players whose surname appears in the scoreboard team name."""
    parts = [p for p in re.split(r"[ /\-]+", _norm(team)) if len(p) >= 4]
    return [o["couleur"] for o in obs if any(p in o["nom"].split() or o["nom"].endswith(p) for p in parts)]


def assign_teams(teams: list[str], obs: list[dict], clusters: list[np.ndarray]) -> tuple[list[str], int]:
    """Order `teams` to match `clusters` (2 torso-colour centroids). Picks the permutation that agrees
    with most VLM observations; a vote agrees with a cluster when its colour word is closer to it."""
    if len(teams) != 2 or len(clusters) != 2:
        return teams, 0

    def agree(team, cluster, other):
        return sum(rgb_distance(cluster, c) < rgb_distance(other, c) for c in team_votes(team, obs))

    keep = agree(teams[0], clusters[0], clusters[1]) + agree(teams[1], clusters[1], clusters[0])
    swap = agree(teams[1], clusters[0], clusters[1]) + agree(teams[0], clusters[1], clusters[0])
    return (list(teams), keep) if keep >= swap else ([teams[1], teams[0]], swap)


def rgb_distance(rgb: np.ndarray, word: str) -> float:
    return float(np.linalg.norm(np.asarray(rgb, float) - np.array(PALETTE[word], float)))


def colour_word(rgb: np.ndarray) -> str:
    return min(PALETTE, key=lambda w: rgb_distance(rgb, w))


def score_story(states: list["ScoreState | None"]) -> str:
    """Plain sentences naming who leads/won each set, from the last valid read (the 9B VLM misreads
    compact score notations, so the narrative is generated here, not by the model)."""
    last = next((s for s in reversed(states) if s is not None), None)
    if last is None:
        return "Tableau de score illisible."
    a, b = last.teams
    out = []
    n = len(last.games[0])
    for i, (ga, gb) in enumerate(zip(*last.games)):
        current = i == n - 1
        if not current:
            w = a if ga > gb else b
            out.append(f"Set {i + 1} gagné par {w} ({max(ga, gb)}-{min(ga, gb)}).")
        elif last.tiebreak():
            pa, pb = int(last.points[0]), int(last.points[1])
            lead = a if pa > pb else b if pb > pa else None
            out.append(f"Set {i + 1} à 6-6, tie-break en cours : {a} {pa}, {b} {pb}"
                       + (f" ({lead} mène)." if lead else "."))
        else:
            lead = a if ga > gb else b if gb > ga else None
            out.append(f"Set {i + 1} en cours : {a} {ga} jeux, {b} {gb} jeux"
                       + (f" ({lead} mène)." if lead else "."))
    out.append("C'est le dernier score visible dans la vidéo : la fin du match n'y figure pas forcément.")
    return " ".join(out)
