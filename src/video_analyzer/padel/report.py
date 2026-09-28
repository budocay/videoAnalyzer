"""Match statistics, heatmaps and Markdown report.

Players are named "<team> · côté droit|côté gauche": in padel each player keeps a side for the
whole match — right = the "drive" player, left = the "revés" player, *from the player's own
perspective* (facing the net). "drive"/"revés" are not used as labels: the VLM mistakes them for strokes. Far team faces the camera, so its right is image-left (x < 5); near team's right is x > 5.

Positions are expressed in the player's own half: depth = distance to the net (0..10 m),
lateral = 0 (own left wall) .. 10 (own right wall), so both halves overlay in one heatmap.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.signal import medfilt, savgol_filter

from .players import ShotTracks
from .rules import NET_Y, NET_ZONE_MAX, SERVICE_LINE_FROM_NET, STROKES

STROKE_LABELS = {k: v.split(" :")[0] for k, v in STROKES.items()}
# bandeja vs víbora hinges on spin/contact that broadcast resolution does not show reliably
# (the VLM said "víbora" 106 times vs 4 bandejas): report them together, keep the detail in JSON.
REPORT_STROKES = {"service": "service", "coup_droit": "coup droit", "revers": "revers",
                  "volee_cd": "volée CD", "volee_revers": "volée revers",
                  "bandeja_vibora": "bandeja / víbora", "smash": "smash", "lob": "lob",
                  "sortie_vitre": "sortie de vitre"}


def report_key(stroke: str) -> str:
    return "bandeja_vibora" if stroke in ("bandeja", "vibora") else stroke
MAX_GAP_FRAMES = 15  # interpolate position gaps up to 0.5 s; longer gaps are not counted


def slot_roles(tr: ShotTracks) -> dict[int, str]:
    """slot → 'côté droit' (drive player) / 'côté gauche' (revés player) from mean lateral position."""
    roles = {}
    for pair, far in (((0, 1), True), ((2, 3), False)):
        xs = [np.nanmean(tr.pos[:, s, 0]) if np.isfinite(tr.pos[:, s, 0]).any() else np.nan for s in pair]
        if np.isnan(xs).any():
            order = pair
        else:
            order = tuple(sorted(pair, key=lambda s: xs[pair.index(s)]))  # image-left first
        left, right = order
        # far team faces the camera: its right side is image-left
        roles[left], roles[right] = ("côté droit", "côté gauche") if far else ("côté gauche", "côté droit")
    return roles


def own_half(x: np.ndarray, y: np.ndarray, far: bool) -> tuple[np.ndarray, np.ndarray]:
    depth = np.abs(y - NET_Y)
    lateral = (10 - x) if far else x  # 0 = own left wall
    return lateral, depth


def smooth_track(p: np.ndarray, fps: float) -> list[np.ndarray]:
    """Split on long gaps, interpolate short ones, median + Savitzky-Golay filter. Returns pieces."""
    ok = np.isfinite(p[:, 0])
    pieces, i, n = [], 0, len(p)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (ok[j + 1] or (not ok[j + 1] and np.any(ok[j + 1:j + 1 + MAX_GAP_FRAMES]))):
            j += 1
        seg = p[i:j + 1].copy()
        t = np.arange(len(seg))
        good = np.isfinite(seg[:, 0])
        for c in (0, 1):
            seg[:, c] = np.interp(t, t[good], seg[good, c])
            if len(seg) >= 9:
                seg[:, c] = savgol_filter(medfilt(seg[:, c], 5), min(15, len(seg) // 2 * 2 - 1), 2)
        pieces.append(seg)
        i = j + 1
    return pieces


def player_stats(rallies: list[dict], tracks: dict[int, ShotTracks]) -> dict[str, dict]:
    stats = defaultdict(lambda: {"frames": 0, "net": 0, "transition": 0, "fond": 0, "distance_m": 0.0,
                                 "seconds": 0.0, "positions": []})
    for r in rallies:
        tr = tracks.get(r["shot_id"])
        if tr is None:
            continue
        for slot_s, label in r["players"].items():
            slot = int(slot_s)
            far = slot < 2
            st = stats[label]
            for seg in smooth_track(tr.pos[:, slot], tr.fps):
                lat, depth = own_half(seg[:, 0], seg[:, 1], far)
                st["frames"] += len(seg)
                st["net"] += int((depth < NET_ZONE_MAX).sum())
                st["transition"] += int(((depth >= NET_ZONE_MAX) & (depth < SERVICE_LINE_FROM_NET)).sum())
                st["fond"] += int((depth >= SERVICE_LINE_FROM_NET).sum())
                st["distance_m"] += float(np.hypot(*np.diff(seg, axis=0).T).sum())
                st["seconds"] += len(seg) / tr.fps
                st["positions"].append(np.c_[lat, depth][::3])
    out = {}
    for label, st in stats.items():
        f = max(st["frames"], 1)
        out[label] = {
            "temps_analyse_s": round(st["seconds"], 1),
            "distance_m": round(st["distance_m"], 1),
            "vitesse_moy_m_s": round(st["distance_m"] / max(st["seconds"], 1e-6), 2),
            "part_filet": round(st["net"] / f, 3),
            "part_transition": round(st["transition"] / f, 3),
            "part_fond": round(st["fond"] / f, 3),
            "_positions": np.vstack(st["positions"]) if st["positions"] else np.zeros((0, 2)),
        }
    return out


def stroke_stats(rallies: list[dict]) -> dict[str, Counter]:
    per = defaultdict(Counter)
    for r in rallies:
        for h in r["hits"]:
            k = report_key(h["stroke"])
            per[h["player"]][k if k in REPORT_STROKES else "?"] += 1
    return per


def heatmaps(pstats: dict[str, dict], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(pstats)
    if not labels:
        return
    fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 4.2), squeeze=False)
    for ax, label in zip(axes[0], labels):
        pos = pstats[label]["_positions"]
        if len(pos):
            ax.hist2d(pos[:, 0], pos[:, 1], bins=[20, 20], range=[[0, 10], [0, 10]], cmap="magma")
        ax.axhline(NET_ZONE_MAX, color="w", ls=":", lw=1)
        ax.axhline(SERVICE_LINE_FROM_NET, color="w", lw=1)
        ax.plot([5, 5], [0, SERVICE_LINE_FROM_NET], color="w", lw=0.5)  # centre service line (net → service line)
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, 10)
        ax.set_ylim(10, 0)  # net at the top
        ax.set_xlabel("gauche ← → droite (m)")
        ax.set_ylabel("distance au filet (m)")
        ax.set_aspect("equal")
    fig.suptitle("Présence sur son demi-terrain (filet en haut)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def markdown(meta: dict, rallies: list[dict], pstats: dict, strokes: dict, summary: str | None,
             heatmap_name: str | None) -> str:
    from ..timeline import fmt

    md = [f"# Analyse padel — {meta['title']}", ""]
    md += [f"Vidéo : `{meta['video']}` · {fmt(meta['duration'])} · {meta['n_shots']} plans dont "
           f"{meta['n_rallies']} plans d'échange ({fmt(meta['rally_seconds'])} de jeu analysé, "
           f"{fmt_pct(meta['rally_seconds'] / meta['duration'])} de la vidéo) · VLM `{meta['vlm']}`", ""]
    if meta.get("score"):
        md += [f"**Score lu sur le tableau :** {meta['score']}", ""]
    md += ["## Résumé du match", ""]
    if meta.get("score_story"):
        md += ["### Déroulé", "", meta["score_story"], ""]
    if summary:
        md += [summary, ""]

    teams = meta["teams"]
    md += ["## Équipes", ""]
    for t in teams:
        won = sum(1 for r in rallies if r.get("winner") == t)
        md.append(f"- **{t}** : maillot {meta['team_colours'].get(t, '?')} · points gagnés identifiés : {won}")
    unknown = sum(1 for r in rallies if not r.get("winner"))
    md += ["", f"Gagnant inconnu pour {unknown} échange(s) sur {len(rallies)} : le montage saute "
           "des points, le vainqueur n'est retenu que si le score avance d'exactement un point.", ""]

    md += ["## Joueurs", "",
           "| Joueur | Frappes | Distance | Vitesse moy. | Filet | Transition | Fond |",
           "|---|---|---|---|---|---|---|"]
    for label in sorted(pstats):
        s = pstats[label]
        n = sum(strokes.get(label, {}).values())
        md.append(f"| {label} | {n} | {s['distance_m']:.0f} m | {s['vitesse_moy_m_s']:.1f} m/s | "
                  f"{fmt_pct(s['part_filet'])} | {fmt_pct(s['part_transition'])} | {fmt_pct(s['part_fond'])} |")
    md += ["", "Côté droit = joueur de drive, côté gauche = joueur de revés (vu depuis sa moitié de terrain). "
           "Filet = moins de 4 m du filet ; transition = jusqu'à la ligne de service (6.95 m) ; "
           "fond = au-delà. Distances sur les seuls plans d'échange, trajectoires lissées.", ""]
    if heatmap_name:
        md += [f"![Heatmaps]({heatmap_name})", ""]

    md += ["## Coups", "", "| Joueur | " + " | ".join(REPORT_STROKES.values()) + " |",
           "|---|" + "---|" * len(REPORT_STROKES)]
    for label in sorted(strokes):
        c = strokes[label]
        md.append(f"| {label} | " + " | ".join(str(c.get(k, 0)) for k in REPORT_STROKES) + " |")
    clf = meta.get("classifier")
    how = (f"classifieur entraîné sur {clf['n']} frappes annotées (précision validée {100 * clf['cv']:.0f} %, "
           f"contre {100 * clf['vlm']:.0f} % pour le VLM seul)" if clf and clf.get("used") else
           "pose au moment de l'impact puis choix du VLM sur des recadrages, limité aux coups compatibles avec "
           "la hauteur et la distance mesurées")
    md += ["", f"Type de coup : {how}. Bandeja et víbora sont regroupées : l'effet qui les distingue n'est pas "
           "visible de façon fiable à cette résolution (détail par frappe dans le JSON). Sans suivi de balle, "
           "lobs et sorties de vitre ne sont reconnus que par le geste.", ""]

    md += ["## Échanges", ""]
    for r in rallies:
        head = f"### {fmt(r['start'])} – {fmt(r['end'])} ({r['end'] - r['start']:.0f} s, {len(r['hits'])} frappes)"
        md += [head, ""]
        if r.get("score_before"):
            md.append(f"Score avant : {r['score_before']}  ")
        if r.get("score_after"):
            md.append(f"Score après : {r['score_after']}  ")
        md.append(f"Côtés : fond = {r['teams_by_side']['fond']}, proche = {r['teams_by_side']['proche']}  ")
        if r.get("winner"):
            md.append(f"Point pour : **{r['winner']}**  ")
        seq = " → ".join(f"{h['team_short']} {h['player'].split(' · ')[-1]} : "
                         f"{REPORT_STROKES.get(report_key(h['stroke']), h['stroke'])}"
                         for h in r["hits"])
        if seq:
            md.append(f"Séquence : {seq}")
        if r.get("comment"):
            md += ["", r["comment"]]
        md.append("")
    return "\n".join(md)
