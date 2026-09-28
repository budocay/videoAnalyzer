"""Padel match pipeline. Every stage is cached under <cache>/<video_key>/padel/.

1. shots   : cuts + VLM classification → rally shots (match camera, live, not replay)
2. court   : homography from court lines (best of several rally frames)
3. score   : scoreboard location + reading at each rally start; names/shirt colours from close-ups
4. players : YOLO-pose tracking on rally shots (VLM unloaded meanwhile)
5. strokes : audio+pose hits, rule-constrained decoding, pose heuristics, VLM confirmation
6. report  : team/player stats, heatmaps, per-rally sequence, VLM match summary
"""
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..extract import VideoInfo, video_key
from ..memory import free_mlx
from ..vision import load_vlm
from . import court as C
from . import learn as L
from . import report as R
from . import score as SC
from . import shots as S
from . import strokes as ST
from .players import track_rallies
from .rules import RULE_NOTES

SCHEMA_VERSION = "padel-1.0"

SUMMARY_PROMPT = (
    "Tu es analyste de padel. Règles utiles :\n{rules}\n\n"
    "Données extraites automatiquement d'un montage de temps forts (des points manquent) du match "
    "{teams}.\n"
    "Déroulé du score (déjà rédigé, ne le réécris pas et ne le contredis pas) : {score}\n"
    "Joueurs désignés par équipe et côté (côté droit = joueur de drive, côté gauche = joueur de revés) ; "
    "ce ne sont pas des coups. Les pourcentages filet/transition/fond sont des TEMPS DE PRÉSENCE, pas des "
    "taux de réussite.\n\n"
    "Points dont le vainqueur est connu (lecture du tableau avant/après) :\n{winners}\n\n"
    "Statistiques par joueur :\n{players}\n\n"
    "Rédige en français, en Markdown sobre (titres ###, pas d'emoji, pas de titre principal, pas de phrase "
    "d'introduction), 2 parties courtes : \"### Style de jeu\" de chaque équipe (présence au filet, coups "
    "dominants, comparaison des deux équipes) ; \"### Moments clés\" : 3 à 5 points pris UNIQUEMENT dans la "
    "liste des points au vainqueur connu, avec leur timecode. Ne parle pas du score des sets. N'invente aucun "
    "coup ni score absent des données."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _calibrate(info: VideoInfo, rallies, cache: Path) -> C.Court:
    path = cache / "court.json"
    if path.exists():
        return C.load(path)
    best = None
    # middle of the 6 longest rally shots; a long shot (continuous fixed camera: one shot for the whole match)
    # gets several frames, since players can hide the lines at any given instant
    times = []
    for s in sorted(rallies, key=lambda s: -s.duration)[:6]:
        k = int(min(8, max(1, s.duration // 60)))
        times += [s.start + s.duration * (i + 0.5) / k for i in range(k)]
    for t in times:
        f = SC._frame(info, t)
        if f is None:
            continue
        try:
            c = C.calibrate(f)
        except C.CourtNotFound:
            continue
        if best is None or c.residual_px < best.residual_px:
            best = c
    if best is None:
        raise C.CourtNotFound("terrain non calibré sur les plans d'échange")
    C.save(best, path)
    return best


def _scores(vlm, info: VideoInfo, shots, rallies, cache: Path, tmp: Path) -> dict:
    """Score before each rally (its first frames) and after it: the last frame showing the scoreboard
    before the next rally (close-ups after a point usually show the updated score)."""
    path = cache / "scores.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "corner" not in out:
        longest = max(rallies, key=lambda s: s.duration)
        out = {"corner": SC.locate(vlm, info, longest.start + 1.0, tmp), "reads": {}}
    out.setdefault("reads_after", {})
    if out["corner"]:
        for i, s in enumerate(rallies):
            if str(s.id) not in out["reads"]:
                st = SC.read(vlm, info, s.start + 0.5, out["corner"], tmp)
                out["reads"][str(s.id)] = asdict(st) if st else None
            if str(s.id) not in out["reads_after"]:
                nxt_start = rallies[i + 1].start if i + 1 < len(rallies) else info.duration
                between = [x for x in shots if s.end <= x.start < nxt_start and x.scoreboard]
                t = (between[-1].end - 0.4) if between else (s.end - 0.4)
                st = SC.read(vlm, info, t, out["corner"], tmp)
                out["reads_after"][str(s.id)] = asdict(st) if st else None
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _players(vlm, shots, cache: Path, tmp: Path) -> list[dict]:
    path = cache / "players.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    found = SC.identify_players(vlm, shots, tmp)
    path.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    return found


def _team_by_side(tracks, teams: list[str], obs: list[dict]) -> tuple[dict[int, dict[str, str]], dict[str, str]]:
    """shot_id → {'fond': team, 'proche': team}, and team → colour word.
    Torso colours of each side are clustered in 2 (2-means); names are attached to clusters by the
    VLM shirt observations (majority of agreeing observations, see score.assign_teams)."""
    side_col = {sid: (np.nanmean(tr.color[:2], axis=0), np.nanmean(tr.color[2:], axis=0))
                for sid, tr in tracks.items()}
    cols = np.array([c for pair in side_col.values() for c in pair if np.isfinite(c).all()])
    names = teams if len(teams) == 2 else ["Équipe A", "Équipe B"]
    if len(cols) < 2:
        return {sid: {"fond": names[0], "proche": names[1]} for sid in side_col}, {}
    c0, c1 = cols[np.argmin(cols.sum(1))], cols[np.argmax(cols.sum(1))]
    for _ in range(10):
        lab = np.linalg.norm(cols - c0, axis=1) > np.linalg.norm(cols - c1, axis=1)
        c0, c1 = cols[~lab].mean(0), cols[lab].mean(0)
    names, votes = SC.assign_teams(names, obs, [c0, c1])
    log(f"  maillots : {names[0]} ≈ {SC.colour_word(c0)} {np.round(c0).astype(int).tolist()}, "
        f"{names[1]} ≈ {SC.colour_word(c1)} {np.round(c1).astype(int).tolist()} ({votes} lectures VLM concordantes)")
    out = {}
    for sid, (far, near) in side_col.items():
        if not (np.isfinite(far).all() and np.isfinite(near).all()):
            out[sid] = {"fond": names[0], "proche": names[1]}
            continue
        far_is_0 = np.linalg.norm(far - c0) + np.linalg.norm(near - c1) <= np.linalg.norm(far - c1) + np.linalg.norm(near - c0)
        out[sid] = {"fond": names[0], "proche": names[1]} if far_is_0 else {"fond": names[1], "proche": names[0]}
    return out, {names[0]: SC.colour_word(c0), names[1]: SC.colour_word(c1)}


def _score_text(d: dict | None) -> str:
    if not d:
        return ""
    return " · ".join(f"{t} {' '.join(map(str, g))} [{p}]{' (serv.)' if s else ''}"
                      for t, g, p, s in zip(d["teams"], d["games"], d["points"], d["serving"]))


def run(info: VideoInfo, out_dir: Path, cache_root: Path, vlm_repo: str, model_dir: Path,
        refine_strokes: bool | None = None, summary: bool = True) -> tuple[Path, Path]:
    """refine_strokes: None = automatic (per-hit VLM check only when no trained classifier beats it)."""
    t_all = time.perf_counter()
    cache = cache_root / video_key(info.path) / "padel"
    cache.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="padel_"))

    log("• 1/6 Plans : détection des coupes")
    shots = S.detect(info, cache, log)
    log(f"• Chargement {vlm_repo}")
    vlm = load_vlm(vlm_repo)
    log("• 1/6 Plans : classification par le VLM")
    S.classify(vlm, shots, cache, log)
    rallies = [s for s in shots if s.rally]
    kinds = {}
    for s in shots:
        k = "échange" if s.rally else ("replay/ralenti" if s.replay else s.kind)
        kinds[k] = kinds.get(k, 0) + 1
    log(f"  {len(rallies)} plans d'échange ({sum(s.duration for s in rallies):.0f} s) · parasites retirés : "
        + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()) if k != "échange"))
    if not rallies:
        raise RuntimeError("aucun plan de caméra de match trouvé")

    log("• 2/6 Terrain")
    court = _calibrate(info, rallies, cache)
    log(f"  méthode {court.method}, erreur ligne centrale {court.residual_px:.1f} px")

    log("• 3/6 Score et équipes (VLM)")
    scores = _scores(vlm, info, shots, rallies, cache, tmp)
    players_found = _players(vlm, shots, cache, tmp)
    def _states(key):
        seq = [SC.ScoreState(**scores[key][str(s.id)]) if scores[key].get(str(s.id)) else None for s in rallies]
        return seq

    # before/after reads interleaved form one chronological sequence: clean it as a whole
    seq = SC.clean_sequence([x for pair in zip(_states("reads"), _states("reads_after")) for x in pair])
    states = {s.id: seq[2 * i] for i, s in enumerate(rallies)}
    states_after = {s.id: seq[2 * i + 1] for i, s in enumerate(rallies)}
    team_names = next((st.teams for st in seq if st), [])
    match_score = SC.score_line(seq)
    story = SC.score_story(seq)
    log(f"  {story}")
    log(f"  tableau {scores['corner']}, {sum(1 for v in seq if v)}/{len(seq)} lectures valides · "
        f"équipes {team_names} · {len(players_found)} noms lus sur les maillots")

    log("• 4/6 Joueurs (YOLO-pose, VLM déchargé)")
    vlm.close()
    del vlm
    free_mlx()
    tracks = {t.shot_id: t for t in track_rallies(info, court, shots, cache, model_dir, log)}

    log("• 5/6 Frappes")
    if refine_strokes is None:
        refine_strokes = L.use_vlm_for_strokes(cache_root)
        log("  types de coups : " + ("vérification par le VLM (pas de classifieur meilleur)" if refine_strokes else
                                     "classifieur entraîné (meilleur que le VLM ; --vlm-strokes pour forcer le VLM)"))
    hits_path = cache / ("hits_vlm.json" if refine_strokes else "hits.json")
    vlm = None
    if hits_path.exists():
        all_hits = {int(k): [ST.Hit(**h) for h in v] for k, v in json.loads(hits_path.read_text(encoding="utf-8")).items()}
        start = {s.id: s.start for s in rallies}
        for sid, hs in all_hits.items():  # re-apply the current pose rules to cached labels
            for i, h in enumerate(hs):
                ST.reconcile(h, tracks[sid], i == 0, start[sid])
    else:
        all_hits = {}
        if refine_strokes or summary:
            vlm = load_vlm(vlm_repo)
        for k, s in enumerate(rallies, 1):
            tr = tracks[s.id]
            hits = ST.detect_hits(info, tr, s.duration)
            for i, h in enumerate(hits):
                ST.classify_stroke(h, tr, i == 0, s.start)
            if refine_strokes and hits:
                t = time.perf_counter()
                hits = ST.refine_with_vlm(vlm, info, hits, tr, tmp, refs=L.references())
                log(f"  [{k}/{len(rallies)}] plan {s.id} : {len(hits)} frappes confirmées ({time.perf_counter() - t:.0f} s)")
            all_hits[s.id] = hits
        hits_path.write_text(json.dumps({k: ST.hits_to_dicts(v) for k, v in all_hits.items()}, ensure_ascii=False), encoding="utf-8")

    clf_info: dict = {}
    all_hits = L.apply_classifier(all_hits, tracks, cache_root, log, clf_info)

    log("• 6/6 Rapport")
    sides, colours = _team_by_side(tracks, team_names, players_found)
    rally_docs = []
    for i, s in enumerate(rallies):
        tr = tracks[s.id]
        side_team = sides[s.id]
        roles = R.slot_roles(tr)
        players = {str(sl): f"{side_team['fond' if sl < 2 else 'proche']} · {roles[sl]}" for sl in range(4)}
        nxt = states.get(rallies[i + 1].id) if i + 1 < len(rallies) else None
        w = SC.point_winner(states.get(s.id), states_after.get(s.id))
        if w is None:
            w = SC.point_winner(states.get(s.id), nxt)
        winner = team_names[w] if w is not None and team_names else None
        hits = []
        for h in all_hits.get(s.id, []):
            d = asdict(h)
            d["player"] = players[str(h.slot)]
            d["team_short"] = side_team[h.side].split("/")[0].strip().title()
            hits.append(d)
        rally_docs.append({
            "shot_id": s.id, "start": s.start, "end": s.end,
            "score_before": _score_text(asdict(states[s.id]) if states[s.id] else None),
            "score_after": _score_text(asdict(states_after[s.id]) if states_after[s.id] else None),
            "winner": winner, "teams_by_side": side_team, "players": players, "hits": hits,
        })

    pstats = R.player_stats(rally_docs, tracks)
    strokes = R.stroke_stats(rally_docs)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    heat = out_dir / f"{stem}_padel_heatmaps.png"
    R.heatmaps(pstats, heat)

    text = None
    if summary:
        vlm = vlm or load_vlm(vlm_repo)
        from ..timeline import fmt as _fmt
        rl = "\n".join(f"[{_fmt(r['start'])}] {r['score_before'] or 'score illisible'} → point pour {r['winner']} "
                       f"({len(r['hits'])} frappes, dernière : "
                       + (f"{r['hits'][-1]['player']} {R.REPORT_STROKES[R.report_key(r['hits'][-1]['stroke'])]}" if r["hits"] else "?")
                       + ")" for r in rally_docs if r["winner"])
        pl = "\n".join(f"- {k} : {v['distance_m']} m parcourus ; présence filet {R.fmt_pct(v['part_filet'])}, "
                       f"fond {R.fmt_pct(v['part_fond'])} ; coups "
                       + ", ".join(f"{R.REPORT_STROKES[c]} {n}" for c, n in strokes.get(k, {}).most_common())
                       for k, v in sorted(pstats.items()))
        a = vlm.ask(SUMMARY_PROMPT.format(rules="\n".join(f"- {n}" for n in RULE_NOTES),
                                          teams=" et ".join(team_names) or "deux équipes",
                                          score=story,
                                          winners=rl or "aucun", players=pl), max_tokens=1200)
        text = a.text
    if vlm is not None:
        vlm.close()

    meta = {"title": stem, "video": info.path.name, "duration": info.duration, "n_shots": len(shots),
            "n_rallies": len(rallies), "rally_seconds": sum(s.duration for s in rallies), "vlm": vlm_repo,
            "teams": team_names, "team_colours": colours, "score": match_score, "score_story": story,
            "classifier": clf_info or None}
    md_path = out_dir / f"{stem}_padel.md"
    md_path.write_text(R.markdown(meta, rally_docs, pstats, strokes, text, heat.name), encoding="utf-8")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "court": {"method": court.method, "residual_px": court.residual_px, "image_corners": court.image_corners},
        "shots": [{**{k: v for k, v in asdict(s).items() if k != "keyframes"}, "rally": s.rally} for s in shots],
        "players_found": players_found,
        "rallies": rally_docs,
        "player_stats": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in pstats.items()},
        "stroke_stats": {k: dict(v) for k, v in strokes.items()},
        "summary": text,
        "timings": {"total_s": round(time.perf_counter() - t_all, 1)},
    }
    json_path = out_dir / f"{stem}_padel.json"
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"✔ {md_path}\n✔ {json_path}\n✔ {heat}\n  total {time.perf_counter() - t_all:.0f} s")
    return md_path, json_path
