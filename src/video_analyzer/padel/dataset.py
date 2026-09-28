"""Annotation dataset: hit selection, static HTML labelling page, label storage.

Labels live in the repo (`data/labels/<video_key>.json`): they are the ground truth used by
`eval` and `train`, and must survive cache clean-ups. One entry per hit key
("shot:frame:slot"), with the annotator ("user" or "claude" for bootstrap labels).

The labelling page is a single self-contained HTML file + JPEG strips (no server): keyboard
shortcuts, autosave in localStorage, "Exporter" downloads the labels JSON, which
`video-analyzer padel import-labels` merges into the dataset.
"""
import json
import random
from collections import Counter
from pathlib import Path

from ..extract import VideoInfo, video_key
from .players import ShotTracks
from .rules import STROKES
from .strokes import Hit, hit_crops, reconcile

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EXTRA_LABELS = {"bandeja_ou_vibora": "bandeja ou víbora (je ne sais pas trancher)",
                "pas_une_frappe": "pas une frappe (rebond, vitre, joueur immobile…)",
                "incertain": "incertain / image inexploitable"}
LABELS = {**{k: v.split(" :")[0] for k, v in STROKES.items()}, **EXTRA_LABELS}
STRIP_OFFSETS = (-0.3, -0.15, 0.0, 0.15, 0.3)


def labels_path(key: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "labels" / f"{key}.json"


def load_labels(key: str, data_dir: Path = DATA_DIR) -> dict[str, dict]:
    p = labels_path(key, data_dir)
    return json.loads(p.read_text(encoding="utf-8"))["labels"] if p.exists() else {}


def save_labels(key: str, video_name: str, labels: dict[str, dict], data_dir: Path = DATA_DIR) -> Path:
    p = labels_path(key, data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"video": video_name, "key": key, "labels": labels}, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_hits(key: str, cache_root: Path) -> tuple[list[Hit], dict[int, ShotTracks]]:
    """Hits of the last `padel` run of video `key`, with current pose rules re-applied."""
    cache = cache_root / key / "padel"
    path = cache / "hits_vlm.json"
    if not path.exists():
        path = cache / "hits.json"
    if not path.exists():
        raise FileNotFoundError("lance d'abord `video-analyzer padel <vidéo>`")
    shots = {d["id"]: d for d in json.loads((cache / "shots.json").read_text(encoding="utf-8"))}
    tracks, hits = {}, []
    for sid_s, hs in json.loads(path.read_text(encoding="utf-8")).items():
        sid = int(sid_s)
        tr = tracks.setdefault(sid, ShotTracks.load(cache / "tracks" / f"shot{sid:04d}.npz"))
        for i, d in enumerate(hs):
            h = Hit(**d)
            reconcile(h, tr, i == 0, shots[sid]["start"])
            hits.append(h)
    return hits, tracks


def select_for_annotation(hits: list[Hit], n: int, already: set[str], seed: int = 0) -> list[Hit]:
    """Active-learning order: disagreements of the pose rules with the VLM or the classifier first (least
    confident classifier first), then round-robin over predicted classes so rare strokes get labelled too."""
    rng = random.Random(seed)
    todo = [h for h in hits if h.key not in already]
    rng.shuffle(todo)
    from .classifier import target  # bandeja / víbora are one class for the classifier

    disagree = [h for h in todo if h.stroke_pose and
                any(o and target(o) != target(h.stroke_pose) for o in (h.stroke_vlm, h.stroke_clf))]
    disagree.sort(key=lambda h: h.clf_prob if h.stroke_clf else 1.0)
    rest = [h for h in todo if h not in disagree]
    by_class: dict[str, list[Hit]] = {}
    for h in rest:
        by_class.setdefault(h.stroke, []).append(h)
    rr = []
    while any(by_class.values()):
        for c in sorted(by_class):
            if by_class[c]:
                rr.append(by_class[c].pop())
    half = n // 2
    picked = disagree[:half]
    picked += rr[:n - len(picked)]
    return picked[:n]


def export_page(info: VideoInfo, hits: list[Hit], tracks: dict[int, ShotTracks], out_dir: Path,
                log=print) -> Path:
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    items = []
    for n, h in enumerate(hits, 1):
        if n % 20 == 0 or n == len(hits):
            log(f"  images {n}/{len(hits)}")
        name = f"{h.key.replace(':', '_')}.jpg"
        path = img_dir / name
        if not path.exists():
            ims = hit_crops(info, h, tracks[h.shot_id], size=220, offsets=STRIP_OFFSETS, context=True)
            if not ims:
                continue
            strip = Image.new("RGB", (sum(i.width for i in ims) + 4 * (len(ims) - 1), ims[0].height), (20, 20, 20))
            x = 0
            for im in ims:
                strip.paste(im, (x, 0))
                x += im.width + 4
            strip.save(path, quality=88)
            ims[3].save(img_dir / f"{h.key.replace(':', '_')}_impact.jpg", quality=92)  # reference crop
        items.append({"key": h.key, "img": f"img/{name}", "t": h.t, "side": h.side,
                      "dist": h.dist_net, "height": h.height, "guess": h.stroke})
    key = video_key(info.path)
    page = out_dir / "annoter.html"
    page.write_text(_HTML.replace("__DATA__", json.dumps({"video": info.path.name, "key": key, "items": items,
                                                          "labels": LABELS}, ensure_ascii=False)),
                    encoding="utf-8")
    return page


def import_labels(path: Path, data_dir: Path = DATA_DIR, annotator: str = "user") -> tuple[Path, Counter]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    key, video = d["key"], d.get("video", "")
    labels = load_labels(key, data_dir)
    for k, v in d["labels"].items():
        if v in LABELS:
            labels[k] = {"label": v, "annotator": annotator}
    return save_labels(key, video, labels, data_dir), Counter(v["label"] for v in labels.values())


_HTML = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Annotation des frappes</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--acc:#ffd23f;--card:#1c1c1c}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 -apple-system,system-ui,sans-serif}
header{padding:12px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}
header b{color:var(--acc)} main{padding:16px;max-width:1500px;margin:auto}
#strip{width:100%;border-radius:6px;background:#000}
.meta{color:var(--mut);margin:8px 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px}
button{background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:10px;text-align:left;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)} button.on{background:var(--acc);color:#111;border-color:var(--acc)}
kbd{display:inline-block;min-width:18px;padding:0 5px;margin-right:6px;border:1px solid #555;border-radius:4px;text-align:center;font-size:12px}
.nav{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap} .nav button{text-align:center}
#guess{color:var(--mut)} .hint{color:var(--mut);font-size:13px;margin-top:10px}
</style></head><body>
<header><span>Vidéo <b id="vid"></b></span><span id="prog"></span>
<button id="exp">Exporter les annotations (JSON)</button>
<label><input type="checkbox" id="show"> afficher la proposition du modèle</label></header>
<main>
<img id="strip" alt="frappe">
<div class="meta" id="meta"></div>
<div class="grid" id="btns"></div>
<div class="nav"><button id="prev">← précédente</button><button id="next">suivante →</button>
<button id="todo">prochaine non annotée</button></div>
<p class="hint">Image 1 : vue d'ensemble, frappeur encadré. Images 2 à 6 : −0.3 s, −0.15 s, impact, +0.15 s, +0.3 s.
Clavier : chiffres/lettres affichés, ← → pour naviguer. Sauvegarde automatique dans ce navigateur ;
clique « Exporter » puis lance <code>video-analyzer padel import-labels &lt;fichier&gt;</code>.</p>
</main>
<script>
const D = __DATA__;
const KEYS = "1234567890qwertyuiop";
const store = "labels:" + D.key;
let labels = {}; try { labels = JSON.parse(localStorage.getItem(store) || "{}"); } catch (e) {}
let i = 0;
const codes = Object.keys(D.labels);
document.getElementById("vid").textContent = D.video;
const btns = document.getElementById("btns");
codes.forEach((c, j) => { const b = document.createElement("button"); b.dataset.c = c;
  b.innerHTML = `<kbd>${KEYS[j] || ""}</kbd>${D.labels[c]}`; b.onclick = () => setLabel(c); btns.appendChild(b); });
function save() { try { localStorage.setItem(store, JSON.stringify(labels)); } catch (e) {} }
function render() {
  const it = D.items[i]; if (!it) return;
  document.getElementById("strip").src = it.img;
  const m = Math.floor(it.t / 60), s = (it.t % 60).toFixed(1).padStart(4, "0");
  document.getElementById("meta").innerHTML = `Frappe ${i + 1}/${D.items.length} · ${m}:${s} · joueur ${it.side === "fond" ? "au fond de l'image" : "au premier plan"} · `
    + `${it.dist.toFixed(1)} m du filet · hauteur mesurée : ${it.height || "?"}`
    + (document.getElementById("show").checked ? ` · <span id="guess">proposition : ${D.labels[it.guess] || it.guess}</span>` : "");
  [...btns.children].forEach(b => b.classList.toggle("on", labels[it.key] === b.dataset.c));
  document.getElementById("prog").textContent = `${Object.keys(labels).length} annotées / ${D.items.length}`;
}
function setLabel(c) { labels[D.items[i].key] = c; save(); if (i < D.items.length - 1) i++; render(); }
document.getElementById("prev").onclick = () => { i = Math.max(0, i - 1); render(); };
document.getElementById("next").onclick = () => { i = Math.min(D.items.length - 1, i + 1); render(); };
document.getElementById("todo").onclick = () => { const k = D.items.findIndex(x => !labels[x.key]); if (k >= 0) { i = k; render(); } };
document.getElementById("show").onchange = render;
document.getElementById("exp").onclick = () => {
  const blob = new Blob([JSON.stringify({ video: D.video, key: D.key, labels }, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `labels_${D.key}.json`; a.click(); };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
  else { const j = KEYS.indexOf(e.key.toLowerCase()); if (j >= 0 && j < codes.length) setLabel(codes[j]); } });
const first = D.items.findIndex(x => !labels[x.key]); i = first >= 0 ? first : 0; render();
</script></body></html>"""
