"""Learning loop: labelled hits → evaluation of each method → classifier training → use in pipeline.

- `evaluate(key)`: accuracy of pose rules, VLM, final label and classifier (cross-validated
  predictions from the last training) against the labels; hit precision = share of detected
  hits that are real hits. Written to data/eval/<key>.md.
- `train_all()`: pools every labelled video, cross-validates, saves the model to
  <cache>/models/stroke_clf.pt and a report to data/eval/entrainement.md.
- `apply_classifier(hits)`: the pipeline uses the classifier only if its cross-validated accuracy
  beats the VLM's on the same labels; the VLM label stays in `stroke_vlm` for comparison.
- `references()`: one validated impact crop per stroke for few-shot VLM prompts.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

from . import classifier as CL
from .dataset import DATA_DIR, load_hits, load_labels
from .strokes import Hit

MODEL_NAME = "stroke_clf.pt"
REJECT_PROB = 0.7  # classifier says "not a hit" with at least this probability → hit dropped


def _labelled_keys(data_dir: Path) -> list[str]:
    return sorted(p.stem for p in (data_dir / "labels").glob("*.json"))


def collect(cache_root: Path, data_dir: Path = DATA_DIR) -> list[CL.Sample]:
    samples = []
    for key in _labelled_keys(data_dir):
        labels = load_labels(key, data_dir)
        try:
            hits, tracks = load_hits(key, cache_root)
        except FileNotFoundError:
            continue
        for h in hits:
            lab = labels.get(h.key)
            y = CL.target(lab["label"]) if lab else None
            if y is None:
                continue
            x = CL.hit_features(h, tracks[h.shot_id])
            if x is not None:
                samples.append(CL.Sample(x, y, key, h.key, h.stroke_vlm, h.stroke_pose))
    return samples


def train_all(cache_root: Path, data_dir: Path = DATA_DIR) -> tuple[Path, dict]:
    samples = collect(cache_root, data_dir)
    bundle, report = CL.train(samples)
    path = cache_root / "models" / MODEL_NAME
    CL.save(bundle, path)
    out = data_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "entrainement.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out / "entrainement.md").write_text(_train_md(report), encoding="utf-8")
    return path, report


def _pct(a) -> str:
    v, n = a
    return f"{100 * v:.0f} % (n={n})" if n else "—"


def _conf_md(conf: dict[str, dict[str, int]]) -> list[str]:
    cls = list(conf)
    md = ["| vrai \\ prédit | " + " | ".join(cls) + " |", "|---|" + "---|" * len(cls)]
    for c in cls:
        md.append(f"| **{c}** | " + " | ".join(str(conf[c][d]) for d in cls) + " |")
    return md


def _train_md(r: dict) -> str:
    md = ["# Entraînement du classifieur de coups", "",
          f"{r['n']} frappes annotées utilisées, validation croisée en {r['folds']} plis.", "",
          "| Méthode | Précision sur les mêmes frappes |", "|---|---|"]
    for k, v in r["accuracy"].items():
        md.append(f"| {k} | {_pct(v)} |")
    md += ["", f"Exemples par classe : {r['counts']}"]
    if r["left_out"]:
        md.append(f"Classes écartées (moins de {CL.MIN_PER_CLASS} exemples) : {r['left_out']}")
    md += ["", "## Matrice de confusion (classifieur, validation croisée)", ""] + _conf_md(r["confusion_classifier"])
    return "\n".join(md) + "\n"


def evaluate(key: str, cache_root: Path, data_dir: Path = DATA_DIR) -> tuple[Path, dict]:
    labels = load_labels(key, data_dir)
    hits, _ = load_hits(key, cache_root)
    rep_path = data_dir / "eval" / "entrainement.json"
    cv = json.loads(rep_path.read_text(encoding="utf-8"))["cv_predictions"] if rep_path.exists() else {}
    rows = [(h, labels[h.key]) for h in hits if h.key in labels]
    annot = Counter(v["annotator"] for _, v in rows)
    real = [(h, CL.target(v["label"])) for h, v in rows if v["label"] != "incertain"]
    n_not_hit = sum(1 for _, y in real if y == "pas_une_frappe")
    strokes = [(h, y) for h, y in real if y != "pas_une_frappe"]

    def acc(get):
        pairs = [(CL.target(get(h)), y) for h, y in strokes if get(h)]
        return (sum(a == b for a, b in pairs) / len(pairs), len(pairs)) if pairs else (float("nan"), 0)

    res = {
        "annotated": len(rows), "annotators": dict(annot),
        "hit_precision": ((len(real) - n_not_hit) / len(real), len(real)) if real else (float("nan"), 0),
        "accuracy": {"pose + VLM (pipeline sans classifieur)": acc(lambda h: h.stroke), "vlm": acc(lambda h: h.stroke_vlm),
                     "pose": acc(lambda h: h.stroke_pose),
                     "classifieur (validation croisée)": acc(lambda h: cv.get(h.key))},
    }
    classes = sorted({y for _, y in strokes})
    res["confusion_final"] = CL.confusion([y for _, y in strokes], [CL.target(h.stroke) for h, _ in strokes], classes)
    md = [f"# Évaluation — {key}", "",
          f"{len(rows)} frappes annotées ({', '.join(f'{k} : {v}' for k, v in annot.items())}).", "",
          f"Précision de la détection de frappes : {_pct(res['hit_precision'])} des frappes détectées sont de vraies frappes.",
          "", "| Méthode | Type de coup correct |", "|---|---|"]
    md += [f"| {k} | {_pct(v)} |" for k, v in res["accuracy"].items()]
    md += ["", "Bandeja et víbora sont comptées comme une seule classe.", "",
           "## Matrice de confusion (pose + VLM)", ""] + _conf_md(res["confusion_final"])
    if annot.get("claude"):
        md += ["", "Attention : une partie des annotations a été faite par Claude (amorçage), pas par un humain."]
    out = data_dir / "eval" / f"{key}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out, res


def apply_classifier(all_hits: dict[int, list[Hit]], tracks, cache_root: Path, log=print,
                     info: dict | None = None) -> dict[int, list[Hit]]:
    """`info` (optional) receives {"n", "cv", "vlm", "used"} for the report."""
    bundle = CL.load(cache_root / "models" / MODEL_NAME)
    if bundle is None:
        return all_hits
    use = bundle["cv_accuracy"] >= (bundle.get("vlm_accuracy") or 0)
    if info is not None:
        info.update(n=bundle["n"], cv=bundle["cv_accuracy"], vlm=bundle.get("vlm_accuracy") or 0.0, used=use)
    log(f"  classifieur de coups : {bundle['n']} exemples, précision validée {100 * bundle['cv_accuracy']:.0f} % "
        f"(VLM {100 * (bundle.get('vlm_accuracy') or 0):.0f} %) → {'utilisé' if use else 'non utilisé'}")
    out = {}
    for sid, hs in all_hits.items():
        kept = []
        for h in hs:
            r = CL.predict(bundle, h, tracks[sid])
            if r:
                h.stroke_clf, h.clf_prob = r
                if use:
                    if h.stroke_clf == "pas_une_frappe":
                        if h.clf_prob >= REJECT_PROB:
                            continue
                    elif h.stroke_clf == "bandeja_vibora":
                        h.stroke = h.stroke_vlm if h.stroke_vlm in ("bandeja", "vibora") else "vibora"
                        h.confidence = h.clf_prob
                    else:
                        h.stroke, h.confidence = h.stroke_clf, h.clf_prob
            kept.append(h)
        out[sid] = kept
    return out


def references(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """stroke → one validated impact crop (human labels preferred) for few-shot VLM prompts."""
    ref_dir = data_dir / "references"
    return {p.stem: p for p in ref_dir.glob("*.jpg")} if ref_dir.exists() else {}


def build_references(annotation_dirs: list[Path], data_dir: Path = DATA_DIR) -> dict[str, Path]:
    ref_dir = data_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    chosen: dict[str, tuple[int, Path]] = {}
    for key in _labelled_keys(data_dir):
        for hkey, v in load_labels(key, data_dir).items():
            lab = v["label"]
            if lab in ("incertain", "pas_une_frappe", "bandeja_ou_vibora"):
                continue
            rank = 0 if v["annotator"] == "user" else 1
            for d in annotation_dirs:
                img = d / "img" / f"{hkey.replace(':', '_')}_impact.jpg"
                if img.exists() and (lab not in chosen or rank < chosen[lab][0]):
                    chosen[lab] = (rank, img)
    for lab, (_, img) in chosen.items():
        shutil.copy(img, ref_dir / f"{lab}.jpg")
    return references(data_dir)
