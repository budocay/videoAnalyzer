import json

import numpy as np

from video_analyzer.padel import classifier as CL
from video_analyzer.padel import dataset as DS
from video_analyzer.padel.players import ShotTracks
from video_analyzer.padel.strokes import Hit


def _hit(slot=2, arm="droit", stroke="coup_droit", vlm="", pose="", frame=20, shot=0):
    h = Hit(shot, 1.0, frame, slot, "proche" if slot >= 2 else "fond", 5.0, 16.0, 6.0, 0.4, 5.0, arm,
            stroke=stroke, height="bas")
    h.stroke_vlm, h.stroke_pose = vlm, pose
    return h


def _tracks(slot=2, mirror=False):
    """A standing player (~200 px tall) whose right wrist moves outwards."""
    T = 41
    k = np.zeros((T, 4, 17, 3))
    base = {0: (0, -180), 5: (-25, -150), 6: (25, -150), 7: (-35, -110), 8: (35, -110), 9: (-40, -70),
            10: (40, -70), 11: (-15, -80), 12: (15, -80), 13: (-15, -40), 14: (15, -40), 15: (-15, 0), 16: (15, 0)}
    for j, (x, y) in base.items():
        k[:, slot, j] = [500 + (-x if mirror else x), 800 + y, 0.9]
    k[:, slot, 10 if not mirror else 10, 0] += np.linspace(0, 60, T) * (-1 if mirror else 1)
    pos = np.full((T, 4, 2), np.nan)
    return ShotTracks(0, 0.0, 30.0, pos, k, np.zeros((4, 3)))


def test_target_merges_overheads_and_ignores_uncertain():
    assert CL.target("vibora") == CL.target("bandeja") == CL.target("bandeja_ou_vibora") == "bandeja_vibora"
    assert CL.target("incertain") is None and CL.target("revers") == "revers"


def test_features_are_view_invariant():
    near = CL.hit_features(_hit(slot=2), _tracks(slot=2))
    far = CL.hit_features(_hit(slot=0), _tracks(slot=0, mirror=True))  # same body seen from the front
    assert near.shape[0] == ((2 * CL.WINDOW) // CL.STEP + 1) * 7 * 2 + 7
    assert np.allclose(near[:-7], far[:-7], atol=1e-6)


def test_train_learns_separable_classes():
    rng = np.random.default_rng(0)
    samples = [CL.Sample(rng.normal(c * 3.0, 1.0, 20).astype(np.float32), y, "v", f"{y}{i}", "", "")
               for c, y in enumerate(["revers", "smash"]) for i in range(20)]
    bundle, report = CL.train(samples, k=4)
    assert report["accuracy"]["classifieur_cv"][0] > 0.9
    assert bundle["classes"] == ["revers", "smash"]


def test_import_labels_roundtrip(tmp_path):
    f = tmp_path / "export.json"
    f.write_text(json.dumps({"video": "v.mp4", "key": "v-abc", "labels": {"1:2:3": "smash", "4:5:6": "bidon"}}))
    path, counts = DS.import_labels(f, data_dir=tmp_path)
    saved = json.loads(path.read_text())["labels"]
    assert saved == {"1:2:3": {"label": "smash", "annotator": "user"}} and counts == {"smash": 1}


def test_selection_puts_disagreements_first():
    hits = [_hit(stroke="revers", vlm="revers", pose="revers", frame=i) for i in range(10)]
    hits += [_hit(stroke="smash", vlm="smash", pose="bandeja", frame=100 + i) for i in range(4)]
    picked = DS.select_for_annotation(hits, 8, already=set())
    assert {h.frame for h in picked[:4]} == {100, 101, 102, 103}


def test_selection_uses_classifier_least_confident_first():
    hits = [_hit(stroke="revers", pose="revers", frame=i) for i in range(10)]
    for h in hits:
        h.stroke_clf, h.clf_prob = "revers", 0.9
    unsure, sure = _hit(pose="revers", frame=100), _hit(pose="revers", frame=101)
    unsure.stroke_clf, unsure.clf_prob = "lob", 0.3
    sure.stroke_clf, sure.clf_prob = "lob", 0.8
    same = _hit(pose="vibora", frame=102)  # bandeja/víbora are one class: not a disagreement
    same.stroke_clf, same.clf_prob = "bandeja_vibora", 0.1
    picked = DS.select_for_annotation(hits + [sure, same, unsure], 8, already=set())
    assert [h.frame for h in picked[:2]] == [100, 101]


def test_collect_uses_saved_features_without_cache(tmp_path):
    from video_analyzer.padel import learn as L

    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "v-abc.json").write_text(json.dumps({"video": "v.mp4", "key": "v-abc", "labels": {
        "1:10:2": {"label": "smash", "annotator": "user"}, "1:20:2": {"label": "incertain", "annotator": "user"}}}))
    missing = []
    assert L.collect(tmp_path / "no_cache", tmp_path, missing) == [] and missing == ["v-abc"]
    x = np.arange(5, dtype=np.float32)
    L._save_features("v-abc", [("1:10:2", x, "", "bandeja"), ("1:20:2", x, "", "lob")], tmp_path)
    missing = []
    s = L.collect(tmp_path / "no_cache", tmp_path, missing)
    assert missing == [] and [(o.key, o.y, o.pose) for o in s] == [("1:10:2", "smash", "bandeja")]  # incertain dropped
    assert np.array_equal(s[0].x, x)


def test_classifier_beats_vlm_handles_missing_vlm_measure():
    from video_analyzer.padel.learn import classifier_beats_vlm

    assert classifier_beats_vlm({"cv_accuracy": 0.44, "vlm_accuracy": 0.22})
    assert not classifier_beats_vlm({"cv_accuracy": 0.20, "vlm_accuracy": 0.22})
    assert classifier_beats_vlm({"cv_accuracy": 0.44, "vlm_accuracy": float("nan")})  # hits analysed without VLM
    assert not classifier_beats_vlm(None)
