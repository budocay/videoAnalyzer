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
