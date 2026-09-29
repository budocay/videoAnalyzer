import numpy as np

from video_analyzer.padel.players import ShotTracks
from video_analyzer.padel.report import own_half, slot_roles
from video_analyzer.padel.score import ScoreState, parse_rows, point_winner
from video_analyzer.padel.shots import split_shots
from video_analyzer.padel.strokes import NOT_HIT_SCORE, _viterbi


def st(games, points):
    return ScoreState(["A", "B"], [True, False], games, points)


def test_depth_rows_from_near_line_net_and_vanishing_point():
    """Low camera: the far service row is derived, not detected. Check against a known homography."""
    import cv2

    from video_analyzer.padel.court import row_of

    court = np.float32([[0, 0], [10, 0], [10, 20], [0, 20]])
    img = np.float32([[610, 712], [1310, 712], [2170, 1065], [-230, 1065]])  # like the Lyon stream
    Hi = cv2.getPerspectiveTransform(court, img)
    v = lambda y: cv2.perspectiveTransform(np.float64([[[5.0, y]]]), Hi)[0, 0, 1]
    left = cv2.perspectiveTransform(np.float64([[[0, 0]], [[0, 20]]]), Hi).reshape(-1)
    right = cv2.perspectiveTransform(np.float64([[[10, 0]], [[10, 20]]]), Hi).reshape(-1)
    line = lambda s: np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0])
    vp = np.cross(line(left), line(right))
    v_inf = vp[1] / vp[2]
    for y in (0.0, 3.05, 6.0):
        assert abs(row_of(y, v(16.95), v(10.0), v_inf) - v(y)) < 0.01


def test_pose_fps_caps_high_frame_rates():
    from types import SimpleNamespace

    from video_analyzer.padel.players import pose_fps

    assert pose_fps(SimpleNamespace(native_fps=60.0)) == 30.0
    assert pose_fps(SimpleNamespace(native_fps=50.0)) == 30.0
    assert pose_fps(SimpleNamespace(native_fps=29.97)) == 29.97  # settings were tuned at this rate
    assert pose_fps(SimpleNamespace(native_fps=25.0)) == 25.0


def test_jump_cuts_isolated_spikes_only():
    from video_analyzer.padel.shots import jump_cuts

    fps = 30
    m = np.full(3000, 0.3)
    m[300] = 4.0                 # jump cut: one-frame spike → cut
    m[900:910] = 4.0             # pan: sustained motion → no cut
    m[1500], m[1530] = 4.0, 4.0  # burst 1 s apart (flash) → only the first
    assert jump_cuts(m, fps) == [300, 1500]


def test_split_shots_merges_short_fragments():
    fps = 30
    d = np.zeros(300)
    d[[100, 105, 200]] = 0.8  # 105 is a flash 5 frames after a cut → merged
    assert split_shots(d, fps) == [(0, 100), (100, 200), (200, 300)]


def test_viterbi_enforces_team_alternation():
    far, near = 0, 1
    cands = [(0.0, np.array([5.0, 1.0])),   # far swing, weaker
             (0.8, np.array([6.0, 0.5])),   # far again: only one of the two can be a hit → the stronger
             (1.6, np.array([0.5, 7.0])),   # near hits
             (2.4, np.array([4.0, 0.2]))]   # far hits
    assert _viterbi(cands) == [None, far, near, far]


def test_viterbi_rejects_weak_evidence_and_min_gap():
    cands = [(0.0, np.array([NOT_HIT_SCORE / 2, 0.1])), (0.2, np.array([0.1, 9.0])), (0.3, np.array([5.0, 0.1]))]
    got = _viterbi(cands)
    assert got[0] is None and got[1] == 1 and got[2] is None  # 0.3 s after a hit: too close


def test_point_winner_padel_scoring():
    assert point_winner(st([[2], [3]], ["15", "30"]), st([[2], [3]], ["30", "30"])) == 0
    assert point_winner(st([[2], [3]], ["40", "30"]), st([[2], [3]], ["40", "40"])) == 1
    assert point_winner(st([[2], [3]], ["A", "40"]), st([[2], [3]], ["40", "40"])) == 1  # advantage lost
    assert point_winner(st([[2], [3]], ["40", "15"]), st([[3], [3]], ["0", "0"])) == 0   # game
    assert point_winner(st([[5], [4]], ["40", "0"]), st([[6, 0], [4, 0]], ["0", "0"])) == 0  # set
    assert point_winner(st([[2], [3]], ["0", "30"]), st([[2], [3]], ["30", "30"])) is None  # 2 points skipped
    assert point_winner(st([[6], [6]], ["3", "4"]), st([[6], [6]], ["3", "5"])) == 1  # tie-break
    assert point_winner(None, st([[0], [0]], ["0", "0"])) is None


def test_parse_rows_from_vlm_json():
    rows = [{"equipe": "Tapia / Coello", "service": True, "valeurs": ["0", "30"]},
            {"equipe": "Galan / Chingotto", "service": False, "valeurs": ["0", "15"]}]
    s = parse_rows(rows)
    assert s.teams == ["TAPIA / COELLO", "GALAN / CHINGOTTO"]
    assert s.games == [[0], [0]] and s.points == ["30", "15"] and s.serving == [True, False]
    assert parse_rows([rows[0]]) is None


def test_slot_roles_depend_on_side_facing():
    pos = np.full((10, 4, 2), np.nan)
    pos[:, 0] = [2, 3]; pos[:, 1] = [8, 3]    # far team: image-left / image-right
    pos[:, 2] = [2, 17]; pos[:, 3] = [8, 17]  # near team
    tr = ShotTracks(0, 0.0, 30.0, pos, np.zeros((10, 4, 17, 3)), np.zeros((4, 3)))
    r = slot_roles(tr)
    # far team faces the camera: image-left is its right (drive); near team: image-right is drive
    assert r == {0: "côté droit", 1: "côté gauche", 2: "côté gauche", 3: "côté droit"}


def test_own_half_normalises_both_sides():
    lat_f, dep_f = own_half(np.array([2.0]), np.array([3.0]), far=True)
    lat_n, dep_n = own_half(np.array([8.0]), np.array([17.0]), far=False)
    assert dep_f[0] == dep_n[0] == 7.0
    assert lat_f[0] == lat_n[0] == 8.0  # both on their own right side


def test_assign_teams_survives_noisy_shirt_readings():
    from video_analyzer.padel.score import assign_teams
    teams = ["TAPIA / COELLO", "GALAN / CHINGOTTO"]
    obs = [{"nom": n, "couleur": c} for n, c in [
        ("CHINGOTTO", "orange"),  # wrong: another player's shirt in the frame
        ("TAPIA", "orange"), ("AGUSTIN TAPIA", "rouge"), ("COELLO", "orange"),
        ("ALE GALAN", "noir"), ("GALAN", "noir"), ("CHINGOTTO", "noir"), ("SALVO", "noir")]]
    dark, orange = np.array([50.0, 45, 55]), np.array([200.0, 80, 55])
    order, votes = assign_teams(teams, obs, [dark, orange])
    assert order == ["GALAN / CHINGOTTO", "TAPIA / COELLO"] and votes == 6


def test_clean_sequence_drops_misreads():
    from video_analyzer.padel.score import clean_sequence
    seq = [st([[6, 4, 1], [3, 6, 3]], ["30", "40"]),
           st([[6, 4], [3, 6]], ["0", "3"]),          # dropped a set column → invalid / backwards
           st([[6, 4, 2], [3, 6, 4]], ["0", "15"]),
           st([[6, 4, 6], [3, 6, 6]], ["6", "1"])]    # tie-break: integer points are valid
    out = clean_sequence(seq)
    assert out[1] is None and out[0] is not None and out[2] is not None and out[3] is not None


def test_score_line():
    from video_analyzer.padel.score import score_line
    s = score_line([st([[6, 4, 6], [3, 6, 6]], ["6", "1"])])
    assert s == "A vs B — set 1 : 6-3 · set 2 : 4-6 · set 3 : 6-6 (tie-break 6-1, dernier score lu)"


def test_allowed_strokes_follow_height_and_distance():
    from video_analyzer.padel.strokes import Hit, allowed_strokes
    h = Hit(0, 0.0, 0, 2, "proche", 5, 18, 8.0, 0.3, 5.0, "droit", height="bas")
    a = allowed_strokes(h)
    assert "vibora" not in a and "volee_cd" not in a and "coup_droit" in a and "service" in a
    h.height, h.dist_net = "tete", 2.0
    assert set(allowed_strokes(h)) == {"bandeja", "vibora", "smash"}


def test_score_story_names_set_winners_and_tiebreak_leader():
    from video_analyzer.padel.score import score_story
    s = score_story([st([[6, 4, 6], [3, 6, 6]], ["6", "1"])])
    assert "Set 1 gagné par A (6-3)." in s and "Set 2 gagné par B (6-4)." in s
    assert "tie-break en cours : A 6, B 1 (A mène)." in s
