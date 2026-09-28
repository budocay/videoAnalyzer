import json
from pathlib import Path

from video_analyzer.extract import Frame, frame_times
from video_analyzer.timeline import (SCHEMA_VERSION, assign_frames, assign_transcript,
                                     build_document, fmt, plan_segments, to_markdown)
from video_analyzer.transcribe import Utterance
from video_analyzer.vision import clean


def frames(duration, fps):
    return [Frame(i, t, Path(f"f_{i:06d}.jpg")) for i, t in enumerate(frame_times(duration, fps))]


def test_fmt():
    assert fmt(0) == "00:00"
    assert fmt(65.9) == "01:05"
    assert fmt(3725) == "01:02:05"


def test_plan_segments_exact_and_partial():
    assert [(s.start, s.end) for s in plan_segments(10, 5)] == [(0, 5), (5, 10)]
    assert [(s.start, s.end) for s in plan_segments(12.3, 5)] == [(0, 5), (5, 10), (10, 12.3)]
    assert [(s.start, s.end) for s in plan_segments(3, 5)] == [(0, 3)]
    assert plan_segments(0, 5) == []


def test_frame_times():
    assert frame_times(30, 1) == [float(k) for k in range(30)]
    assert len(frame_times(10, 2)) == 20
    assert frame_times(0.2, 1) == [0.0]


def test_assign_frames_by_timestamp():
    segs = plan_segments(12, 5)
    assign_frames(segs, frames(12, 1))
    assert [[f.t for f in s.frames] for s in segs] == [
        [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]


def test_assign_frames_overflow_goes_to_last():
    segs = plan_segments(10, 5)
    assign_frames(segs, [Frame(0, 0.0, Path("a")), Frame(1, 10.0, Path("b"))])
    assert [f.t for f in segs[-1].frames] == [10.0]


def test_transcript_heard_is_not_duplicated_but_context_overlaps():
    segs = plan_segments(15, 5)
    utts = [Utterance(0.5, 3, "a"), Utterance(4, 7, "b"), Utterance(11, 14, "c")]
    assign_transcript(segs, utts)
    assert [s.heard for s in segs] == ["a", "b", "c"]  # midpoint of b = 5.5 → segment 1
    assert segs[0].context == "a b"
    assert segs[1].context == "b"


def test_document_schema_is_stable_and_serializable():
    segs = plan_segments(10, 5)
    assign_frames(segs, frames(10, 1))
    assign_transcript(segs, [Utterance(1, 2, "bonjour")])
    segs[0].seen, segs[1].seen = "un carré", "un cercle"
    doc = build_document(video={"name": "v.mp4", "duration": 10.0}, config={"fps": 1.0, "segment": 5.0},
                         models={"vlm": "m", "whisper": "w"}, transcript=[Utterance(1, 2, "bonjour")],
                         segments=segs, summary=None, timings={})
    assert doc["schema_version"] == SCHEMA_VERSION
    assert set(doc) == {"schema_version", "video", "config", "models", "audio", "transcript",
                        "segments", "summary", "timings", "extensions"}
    assert set(doc["segments"][0]) == {"id", "start", "end", "frames", "seen", "heard", "stats"}
    assert doc["segments"][0]["frames"][1] == {"index": 1, "t": 1.0}
    json.dumps(doc)
    md = to_markdown(doc)
    assert "### 00:05 – 00:10" in md and "**Entendu :** bonjour" in md


def test_clean_strips_think_blocks():
    assert clean("<think>hmm\nok</think>\n Réponse.") == "Réponse."
    assert clean("</think>Réponse.") == "Réponse."


def test_fit_longest_side_even_no_upscale():
    from video_analyzer.extract import fit
    assert fit(1920, 1080, 640) == (640, 360)
    assert fit(2160, 3840, 640) == (360, 640)  # portrait keeps the same token budget
    assert fit(853, 479, 640) == (640, 360)
    assert fit(320, 240, 640) == (320, 240)


def test_transpose_matches_ffmpeg_autorotate():
    from video_analyzer.extract import _transpose
    assert _transpose(-90) == ["transpose=clock"]
    assert _transpose(90) == ["transpose=cclock"]
    assert _transpose(180) == ["hflip", "vflip"]
    assert _transpose(0) == []


def test_filter_utterances_drops_hallucinations_and_clamps():
    from video_analyzer.transcribe import filter_utterances
    raw = [
        {"start": 0.0, "end": 6.0, "text": " ."},              # no letters
        {"start": 7.0, "end": 9.5, "text": " Merci."},         # falls in a VAD gap
        {"start": 1.5, "end": 6.4, "text": " Bonjour."},
        {"start": 9.8, "end": 30.0, "text": " Fin."},          # end past the audio
    ]
    speech = [(1.6, 6.37), (9.89, 11.04)]
    got = filter_utterances(raw, speech, duration=11.65)
    assert [(u.start, u.end, u.text) for u in got] == [(1.5, 6.4, "Bonjour."), (9.8, 11.65, "Fin.")]
    # Whisper on a pure tone: "Thank you." with no detected speech → dropped
    assert filter_utterances([{"start": 0.0, "end": 29.98, "text": " Thank you."}], [], duration=2.02) == []
