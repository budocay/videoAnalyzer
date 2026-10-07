# video-analyzer — notes for future sessions

Start with docs/REPRISE.md (French handoff: history, status, what is untested, next steps).

Local-only video analysis. Developed on Apple Silicon (M1 Max, 32 GB) with MLX; also runs on
Windows / Linux / Intel Mac through the "portable" backend (see below). **No cloud calls, ever.** Hugging Face is only used to download weights once; `hub.py` then sets
`HF_HUB_OFFLINE=1` (verified: a run with `HF_ENDPOINT=http://127.0.0.1:9` succeeds).
Must work on **any video** ffmpeg can decode (user requirement): orientation, HDR, codec, no audio, short/long.

## Layout

```
src/video_analyzer/
  cli.py        argparse entry point (`video-analyzer`), --dry-run
  config.py     model presets (4b / 9b / qwen3vl), defaults, prompts
  extract.py    ffprobe/ffmpeg: VideoInfo (displayed size, rotation, HDR), frames via VideoToolbox or
                software fallback, 16 kHz mono wav, video_key()
  transcribe.py Silero VAD (mlx-audio) + mlx-whisper + hallucination filter + JSON cache
  hub.py        offline switch; must run before anything imports huggingface_hub
  vision.py     VLM wrapper: load once, ask(prompt, images) -> Answer(text, timings, tokens, peak)
  timeline.py   pure logic: segments, frame/transcript assignment, JSON schema, Markdown
  pipeline.py   orchestration + per-segment progress
  plan.py       dry-run plan + memory estimate
  memory.py     mx.clear_cache / peak helpers
tests/          pytest, no model needed (timeline, schema, VAD filter, fit/rotation, <think> cleaning)
video_analyze.py  original single-file draft, kept for reference only
samples/        test_synth.mp4, 20260924_114504.mp4 (user's real clip: Samsung HEVC HLG portrait),
                matrix/ (vp9, av1, prores, hdr10 pq, rot90, short odd mkv, avi, mp3+cover) + outputs
```

Environment: two machines, synchronised through git (pull first, push when done).
- Windows PC (Ryzen 7 5800X3D + RX 7900 XT 20 GB): `.venv` Python 3.12 + AMD ROCm 7.2.1 torch, made by install.cmd.
  Tests `.venv\Scripts\python -m pytest -q`, CLI `.\va.cmd`. Has the caches of the .mkv match, train1, train2.
- Mac M1 Max 32 GB (back in use since 2026-10-07): `.venv` Python 3.14 + MLX. Tests `.venv/bin/python -m pytest -q`,
  CLI `./va.sh`. Has the cache of the Paris final (.mp4, key …5eacad63229d).
Do not migrate to uv. Videos and caches never travel through git; labelled-hit features do (data/features/).

## Decisions

- Frame timestamps = `index / fps`, computed by us (ffmpeg `fps` filter grid), never asked to the model.
- Segments are fixed windows `[k*segment, (k+1)*segment)`; one VLM call per segment with all its frames.
- Transcript fusion: `heard` = utterances assigned by **midpoint** (no duplication across segments);
  the VLM prompt `context` = every utterance **overlapping** the window.
- Whisper runs first and is fully released before the VLM loads (see pitfalls).
- Output JSON has `schema_version` (currently `1.0`). Bump it on any breaking change. Phase 2
  writes into `extensions` (e.g. `extensions.pose`) without touching existing keys.
- Cache under `~/.cache/video-analyzer/<stem>-<sha1(name,size,first/last MiB)[:12]>/`:
  `audio_16k.wav`, `transcript_v2_<model>_<lang>.json` (utterances + VAD regions + language),
  `frames_fps<fps>_s<size>/` (`.done` marker holds the extraction method).
  Delete that folder to force recomputation.
- Default VLM is the 9B; the 4B is the smoke-test / fast option (`--model 4b`).
- Greedy decoding (`temperature=0`) for reproducible descriptions.
- `--size` = longest side (not width): portrait and landscape cost the same tokens (~220/frame at 640).
- Frame extraction: VideoToolbox first (`-noautorotate`, `scale_vt` in stored orientation, HDR→bt709
  tone mapping via scale_vt color_* options, `hwdownload,format=nv12|p010le` chosen from pix_fmt,
  then software `transpose` like ffmpeg autorotate). Any failure or non-4:2:0 input → software path.
- Audio: VAD first; no speech → Whisper skipped. Utterances kept only if they overlap VAD speech
  (≥ min(0.2 s, half their length)), clamped to the audio duration, and contain a letter.
- Summary: single pass up to 40 segments, else chunk summaries then a global one.
- Prompts forbid guessing intent/place type and ask the summary for plain Markdown (no emoji/table/intro).

## Verified library facts (mlx-vlm 0.7.2, mlx-whisper 0.4.3, mlx 0.32.2)

- `mlx_vlm.load(repo) -> (model, processor)`; config is `model.config` (no need for `load_config`).
- `apply_chat_template(processor, config, prompt, num_images=N, **kw)`: kw are forwarded to the HF
  template. mlx-vlm already defaults `enable_thinking=False` when the template accepts it
  (`prompt_utils.py` ~l.814). Qwen3.5's template then emits an empty `<think>\n\n</think>` block.
  Without that flag the template turns thinking ON. We pass it explicitly and still strip `<think>`.
- `qwen3_5` is `LIST_WITH_IMAGE_FIRST` and not in `SINGLE_IMAGE_ONLY_MODELS`: multi-image works.
  mlx-vlm also has a native `video=` path for qwen3_5; not used (we want our own timestamps).
- `generate(...) -> GenerationResult` with `.text`, `.prompt_tokens`, `.generation_tokens`,
  `.generation_tps`, `.peak_memory` (GB), `.finish_reason`.
- `mlx_whisper.transcribe(path, path_or_hf_repo=, language=) -> {"text", "segments", "language"}`.
- Qwen3.5 vision: patch 16, merge 2 → one token per 32×32 px. 640×360 ≈ 220 tokens/frame,
  960×540 ≈ 510.

## Pitfalls hit

- Whisper large-v3-turbo hallucinates on non-speech ("Thank you." over a 440 Hz tone, end=29.98 s on a
  2 s clip) with `no_speech_prob=0.00`: its own confidence is useless → Silero VAD gate.
- ffmpeg skips autorotate when frames stay on the GPU (videotoolbox_vld): rotate manually after hwdownload.
- `hwdownload,format=nv12|p010le` does not negotiate; pick the exact format (10-bit → p010le).
- Software decode of 4K HEVC 60 fps is very slow (22 s for 12 frames of an 11 s clip); VideoToolbox: 2 s.
- Phone videos store landscape pixels + display-matrix rotation: ffprobe width/height must be swapped.
- Every `load()` hits the Hub for a revision check even when cached → HF_HUB_OFFLINE once cached.

- **mlx-whisper keeps the model in `ModelHolder` (class attribute).** `gc.collect()` + `mx.clear_cache()`
  free nothing; ~1.6 GB stayed resident under the VLM. Fixed by resetting `ModelHolder.model`.
- huggingface_hub 1.x (xet) stores blobs as `blobs/<xx>/<hash>` behind symlinks: `du` on the cache
  under-reports; use `stat -L` / `hf cache ls`. `.incomplete` files are preallocated, so their size is
  not a progress indicator.
- Homebrew ffmpeg 9 has no `drawtext` (no freetype), and `drawbox` does not re-evaluate `t` per frame.
  Use `overlay=x='t*100'` for moving test shapes.
- The original draft's `free_mlx()` fell back to `mx.metal.clear_cache` (deprecated); use `mx.clear_cache`.
- The 4B over-interprets synthetic content (calls testsrc2 animations "playback cursors") and the
  summary invents intent ("stability test"). The 9B is more literal but hallucinated a "pause at 00:03"
  in a uniform motion. On the real office clip the 9B is very accurate (red/white mechanical keyboard,
  earbuds, yellow mug, glass door, even reads "11:45" on a laptop screen); the 4B said "static camera"
  while it moved and guessed the place type ("coworking or café").

## Measured on M1 Max 32 GB (1 fps, 5 s segments, 640 px longest side)

- Real 11.6 s clip (4K HEVC HLG portrait): 9B total 52 s (10.1–10.8 s/segment, peak 11.88 GB);
  4B total 36 s (5.4–6.9 s/segment, peak 6.56 GB). VAD+Whisper ~5 s, extraction 2 s.

- Synthetic 30 s: 4B: ~4.8 s/segment, VLM peak 6.51 GB. 9B: ~9.8 s/segment, VLM peak 11.83 GB. Whisper turbo: 2.4 s / 30 s, 2.5 GB.
- Time per segment scales ~linearly with prompt tokens (4B, 8 frames @ 960 px = 4.2k tokens: 12.8 s, 7.19 GB).
- Dry-run memory model: weights + 1.1 GB + 0.22 GB per 1k prompt tokens (matches both models within 0.1 GB).

## Phase 2 — padel (`video-analyzer padel`, package `src/video_analyzer/padel/`)

User wants a full match analysis, working on highlight montages too (parasites removed by the local VLM),
grounded in padel rules, and accepts my changes without asking. Test file: samples/padel_paris_final_hl.mp4
(beIN highlights, Premier Padel Paris Major 2026 final, Tapia/Coello red vs Galán/Chingotto black, 15 min).

Modules: `video_io` (ffmpeg rawvideo streaming) · `shots` · `court` · `players` · `strokes` · `score` ·
`rules` (FIP geometry, scoring, stroke vocabulary) · `report` · `pipeline`. Tests: tests/test_padel.py.

Decisions / facts measured on the Paris final:
- Full-rate decode: software multithreaded ffmpeg (15 s for 15 min 1080p H.264) beats VideoToolbox (95 s,
  per-frame GPU download). VideoToolbox stays for sparse HDR extraction.
- Cuts: 512-bin RGB histogram TV distance at 96x54, threshold 0.3 (bimodal: p99 0.12, p99.5 0.50). 182 shots.
- Shot filter: VLM type + correlation with match-camera reference (64x36 gray, threshold 0.6: match cam
  0.73–0.92 as shadows move, others ≤ 0.45) + "no scoreboard ⇒ replay" when ≥70 % of match-cam shots show it.
  The VLM alone called a net-level presentation shot and a drone shot "principale".
- Court: service lines + centre line (top-hat white) + sidelines (Canny/Hough through far service-line ends)
  → 4 exact points → homography; residual 1.9 px on the centre line. Failed attempts: floor colour mask
  (blue boards/glass share the hue), far-wall row (sponsor boards), net width (mesh overhangs 10 m by ~5 %).
- Pose: yolo11n-pose at imgsz 1920 (1280 misses far players), ~60 ms/frame on MPS. Feet = ankle midpoint.
- Hits: audio onsets include bounces/glass; wrist speed normalised by each player's median (far players jitter);
  Viterbi with team alternation + 0.45 s min gap. VLM confirms/labels on 3 crops (−0.2/0/+0.2 s).
- Players are "<team> · drive|revés" (right/left half from their own perspective; far team faces camera).
- Team ↔ colour from names printed on shirts in close-ups (VLM), matched to torso-colour of tracks.

Pitfalls hit in phase 2:
- VLM shirt reading attributes other players' shirt colours → majority vote failed (both teams "orange").
  Fixed: foreground-player prompt, accent-stripped names, 2-means on track torso colours, and the team↔cluster
  permutation that agrees with most VLM observations (score.assign_teams).
- Reading the score only at rally starts gave 2/32 point winners in a montage; reading also the last
  scoreboard frame before the next rally (close-ups show the update) gives 18/32. Reads are validated
  (point format, tie-break ints, monotonic progress) — the VLM sometimes drops a set column.
- The 9B summary invented set scores and mistook "drive/revés" labels for strokes → score narrative is
  generated in code (score.score_story); players are labelled "côté droit/gauche"; the VLM only writes style +
  moments picked from rallies with a known winner.
- VLM picked "víbora" for low shots and 106 víboras vs 4 bandejas; confidence always ~0.95 (useless).
  Fixed: VLM chooses only among strokes allowed by pose height (±2 frames around the speed peak; ±5 caught
  the racket preparation) and distance to net; bandeja/víbora merged in the report.
- Full cold run ≈ 50 min for the 15-min montage; report-only rerun 30 s (everything cached).
- Windows pipes: `Popen(bufsize=<large>)` + `read(size)` capped 1440p rawvideo at 20 img/s (Python 1 core, GPU idle);
  `bufsize=0` + `readinto` a preallocated frame in a reader thread → 95 img/s decode, pose 11.6 → 48 img/s on the
  RX 7900 XT. fp16 rejected (moves keypoints). Then ffmpeg downscales to IMGSZ itself (ultralytics' CPU letterbox
  fought ffmpeg for cores) and pose is sampled at POSE_FPS=30 → 73 img/s = 2.4 s of video/s on 1440p60.
- samples/train2.mp4 (FIP Platinum Lyon, Sager/Cepero vs Leal/Guerrero, 16 min) = low fixed camera behind the near
  glass, **condensed** (dead time cut, no visible cut: histogram distance 0.03–0.09 at the cuts). Fixes it needed:
  jump cuts from isolated motion spikes (shots.jump_cuts, 77 cuts ≈ one per point); court method `sidelines_low`
  (far half only seen through the net mesh, whose strands make every row look like a line → depth from near service
  line + net base + sidelines' vanishing point, court.row_of; TV method runs first, identical results on 22 TV
  frames); tracking reach grows 8 m/s while a slot is unseen (a fixed 2.5 m lost players for good on long shots:
  far players present 3 % → 34 %). Audio is useless there for rally splitting (adjacent courts: no gap > 4 s).
  The scoreboard bbox can be asked to Qwen3-VL (normalised 0–1000 `bbox_2d`, exact on Lyon).
- All stroke/hit settings are per *frame* (wrist speed/frame, smash speed > 0.45, ±2 frames, classifier ±15 frames),
  tuned at ~30 img/s: never feed tracks at another rate (players.pose_fps). hit_crops decodes at native fps.

Learning loop (padel/dataset.py, classifier.py, learn.py; CLI `video-analyzer padel annotate|import-labels|eval|train`):
- Ground truth in data/labels/<key>.json (repo, not cache). Bootstrap: 150 hits labelled by Claude from image
  strips (annotator "claude", 42 "incertain"); the user's labels must replace them (annotate re-offers them).
- First measurement (Claude labels): hit precision 77 %; stroke accuracy pose 27 %, VLM 27 %, pose+VLM 33 %;
  forehand/backhand from pose ≈ chance on both sides (YOLO L/R or the hip-axis rule is unreliable).
  Classifier (MLP on canonicalised pose sequence) 50 % CV and rejects ~25 % false hits → pipeline uses it
  automatically when cv_accuracy ≥ VLM accuracy (model at <cache>/models/stroke_clf.pt).
- Per-hit VLM check is automatic (`learn.use_vlm_for_strokes`): skipped when the classifier beats the VLM, since
  it overwrites the VLM label anyway (~5 s/hit, ~7.2k tokens on Ollama → >1 h for a 29-min match). `--vlm-strokes`
  forces it. NaN vlm_accuracy (training hits without VLM) counts as 0. Annotation order = pose vs VLM/classifier
  disagreements, least confident classifier first. Bundle stores pose_fps.
- Hits keep stroke_pose / stroke_vlm / stroke_clf separately for evaluation; `reconcile` re-applies pose rules
  to cached hits. Feature canonicalisation is unit-tested (front view == mirrored back view).

Next steps: human labels (300–500 hits) · audio classifier hit/bounce/glass · ball tracking (TrackNet-style) for lobs/glass exits and winners/errors; audio classifier
hit vs bounce; per-player identity by name (jersey OCR in close-ups is not linked to court slots yet);
continuous (non-montage) match to validate rally/point segmentation; stroke labels to fine-tune.

## Cross-platform (backend.py, scripts/)

- `backend.name()`: "mlx" on macOS arm64 with mlx installed, else "portable"; override VIDEO_ANALYZER_BACKEND.
- Portable = VLM via local Ollama HTTP API (`vision.OllamaVLM`, /api/chat, base64 images, temperature 0,
  keep_alive 0 on close) with `qwen3-vl:8b-instruct` (≥ 8 GB NVIDIA VRAM) or `qwen3-vl:4b-instruct`
  (tags verified on ollama.com); faster-whisper `large-v3-turbo` (→ mobiuslabsgmbh/faster-whisper-large-v3-turbo,
  CTranslate2 `model.bin`), device auto → CPU int8 fallback on any CUDA error; VAD = faster_whisper.vad (Silero ONNX).
  Pose on cuda/mps/cpu (`backend.torch_device`). VideoToolbox only on darwin.
- Verified on this Mac with VIDEO_ANALYZER_BACKEND=portable and the user's Ollama: 11.6 s clip → 75 s,
  ~5.5k prompt tokens/segment (Ollama's qwen3-vl tiles images finer than mlx-vlm's 1.3k).
- pyproject uses environment markers (MLX deps only on darwin/arm64, faster-whisper elsewhere).
- All file IO is explicit UTF-8 and the CLI reconfigures stdout (Windows cp1252 consoles).
- Installers: install.sh (brew / apt / dnf / pacman / zypper + ollama.com script), install.cmd →
  scripts/install.ps1 (winget: Python.Python.3.12, Gyan.FFmpeg, Ollama.Ollama), both → scripts/bootstrap.py
  (stdlib only: venv, torch CUDA/CPU index choice, pip -e, models, doctor, tests, --demo). `--model` is saved in
  settings.json (read by config.default_vlm_choice). `video-analyzer doctor` = environment check.
- scripts/package.py → dist/video-analyzer.zip (code + data/labels,eval,references + trained stroke_clf from
  the cache into models/, installed by bootstrap). Fresh install from the zip verified on macOS arm64 only;
  Windows and Linux installers are untested on real machines (no pwsh here to even lint install.ps1).

## Documentation

- README.md = user/colleague-facing (prerequisites, install per OS, doctor, usage, troubleshooting, update/uninstall).
- docs/TECHNIQUE.md = internals, measurements, schemas. Keep measured numbers there, not in the README.
- Latest learning numbers (user labels, 315 hits on the Paris final): classifier 44 % CV vs VLM 22 % / pose 23 %.
- 2nd match (samples/padel_paris_final_hl.mkv, despite its name Coello/Tapia vs Stupaczuk/Lebrón, VP9 1440p60,
  29 min): 196 user labels, 71 are "pas_une_frappe" (hit precision ~62 %). The Paris-final classifier scored 27 % on it
  (47 % CV on its own match) and rejected 0/71 false hits → CV on one match overestimates. Retrained on this match only:
  ~35 % on strokes, 43/71 false hits rejected, no bandeja_vibora class (2 examples).
- The Paris final's 315 labels were recovered on 2026-10-07 by running `padel train` on the Mac (its cache is there):
  data/features/padel_paris_final_hl-5eacad63229d.npz, 251 usable hits → 604 labelled hits over 4 matches.
- **Honest metric = held-out match** (`classifier.train` → report["by_video"], bundle["unseen_match_accuracy"]):
  k-fold CV 46 % but 35 % on an unseen match; stroke type alone 18–30 %. Adding the 251 Paris hits changed the other
  matches by −3…+4 points (noise): more labels of the same kind do not help. On an unseen match (3 seeds):
  stroke family fond/volée/haut/service 57 % (majority 38 %) = real signal; forehand vs backhand 52 % (majority 52 %)
  = chance; hit vs not-hit 73 % (majority 70 %) = almost nothing. The pose keypoints carry no reliable side
  information → next lever is a new input (racket detection / image crops / ball), not more annotation.

## AMD / NVIDIA GPUs (portable backend)

- `backend.gpu()` → GPU(vendor, name, vram_gb): nvidia-smi; AMD Linux via sysfs mem_info_vram_total; AMD Windows via
  the display-class registry key HardwareInformation.qwMemorySize (WMI AdapterRAM is capped at 4 GB). Biggest VRAM wins.
- Whisper engine (`backend.whisper_engine`): mlx | faster-whisper (NVIDIA CUDA or CPU) | transformers (AMD with ROCm
  torch: `openai/whisper-large-v3-turbo`, pipeline dtype=fp16, return_timestamps chunks, return_language → mapped
  "french"→"fr" via TO_LANGUAGE_CODE). Verified on this Mac via MPS (same transcript as the other engines).
- ROCm torch is seen as device "cuda" (torch.version.hip set) → YOLO/transformers need no special code.
- bootstrap: AMD Windows → Python 3.12 venv (install.ps1 pins 3.12 when a Radeon RX/PRO is present) + AMD wheels
  from repo.radeon.com rocm-rel-7.2.1 (URLs checked HTTP 200); AMD Linux → pytorch.org rocm7.0 index. torch is
  installed before `pip install -e .` so ultralytics keeps it; afterwards torch.cuda.is_available() is checked.
- User's second machine: Ryzen 7 5800X3D + Sapphire RX 7900 XT Pulse (20 GB), Windows. Nothing GPU-related has been
  run on real NVIDIA/AMD hardware yet: first real test = `install.cmd --demo` + `va.cmd doctor` there.
