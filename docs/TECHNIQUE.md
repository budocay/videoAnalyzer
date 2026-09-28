# video-analyzer — documentation technique

Fonctionnement interne, mesures et formats. Pour l'installation et l'usage, voir le [README](../README.md).

## Sommaire

1. [Architecture et moteurs](#1-architecture-et-moteurs)
2. [Pipeline « vu / entendu »](#2-pipeline--vu--entendu-)
3. [Vidéos supportées](#3-vidéos-supportées)
4. [Analyse de padel](#4-analyse-de-padel)
5. [Boucle d'apprentissage](#5-boucle-dapprentissage)
6. [Modèles](#6-modèles)
7. [Performances mesurées](#7-performances-mesurées)
8. [Schémas JSON](#8-schémas-json)
9. [Organisation du code et tests](#9-organisation-du-code-et-tests)

---

## 1. Architecture et moteurs

| | Moteur `mlx` (Mac Apple Silicon) | Moteur `portable` (Windows, Linux, Mac Intel) |
|---|---|---|
| Modèle de vision (VLM) | mlx-vlm, Qwen3.5 9B/4B 8 bits | Ollama local (`/api/chat`), `qwen3-vl:8b/4b-instruct` |
| Transcription | mlx-whisper large-v3-turbo | faster-whisper large-v3-turbo (CUDA, sinon CPU int8) |
| Détection de parole | Silero VAD (mlx-audio) | Silero VAD (ONNX, inclus dans faster-whisper) |
| Pose (padel) | YOLO11n-pose, PyTorch MPS | YOLO11n-pose, PyTorch CUDA ou CPU |
| Décodage des images | VideoToolbox (matériel) puis logiciel en secours | ffmpeg logiciel |

Le moteur est choisi automatiquement (`src/video_analyzer/backend.py`). On peut forcer le choix avec
`VIDEO_ANALYZER_BACKEND=mlx|portable`. Le modèle de vision par défaut vient de `--model`, sinon de
`settings.json` (écrit par l'installateur), sinon de la machine (9B sur Mac ; 8B avec une carte NVIDIA
d'au moins 8 Go ; 4B sinon).

Fonctionnement hors ligne : une fois les modèles en cache, `HF_HUB_OFFLINE=1` est activé automatiquement.
Vérifié avec un Hub injoignable (`HF_ENDPOINT=http://127.0.0.1:9`).

---

## 2. Pipeline « vu / entendu »

```
ffmpeg ─┬─ images à N fps (rotation, tone-mapping HDR ; horodatage = index / fps, jamais demandé au modèle)
        └─ audio mono 16 kHz ─ VAD Silero ─ Whisper ─ transcription horodatée, filtrée par la VAD
                                                  │
VLM : 1 appel par segment de N s, avec les images du segment et les paroles qui le recouvrent
                                                  │
                      <nom>_analyse.json (schéma versionné) + <nom>_analyse.md (+ résumé)
```

- **Segments** : fenêtres fixes `[k·N, (k+1)·N)`.
- **Paroles** : la colonne « entendu » contient les énoncés dont le milieu tombe dans le segment, pour ne pas
  les dupliquer. Le prompt du VLM reçoit, lui, tous les énoncés qui recouvrent le segment.
- **Mémoire** : Whisper est entièrement libéré avant le chargement du VLM. mlx-whisper garde son modèle dans un
  attribut de classe, qu'il faut remettre à zéro.
- **Hallucinations** : large-v3-turbo invente du texte sur les sons sans parole (« Thank you. » sur un bip,
  avec `no_speech_prob = 0`). La VAD passe donc en premier : sans parole, Whisper n'est pas lancé ; sinon, seuls
  les énoncés qui recouvrent de la parole détectée sont gardés.
- **Résumé** : en une passe jusqu'à 40 segments, au-delà par résumés partiels puis un résumé global.
- **Génération** : décodage glouton (`temperature=0`). Le mode « thinking » de Qwen3.5 est désactivé via le
  chat template, et les balises `<think>` résiduelles sont retirées.
- **Taille des images** (`--size`) : c'est le côté long. Portrait et paysage coûtent donc le même nombre de
  tokens (~220 par image à 640 px).

---

## 3. Vidéos supportées

Tout ce que ffmpeg décode. Formats testés sur Mac :

| Cas | Exemple | Extraction |
|---|---|---|
| H.264 paysage SDR | mp4 1280x720 | VideoToolbox |
| HEVC 10 bits HDR HLG, portrait (rotation -90°), 4K 60 fps | vidéo Samsung | VideoToolbox + tone-mapping bt709 |
| HEVC 10 bits HDR10 (PQ) | mp4 | VideoToolbox + tone-mapping bt709 |
| Rotation 90° en métadonnée | mp4 | VideoToolbox + rotation |
| VP9 sans audio | webm | VideoToolbox |
| Clip de 2 s, dimensions impaires | mkv | VideoToolbox |
| ProRes 422 10 bits, audio PCM | mov | logiciel (4:2:2 non géré par le chemin matériel) |
| AV1 | mp4 | logiciel (pas de décodage AV1 matériel sur M1) |
| MPEG-4 Part 2, timestamps décalés | avi | logiciel |
| Audio seul avec pochette | mp3 | refusé proprement : « Pas de flux vidéo » |

- **HDR** : le tone-mapping HLG/PQ vers bt709 passe par `scale_vt` (VideoToolbox, macOS). Le ffmpeg standard
  n'a ni `zscale` ni `libplacebo`, donc le chemin logiciel ne tone-mappe pas. Le HLG reste correct, le PQ sort délavé.
- **Lecture complète du flux** (padel) : le décodage logiciel multi-thread est plus rapide que VideoToolbox
  (15 s contre 95 s pour 15 min en 1080p H.264), car le transfert de chaque image depuis le GPU domine.

---

## 4. Analyse de padel

| Étape | Méthode |
|---|---|
| 1. Plans | Coupes par distance d'histogrammes RGB (96x54, seuil 0.3). Chaque plan est classé par le VLM (caméra de match, autre angle, gros plan, public, graphique, replay). Deux garde-fous : corrélation avec l'image de référence de la caméra de match (seuil 0.6), et « pas de bandeau de score = replay ». |
| 2. Terrain | Lignes de service et ligne centrale (top-hat), puis jonctions sol/vitres latérales (Canny + Hough). On obtient 4 points exacts, donc la correspondance image ↔ terrain de 20×10 m, validée sur la ligne centrale. Le filet sert de secours. |
| 3. Score | Le VLM localise puis lit le bandeau avant et après chaque échange. Les lectures sont validées (format, tie-break, progression monotone). Le gagnant d'un point est déduit des règles de comptage. |
| 4. Équipes | Couleur du torse des joueurs regroupée en 2 groupes. Les noms lus sur les maillots dans les gros plans sont rattachés au groupe qui recueille le plus de lectures concordantes. |
| 5. Joueurs | YOLO11n-pose à pleine cadence (`imgsz=1920`). Pieds = milieu des chevilles, projetés sur le terrain. 2 joueurs par côté du filet, suivis par appariement hongrois. |
| 6. Frappes | Impacts sonores (1.5–6 kHz) confirmés par la vitesse du poignet, rapportée au bruit propre à chaque joueur. Décodage de Viterbi imposant l'alternance des équipes (≥ 0.45 s entre deux frappes). |
| 7. Coups | Classifieur entraîné sur les annotations s'il bat le VLM. Sinon, la pose (hauteur de frappe, distance au filet) restreint les coups possibles et le VLM choisit sur 3 recadrages 1080p, avec des exemples de référence. |
| 8. Rapport | Stats par joueur et par équipe, heatmaps sur le demi-terrain, séquence de chaque échange. Déroulé du score rédigé par le code ; style et moments clés rédigés par le VLM. |

Géométrie et règles (`padel/rules.py`, règlement FIP) :
- terrain de 20 × 10 m, filet à 0.88 m au centre ;
- lignes de service à 6.95 m du filet ;
- zones du rapport : filet < 4 m, transition jusqu'à 6.95 m, fond au-delà ;
- un joueur est désigné par son côté vu depuis sa propre moitié : droit = drive, gauche = revés.

Mesures sur la finale du Premier Padel Paris Major 2026 (montage beIN de 15 min 10, 1080p, M1 Max, VLM 9B) :

| Étape | Résultat | Temps |
|---|---|---|
| Coupes et images clés | 182 plans | 36 s |
| Classification des plans | 32 échanges gardés (7 min 22), 150 plans parasites retirés | ~16 min |
| Calibration du terrain | erreur de 1.9 px sur la ligne centrale | < 5 s |
| Lecture du score | 48/64 lectures valides : 6-3, 4-6, 6-6 (tie-break 6-1) | ~6 min |
| Pose YOLO (13 200 images) | 4 joueurs présents 85 à 100 % du temps, 22 ms/image | ~5 min |
| Frappes et vérification VLM | 315 frappes détectées ; gagnant connu pour 18 points sur 32 | ~24 min |
| Rapport | | ~30 s |

---

## 5. Boucle d'apprentissage

```
padel annotate → page HTML locale → export JSON → padel import-labels → data/labels/
padel train → classifieur (validation croisée) → padel eval → padel <vidéo> (classifieur utilisé s'il bat le VLM)
```

- **Page d'annotation** : une vue d'ensemble avec le frappeur encadré, puis 5 recadrages de −0.3 s à +0.3 s.
  Réponses au clavier, sauvegarde dans le navigateur. La proposition du modèle est masquée par défaut. Les frappes
  où la pose et le VLM se contredisent passent en premier.
- **Caractéristiques du classifieur** : séquence de pose du frappeur sur ±0.5 s (7 points clés, un pas sur 3).
  Elle est centrée sur les hanches et normalisée par la taille du corps ; les joueurs vus de face sont
  ramenés à une vue de dos, les gauchers à « bras raquette = droit ». S'y ajoutent distance au filet, position
  latérale, hauteur de frappe et vitesse du poignet.
- **Modèle** : petit réseau PyTorch (MLP), régularisé, entraîné en quelques secondes. Il peut aussi répondre
  « pas une frappe », ce qui écarte les fausses détections.
- **Évaluation** : proportion de vraies frappes parmi les détections, et précision du type de coup pour la pose,
  le VLM, pose + VLM et le classifieur (prédictions en validation croisée), avec la matrice de confusion.

Résultats sur la finale de Paris :

| Annotations | Classifieur (validation croisée) | VLM seul | Pose seule |
|---|---|---|---|
| 150 (amorçage par Claude) | 50 % | 27 % | 27 % |
| 150 (humain) | 36 % | 25 % | 26 % |
| **315 (humain)** | **44 %** | 22 % | 23 % |

Les chiffres sont mesurés sur les mêmes frappes. Avec 315 annotations, le classifieur écarte 31 fausses
frappes sur 54 et reconnaît bien services, lobs et smashs ; coup droit et volées restent souvent confondus.
Le coup droit ou revers déduit de la pose seule est au niveau du hasard (~50 %), pour les joueurs vus de face
comme pour ceux vus de dos.

Prochaines étapes : 500 à 1000 frappes annotées sur plusieurs matchs, un classifieur audio frappe / rebond / vitre,
puis le suivi de la balle (lobs, sorties de vitre, coups gagnants et fautes).

---

## 6. Modèles

| Rôle | Mac Apple Silicon | Autres machines |
|---|---|---|
| VLM principal | `mlx-community/Qwen3.5-9B-MLX-8bit` (10.45 Go) | `qwen3-vl:8b-instruct` (6.1 Go) |
| VLM rapide | `mlx-community/Qwen3.5-4B-MLX-8bit` (5.16 Go) | `qwen3-vl:4b-instruct` (3.3 Go) |
| Transcription | `mlx-community/whisper-large-v3-turbo` (1.61 Go) | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (~1.6 Go) |
| Détection de parole | `mlx-community/silero-vad` (2.2 Mo) | inclus dans faster-whisper |
| Pose | `yolo11n-pose.pt` (6.3 Mo, releases GitHub d'Ultralytics) | idem |
| Coups de padel | `stroke_clf.pt` (48 Ko, entraîné sur `data/labels/`) | idem |

Les noms des dépôts Hugging Face et des tags Ollama ont été vérifiés avant usage.

---

## 7. Performances mesurées

MacBook Pro M1 Max, 32 Go, macOS 27, mlx 0.32.2, mlx-vlm 0.7.2. Réglage par défaut : 1 fps, segments de 5 s,
images de 640 px (~1 280 tokens de prompt par segment).

Vidéo réelle de 11.6 s (HEVC 4K 60 fps HDR HLG, portrait, parole en français) :

| | 4B | 9B | Portable (Ollama 4B, CPU + MPS) |
|---|---|---|---|
| Extraction des images | 1.9 s | 2.0 s | 2.6 s |
| VAD + Whisper | 5 s | 4 s | 5.5 s |
| Par segment de 5 s | 5.4 – 6.9 s | 10.1 – 10.8 s | ~21 s (~5 500 tokens/segment) |
| Pic mémoire du VLM | 6.56 Go | 11.88 Go | non mesuré |
| Total avec résumé | 36 s | 52 s | 75 s |

- **Ordre de grandeur** : ~2 s de calcul par seconde de vidéo avec le 9B, ~1 s avec le 4B.
- **Temps par segment** : à peu près proportionnel au nombre de tokens de prompt (images × résolution).
  Ollama découpe les images de qwen3-vl plus finement que mlx-vlm, d'où environ 4 fois plus de tokens.
- **Estimation mémoire du `--dry-run`** : poids + 1.1 Go + 0.22 Go par millier de tokens de prompt (juste à 0.1 Go près).

---

## 8. Schémas JSON

`<nom>_analyse.json` (`schema_version` 1.0) :

```jsonc
{
  "schema_version": "1.0",
  "video":  { "path", "name", "key", "duration", "width", "height", "native_fps", "has_audio",
              "rotation", "codec", "pix_fmt", "hdr" },          // width/height = taille affichée
  "config": { "fps", "segment", "size", "lang", "audio", "summary", "frame_extraction" },
  "models": { "vlm", "whisper", "vad" },
  "audio":  { "language", "speech": [ { "start", "end" } ] },   // zones de parole (VAD)
  "transcript": [ { "start", "end", "text" } ],
  "segments": [ {
      "id", "start", "end",
      "frames": [ { "index", "t" } ],          // t en secondes = index / fps
      "seen":  "description VLM",
      "heard": "paroles dont le milieu tombe dans le segment",
      "stats": { "seconds", "prompt_tokens", "generation_tokens", "peak_gb", "finish_reason" }
  } ],
  "summary": "…" | null,
  "timings": { "transcribe_s", "extract_s", "vlm_load_s", "vlm_segments_s", "summary_s", "total_s", "vlm_peak_gb" },
  "extensions": {}
}
```

`<nom>_padel.json` (`schema_version` padel-1.0) : `meta` (équipes, couleurs, score, classifieur),
`court`, `shots`, `players_found`, `rallies` (score avant/après, gagnant, joueurs, frappes détaillées
avec `stroke`, `stroke_pose`, `stroke_vlm`, `stroke_clf`), `player_stats`, `stroke_stats`, `summary`, `timings`.

Toute modification incompatible incrémente `schema_version`.

---

## 9. Organisation du code et tests

```
src/video_analyzer/
  cli.py          commandes : description, doctor, padel (analyse, annotate, import-labels, eval, train)
  backend.py      choix du moteur (mlx / portable), GPU
  config.py       modèles, réglages par défaut, prompts
  extract.py      ffprobe/ffmpeg : infos vidéo, images horodatées, audio 16 kHz
  transcribe.py   VAD + Whisper + filtre anti-hallucination + cache
  vision.py       VLM MLX ou Ollama : load une fois, ask(prompt, images)
  timeline.py     segmentation, fusion vu/entendu, JSON, Markdown
  pipeline.py     orchestration de la description
  padel/          shots, court, players, strokes, score, rules, report, pipeline,
                  dataset, classifier, learn, video_io
scripts/          bootstrap.py (installation commune), install.ps1, package.py
install.sh, install.cmd, va.sh, va.cmd
data/             labels/, eval/, references/, annotation/
tests/            tests unitaires, sans modèle
```

```bash
.venv/bin/python -m pytest -q
```

Les tests couvrent : segmentation, fusion vu/entendu, schéma, filtre anti-hallucination, redimensionnement
et rotation, règles de score, décodage des frappes, invariance des caractéristiques (vue de face = vue de dos en
miroir), import des annotations, client Ollama (contre un faux serveur) et choix du moteur.
