# Reprise du projet sur une autre machine

Récapitulatif de tout ce qui a été fait et décidé, pour reprendre le travail ailleurs, par exemple sur le PC
Windows (Ryzen 7 5800X3D + Radeon RX 7900 XT 20 Go). Mis à jour le 28 septembre 2026.

> **Pour Claude Code sur la nouvelle machine** : lire ce fichier, puis `CLAUDE.md` (décisions techniques et pièges),
> puis `docs/TECHNIQUE.md` (fonctionnement détaillé et mesures). Tout y est ; l'historique de conversation ne suit pas.

---

## 1. Le projet en bref

Outil d'**analyse vidéo 100 % locale**, sans aucun appel cloud (contrainte forte depuis le départ).

- **Description d'une vidéo** : timeline « vu / entendu » en Markdown + JSON, résumé optionnel.
  ```
  va.cmd ma_video.mp4 --lang fr --summary
  ```
- **Analyse de match de padel** : tri des plans parasites par le modèle de vision local, calibration du terrain,
  score lu sur le bandeau, pose et suivi des 4 joueurs, détection et type des frappes, rapport + heatmaps.
  ```
  va.cmd padel match.mp4
  ```
- **Boucle d'apprentissage** : page d'annotation des frappes, puis évaluation chiffrée et classifieur de coups
  entraîné sur les annotations.
- **Multi-plateforme** : MLX sur Mac Apple Silicon ; Ollama + Whisper + PyTorch sur Windows/Linux, avec GPU NVIDIA
  (CUDA) ou AMD (ROCm).

Dépôt : `git@github.com:budocay/videoAnalyzer.git` (branche `main`).

---

## 2. Chronologie de ce qui a été fait

1. **Phase 1 : description vu / entendu** (Mac M1 Max 32 Go)
   - Refonte du premier script `video_analyze.py` en package `src/video_analyzer/`.
   - Modèles vérifiés sur Hugging Face : Qwen3.5-9B et 4B en MLX 8 bits, whisper-large-v3-turbo.
   - Correctifs importants :
     - Whisper n'était pas libéré de la mémoire (1.6 Go restaient occupés) ;
     - Whisper inventait des phrases sur les sons sans parole, d'où l'ajout d'une détection de parole (Silero VAD) ;
     - le programme se reconnectait à Hugging Face à chaque lancement, d'où le mode hors ligne automatique.
2. **« N'importe quelle vidéo »** : portrait, rotation, HDR (tone-mapping), HEVC 4K, VP9, AV1, ProRes, vidéo
   sans audio, clip très court… Tous testés.
3. **Phase 2 : padel**, sur la finale du Premier Padel Paris Major 2026 (montage beIN de 15 min, Tapia/Coello
   en rouge contre Galán/Chingotto en noir).
   - 182 plans ; 32 échanges gardés, 150 plans parasites retirés.
   - Terrain calé à 1.9 px près.
   - Score lu : 6-3, 4-6, 6-6, tie-break 6-1 (dernier score visible dans le montage).
   - Gagnant identifié pour 18 points sur 32.
   - 4 joueurs suivis, environ 600 m parcourus chacun.
4. **Apprentissage** : 315 frappes annotées à la main, puis un classifieur de coups entraîné dessus.
5. **Installation plug and play** :
   - `install.cmd` (Windows), `install.sh` (macOS/Linux), commande `doctor` ;
   - archive à transmettre (`scripts/package.py`) ;
   - README complet et documentation technique.
6. **Git** : dépôt créé et poussé sur GitHub.
7. **Cartes AMD/NVIDIA** :
   - détection de la carte et de sa mémoire ;
   - PyTorch ROCm (Windows : paquets officiels AMD 7.2.1 ; Linux : ROCm 7.0) ou CUDA ;
   - transcription sur GPU AMD via transformers ;
   - modèle 8B choisi automatiquement dès 8 Go de mémoire vidéo.

---

## 3. Où on en est (chiffres clés)

### Reconnaissance des types de coups (padel)

| Annotations | Classifieur | Modèle de vision seul | Pose seule |
|---|---|---|---|
| 150 | 36 % | 25 % | 26 % |
| **315** | **44 %** | 22 % | 23 % |

- **Ce qui marche bien** : le classifieur écarte bien les fausses frappes (31 sur 54) et reconnaît services,
  lobs et smashs.
- **Ce qui coince** : coup droit (4 sur 23) et volées coup droit / revers encore souvent confondus ; trop peu
  d'exemples de sortie de vitre (3).
- **Toutes les frappes de la finale de Paris sont annotées.** Pour progresser, il faut **d'autres vidéos**.

### Performances sur le Mac M1 Max

| Tâche | Temps |
|---|---|
| Description, modèle 9B | ~2 s de calcul par seconde de vidéo |
| Description, modèle 4B | ~1 s de calcul par seconde de vidéo |
| Padel, 15 min de montage (première fois) | ~50 min |
| Padel, rapport régénéré depuis le cache | ~30 s |

### Ce qui est vérifié, et ce qui ne l'est pas encore

| Élément | État |
|---|---|
| Tout le programme sur Mac Apple Silicon, installateur compris (depuis l'archive, dossier vierge) | ✅ testé |
| Moteur Windows/Linux (Ollama + faster-whisper), exécuté sur le Mac | ✅ testé |
| Transcription via transformers (chemin AMD), exécutée sur le GPU du Mac | ✅ testé |
| `install.cmd` / `install.ps1` sur un vrai Windows | ❌ jamais exécuté (pas de PowerShell sur le Mac pour vérifier) |
| PyTorch ROCm sur la RX 7900 XT | ❌ jamais exécuté (pas de carte AMD disponible) |
| Installateur Linux, cartes NVIDIA | ❌ jamais exécutés |

**La première installation sur le PC AMD est donc le premier vrai test Windows + ROCm.**

---

## 4. Reprendre sur le PC Windows AMD : checklist

### 4.1 Avant d'installer

- [ ] **Windows 11** (ROCm pour Windows ne gère pas Windows 10).
- [ ] **Pilote AMD Adrenalin 26.2.2 ou plus récent** (exigé par PyTorch ROCm 7.2.1).
- [ ] **Git** et une clé SSH GitHub. Sans clé SSH, cloner en HTTPS :
      `git clone https://github.com/budocay/videoAnalyzer.git`.
- [ ] Environ **15 Go** de libre.

### 4.2 Installer

```bat
git clone git@github.com:budocay/videoAnalyzer.git
cd videoAnalyzer
install.cmd --model 9b --demo
va.cmd doctor
```

L'installateur installe Python 3.12 (imposé par AMD), ffmpeg, Ollama, PyTorch ROCm et les modèles, dont
`qwen3-vl:8b-instruct` (6.1 Go) et `openai/whisper-large-v3-turbo` (1.6 Go).

### 4.3 Ce que `doctor` doit afficher

```
OK  carte graphique AMD Radeon RX 7900 XT · 20 Go
OK  pose (torch)   accélération ROCm
OK  transcription  transformers · GPU AMD (ROCm)
OK  VLM (Ollama)   qwen3-vl:8b-instruct prêt
```

Si une ligne diffère, par exemple « accélération CPU », l'analyse fonctionne quand même, mais plus lentement.
**Envoyer la sortie complète de `doctor` à Claude** pour corriger.

### 4.4 Récupérer ce qui ne passe pas par Git

Git ne transporte **ni les vidéos, ni les caches, ni les résultats**. À copier depuis le Mac (clé USB, réseau…) :

| Quoi | Depuis le Mac | Vers le PC | Pourquoi |
|---|---|---|---|
| Vidéo de la finale (416 Mo) | `samples/padel_paris_final_hl.mp4` | `videoAnalyzer\samples\` | **garder exactement le même nom** : le cache est lié au nom, à la taille et au contenu |
| Cache d'analyse padel (45 Mo) | `~/.cache/video-analyzer/padel_paris_final_hl-5eacad63229d/` | `%USERPROFILE%\.cache\video-analyzer\` | évite ~50 min de recalcul et permet de **réentraîner** le classifieur |
| Autres vidéos (optionnel) | `samples/20260924_114504.mp4`, `samples/train1.mp4` (625 Mo, pas encore analysée) | `videoAnalyzer\samples\` | |

Sans la vidéo et son cache, `padel train` ne trouvera pas les frappes annotées : il lui faut les données de pose
de la vidéo. Le classifieur **déjà entraîné** est, lui, dans le dépôt (`models/`) et installé automatiquement.

Test rapide une fois copiés :

```bat
va.cmd padel samples\padel_paris_final_hl.mp4 --out samples\out_padel
```

Le rapport doit se régénérer en une minute environ, tout venant du cache.

---

## 5. Commandes utiles

| Action | Commande (Windows) |
|---|---|
| Vérifier l'installation | `va.cmd doctor` |
| Décrire une vidéo | `va.cmd video.mp4 --lang fr --summary` |
| Analyser un match | `va.cmd padel match.mp4` |
| Analyse plus rapide (coups sans le modèle de vision) | `va.cmd padel match.mp4 --no-vlm-strokes` |
| Page d'annotation | `va.cmd padel annotate match.mp4 -n 200` |
| Importer des annotations | `va.cmd padel import-labels %USERPROFILE%\Downloads\labels_<nom>.json` |
| Réentraîner le classifieur | `va.cmd padel train` |
| Mesurer la précision | `va.cmd padel eval match.mp4` |
| Créer l'archive à transmettre | `.venv\Scripts\python scripts\package.py` |
| Tests automatiques | `.venv\Scripts\python -m pytest -q` |

Sous macOS/Linux, remplacer `va.cmd` par `./va.sh`.

Règles d'annotation (important pour la qualité) :
- **`0` = « pas une frappe »** quand le joueur encadré ne frappe pas ;
- « incertain » seulement pour une image inexploitable.

---

## 6. Décisions et choix à connaître

- **Contrainte absolue : 100 % local.** Internet sert uniquement à télécharger les modèles pendant l'installation.
  Ollama n'écoute que sur 127.0.0.1.
- **Modèle de vision** : 9B/8B par défaut quand la machine le permet. `--model` est mémorisé dans `settings.json`.
- **Padel** :
  - les joueurs sont désignés par **équipe + côté** (côté droit = drive, côté gauche = revés), pas par leur nom ;
  - **bandeja et víbora sont regroupées** : pas distinguables de façon fiable à la résolution d'une retransmission ;
  - le **déroulé du score est rédigé par le code**, parce que le modèle de vision se trompait sur qui menait.
- **Classifieur** : le rapport ne l'utilise que s'il bat le modèle de vision en validation croisée.
- **Commits** en français avec emoji. L'identité Git du dépôt est `budocay`.
- Environnement Python : `.venv` dans le dossier du projet, pas de uv.

---

## 7. Prochaines étapes possibles

Par ordre d'intérêt :

1. **Valider l'installation sur le PC AMD** (section 4) et corriger ce que `doctor` signale.
2. **Mesurer les performances sur la RX 7900 XT** : relancer la finale de Paris sans cache et comparer aux ~50 min du Mac.
3. **Nouvelles vidéos annotées** :
   - `samples/train1.mp4` est déjà là, à analyser (`va.cmd padel samples\train1.mp4`) puis annoter ;
   - idéalement, un de tes matchs filmé en continu (caméra fixe, en hauteur, derrière un fond de court) ;
   - objectif : 500 à 1000 frappes au total, au moins 50 par type de coup.
4. **Accélérations restantes** : décodage vidéo sur GPU (NVDEC pour NVIDIA, D3D11VA/AMF pour AMD), traitement de
   la pose par lots plus gros.
5. **Classifieur audio** frappe / rebond / vitre, pour fiabiliser la détection des frappes.
6. **Suivi de la balle** (type TrackNet) : lobs, sorties de vitre, coups gagnants et fautes.
7. **Nom des joueurs** : relier les noms lus sur les maillots aux joueurs suivis sur le terrain.

---

## 8. Phrase pour relancer Claude Code sur la nouvelle machine

À coller au début de la première session :

> Lis `docs/REPRISE.md`, `CLAUDE.md` et `docs/TECHNIQUE.md` pour reprendre le projet. Je suis sur le PC Windows
> avec la Radeon RX 7900 XT. Voici la sortie de `va.cmd doctor` : [coller la sortie]. On commence par la section 4
> de REPRISE.md.
