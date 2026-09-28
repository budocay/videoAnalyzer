# video-analyzer

Analyse vidéo **100 % locale** : aucune image, aucun son ni aucun résultat ne quitte la machine.

- **Description d'une vidéo quelconque** : une timeline horodatée « ce qui est vu / ce qui est entendu »,
  en Markdown et en JSON, avec un résumé global optionnel.
- **Analyse d'un match de padel** : les plans parasites (gros plans, public, replays…) sont retirés.
  On obtient le score lu sur le bandeau, les positions et la distance parcourue par chaque joueur, les
  heatmaps, les frappes avec leur type de coup, et un rapport rédigé.

Fonctionne sur **Mac Apple Silicon, Windows et Linux**. Un script installe tout.

---

## Sommaire

1. [Prérequis](#1-prérequis)
2. [Installation](#2-installation)
3. [Vérifier l'installation](#3-vérifier-linstallation)
4. [Utilisation](#4-utilisation)
5. [Résultats et fichiers produits](#5-résultats-et-fichiers-produits)
6. [Dépannage](#6-dépannage)
7. [Mettre à jour, transmettre, désinstaller](#7-mettre-à-jour-transmettre-désinstaller)
8. [Confidentialité et limites](#8-confidentialité-et-limites)
9. [Pour aller plus loin](#9-pour-aller-plus-loin)

---

## 1. Prérequis

### Machine

| Machine | Moteur utilisé | Vitesse indicative |
|---|---|---|
| **Mac Apple Silicon** (M1 à M4), 16 Go de mémoire ou plus | MLX, sur le GPU Apple | description : ~2 s de calcul par seconde de vidéo · padel : ~50 min pour 15 min de vidéo |
| **PC avec carte NVIDIA**, 8 Go de mémoire vidéo ou plus | Ollama + faster-whisper + PyTorch CUDA, sur le GPU | non mesurée, du même ordre qu'un Mac |
| **PC avec carte AMD Radeon** (RX 7000/9000) | Ollama + Whisper (transformers) + PyTorch ROCm, sur le GPU | non mesurée |
| **PC sans carte dédiée**, Mac Intel | Ollama + faster-whisper, sur le processeur | lent : plusieurs heures pour un match de padel de 15 min |

Le script détecte la machine et choisit le moteur tout seul.

### Cartes AMD Radeon

Toute la chaîne peut tourner sur une Radeon :
- le modèle de vision, via Ollama ;
- la pose et la transcription, via PyTorch ROCm.

L'installateur détecte la carte et installe ce qu'il faut. Conditions fixées par AMD :

| | Windows | Linux |
|---|---|---|
| Système | **Windows 11** | Ubuntu, Debian, RHEL… récents |
| Pilote | **AMD Adrenalin 26.2.2 ou plus récent** | pilote `amdgpu` du noyau |
| PyTorch | ROCm 7.2.1, paquets AMD (`repo.radeon.com`) | ROCm 7.0 (index pytorch.org) |
| Python | **3.12 obligatoire** (installé automatiquement) | 3.11 ou plus |
| Cartes | RX 7000 (dont 7900 XTX/XT/GRE, 7800 XT, 7700 XT) et RX 9000 | idem |

AMD précise que sous Windows, « toute la pile ROCm n'est pas encore prise en charge ». Si PyTorch ne voit pas la
carte, `doctor` le signale : le modèle de vision reste alors sur la carte (Ollama), mais la pose et la
transcription passent sur le processeur.

### Espace disque

| Machine | Modèles | Environnement Python | Total |
|---|---|---|---|
| Mac Apple Silicon | ~12 Go (9B) ou ~7 Go (4B) | ~2 Go | **~14 Go** |
| PC avec GPU NVIDIA | ~8 Go | ~5 Go (PyTorch CUDA) | **~13 Go** |
| PC avec GPU AMD | ~8 Go | ~6 Go (PyTorch ROCm) | **~14 Go** |
| PC sans GPU | ~5 Go | ~2 Go | **~7 Go** |

Il faut aussi prévoir la place des vidéos et d'un cache d'analyse de quelques centaines de Mo par vidéo.

### Logiciels

Rien à installer à la main, **sauf** :

| Système | À avoir avant de lancer l'installateur |
|---|---|
| macOS | [Homebrew](https://brew.sh) |
| Windows 10 / 11 | **winget** (application « App Installer » du Microsoft Store, présente par défaut sur Windows 11) |
| Linux | `sudo` et l'un des gestionnaires de paquets `apt`, `dnf`, `pacman` ou `zypper` |

L'installateur ajoute lui-même, s'ils manquent : **Python 3.11+**, **ffmpeg** et **Ollama**
(Ollama seulement hors Mac Apple Silicon).

Une connexion internet n'est nécessaire **que pendant l'installation**, pour télécharger les modèles.
Ensuite, tout fonctionne hors ligne.

---

## 2. Installation

Décompresser `video-analyzer.zip`, ou cloner le dépôt, puis suivre la section de votre système.

### Windows

1. Ouvrir le dossier `video-analyzer`.
2. **Double-cliquer sur `install.cmd`.**
   - Si Windows affiche « Windows a protégé votre ordinateur », cliquer sur *Informations
     complémentaires* puis *Exécuter quand même*.
   - Accepter les fenêtres d'autorisation de winget (installation de Python, ffmpeg et Ollama).
3. Attendre le message **« Installation terminée »**. La fenêtre reste ouverte jusqu'à l'appui sur une touche.

Pour lancer aussi l'analyse d'une vidéo de démonstration à la fin, ouvrir un terminal dans le dossier et taper :

```bat
install.cmd --demo
```

### macOS

```bash
cd video-analyzer
./install.sh --demo
```

Si `./install.sh` répond *permission denied*, lancer `chmod +x install.sh va.sh` puis réessayer.

### Linux

```bash
cd video-analyzer
./install.sh --demo
```

Le mot de passe administrateur est demandé pour installer ffmpeg, Python et Ollama. Ollama est
installé avec son script officiel et démarre comme service.

### Ce que fait l'installateur

1. Installe les prérequis manquants (Python, ffmpeg, Ollama).
2. Crée un environnement Python isolé dans le dossier du projet (`.venv/`), sans toucher au Python du système.
3. Installe PyTorch adapté à la machine (CUDA si une carte NVIDIA est détectée, sinon CPU), puis le projet.
4. Télécharge les modèles : VLM Qwen, Whisper pour la transcription, YOLO pour la pose. Il installe aussi
   le classifieur de coups de padel fourni avec le projet.
5. Lance le diagnostic (`doctor`) et les tests automatiques.
6. Avec `--demo` : génère une vidéo de 10 s et l'analyse de bout en bout.

L'installateur peut être relancé sans risque : ce qui est déjà installé est réutilisé.

### Options de l'installateur

| Option | Effet |
|---|---|
| `--demo` | analyse une vidéo de démonstration à la fin |
| `--model 4b` ou `--model 9b` | taille du modèle de vision (mémorisée pour tous les lancements suivants) |
| `--skip-models` | ne télécharge pas les modèles maintenant (ils le seront au premier lancement) |
| `--video FICHIER` | analyse cette vidéo à la fin de l'installation |

Modèle de vision choisi par défaut :

| Machine | Modèle | Taille |
|---|---|---|
| Mac Apple Silicon | Qwen3.5-9B (MLX, 8 bits) | 10.5 Go |
| PC avec GPU NVIDIA ou AMD ≥ 8 Go | `qwen3-vl:8b-instruct` (Ollama) | 6.1 Go |
| Autres machines | `qwen3-vl:4b-instruct` (Ollama) | 3.3 Go |

Utiliser `--model 4b` sur un Mac de 16 Go ou pour aller plus vite. Le 4B est environ deux fois plus rapide
et un peu moins précis.

---

## 3. Vérifier l'installation

Depuis le dossier du projet :

| Système | Commande |
|---|---|
| Windows | `va.cmd doctor` |
| macOS / Linux | `./va.sh doctor` |

Exemple de sortie sur un Mac :

```
video-analyzer · Python 3.14.7 · Darwin arm64
OK  moteur         mlx (Apple Silicon, GPU Metal)
OK  ffmpeg         ffmpeg version 9.0.2
OK  ffprobe        ffprobe version 9.0.2
OK  pose (torch)   périphérique mps
OK  modèle HF      mlx-community/Qwen3.5-9B-MLX-8bit
OK  modèle HF      mlx-community/whisper-large-v3-turbo
OK  modèle HF      mlx-community/silero-vad
OK  pose (YOLO)    ~/.cache/video-analyzer/models/yolo11n-pose.pt

Tout est prêt.
```

- `OK` : l'élément est prêt.
- `--` : l'élément n'est pas bloquant et sera téléchargé au premier usage.
- `ERR` : l'élément est à corriger (voir [Dépannage](#6-dépannage)).

---

## 4. Utilisation

Toutes les commandes se lancent **depuis le dossier du projet**, avec le lanceur :

- Windows : `va.cmd …`
- macOS / Linux : `./va.sh …`

Les exemples ci-dessous utilisent `./va.sh`. Sous Windows, remplacer par `va.cmd`.

> Pour taper `video-analyzer …` directement, activer d'abord l'environnement :
> `source .venv/bin/activate` (macOS/Linux) ou `.venv\Scripts\activate` (Windows).

### 4.1 Décrire une vidéo (vu / entendu)

```bash
./va.sh ma_video.mp4 --lang fr --summary
```

Les fichiers produits sont `ma_video_analyse.md` et `ma_video_analyse.json`, à côté de la vidéo.

| Option | Défaut | Rôle |
|---|---|---|
| `--summary` | non | ajoute un résumé global |
| `--lang fr` | automatique | langue parlée ; à préciser si la détection automatique se trompe |
| `--model 4b` / `9b` | selon la machine | taille du modèle de vision |
| `--fps 2` | 1 | images analysées par seconde (plus = plus précis et plus lent) |
| `--segment 4` | 5 | durée d'un segment de description, en secondes |
| `--size 960` | 640 | taille des images envoyées au modèle (côté long, en pixels) |
| `--no-audio` | non | ignore le son |
| `--out DOSSIER` | dossier de la vidéo | où écrire les résultats |
| `--dry-run` | non | affiche le plan et une estimation de la mémoire, sans rien calculer |

Tous les formats lus par ffmpeg sont acceptés : mp4, mov, mkv, webm, avi… Les vidéos de téléphone
(portrait, HDR, 4K 60 fps) aussi.

### 4.2 Analyser un match de padel

```bash
./va.sh padel match.mp4
```

Cela produit `match_padel.md` (le rapport), `match_padel.json` (les données) et `match_padel_heatmaps.png`.

Le rapport contient :
- le score par set lu sur le bandeau de la retransmission, et le gagnant de chaque point quand on peut le déterminer ;
- pour chaque joueur, désigné par son équipe et son côté (côté droit = drive, côté gauche = revés) :
  frappes, distance parcourue, temps passé au filet, en transition et au fond ;
- la répartition des coups (service, coup droit, revers, volées, bandeja/víbora, smash, lob…) ;
- les heatmaps de présence sur le terrain ;
- la séquence de chaque échange et un commentaire de match.

Bonnes conditions de prise de vue : **caméra fixe, en hauteur, derrière un fond de court**, comme
les retransmissions officielles ou un téléphone posé en haut de la vitre. Les montages de « temps
forts » fonctionnent aussi, car les plans parasites sont retirés automatiquement.

| Option | Rôle |
|---|---|
| *(par défaut)* | types de coups par le classifieur entraîné s'il est plus fiable que le modèle de vision (c'est le cas aujourd'hui), sinon vérification de chaque frappe par le modèle de vision |
| `--vlm-strokes` | force la vérification de chaque frappe par le modèle de vision (lent : quelques secondes par frappe) |
| `--no-vlm-strokes` | jamais de vérification par le modèle de vision |
| `--no-summary` | pas de commentaire rédigé |
| `--model 4b` / `9b` | taille du modèle de vision |
| `--out DOSSIER` | où écrire les résultats |

La première analyse est longue, mais **tout est mis en cache**. Une nouvelle génération du rapport prend ensuite
une trentaine de secondes, et une analyse interrompue reprend là où elle s'est arrêtée.

### 4.3 Améliorer la reconnaissance des coups (annotation)

Le type de coup est reconnu par un classifieur qui apprend de vos annotations. Plus il y a
d'annotations, meilleur il devient.

1. **Générer la page d'annotation** d'une vidéo déjà analysée :
   ```bash
   ./va.sh padel annotate match.mp4 -n 200
   ```
   Une page s'ouvre dans le navigateur. Elle fonctionne sans serveur et sans internet.
2. **Annoter** : pour chaque frappe, regarder le joueur encadré en jaune et appuyer sur la touche du coup.
   - **`0` = « pas une frappe »** quand le joueur encadré ne frappe pas. C'est important : c'est ce qui
     apprend au programme à écarter les fausses détections.
   - « incertain » seulement si l'image est inexploitable.
   - La progression est sauvegardée dans le navigateur : on peut fermer la page et revenir plus tard.
3. **Exporter** : bouton « Exporter les annotations (JSON) ». Le fichier arrive dans Téléchargements.
4. **Importer, entraîner, mesurer, puis relancer l'analyse** :
   ```bash
   ./va.sh padel import-labels ~/Downloads/labels_<nom>.json
   ./va.sh padel train
   ./va.sh padel eval match.mp4
   ./va.sh padel match.mp4
   ```

`eval` affiche la précision de chaque méthode. Le rapport n'utilise le classifieur que s'il fait mieux
que le modèle de vision seul. Viser au moins 30 à 50 exemples par type de coup, sur plusieurs matchs.

---

## 5. Résultats et fichiers produits

| Emplacement | Contenu |
|---|---|
| à côté de la vidéo (ou `--out`) | `*_analyse.md/.json`, `*_padel.md/.json`, `*_padel_heatmaps.png` |
| `data/labels/` | vos annotations (la vérité terrain, à conserver et partager) |
| `data/eval/` | rapports d'évaluation et d'entraînement |
| `data/annotation/` | pages d'annotation générées (régénérables) |
| `~/.cache/video-analyzer/<vidéo>-<hash>/` | cache d'analyse par vidéo (frames, audio, transcript, pose…) |
| `~/.cache/video-analyzer/models/` | poids YOLO et classifieur de coups entraîné |
| `~/.cache/huggingface/hub/` | modèles téléchargés depuis Hugging Face |
| Ollama (hors Mac Apple Silicon) | modèle de vision, visible avec `ollama list` |

Sous Windows, `~` correspond à `C:\Users\<nom>`.

Pour **forcer un recalcul complet** d'une vidéo, supprimer son dossier dans `~/.cache/video-analyzer/`.

---

## 6. Dépannage

Toujours commencer par `./va.sh doctor` (ou `va.cmd doctor`), qui indique ce qui manque.

| Symptôme | Solution |
|---|---|
| `command not found: video-analyzer` | Utiliser le lanceur `./va.sh` / `va.cmd` depuis le dossier du projet, ou activer l'environnement (voir [4](#4-utilisation)). |
| `ffmpeg introuvable` | Relancer l'installateur. Sous Windows, fermer et rouvrir le terminal pour que le PATH soit à jour. |
| `Ollama injoignable sur http://127.0.0.1:11434` | Lancer l'application Ollama (Windows/macOS) ou `ollama serve` (Linux), puis réessayer. |
| `modèle Ollama absent : lance ollama pull …` | Exécuter la commande indiquée, par exemple `ollama pull qwen3-vl:4b-instruct`. |
| Analyse très lente sur PC | Normal sans carte NVIDIA. Utiliser `--model 4b`. |
| Carte NVIDIA présente mais `doctor` indique « CPU » | Mettre à jour le pilote NVIDIA, puis relancer l'installateur, qui réinstallera PyTorch CUDA. |
| Carte AMD : `doctor` indique « accélération CPU » | Windows : mettre à jour le pilote Adrenalin (≥ 26.2.2) et vérifier Windows 11, puis relancer l'installateur. Linux : `sudo usermod -aG render,video $USER`, se reconnecter. |
| Carte AMD : Python 3.12 absent | `install.cmd` l'installe tout seul (winget). S'il échoue : `winget install Python.Python.3.12`. |
| Transcription lente malgré une carte NVIDIA | faster-whisper bascule sur le processeur quand les bibliothèques CUDA (cuBLAS/cuDNN) manquent ; le résultat reste correct. |
| Windows : « l'exécution de scripts est désactivée » | Passer par `install.cmd` et non `install.ps1` : il contourne cette restriction pour la seule installation. |
| Téléchargement de modèle interrompu | Relancer l'installateur : les téléchargements reprennent. |
| Mac : mémoire insuffisante, machine qui rame | Réinstaller avec `--model 4b` (~7 Go de mémoire au lieu de ~12). |
| Padel : « terrain non calibré » | La caméra n'est pas derrière un fond de court, ou les lignes ne sont pas visibles. Voir les conditions de prise de vue en 4.2. |
| Couleurs d'une vidéo HDR délavées | Sur Mac, le tone-mapping matériel s'en charge. Ailleurs, le ffmpeg standard ne sait pas convertir le HDR10 ; un avertissement s'affiche. |

---

## 7. Mettre à jour, transmettre, désinstaller

**Mettre à jour** : remplacer le code (nouvelle archive ou `git pull`) en gardant `data/`, puis relancer
l'installateur. L'environnement et les modèles existants sont réutilisés.

**Transmettre le projet** à quelqu'un :

```bash
python3 scripts/package.py
```

La commande produit `dist/video-analyzer.zip` (moins de 1 Mo). L'archive contient le code, les installateurs, les
annotations et le classifieur de coups entraîné, sans l'environnement Python, les vidéos ni les caches.

**Désinstaller** :

1. Supprimer le dossier du projet (il contient `.venv/`).
2. Supprimer les caches : `~/.cache/video-analyzer/` et, si aucun autre outil ne s'en sert, les modèles dans
   `~/.cache/huggingface/hub/` (dossiers `models--mlx-community--*` et `models--mobiuslabsgmbh--*`).
3. Hors Mac Apple Silicon, supprimer le modèle Ollama : `ollama rm qwen3-vl:8b-instruct` (ou `:4b-instruct`).
   Ollama, Python et ffmpeg peuvent être gardés ou désinstallés comme n'importe quel logiciel.

---

## 8. Confidentialité et limites

**Confidentialité**
- Tous les calculs sont faits sur la machine. Aucune vidéo, image, transcription ni aucun résultat n'est envoyé en ligne.
- Internet sert uniquement à télécharger les modèles pendant l'installation. Ensuite, le programme se coupe
  lui-même du Hub Hugging Face.
- Ollama est un service local qui n'écoute que sur `127.0.0.1`.

**Limites actuelles**
- **Descriptions** : le modèle de vision peut se tromper ou sur-interpréter. Le 9B est nettement plus fiable que le 4B.
- **Pas de suivi de la balle** en padel. Les lobs et sorties de vitre ne sont reconnus que par le geste, et les
  coups gagnants ou fautes ne sont pas détectés.
- **Types de coups** : environ 44 % corrects avec 315 frappes annotées sur un seul match, contre 22 % pour le modèle
  de vision seul. Ce taux progresse avec les annotations (voir 4.3).
- **Joueurs** : identifiés par équipe et par côté, pas par leur nom.
- **Montages de temps forts** : le gagnant d'un point n'est connu que si le score affiché avance d'exactement un point.
- **Tests** : les installateurs Windows et Linux et la prise en charge des cartes AMD et NVIDIA sont écrits mais
  n'ont pas encore été testés sur une vraie machine de ces systèmes. L'installation Mac, le moteur Ollama/faster-whisper
  et la transcription via transformers ont, eux, été vérifiés de bout en bout.

---

## 9. Pour aller plus loin

- [`docs/REPRISE.md`](docs/REPRISE.md) : récapitulatif du projet et checklist pour reprendre sur une autre machine.
- [`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) : fonctionnement détaillé (pipeline, padel, apprentissage),
  performances mesurées, formats testés, schéma JSON.
- [`CLAUDE.md`](CLAUDE.md) : décisions de conception et pièges rencontrés, pour les développeurs.
- Tests automatiques (aucun modèle requis) :
  ```bash
  .venv/bin/python -m pytest -q        # Windows : .venv\Scripts\python -m pytest -q
  ```
