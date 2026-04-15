<div align="center">

# RVC-WebUI-MacOS

**Une `.app` macOS native de Retrieval-based Voice Conversion.**
Interface SwiftUI + backend Python embarqué. Pas de navigateur, pas de réseau, pas de pip install.

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCKPRO/RVC-WebUI-MacOS?style=for-the-badge)](../../LICENSE)

[**English**](../../README.md) · [**日本語**](../jp/README.ja.md) · [**中文简体**](../cn/README.cn.md) · [**한국어**](../kr/README.ko.md) · [**Français**](./README.fr.md) · [**Português**](../pt/README.pt.md) · [**Türkçe**](../tr/README.tr.md)

</div>

---

## De quoi s'agit-il

RVC-WebUI-MacOS reconditionne le [Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI) en une **`.app` autonome unique** pour Apple Silicon. Tout — PyTorch, fairseq, l'ensemble des modèles préentraînés (HuBERT, RMVPE, UVR5, pretrained_v2) — est livré à l'intérieur du bundle. Le premier lancement est un double-clic ; pas de conda, pas de pip, pas de Homebrew, pas d'URL localhost, et aucune connexion Internet n'est requise après téléchargement.

Le projet d'origine utilise Gradio dans un navigateur et FreeSimpleGUI pour la fenêtre de conversion en temps réel. Ce fork remplace les deux par une **interface SwiftUI** qui communique avec un **backend Python lancé en sous-processus** via JSON-RPC sur stdin/stdout.

## Fonctionnalités

- **Entièrement hors ligne** — tous les poids ML se trouvent dans le bundle. Aucune étape de téléchargement d'actifs, aucune récupération depuis HuggingFace.
- **Apple Silicon en priorité** — backend PyTorch MPS prêt à l'emploi. Retombe correctement sur le CPU quand MPS ne gère pas une opération.
- **Moniteur de ressources toujours visible** — utilisation CPU / mémoire unifiée / MPS dans la barre d'outils, rafraîchie chaque seconde.
- **Barres de progression honnêtes** — pourcentage par tâche, étiquette de phase, ETA. Les boutons d'annulation n'apparaissent que là où l'opération est réellement interruptible.
- **Toutes les fonctionnalités RVC dans une seule app** :
  - Inférence sur un fichier unique et par lot
  - Séparation voix/instruments UVR5 avec guide de choix (quel HP/DeEcho/DeReverb choisir et pourquoi)
  - Chaîne automatique optionnelle de polissage (second passage DeReverb après extraction de la voix)
  - Pipeline d'entraînement complet : prétraitement → extraction F0 / de caractéristiques → entraînement → index
  - Gestion des modèles : comparer, fusionner, extraire (slim), éditer les informations
  - Export ONNX
  - Changeur de voix en temps réel avec sélection de périphériques + mise à jour à chaud des paramètres
- **Disposition lisible par un humain** — chaque fichier utilisateur vit sous `~/Documents/RVC-WebUI/`, rien n'est éparpillé dans des dossiers Application Support cachés.
- **Des valeurs par défaut qui ne dégradent pas l'audio** — la sortie est en FLAC (sans perte) ; WAV / MP3 / M4A restent disponibles.

## Configuration requise

| | Minimum | Recommandée |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma ou ultérieur |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro ou supérieur |
| RAM | 8 Go | 16 Go et plus (l'entraînement est gourmand) |
| Disque | 8 Go libres | 20 Go et plus pour l'entraînement |

Les Macs Intel ne sont **pas pris en charge** — le PyTorch embarqué est uniquement ARM64.

## Installation

### Pour les utilisateurs finaux

1. Téléchargez `RVC-WebUI.app.zip` depuis la dernière [Release](https://github.com/RTCKPRO/RVC-WebUI-MacOS/releases).
2. Décompressez, glissez `RVC-WebUI.app` dans `/Applications`.
3. Double-cliquez pour lancer. Au premier démarrage, Gatekeeper peut demander confirmation — clic droit sur l'app → **Ouvrir** → **Ouvrir** dans la boîte de dialogue.

Au premier lancement, l'app crée `~/Documents/RVC-WebUI/` ainsi que les sous-dossiers pour vos entrées, sorties, modèles et journaux. C'est le seul endroit où elle écrit.

### Pour les développeurs / compilation depuis les sources

```bash
# Prérequis : Homebrew, Xcode CLT, Miniforge/conda
brew install xcodegen conda-pack

# 1. Clone
git clone https://github.com/RTCKPRO/RVC-WebUI-MacOS.git
cd RVC-WebUI-MacOS

# 2. Créer l'environnement conda (Python 3.10 + PyTorch MPS + fairseq etc.)
./setup_conda_env.sh
conda activate rvc

# 3. (Facultatif) Test rapide du backend Python seul
python tools/test_rpc.py
# attendu : notification "ready" → réponse initialize → resource_stats chaque seconde

# 4. Construire le bundle .app complet
./build_app.sh
# Produit : build/RVC-WebUI.app  (environ 4 Go incluant PyTorch et tous les modèles)
```

Options de construction :

- `--skip-conda` — réutiliser l'environnement Python déjà packé (`build/python_env/`)
- `--skip-xcode` — réutiliser le binaire Swift déjà construit
- `--skip-sign` — ignorer la signature de code (correct en dev local, à éviter pour la distribution)

Pour les builds signés pour distribution :

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

## Architecture

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│   NavigationSplitView + TabView              │
│   barre d'outils : moniteur CPU / MEM / MPS  │
└───────────────────┬──────────────────────────┘
                    │ JSON-RPC 2.0 sur stdio
                    │ (ni réseau, ni socket)
┌───────────────────▼──────────────────────────┐
│      Sous-processus Python (rpc_server.py)   │
│   VC · UVR5 · Entraînement · Temps réel · ONNX │
│   Échantillonnage via psutil + torch.mps     │
└──────────────────────────────────────────────┘
```

- Frontend : `RVCApp/` — SwiftUI, généré avec `xcodegen` à partir de `project.yml`
- Pont : `RVCApp/RVCApp/Bridge/PythonBridge.swift` — lance le sous-processus Python, dispatche les appels RPC, achemine les notifications de progression / ressources vers l'état `@Published`
- Backend : `rpc_server.py` + `rpc_training.py` — les méthodes JSON-RPC enveloppent `infer/modules/vc`, `infer/modules/uvr5` et les scripts d'entraînement ; stdout est bufferisé ligne par ligne pour une première réponse rapide
- Assets : `assets/hubert/`, `assets/rmvpe/`, `assets/pretrained_v2/`, `assets/uvr5_weights/` — tous copiés dans `.app/Contents/Resources/rvc_backend/assets/` à la construction
- Runtime Python : `build/python_env/` via `conda-pack`, puis embarqué dans `.app/Contents/Resources/python/`

Voir [`BUILD_NATIVE_APP.md`](../../BUILD_NATIVE_APP.md) pour le pipeline de build complet et les notes d'architecture.

## Arborescence des fichiers

**Dans le bundle** (`RVC-WebUI.app/Contents/Resources/`) — lecture seule :

```
rvc_backend/    # Code Python + assets, copiés depuis le dépôt
python/         # Runtime Python 3.10 embarqué avec toutes les dépendances
```

**Dans votre dossier personnel** (`~/Documents/RVC-WebUI/`) — toutes vos données :

```
input/
  audio/          # Déposez ici les fichiers pour l'inférence
  training/       # Jeux de données d'entraînement
output/
  inference/      # Résultats de conversion fichier unique (FLAC par défaut)
  batch/          # Résultats de conversion par lot
  separation/     # vocals/ et accompaniment/ d'UVR5
  onnx/           # Exports ONNX
models/           # Vos modèles de voix .pth entraînés
indices/          # Fichiers FAISS .index
logs/             # Checkpoints + journaux d'entraînement, un dossier par expérience
configs/inuse/    # Configuration d'exécution
temp/             # Espace tampon, vidé au démarrage
```

## Dépannage

**"RVC-WebUI.app est endommagé et ne peut pas être ouvert"** — Les builds signés en ad-hoc font tomber Gatekeeper sur un téléchargement neuf. Solution :
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**"No supported NVIDIA GPU found"** — Attendu. L'app tourne sur MPS ; c'est une ligne de journal d'un chemin de code amont, pas une erreur.

**L'entraînement échoue immédiatement à l'extraction de caractéristiques** — Corrigé dans ce fork. Si vous compilez depuis un checkout très ancien, assurez-vous que `infer/lib/torch_compat.py` existe et est importé avant `fairseq` dans `extract_feature_print.py`, `infer/modules/vc/utils.py` et `infer/lib/rtrvc.py`. Ce shim désactive la valeur par défaut `weights_only=True` de PyTorch 2.6+ qui fait trébucher le chargeur HuBERT de fairseq.

**Plus assez de mémoire MPS pendant l'entraînement** — abaissez `batch_size_per_gpu`, fermez d'autres applications, ou définissez `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (déjà défini au lancement, mais à vérifier dans `~/Documents/RVC-WebUI/logs/<exp>/train.log`).

**Premier lancement lent** — l'import à froid de fairseq + torch prend environ 3 s sur M1, 2 s sur M3. Le splash affiche « en attente du backend » jusqu'à ce que `alive` arrive ; rien à faire.

## Développement

Le projet SwiftUI est régénéré à chaque build par xcodegen depuis `RVCApp/project.yml`, ne modifiez donc pas `RVCApp.xcodeproj` à la main. Ouvrez `RVCApp.xcodeproj` dans Xcode puis Run — en mode dev, l'app lance le `rpc_server.py` du dépôt via votre environnement conda actif (et non le Python embarqué), ce qui rend l'itération beaucoup plus rapide.

Modifications côté Python :
- Les sources sont à la racine du dépôt (`rpc_server.py`, `rpc_training.py`, `infer/`, `rvc/`, `configs/`, `i18n/`, `tools/`)
- `./build_app.sh --skip-conda --skip-xcode` resynchronise le backend Python dans un `.app` existant sans reconstruire le binaire Swift ni repacker Python
- Pour une itération rapide contre un `.app` déjà construit, `rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/` suffit

## Crédits

- Framework de conversion vocale amont : [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- Blocs de construction : [ContentVec](https://github.com/auspicious3000/contentvec), [VITS](https://github.com/jaywalnut310/vits), [HIFIGAN](https://github.com/jik876/hifi-gan), [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), [audio-slicer](https://github.com/openvpi/audio-slicer), [RMVPE](https://github.com/Dream-High/RMVPE) (modèle préentraîné par [yxlllc](https://github.com/yxlllc/RMVPE) et [RVC-Boss](https://github.com/RVC-Boss))
- Fork macOS initial : [Nevil Patel](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- Refonte native `.app` : ce dépôt

## Licence

MIT. Voir [LICENSE](../../LICENSE).
