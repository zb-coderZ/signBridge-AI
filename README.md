# SignBridge AI: An Intelligent Sign Language Recognition and Communication System

Webcam app that recognises **Pakistan Sign Language (PSL)** and **American Sign Language (ASL)**, converts signs to text and
speech, and supports a **two-way mode** so a hearing person can reply (speech/typing → text, word → sign animation).
Laptop webcam only. Landmark-based (MediaPipe hand landmarks → small classifiers); the models never see raw pixels.

> **Read this first.** The project ships with a **DEMO model trained on synthetic landmarks** so everything runs immediately.
> The demo's "signs" are coarse finger-extension patterns, **not real ASL/PSL signs**, and its accuracy numbers say nothing about
> real sign-language accuracy. To get a real system you must download/record real data and retrain (Step 8 below).
> The UI shows a "DEMO MODEL" banner whenever demo models are active.

---------------------------------------------------------------------------------------------------

## Contents

1. [Project structure (what every file is)](#1-project-structure)
2. [Step-by-step setup](#2-step-by-step-setup-from-zero)
3. [Running the app and using each feature](#3-using-the-app)
4. [Datasets: where to download each one](#4-datasets)
5. [Training, evaluation, retraining commands](#5-training-and-evaluation-commands)
6. [Configuration](#6-configuration-configyaml)
7. [Design decisions](#7-design-decisions-and-assumptions)
8. [Results table](#8-results)
9. [Tests](#9-tests)
10. [Troubleshooting](#10-troubleshooting)
11. [Known limitations](#11-known-limitations)

---------------------------------------------------------------------------------------------------

## 1. Project structure

```
signbridge_ai/
├── README.md                 this file
├── config.yaml               ALL settings (paths, thresholds, training, TTS/STT, LLM)
├── requirements.txt          pinned dependencies (tested on Python 3.12.3)
├── setup.bat / setup.sh      creates the venv (tries Python 3.14, falls back to 3.12), installs, checks, builds demo
├── run_app.bat / run_app.sh  starts the Streamlit app
├── app/
│   ├── streamlit_app.py      the UI: 6 tabs + sidebar settings
│   ├── ui_helpers.py         CSS, top-3 bars, confidence chart, history table, CSV export
│   └── components/video_processor.py   streamlit-webrtc processor (frames → engine)
├── core/                     all logic; no Streamlit dependency except app/
│   ├── config.py             config loading, project-relative paths, logging
│   ├── compat.py             safe import order (streamlit_webrtc → mediapipe → tensorflow)
│   ├── hand_detector.py      MediaPipe Tasks HandLandmarker (auto-downloads hand_landmarker.task) + skeleton drawing
│   ├── landmarks.py          landmark layout, hand-slot assignment, sequence resampling
│   ├── normalize.py          wrist-relative / scale-invariant / handedness canonicalisation, features, augmentation, motion score
│   ├── stability.py          N-consecutive-frame commit, unknown rejection, cooldown, release rule
│   ├── sentence_builder.py   letters/words, space/backspace/clear, rule-based or optional LLM gloss clean-up
│   ├── tts.py                pyttsx3 on a worker thread, graceful when no driver / no Urdu voice
│   ├── stt.py                speech-to-text (Google Web Speech, or offline faster-whisper if installed)
│   ├── language_router.py    PSL / ASL / Auto (higher mean confidence over the last 3 s; manual overrides Auto)
│   ├── practice.py           landmark similarity score + feedback; skeleton rendering and GIF sign animation
│   ├── engine.py             real-time pipeline (camera-agnostic, unit-tested)
│   ├── models_io.py          model loading (trained models override demo models per component)
│   ├── training.py           shared training helpers (split, augmentation, Keras builders, references)
│   ├── dataset.py            .npz landmark container + signer-disjoint split
│   ├── dataset_info.py       per-dataset download instructions
│   └── synthetic.py          synthetic hand generator (demo model + tests ONLY)
├── scripts/
│   ├── inspect_dataset.py    class counts, imbalance, sample previews; prints download help if missing
│   ├── extract_landmarks.py  MediaPipe once over a dataset → data/processed/*.npz
│   ├── record_samples.py     record YOUR OWN samples with the webcam (any label, any language)
│   ├── train_static.py       Random Forest + Keras MLP, one set per language
│   ├── train_dynamic.py      LSTM vs GRU ablation (same data/split/hyper-parameters)
│   ├── evaluate.py           cross-signer test metrics, confusion matrices, latency/FPS → reports/
│   ├── build_demo.py         synthetic data → demo models → evaluation (one command)
│   └── check_env.py          verifies the native libraries import on this interpreter
├── data/  (raw/, processed/, README.md)
├── models/ (demo/ = shipped synthetic models, trained/ = your real models, hand_landmarker.task downloaded at first use)
├── notebooks/ 01_eda.ipynb, 02_results.ipynb
├── reports/ PROJECT_REPORT.md, demo/ (metrics.json, results_table.md, confusion matrices, training logs)
└── tests/                    46 unit/integration tests
```

---------------------------------------------------------------------------------------------------

## 2. Step-by-step setup (from zero)

You need: a laptop with a webcam, ~3 GB free disk, internet for the install (and for the one-time hand-model download).

### Step 1: Install Python
Install **Python 3.12** from https://www.python.org/downloads/ (on Windows tick *"Add python.exe to PATH"* and keep the
*py launcher*). Python 3.14 is tried first by the setup scripts, but MediaPipe/TensorFlow wheels usually lag new Python
releases, so the scripts automatically fall back to 3.12 if 3.14 fails. (I could not test 3.14.2 myself: the development
machine only had Python 3.12.3, so **3.12.3 is the tested interpreter**.)

### Step 2: Unzip the project
Unzip `SignBridge_AI.zip` somewhere with a short path without special characters, e.g. `C:\Projects\signbridge_ai` or `~/signbridge_ai`.

### Step 3: Create the environment and install everything

| Windows (Command Prompt / PowerShell) | Linux / macOS (Terminal) |
|---|---|
| `cd C:\Projects\signbridge_ai` | `cd ~/signbridge_ai` |
| `setup.bat` | `chmod +x setup.sh run_app.sh` then `./setup.sh` |

The script: creates `venv/` → `pip install -r requirements.txt` → runs `scripts/check_env.py` (imports every native library in the
safe order) → if that fails it deletes the venv and retries with the next Python version → builds the demo models if missing.
Expect 5-10 minutes (TensorFlow is large). Success ends with *"Environment OK with Python 3.x"*.

Manual equivalent:
```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py       # must print OK for every line
python scripts/build_demo.py      # only needed if models/demo is empty
```

### Step 4: (Optional) check the demo pipeline without any camera
```bash
python scripts/evaluate.py --demo      # re-computes metrics on the synthetic demo data
pytest -q                              # 46 tests
```

### Step 5: Start the app
`run_app.bat` (Windows) or `./run_app.sh` (Linux/macOS). A browser tab opens at http://localhost:8501.

### Step 6: First camera use
1. Open the **Live translate** tab → click **START** → allow the browser camera permission.
2. The first time, the app downloads `hand_landmarker.task` (~8 MB) from Google's MediaPipe storage into `models/`.
   If you are offline, download it yourself from
   `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
   and save it as `models/hand_landmarker.task`.
3. No webcam / WebRTC trouble? Use the **Demo (no camera)** tab (synthetic landmarks through the real pipeline) or the
   *snapshot* expander on the Live tab.

### Step 7: Get real data (needed for a real system)
See [Datasets](#4-datasets). Minimum viable path for **PSL** (no public dataset needed): record your own signs, see Step 8.

### Step 8: Train on real data
```bash
python scripts/inspect_dataset.py --all
python scripts/extract_landmarks.py --name asl_alphabet --max-per-class 400     # ASL letters (images)
python scripts/record_samples.py --lang PSL --label alif --kind static --signer zb    # repeat per label/person
python scripts/train_static.py  --lang all
python scripts/train_dynamic.py --lang all
python scripts/evaluate.py
```
Real models are written to `models/trained/` and **automatically override** the demo models in the app (per language and per
static/dynamic component). Restart the app after training.

---------------------------------------------------------------------------------------------------

## 3. Using the app

**Sidebar:** language (Auto / PSL / ASL), sign type (Auto / Static only / Dynamic only), confidence thresholds, N stable frames,
cooldown, motion threshold, skeleton overlay, speak-on-commit, speech language (English/Urdu), optional LLM clean-up.

| Tab | What it does |
|---|---|
| **Live translate** | WebRTC video with landmark skeleton. Shows the prediction, stability progress, top-3 bars, confidence-over-time chart, FPS, latency. A sign is *committed* only after N agreeing frames above the threshold; `unknown` is rejected; a cooldown plus a "release" rule prevents repeats while you hold a sign. Sentence builder: Space, Backspace, Clear, Speak, Clean up (gloss → sentence). |
| **Demo (no camera)** | Simulates a (right- or left-handed) virtual signer producing a chosen sign and runs it through the same pipeline; shows per-frame log and the committed sign. |
| **Two-way** | *Hearing → text:* record a reply in the browser (Google Web Speech online, or offline Whisper if you `pip install faster-whisper`), or type it. *Text → sign:* type a word; the app animates the stored reference landmarks (dynamic prototype if the word is a known dynamic label, else fingerspelling letter by letter). |
| **Practice** | Pick a target sign; the live camera hand (or a simulated mistake) is compared with the class-mean reference: similarity 0-100 + per-finger feedback ("Extend your index finger more"). |
| **History** | Table of committed signs (timestamp, language, type, sign, confidence) with CSV export. |
| **Status** | Python/library status, hand-model presence, loaded models (demo vs trained, validation accuracy), dataset folders found, evaluation tables. |

Auto-detect: both languages' models run on every frame; the language with the higher mean top-confidence over the last 3 s is shown
(small hysteresis to avoid flicker). Choosing PSL or ASL manually overrides it. Static/dynamic switching: if recent hand
motion (wrist travel or finger opening/closing over ~15 frames) exceeds `motion_threshold` the dynamic model is used.

---------------------------------------------------------------------------------------------------

## 4. Datasets

None are bundled. `python scripts/inspect_dataset.py --all` prints these instructions for whatever is missing and carries on.

| Dataset | Use | Folder | Source |
|---|---|---|---|
| ASL Alphabet | ASL static letters | `data/raw/asl_alphabet/<CLASS>/*.jpg` | Kaggle `grassknoted/asl-alphabet` (`kaggle datasets download -d grassknoted/asl-alphabet -p data/raw/asl_alphabet --unzip`) |
| WLASL (small subset) | ASL dynamic words | `data/raw/wlasl/<gloss>/*.mp4` | https://github.com/dxli94/WLASL (`start_kit/video_downloader.py`; many YouTube links are dead, keep what downloads) |
| MS-ASL (optional) | ASL dynamic words | `data/raw/msasl/<gloss>/*.mp4` | https://www.microsoft.com/en-us/download/details.aspx?id=100121 |
| PSL (LUMS/CHISEL, Dynamic PSL) | PSL static + dynamic | `data/raw/psl/...` | may need author access; **fallback: `scripts/record_samples.py`** |

Details, folder layouts and the signer-id caveat are in `data/README.md`.

---------------------------------------------------------------------------------------------------

## 5. Training and evaluation commands

```bash
# 1. landmarks (run once per dataset; MediaPipe is not run again during training)
python scripts/extract_landmarks.py --name asl_alphabet --max-per-class 400
python scripts/extract_landmarks.py --path data/raw/wlasl --lang ASL --kind dynamic --source wlasl --max-per-class 30
# 2. your own recordings (adds signers; fallback for PSL)
python scripts/record_samples.py --lang PSL --label alif --kind static  --signer ali   --frames 40
python scripts/record_samples.py --lang PSL --label shukriya --kind dynamic --signer ali --count 20
python scripts/record_samples.py --lang PSL --label unknown --kind static --signer ali   # in-between poses → rejection class
# 3. train
python scripts/train_static.py  --lang PSL        # RF baseline + Keras MLP     → models/trained/
python scripts/train_dynamic.py --lang PSL        # LSTM vs GRU ablation
# 4. evaluate (signer-held-out test split) → reports/metrics.json, results_table.md, confusion_*.png
python scripts/evaluate.py
# demo (synthetic) equivalents
python scripts/build_demo.py
```

Evaluation protocol: whole **signers** are held out for testing (and a separate set for validation) with `GroupShuffleSplit`.
If a dataset has fewer than 3 signer ids the split silently-but-loudly falls back to a random split and every report is labelled
`random_NOT_cross_signer`. Augmentation (small rotations, noise, scale jitter, mirror) is applied **to the training split only**.
Metrics: accuracy, per-class precision/recall/F1, macro-F1, confusion matrix PNG, per-sample model latency, end-to-end pipeline
ms/frame and FPS (camera capture excluded). Every number is produced by `evaluate.py`; nothing is hard-coded.

---------------------------------------------------------------------------------------------------

## 6. Configuration (`config.yaml`)

One file for paths, hand-detector options, thresholds (`confidence_threshold`, `stable_frames`, `cooldown_s`,
`motion_threshold`, ...), training hyper-parameters, demo-generator size, TTS rate / Urdu voice keywords, LLM model & env-var name,
STT backend, STUN servers. Override the file location with the `SIGNBRIDGE_CONFIG` environment variable.
Optional LLM clean-up: set `ANTHROPIC_API_KEY` and tick the sidebar box; without a key (or on error) the rule-based clean-up is used.

---------------------------------------------------------------------------------------------------

## 7. Design decisions and assumptions

* **MediaPipe Tasks `HandLandmarker`** (not the deprecated `mp.solutions`); up to 2 hands; model auto-downloaded.
* **Import order is load-bearing.** On the tested setup the process crashes with a segmentation fault if `mediapipe` or `tensorflow`
  is imported *before* `streamlit_webrtc`. The safe order is `streamlit_webrtc → mediapipe → tensorflow` (`core/compat.py`),
  which also satisfies "mediapipe before tensorflow".
* **`opencv-contrib-python`** is pinned instead of `opencv-python` because MediaPipe depends on the contrib build; installing both breaks `cv2`.
* **Features:** 2 hand slots × 21 × (x,y,z) wrist-relative, divided by palm size (wrist→middle MCP); the Left slot is mirrored so
  one model serves both hands; a lone left hand is moved into slot 0 so one-handed signs are identical for either hand
  (found and fixed with a regression test: left-handed signers initially failed because presence flags and trajectories were slot-specific).
  Rotation is *not* normalised away (orientation distinguishes signs) but augmented. Dynamic features add a mirrored,
  palm-normalised wrist trajectory. Static = 128 dims, dynamic = 30 × 132.
* **Mirror augmentation** flips x *without* swapping slots, which simulates MediaPipe handedness mislabelling.
* **Selfie convention:** live frames and dataset images are both horizontally flipped (`hand_detector.mirror_input`) so handedness labels mean the same thing in training and at run time.
* **Dynamic window:** the last 1.0 s of live frames (`sequence_window_s`) is resampled to 30 frames, so recognition does not depend on camera FPS. Signs are assumed to last about a second.
* **Practice score** = `100·max(0, 1 − d/d_max)` with `d` the mean landmark distance in palm units (`d_max` = 0.6): a transparent heuristic, not a probability.
* **Demo model:** `core/synthetic.py` builds a parametric hand (finger curls, thumb, spread) with per-signer size/proportions/rotation, every third signer left-handed. PSL demo classes are neutral ids `P1..P6`; dynamic demo classes are motion patterns (`WAVE, CIRCLE, NOD, SWIPE, PINCH`).

---------------------------------------------------------------------------------------------------

## 8. Results

All rows below come from `python scripts/evaluate.py --demo` on **SYNTHETIC demo data** (8 virtual signers per language; test = held-out
virtual signers). They show that the pipeline works end to end. **They are not sign-language accuracies.** Real-data results:
*not yet produced, requires datasets / your recordings; run `scripts/evaluate.py` after training and they appear in `reports/metrics.json`.*

| Language | Type | Model | Test accuracy | Macro-F1 | Model latency (ms/sample) | Split | n_test |
|---|---|---|---|---|---|---|---|
| PSL | static | RF | 0.961 | 0.966 | 8.01 | cross_signer (SYNTHETIC) | 640 |
| PSL | static | MLP | 0.958 | 0.963 | 1.66 | cross_signer (SYNTHETIC) | 640 |
| PSL | dynamic | LSTM | 0.990 | 0.990 | 40.95 | cross_signer (SYNTHETIC) | 300 |
| PSL | dynamic | GRU | 1.000 | 1.000 | 50.57 | cross_signer (SYNTHETIC) | 300 |
| ASL | static | RF | 1.000 | 1.000 | 8.84 | cross_signer (SYNTHETIC) | 720 |
| ASL | static | MLP | 1.000 | 1.000 | 2.14 | cross_signer (SYNTHETIC) | 720 |
| ASL | dynamic | LSTM | 0.997 | 0.997 | 47.61 | cross_signer (SYNTHETIC) | 300 |
| ASL | dynamic | GRU | 1.000 | 1.000 | 49.82 | cross_signer (SYNTHETIC) | 300 |

| Language | Type | Pipeline mean (ms/frame) | p95 (ms) | FPS (pipeline only) |
|---|---|---|---|---|
| PSL | static | 1.98 | 2.09 | 505 |
| PSL | dynamic | 13.35 | 46.69 | 75 |
| ASL | static | 1.98 | 2.10 | 505 |
| ASL | dynamic | 12.31 | 37.08 | 81 |

Files: `reports/demo/metrics.json`, `reports/demo/results_table.md`, `reports/demo/confusion_*.png`, `reports/demo/train_*.json`.
Hand-detector latency: not measured (hand model could not be downloaded in the build environment)

---------------------------------------------------------------------------------------------------

## 9. Tests

```bash
pytest -q        # 46 tests: normalisation, stability filter, sentence builder, engine, router, practice, TTS/STT degradation,
                 # dataset split, headless Streamlit smoke test
```
The Streamlit smoke test stubs only `webrtc_streamer` (the real component needs a browser).

---------------------------------------------------------------------------------------------------

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| Segmentation fault on start | Some library imported before `streamlit_webrtc`. Use `run_app.*` / `streamlit run app/streamlit_app.py`, do not import mediapipe/tensorflow first in your own scripts. |
| `pip install` fails on tensorflow/mediapipe | Python too new. Use Python 3.12 (`setup.*` does this automatically). |
| Video does not start | Allow camera permission; close Zoom/Teams; try Chrome; on a network without internet the STUN server is unreachable (localhost still works). |
| "Hand model unavailable" | Download `hand_landmarker.task` manually (Step 6) into `models/`. |
| No speech | Linux needs `espeak`/`espeak-ng` (`sudo apt install espeak-ng`); Windows/macOS use built-in voices. Urdu needs an Urdu voice installed in the OS, otherwise the default voice is used and the UI says so. |
| Speech-to-text error | Google backend needs internet; otherwise `pip install faster-whisper`. |
| Static signs flip to "dynamic" while holding still | Raise *Motion threshold* in the sidebar (or choose *Static only*). The shipped threshold was calibrated on synthetic jitter (see 11). |
| Signs never commit | Lower the confidence threshold or N; check the Status tab, you may still be on the demo model. |

---------------------------------------------------------------------------------------------------

## 11. Known limitations

* **The live camera path was not exercised in development.** The build sandbox had no webcam and could not reach Google's model host,
  so the real `HandLandmarker`, the WebRTC callback and the download step were never run with real video. They were tested only for graceful failure
  (missing model → clear message; app still opens). Everything downstream of the landmarks was tested for real. Expect to fix small things on first camera use.
* **Demo model = synthetic.** Not real signs; metrics are near-perfect because the synthetic task is easy. Do not cite them as sign-language accuracy.
* **Motion threshold is calibrated on synthetic jitter** (held pose, jitter σ = 0.004 → max score ≈ 0.09; synthetic dynamic windows ≥ 0.17, `PINCH` as low as 0.095 at the 5th percentile). Real webcams jitter more or less; retune in the sidebar / `config.yaml`.
  The first static letter commits ~15 frames later than it otherwise would because Auto mode waits for the motion window to fill.
* **ASL Alphabet has no signer ids,** so it cannot give a signer-independent test; and J/Z are motion letters shown as still images.
* **PSL real data is not bundled and may need author access;** the recording tool is the fallback. Many PSL signs are two-handed or involve the face/body, which hand landmarks alone do not capture.
* Only hand landmarks are used (no pose, face or facial expression), which limits recognition of real sign-language words.
* Two-way "text → sign" shows reference *landmark skeletons* (not video of a human signer); letters/words without a reference are skipped.
* Speech-to-text (Google) needs internet; Urdu TTS needs an OS Urdu voice. Optional LLM clean-up needs an API key and internet.
* Per-frame dynamic inference is throttled (every 3rd frame) to keep video smooth on CPU.
