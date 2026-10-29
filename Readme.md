# Lab06_FaceRecognition_SVM

Welcome to the sixth assignment of the Computer Vision and Machine Learning lab course!

This project provides a real-time PyQt6 application that recognizes people from a webcam, an image or a video. Faces are turned into 512-number embeddings by MagFace, and an SVM that **you** train decides who is who. The goal of this lab is to understand multi-class SVM classification, embeddings, cosine similarity, and why a recognition system needs a threshold to say "Unknown".

## 🎯 Assignment Objectives

In this lab, you will complete two functions located in `svm.py`:

```
train_svm(embeddings, labels, kernel, C, gamma)
predict_svm(model, embedding)
```

Students should modify only `svm.py`. Do not modify `main.py`, `gui.py`, `video_thread.py`, `workers.py`, `annotation.py`, `face_features.py`, `dataset_utils.py`, `config.py`, or any other project file. The detailed, step-by-step TODO comments are inside `svm.py`.

The expected processing pipeline is:

```
Camera Frame / Image / Video
     ↓
Face Landmarker (k largest faces)           ← provided
     ↓
Alignment + crop → MagFace → 512-d vector  ← provided
     ↓
SVM predicts the person                     ← predict_svm()
     ↓
Cosine similarity to that person's centroid ← predict_svm()
     ↓
similarity ≥ threshold ?  name : "Unknown"  ← provided (slider)
```

Training (Train button): every photo in `assets/database/<person>/` becomes one embedding labelled with the folder name, then `train_svm()` fits an `SVC` and computes one centroid (normalised mean embedding) per person.

## ✨ Features

- Real-time recognition of the **k largest faces** (k is adjustable: lower it on a slow laptop).
- Three input sources: **Webcam** (default), **Image** and **Video**, with identical processing and annotations.
- Smooth video: MagFace runs in a background thread, labels update asynchronously.
- Bounding box, name, similarity score and optional **facial landmarks** (checkbox), scaled to the image size.
- **Match similarity threshold** slider: the effect is instant.
- Train button with kernel, `C` and `gamma` controls and a configurable database folder.
- **Evaluate on test set**: correct match / wrong match / rejected, correct rejection / false accept, and a threshold sweep.
- Dark and light themes, full-screen mode, High-DPI support.
- Pause / resume, camera-index selection, settings restored between sessions, reset-to-default button.
- FPS, landmark time, MagFace time, face count and frame size indicators.

## 🛠️ Setup & Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment (recommended):

```
python -m venv venv
```

Activate it on Linux/macOS:

```
source venv/bin/activate
```

On Windows PowerShell:

```
venv\Scripts\Activate.ps1
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Download the MagFace model (about 250 MB, too large for git) and save it as `assets/models/magface_iresnet100.onnx`:

   <https://drive.google.com/file/d/1kdoVMRDo__cnB2OKqJu40Qv7rwApBzHt/view?usp=sharing>

5. Run the application:

```
python main.py
```

The application starts even before you finish `svm.py`: until then every face is shown as "Unknown" and **Train** tells you that `train_svm()` is not implemented yet.

## 🎮 Using the Application

### First activity: add yourself

1. Take 5 good selfies (different small head poses, good light, one face per photo).
2. Create `assets/database/<Your Name>/` and put the selfies there.
3. Optionally put one *different* photo of yourself (other day, other light) into `assets/test/<Your Name>/`.
4. Press **Train** and look at the webcam.

### Threshold

The SVM always answers with one of the people it knows, so the *similarity* (cosine similarity between the face and the average embedding of the predicted person) decides whether the answer is trusted.

- Too strict (slider too high): even enrolled people become **Unknown**.
- Too lenient (slider too low): strangers are matched to someone in the database.

### Evaluation

The test folder has the same structure as the database. Its photos are never used for training. For every test image the report counts:

| People in the training set | People *not* in the training set |
| --- | --- |
| **correct match**: accepted, right name | **correct reject**: shown as Unknown |
| **wrong match**: accepted, wrong name | **false accept**: matched to someone |
| **rejected**: shown as Unknown | |

The provided `assets/test/` contains photos of the 100 enrolled people and of 6 people that are *not* in `assets/database/`, so both kinds of errors can be measured. The threshold sweep at the end of the report helps you pick a good slider value.

## 📂 Repository Structure

```
cvml-lab06-svm/
├── .gitignore
├── Readme.md
├── requirements.txt
├── main.py
├── gui.py
├── video_thread.py
├── workers.py
├── annotation.py
├── face_features.py
├── dataset_utils.py
├── config.py
├── svm.py                      # The only file students should modify
└── assets/
    ├── database/<person>/*.jpg # training photos (label = folder name)
    ├── test/<person>/*.jpg     # test photos (never used for training)
    ├── models/                 # face_landmarker.task, magface_iresnet100.onnx, svm_model.joblib
    └── examples/               # standalone scripts, no GUI
```

### Module responsibilities

- `main.py` — application entry point only.
- `gui.py` — PyQt6 user interface, controls, themes, persistence, and rendering.
- `video_thread.py` — input sources, face tracking and asynchronous recognition.
- `workers.py` — background threads for model loading, training and evaluation.
- `annotation.py` — boxes, landmarks and labels, scaled to the image size.
- `face_features.py` — `FaceFeatureExtractor`: landmarks, alignment, MagFace embedding.
- `dataset_utils.py` — dataset reading, embedding cache, model saving and evaluation metrics.
- `config.py` — constants, defaults, themes and shortcuts.
- `svm.py` — the student implementation.

### Standalone examples (`assets/examples/`)

Run them from the project root; they use the same code as the GUI.

```
python assets/examples/01_extract_embedding.py [image]     # photo -> 512-d embedding
python assets/examples/02_train_svm.py                     # database -> trained model
python assets/examples/03_predict_image.py image.jpg       # photo -> name + similarity
python assets/examples/04_evaluate.py                      # test folder -> metrics
```

## ⌨️ Keyboard Shortcuts

- `Space` — Pause / Resume
- `F11` — Full Screen / Windowed
- `R` — Reset Defaults
- `Q` — Quit
