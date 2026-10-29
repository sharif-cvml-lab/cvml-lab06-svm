# Models

| File | In git? | Purpose |
| --- | --- | --- |
| `face_landmarker.task` | yes | MediaPipe FaceLandmarker: finds faces and 478 landmarks |
| `magface_iresnet100.onnx` | **no (about 250 MB)** | MagFace: aligned face -> 512-d embedding |
| `svm_model.joblib` | no, generated | your trained SVM (written by the Train button) |
| `embedding_cache.pkl` | no, generated | embeddings of already processed photos (speeds up re-training) |

Download MagFace and save it here as `magface_iresnet100.onnx`:

<https://drive.google.com/file/d/1kdoVMRDo__cnB2OKqJu40Qv7rwApBzHt/view?usp=sharing>
