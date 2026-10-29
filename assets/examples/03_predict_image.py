"""Example 3 - recognise the person in one photo.

    python assets/examples/03_predict_image.py image.jpg [--threshold 0.45]

photo -> embedding -> predict_svm() -> (name, similarity) -> accept or "Unknown".
Run example 2 (or press Train in the GUI) first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root

from dataset_utils import decide, load_model, predict_identity, read_image
from face_features import FaceFeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--threshold", type=float, default=0.45)
args = parser.parse_args()

model, meta = load_model()
if model is None:
    sys.exit("No trained model found. Run 02_train_svm.py first.")

image = read_image(args.image)
if image is None:
    sys.exit(f"Cannot read image: {args.image}")

result = FaceFeatureExtractor().extract(image)
if result is None:
    sys.exit("No face found in the image.")

name, similarity = predict_identity(model, result["identity_features"])
print(f"SVM prediction  : {name}")
print(f"Cosine similarity to {name}: {similarity:.3f}")
print(f"Threshold       : {args.threshold:.2f}")
print(f"Final decision  : {decide(name, similarity, args.threshold)}")
