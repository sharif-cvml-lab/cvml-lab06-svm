"""Example 2 - build the embedding table of the database and train the SVM.

    python assets/examples/02_train_svm.py [--database DIR] [--kernel linear|rbf] [--C 10]

Every photo of assets/database/<person>/ becomes one 512-d training vector whose
label is the folder name. Your train_svm() (svm.py) is then called and the model
is saved to assets/models/svm_model.joblib - the same file the GUI uses.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root

from config import DEFAULT_DATABASE_DIR, SVM_MODEL_PATH, SvmParameters
from dataset_utils import train_model
from face_features import FaceFeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--database", default=str(DEFAULT_DATABASE_DIR))
parser.add_argument("--kernel", default="linear", choices=["linear", "rbf"])
parser.add_argument("--C", type=float, default=10.0)
parser.add_argument("--gamma", default="scale", choices=["scale", "auto"])
args = parser.parse_args()

extractor = FaceFeatureExtractor()


def show_progress(done, total, message):
    print(f"\r[{done}/{total}] {message:<50}", end="", flush=True)


try:
    model, info = train_model(
        extractor, args.database,
        SvmParameters(kernel=args.kernel, C=args.C, gamma=args.gamma),
        progress=show_progress,
    )
except (NotImplementedError, RuntimeError) as error:
    sys.exit(f"\n{error}")

print(f"\nTrained on {info['num_images']} images of {info['num_classes']} people "
      f"in {info['train_seconds']:.2f} s")
for path, reason in info["skipped"]:
    print(f"  skipped {path}: {reason}")
print(f"Model saved to {SVM_MODEL_PATH}")
