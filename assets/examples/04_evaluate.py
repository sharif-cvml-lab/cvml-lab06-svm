"""Example 4 - measure the recognition quality on the test folder.

    python assets/examples/04_evaluate.py [--test DIR] [--threshold 0.45]

The test photos were never used for training. People that are in the training
set should be matched, people that are not should be rejected as "Unknown".
The report also shows how every number changes with the threshold.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root

from config import DEFAULT_TEST_DIR
from dataset_utils import evaluate_directory, format_report, load_model, model_class_names
from face_features import FaceFeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=str(DEFAULT_TEST_DIR))
parser.add_argument("--threshold", type=float, default=0.45)
args = parser.parse_args()

model, meta = load_model()
if model is None:
    sys.exit("No trained model found. Run 02_train_svm.py first.")

records, skipped = evaluate_directory(FaceFeatureExtractor(), model, args.test)
print(format_report(records, model_class_names(model), args.threshold, skipped))
