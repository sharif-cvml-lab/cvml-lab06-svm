"""Example 1 - from a photo to a 512-number "identity" vector.

    python assets/examples/01_extract_embedding.py [image]

Steps: read image -> find the largest face and its landmarks -> align and crop
the face -> MagFace -> 512-d embedding of unit length.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root

import numpy as np

from config import DEFAULT_DATABASE_DIR
from dataset_utils import list_identities, read_image
from face_features import FaceFeatureExtractor

if len(sys.argv) > 1:
    image_path = Path(sys.argv[1])
else:   # default: the first photo of the first person in the database
    first_person = next(iter(list_identities(DEFAULT_DATABASE_DIR).values()))
    image_path = first_person[0]

image = read_image(image_path)
if image is None:
    sys.exit(f"Cannot read image: {image_path}")

extractor = FaceFeatureExtractor()
result = extractor.extract(image)
if result is None:
    sys.exit("No face found in the image.")

embedding = result["identity_features"]
print(f"Image            : {image_path}")
print(f"Face box         : {result['face_box']}   (x1, y1, x2, y2)")
print(f"Landmarks        : {result['landmarks'].shape[0]} points")
print(f"Embedding shape  : {embedding.shape}")
print(f"Embedding length : {np.linalg.norm(embedding):.3f}   (always 1.000)")
print(f"First 5 numbers  : {np.round(embedding[:5], 4)}")
