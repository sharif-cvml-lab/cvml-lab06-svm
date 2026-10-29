"""Face image -> 512-d identity embedding (fixed infrastructure, do not edit).

Pipeline
    frame --FaceLandmarker--> landmarks of the k largest faces
          --eye/mouth based alignment + crop--> aligned face crop
          --MagFace (ONNX)--> 512-d, L2-normalised embedding

The old OpenCV face detector is no longer used: the MediaPipe FaceLandmarker
finds the faces *and* their landmarks in one step.

Thread note: the landmarker and MagFace are protected by two separate locks, so
the live video thread (landmarks), the recognition worker (MagFace) and the
training thread (both) can share a single extractor safely.
"""

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    FACE_LANDMARKER_FILE,
    MAGFACE_FILE,
    MAGFACE_URL,
    MAX_DETECTED_FACES,
    MIN_FACE_SIZE,
    MODELS_DIR,
)

# MediaPipe landmark indices used for alignment
IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE = 33, 263, 1
IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT = 61, 291


def _find_model_file(models_dir, file_name):
    """Return the path of file_name inside models_dir (case-insensitive), or None."""
    exact = Path(models_dir) / file_name
    if exact.is_file():
        return exact
    if Path(models_dir).is_dir():
        for candidate in Path(models_dir).iterdir():
            if candidate.is_file() and candidate.name.lower() == file_name.lower():
                return candidate
    return None


@dataclass
class DetectedFace:
    """One face found by the landmarker (all coordinates in image pixels)."""
    landmarks: np.ndarray                 # (478, 2) float32
    bbox: tuple                           # (x1, y1, x2, y2) ints: box around all landmarks

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


def _face_from_landmarks(points, image_w, image_h):
    """Landmark array -> DetectedFace; the box is the min/max of the landmarks."""
    x1 = int(max(0, math.floor(points[:, 0].min())))
    y1 = int(max(0, math.floor(points[:, 1].min())))
    x2 = int(min(image_w - 1, math.ceil(points[:, 0].max())))
    y2 = int(min(image_h - 1, math.ceil(points[:, 1].max())))
    return DetectedFace(landmarks=points, bbox=(x1, y1, x2, y2))


class FaceFeatureExtractor:
    """Face image -> 512-dimensional, L2-normalised identity embedding."""

    def __init__(self, models_dir=None, max_faces_limit=MAX_DETECTED_FACES):
        models_dir = Path(models_dir) if models_dir else MODELS_DIR

        landmarker_path = _find_model_file(models_dir, FACE_LANDMARKER_FILE)
        magface_path = _find_model_file(models_dir, MAGFACE_FILE)
        missing = []
        if landmarker_path is None:
            missing.append(FACE_LANDMARKER_FILE)
        if magface_path is None:
            missing.append(f"{MAGFACE_FILE} (download: {MAGFACE_URL})")
        if missing:
            raise FileNotFoundError(
                f"Missing model file(s) in '{models_dir}': {'; '.join(missing)}"
            )

        self.face_landmarker_model_path = str(landmarker_path)
        self.magface_model_path = str(magface_path)
        self.max_faces_limit = max_faces_limit

        # Cache signature: changes whenever one of the model files changes.
        self.signature = (
            f"{landmarker_path.stat().st_size}-{magface_path.stat().st_size}"
        )

        self._landmarker_lock = threading.Lock()
        self._magface_lock = threading.Lock()

        self.magface_net = cv2.dnn.readNetFromONNX(self.magface_model_path)

        base_options = python.BaseOptions(model_asset_path=self.face_landmarker_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces_limit,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ #
    # Step 1: faces + landmarks
    # ------------------------------------------------------------------ #
    def _run_landmarker(self, image_bgr):
        height, width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._landmarker_lock:
            result = self.face_landmarker.detect(mp_image)

        faces = []
        for landmarks in result.face_landmarks:
            points = np.array([(p.x, p.y) for p in landmarks], dtype=np.float32)
            points *= np.array([width, height], dtype=np.float32)
            faces.append(_face_from_landmarks(points, width, height))
        return faces

    def detect_faces(self, image_bgr, max_faces=None, retry_with_padding=False):
        """Return the `max_faces` largest faces (largest landmark box first).

        retry_with_padding: dataset photos are often tight face crops that the
        landmarker misses; with this flag a black border is added and the
        detection is repeated once.
        """
        max_faces = max_faces or self.max_faces_limit
        height, width = image_bgr.shape[:2]

        faces = self._run_landmarker(image_bgr)
        if not faces and retry_with_padding:
            pad = int(0.4 * max(height, width))
            padded = cv2.copyMakeBorder(
                image_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            for face in self._run_landmarker(padded):
                points = face.landmarks - np.float32(pad)
                faces.append(_face_from_landmarks(points, width, height))

        faces = [
            f for f in faces
            if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= MIN_FACE_SIZE
        ]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces]

    # ------------------------------------------------------------------ #
    # Step 2: alignment + crop (same geometry as the original C++ pipeline)
    # ------------------------------------------------------------------ #
    def getFaceBoundingBox(self, y_eye, y_lip, x):
        eye_ratio = 0.45
        lip_ratio = 0.21
        denominator = 1.0 - eye_ratio - lip_ratio
        h = (y_lip - y_eye) / denominator
        w = h
        y1 = y_eye - (eye_ratio * h)
        x1 = x - (w / 2.0)
        return int(round(x1)), int(round(y1)), int(round(w)), int(round(h))

    def crop_face(self, image_bgr, face):
        """Rotate so the eyes are level, then cut the box from getFaceBoundingBox().

        The image is rotated about its centre and the box is computed in the
        rotated coordinates (exactly as before). To stay fast on large frames
        only the box region is warped; pixels of the box that fall outside the
        frame stay black. Returns None if the geometry is degenerate.
        """
        img_h, img_w = image_bgr.shape[:2]
        lm = face.landmarks

        left_eye, right_eye, nose = lm[IDX_LEFT_EYE], lm[IDX_RIGHT_EYE], lm[IDX_NOSE]
        mouth_left, mouth_right = lm[IDX_MOUTH_LEFT], lm[IDX_MOUTH_RIGHT]

        angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        center = (img_w / 2.0, img_h / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        def rotate_point(pt):
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            x_shifted, y_shifted = pt[0] - center[0], pt[1] - center[1]
            x_rot = x_shifted * cos_a + y_shifted * sin_a
            y_rot = -x_shifted * sin_a + y_shifted * cos_a
            return (x_rot + center[0], y_rot + center[1])

        aligned_left_eye = rotate_point(left_eye)
        aligned_right_eye = rotate_point(right_eye)
        aligned_nose = rotate_point(nose)
        aligned_mouth_left = rotate_point(mouth_left)
        aligned_mouth_right = rotate_point(mouth_right)

        y_eye = (aligned_left_eye[1] + aligned_right_eye[1]) * 0.5
        y_lip = (aligned_mouth_left[1] + aligned_mouth_right[1]) * 0.5
        eye_x_center = (aligned_left_eye[0] + aligned_right_eye[0]) * 0.5
        x_center = eye_x_center * 0.7 + aligned_nose[0] * 0.3   # eyes + nose axis

        if y_lip <= y_eye:
            return None

        x1, y1, w, h = self.getFaceBoundingBox(y_eye, y_lip, x_center)
        if w < 16 or h < 16 or w > 4 * max(img_w, img_h) or h > 4 * max(img_w, img_h):
            return None

        # Warp only the box region: shift the rotation so (x1, y1) becomes (0, 0).
        shifted = rotation_matrix.copy()
        shifted[0, 2] -= x1
        shifted[1, 2] -= y1
        warped = cv2.warpAffine(
            image_bgr, shifted, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        # Black out everything that lies outside the original frame.
        cropped = np.zeros((h, w, 3), dtype=image_bgr.dtype)
        box_x1, box_y1 = max(0, x1), max(0, y1)
        box_x2, box_y2 = min(img_w, x1 + w), min(img_h, y1 + h)
        valid_w, valid_h = box_x2 - box_x1, box_y2 - box_y1
        if valid_w <= 0 or valid_h <= 0:
            return None
        offset_x, offset_y = max(0, -x1), max(0, -y1)
        cropped[offset_y:offset_y + valid_h, offset_x:offset_x + valid_w] = \
            warped[offset_y:offset_y + valid_h, offset_x:offset_x + valid_w]
        return cropped

    # ------------------------------------------------------------------ #
    # Step 3: MagFace embedding
    # ------------------------------------------------------------------ #
    def _preprocess_magface_input(self, face_image):
        """Creates an NCHW blob (112x112, BGR, 0..1) for MagFace."""
        return cv2.dnn.blobFromImage(
            face_image, scalefactor=1.0 / 255.0, size=(112, 112),
            mean=(0, 0, 0), swapRB=False, crop=False,
        )

    def embed_crop(self, face_crop):
        """Aligned face crop -> 512-d float32 vector with unit length."""
        blob = self._preprocess_magface_input(face_crop)
        with self._magface_lock:
            self.magface_net.setInput(blob)
            features = self.magface_net.forward().flatten()
        features = features / (np.linalg.norm(features) + 1e-10)
        return features.astype(np.float32)

    def extract_from_face(self, image_bgr, face):
        """Embedding of one detected face, or None if it cannot be cropped."""
        crop = self.crop_face(image_bgr, face)
        if crop is None:
            return None
        return self.embed_crop(crop)

    # ------------------------------------------------------------------ #
    # Convenience: largest face of a still image
    # ------------------------------------------------------------------ #
    def extract(self, image_bgr, retry_with_padding=True):
        """Embedding of the largest face in the image.

        Returns {"face_box": [x1, y1, x2, y2], "landmarks": (478,2) array,
        "identity_features": (512,) array} or None if no usable face is found.
        """
        faces = self.detect_faces(image_bgr, max_faces=1, retry_with_padding=retry_with_padding)
        if not faces:
            return None
        face = faces[0]
        features = self.extract_from_face(image_bgr, face)
        if features is None:
            return None
        return {
            "face_box": list(face.bbox),
            "landmarks": face.landmarks,
            "identity_features": features,
        }
