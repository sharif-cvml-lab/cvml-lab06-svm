"""Video input, face tracking and asynchronous recognition (fixed infrastructure).

Two threads cooperate so the picture stays smooth even though MagFace is slow:

  VideoThread (this QThread)
      reads a frame -> FaceLandmarker (fast) -> tracks the faces between frames
      -> emits frame + overlays to the GUI at the full frame rate.
  RecognitionWorker (plain thread)
      takes ONE aligned face crop at a time -> MagFace -> predict_svm()
      -> posts the raw (name, similarity) back to the video thread.

Every face keeps its last result until the worker refreshes it, so labels never
flicker and the frame rate does not depend on the MagFace speed. The threshold
is applied only when drawing, so moving the slider is instantaneous.
"""

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from annotation import (
    STATUS_NO_MODEL,
    STATUS_PENDING,
    STATUS_READY,
    FaceOverlay,
    landmarks_to_polygon,
)
from config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    MAX_FRAME_SIDE,
    RECOGNITION_REFRESH_SEC,
    SOURCE_IMAGE,
    SOURCE_VIDEO,
    SOURCE_WEBCAM,
    TARGET_FPS,
    TRACK_MAX_MISSED,
    TRACK_MIN_IOU,
    RecognitionParameters,
)
from dataset_utils import predict_identity, read_image


# --------------------------------------------------------------------------- #
# Face tracking (matches faces of consecutive frames by box overlap)
# --------------------------------------------------------------------------- #
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class FaceTrack:
    track_id: int
    face: object                      # latest DetectedFace
    missed: int = 0
    has_result: bool = False
    name: str = ""
    score: float = 0.0
    result_time: float = 0.0


class FaceTracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def reset(self):
        self.tracks = []

    def update(self, faces, purge_unmatched=False):
        """Match new detections to existing tracks (greedy, best IoU first)."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for fi, face in enumerate(faces):
                iou = _iou(track.face.bbox, face.bbox)
                if iou >= TRACK_MIN_IOU:
                    pairs.append((iou, ti, fi))
        pairs.sort(reverse=True)

        used_tracks, used_faces = set(), set()
        for iou, ti, fi in pairs:
            if ti in used_tracks or fi in used_faces:
                continue
            self.tracks[ti].face = faces[fi]
            self.tracks[ti].missed = 0
            used_tracks.add(ti)
            used_faces.add(fi)

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.missed += 1
        for fi, face in enumerate(faces):
            if fi not in used_faces:
                self.tracks.append(FaceTrack(self._next_id, face))
                self._next_id += 1

        limit = 0 if purge_unmatched else TRACK_MAX_MISSED
        self.tracks = [t for t in self.tracks if t.missed <= limit]

    def visible(self):
        return [t for t in self.tracks if t.missed == 0]


# --------------------------------------------------------------------------- #
# Asynchronous recognition
# --------------------------------------------------------------------------- #
@dataclass
class RecognitionJob:
    track_id: int
    generation: int
    crop: np.ndarray
    model: object


@dataclass
class RecognitionResult:
    track_id: int
    generation: int
    name: str = ""
    score: float = 0.0
    millis: float = 0.0
    error: str = ""


class RecognitionWorker(threading.Thread):
    """Processes one crop at a time: MagFace + predict_svm()."""

    def __init__(self, extractor):
        super().__init__(daemon=True)
        self._extractor = extractor
        self._jobs = queue.Queue()
        self.results = queue.Queue()
        self._busy = False

    def is_idle(self):
        return not self._busy

    def submit(self, job):
        if self._busy:
            return False
        self._busy = True
        self._jobs.put(job)
        return True

    def stop(self):
        self._jobs.put(None)

    def run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            started = time.perf_counter()
            result = RecognitionResult(job.track_id, job.generation)
            try:
                embedding = self._extractor.embed_crop(job.crop)
                result.name, result.score = predict_identity(job.model, embedding)
            except Exception as exc:   # a bug in the student's code must not kill the app
                result.error = f"predict_svm() failed: {exc}"
            result.millis = (time.perf_counter() - started) * 1000.0
            self.results.put(result)
            self._busy = False


# --------------------------------------------------------------------------- #
# The video thread
# --------------------------------------------------------------------------- #
@dataclass
class FrameStats:
    fps: float = 0.0              # 0 for still images / paused
    detect_ms: float = 0.0
    recognize_ms: float = 0.0
    faces: int = 0
    width: int = 0
    height: int = 0


class VideoThread(QThread):
    # frame (BGR ndarray), list[FaceOverlay], FrameStats
    frame_ready = pyqtSignal(object, object, object)
    status_message = pyqtSignal(str)       # source / camera status
    error_message = pyqtSignal(str)        # problems in the student's code

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()

        # requests from the GUI thread (protected by _lock)
        self._source_kind = SOURCE_WEBCAM
        self._source_path = ""
        self._camera_index = camera_index
        self._source_version = 0
        self._paused = False
        self._parameters = RecognitionParameters()
        self._extractor = None
        self._model = None
        self._generation = 0
        self._redetect = True
        self._render = True

        self._last_status = ""
        self._last_error = ""

    # ---------------- GUI-thread API ---------------- #
    def set_extractor(self, extractor):
        with self._lock:
            self._extractor = extractor
            self._generation += 1
            self._redetect = True

    def set_model(self, model):
        with self._lock:
            self._model = model
            self._generation += 1
            self._redetect = True

    def set_parameters(self, parameters):
        with self._lock:
            old_k = self._parameters.max_faces
            self._parameters = RecognitionParameters(
                parameters.threshold, parameters.max_faces, parameters.show_landmarks
            )
            if parameters.max_faces != old_k:
                self._redetect = True
            self._render = True

    def set_source(self, kind, path=""):
        with self._lock:
            self._source_kind = kind
            self._source_path = path
            self._source_version += 1

    def set_camera_index(self, camera_index):
        with self._lock:
            self._camera_index = camera_index
            if self._source_kind == SOURCE_WEBCAM:
                self._source_version += 1

    def set_paused(self, paused):
        with self._lock:
            self._paused = paused
            self._render = True

    def stop(self):
        self._running.clear()
        self.wait()

    # ---------------- helpers ---------------- #
    def _snapshot(self):
        with self._lock:
            snap = (
                self._source_kind, self._source_path, self._camera_index,
                self._source_version, self._paused, self._parameters,
                self._extractor, self._model, self._generation,
                self._redetect, self._render,
            )
            self._redetect = False
            self._render = False
            return snap

    def _status(self, message):
        if message != self._last_status:
            self._last_status = message
            self.status_message.emit(message)

    def _error(self, message):
        if message != self._last_error:
            self._last_error = message
            self.error_message.emit(message)

    @staticmethod
    def _fit(frame):
        """Down-scale huge frames once; make sure the array is contiguous."""
        h, w = frame.shape[:2]
        side = max(h, w)
        if side > MAX_FRAME_SIDE:
            scale = MAX_FRAME_SIDE / side
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame)

    def _open_source(self, kind, path, camera_index):
        """Returns (capture or None, still image or None, fps)."""
        if kind == SOURCE_IMAGE:
            image = read_image(path)
            if image is None:
                self._status(f"Could not read image: {path}")
                return None, None, 0.0
            self._status(f"Image: {path}")
            return None, self._fit(image), 0.0
        if kind == SOURCE_VIDEO:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                self._status(f"Could not open video: {path}")
                return None, None, 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1.0 <= fps <= 120.0 else 30.0
            self._status(f"Video: {path}")
            return cap, None, fps
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        if not cap.isOpened():
            cap.release()
            self._status(f"Camera {camera_index} is unavailable. Retrying...")
            return None, None, 0.0
        self._status(f"Camera {camera_index} connected")
        return cap, None, float(TARGET_FPS)

    # ---------------- main loop ---------------- #
    def run(self):
        tracker = FaceTracker()
        worker = None
        worker_extractor = None

        cap = None
        still = None                 # still image (SOURCE_IMAGE)
        frame = None                 # latest frame shown
        opened_version = -1
        opened_generation = -1
        source_fps = float(TARGET_FPS)
        source_ok = False

        detect_ms = 0.0
        recog_ms = 0.0
        fps_ema = 0.0
        last_time = time.perf_counter()
        need_detect = True
        need_render = True

        try:
            while self._running.is_set():
                (kind, path, cam_idx, version, paused, params, extractor,
                 model, generation, redetect, render) = self._snapshot()
                need_detect = need_detect or redetect
                need_render = need_render or render

                # (re)create the recognition worker when the extractor changes
                if extractor is not worker_extractor:
                    if worker is not None:
                        worker.stop()
                        worker = None
                    worker_extractor = extractor
                    if extractor is not None:
                        worker = RecognitionWorker(extractor)
                        worker.start()

                # ---- source change ----
                if version != opened_version:
                    if cap is not None:
                        cap.release()
                    cap, still, source_fps = self._open_source(kind, path, cam_idx)
                    source_ok = cap is not None or still is not None
                    opened_version = version
                    frame = still
                    tracker.reset()
                    need_detect = need_render = True
                    if not source_ok:
                        self.frame_ready.emit(None, [], FrameStats())
                        if kind == SOURCE_WEBCAM:
                            time.sleep(1.0)
                            opened_version = -1      # retry the camera
                        else:
                            time.sleep(0.1)
                        continue

                if generation != opened_generation:      # model / extractor changed
                    opened_generation = generation
                    tracker.reset()
                    need_detect = need_render = True

                if not source_ok:
                    time.sleep(0.05)
                    continue

                loop_start = time.perf_counter()
                live = cap is not None and not paused

                # ---- acquire ----
                if live:
                    ok, new_frame = cap.read()
                    if not ok:
                        if kind == SOURCE_VIDEO:           # end of file: loop
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            tracker.reset()
                            ok, new_frame = cap.read()
                        if not ok:
                            self._status("Frame read failed. Reconnecting...")
                            cap.release()
                            cap = None
                            opened_version = -1
                            time.sleep(0.5)
                            continue
                    frame = self._fit(new_frame)
                    need_detect = True
                    need_render = True
                    self._status(
                        f"Camera {cam_idx} running" if kind == SOURCE_WEBCAM else "Playing video"
                    )
                elif paused:
                    self._status("Paused")

                # ---- detection (landmarks) ----
                if extractor is not None and frame is not None and need_detect:
                    t0 = time.perf_counter()
                    try:
                        faces = extractor.detect_faces(frame, max_faces=params.max_faces)
                    except Exception as exc:
                        faces = []
                        self._error(f"Face detection failed: {exc}")
                    detect_ms = 0.8 * detect_ms + 0.2 * (time.perf_counter() - t0) * 1000.0 \
                        if detect_ms else (time.perf_counter() - t0) * 1000.0
                    tracker.update(faces, purge_unmatched=not live)
                    need_detect = False
                    need_render = True

                # ---- collect finished recognitions ----
                if worker is not None:
                    while True:
                        try:
                            result = worker.results.get_nowait()
                        except queue.Empty:
                            break
                        if result.error:
                            self._error(result.error)
                        if result.generation != generation:
                            continue                       # stale result
                        recog_ms = result.millis if not recog_ms else 0.7 * recog_ms + 0.3 * result.millis
                        for track in tracker.tracks:
                            if track.track_id == result.track_id:
                                track.name, track.score = result.name, result.score
                                track.has_result = True
                                track.result_time = time.perf_counter()
                        need_render = True

                    # ---- schedule the next recognition (one at a time) ----
                    if model is not None and frame is not None and worker.is_idle():
                        self._schedule(worker, extractor, tracker, frame, model, generation, live)

                # ---- emit ----
                if need_render and frame is not None:
                    overlays = self._build_overlays(tracker, model, extractor)
                    now = time.perf_counter()
                    if live:
                        dt = now - last_time
                        inst = 1.0 / dt if dt > 0 else 0.0
                        fps_ema = inst if not fps_ema else 0.9 * fps_ema + 0.1 * inst
                    else:
                        fps_ema = 0.0
                    last_time = now
                    h, w = frame.shape[:2]
                    stats = FrameStats(fps_ema, detect_ms, recog_ms, len(overlays), w, h)
                    self.frame_ready.emit(frame, overlays, stats)
                    need_render = False

                # ---- pacing ----
                if live:
                    period = 1.0 / (source_fps if kind == SOURCE_VIDEO else TARGET_FPS)
                    time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
                else:
                    time.sleep(0.03)
        finally:
            if cap is not None:
                cap.release()
            if worker is not None:
                worker.stop()
            self._status("Stopped")

    # ---------------- recognition scheduling / overlays ---------------- #
    @staticmethod
    def _schedule(worker, extractor, tracker, frame, model, generation, live):
        now = time.perf_counter()
        candidates = []
        for track in tracker.visible():
            if not track.has_result:
                candidates.append((0, track.track_id, track))
            elif live and now - track.result_time >= RECOGNITION_REFRESH_SEC:
                candidates.append((1, track.result_time, track))
        if not candidates:
            return
        candidates.sort(key=lambda c: (c[0], c[1]))
        track = candidates[0][2]
        crop = extractor.crop_face(frame, track.face)
        if crop is None:                         # degenerate geometry
            track.name, track.score = "", 0.0
            track.has_result = True
            track.result_time = now
            return
        worker.submit(RecognitionJob(track.track_id, generation, crop, model))

    @staticmethod
    def _build_overlays(tracker, model, extractor):
        overlays = []
        for track in tracker.visible():
            if extractor is None or model is None:
                status = STATUS_NO_MODEL
            elif track.has_result:
                status = STATUS_READY
            else:
                status = STATUS_PENDING
            overlays.append(FaceOverlay(
                bbox=track.face.bbox,
                landmarks=landmarks_to_polygon(track.face.landmarks),
                status=status, name=track.name, score=track.score,
            ))
        return overlays
