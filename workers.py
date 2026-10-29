"""Background QThreads for model loading, training and evaluation (fixed infrastructure).

They keep the GUI responsive: all heavy work (reading hundreds of images,
running MagFace) happens here, and progress is reported with signals.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_utils import evaluate_directory, model_class_names, train_model
from face_features import FaceFeatureExtractor


class ModelLoadThread(QThread):
    """Loads the FaceLandmarker and MagFace (about a second)."""
    loaded = pyqtSignal(object)      # FaceFeatureExtractor
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.loaded.emit(FaceFeatureExtractor())
        except FileNotFoundError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Could not load the face models: {exc}")


class _CancellableThread(QThread):
    progress = pyqtSignal(int, int, str)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _should_cancel(self):
        return self._cancel

    def _report(self, done, total, message):
        self.progress.emit(done, total, message)


class TrainThread(_CancellableThread):
    """Embeds the database, calls svm.train_svm() and saves the model."""
    finished_training = pyqtSignal(object, object)     # model, info dict
    not_implemented = pyqtSignal(str)

    def __init__(self, extractor, database_dir, svm_params, parent=None):
        super().__init__(parent)
        self._extractor = extractor
        self._database_dir = database_dir
        self._svm_params = svm_params

    def run(self):
        try:
            model, info = train_model(
                self._extractor, self._database_dir, self._svm_params,
                self._report, self._should_cancel,
            )
        except NotImplementedError as exc:
            self.not_implemented.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_training.emit(model, info)


class EvalThread(_CancellableThread):
    """Predicts every image of the test folder with the trained model."""
    finished_evaluation = pyqtSignal(object, object, object)   # records, skipped, known classes

    def __init__(self, extractor, model, test_dir, parent=None):
        super().__init__(parent)
        self._extractor = extractor
        self._model = model
        self._test_dir = test_dir

    def run(self):
        try:
            records, skipped = evaluate_directory(
                self._extractor, self._model, self._test_dir,
                self._report, self._should_cancel,
            )
        except Exception as exc:
            self.failed.emit(f"Evaluation failed: {exc}")
            return
        self.finished_evaluation.emit(records, skipped, model_class_names(self._model))
