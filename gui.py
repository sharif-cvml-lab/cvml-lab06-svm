"""PyQt6 user interface (fixed infrastructure, do not edit).

Layout: one large view (webcam / image / video with annotations) on the left,
a scrollable control panel on the right and a status bar with live metrics.
"""

from pathlib import Path

from PyQt6.QtCore import QRectF, QSettings, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QKeySequence,
    QPainter,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from annotation import paint_overlays
from config import (
    APP_NAME,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_DATABASE_DIR,
    DEFAULT_TEST_DIR,
    DEFAULT_THEME,
    IMAGE_EXTENSIONS,
    MAGFACE_URL,
    MAX_CAMERA_INDEX,
    MAX_FACES_MAX,
    MAX_FACES_MIN,
    ORGANIZATION_NAME,
    SETTINGS_NAME,
    SHORTCUT_FULLSCREEN,
    SHORTCUT_PAUSE,
    SHORTCUT_QUIT,
    SHORTCUT_RESET,
    SOURCE_IMAGE,
    SOURCE_VIDEO,
    SOURCE_WEBCAM,
    SVM_C_MAX,
    SVM_C_MIN,
    SVM_GAMMAS,
    SVM_KERNELS,
    THEMES,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    VIDEO_EXTENSIONS,
    RecognitionParameters,
    SvmParameters,
)
from dataset_utils import format_report, load_model
from video_thread import VideoThread
from workers import EvalThread, ModelLoadThread, TrainThread


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
def build_stylesheet(t):
    return f"""
    QMainWindow, QWidget#root {{ background: {t['window']}; }}
    QWidget {{ color: {t['text']}; font-size: 10pt; }}
    QScrollArea {{ border: none; background: transparent; }}
    QWidget#panel {{ background: transparent; }}
    QGroupBox {{ background: {t['panel']}; border: 1px solid {t['border']};
                 border-radius: 8px; margin-top: 14px; padding: 14px 8px 8px 8px;
                 font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px;
                        color: {t['text_dim']}; }}
    QLabel#dim {{ color: {t['text_dim']}; font-weight: 400; }}
    QLabel#banner {{ background: {t['panel']}; border: 1px solid {t['bad']};
                     border-radius: 6px; padding: 8px; color: {t['bad']}; }}
    QPushButton {{ background: {t['panel_alt']}; border: 1px solid {t['border']};
                   border-radius: 6px; padding: 6px 12px; }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:disabled {{ color: {t['text_dim']}; }}
    QPushButton#primary {{ background: {t['accent']}; color: {t['accent_text']};
                           border: none; font-weight: 600; }}
    QPushButton#primary:disabled {{ background: {t['panel_alt']}; color: {t['text_dim']}; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {t['panel_alt']}; border: 1px solid {t['border']};
        border-radius: 5px; padding: 4px 6px; }}
    QComboBox QAbstractItemView {{ background: {t['panel']}; selection-background-color: {t['accent']}; }}
    QSlider::groove:horizontal {{ height: 6px; background: {t['border']}; border-radius: 3px; }}
    QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {t['accent']}; width: 16px; height: 16px;
                                  margin: -6px 0; border-radius: 8px; }}
    QProgressBar {{ border: 1px solid {t['border']}; border-radius: 5px;
                    text-align: center; background: {t['panel_alt']}; }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 4px; }}
    QStatusBar {{ background: {t['panel']}; color: {t['text_dim']}; }}
    QStatusBar QLabel {{ color: {t['text_dim']}; padding: 0 8px; }}
    QPlainTextEdit {{ background: {t['panel']}; border: 1px solid {t['border']}; }}
    """


# --------------------------------------------------------------------------- #
# The image view
# --------------------------------------------------------------------------- #
class ImageView(QWidget):
    """Draws the current frame scaled to fit, plus the annotations on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame = None          # keeps the numpy buffer alive for the QImage
        self._qimage = None
        self._overlays = []
        self._message = "Starting..."
        self._threshold = 0.45
        self._show_landmarks = True
        self._bg = QColor("#101417")
        self._fg = QColor("#9aa5ad")

    def set_colors(self, background, foreground):
        self._bg, self._fg = QColor(background), QColor(foreground)
        self.update()

    def set_threshold(self, threshold):
        self._threshold = threshold
        self.update()

    def set_show_landmarks(self, show):
        self._show_landmarks = show
        self.update()

    def set_message(self, message):
        self._message = message
        if self._qimage is None:
            self.update()

    def set_frame(self, frame, overlays):
        if frame is None:
            self._frame, self._qimage, self._overlays = None, None, []
        else:
            h, w = frame.shape[:2]
            self._frame = frame
            self._qimage = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
            self._overlays = overlays
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg)
        if self._qimage is None:
            painter.setPen(self._fg)
            font = QFont(self.font())
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        img_w, img_h = self._qimage.width(), self._qimage.height()
        scale = min(self.width() / img_w, self.height() / img_h)
        draw_w, draw_h = img_w * scale, img_h * scale
        x0, y0 = (self.width() - draw_w) / 2.0, (self.height() - draw_h) / 2.0

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(x0, y0, draw_w, draw_h), self._qimage)

        painter.save()
        painter.translate(x0, y0)
        painter.scale(scale, scale)
        paint_overlays(painter, self._overlays, img_w, img_h,
                       self._threshold, self._show_landmarks)
        painter.restore()


class ReportDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Evaluation on the test set")
        self.resize(760, 640)
        layout = QVBoxLayout(self)
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        box.setPlainText(text)
        layout.addWidget(box)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 760)
        self.setMinimumSize(1000, 600)

        self.settings = QSettings(ORGANIZATION_NAME, SETTINGS_NAME)
        self.extractor = None
        self.model = None
        self.model_meta = None
        self._train_thread = None
        self._eval_thread = None
        self._loader = None
        self._loading = True               # ignore widget signals while restoring settings
        self._busy = False                 # a training / evaluation task is running

        self._source_kind = SOURCE_WEBCAM
        self._image_path = ""
        self._video_path = ""
        self._paused = False
        self._theme = DEFAULT_THEME

        self._build_ui()
        self._build_shortcuts()
        self._load_settings()
        self._apply_theme()
        self._loading = False

        # video thread (webcam by default)
        self.video = VideoThread(self.spin_camera.value(), self)
        self.video.frame_ready.connect(self._on_frame)
        self.video.status_message.connect(self._on_source_status)
        self.video.error_message.connect(self._show_banner)
        self.video.set_parameters(self._recognition_params())
        self.video.start()

        # previously trained model (if any)
        model, meta = load_model()
        if model is not None:
            self._set_model(model, meta)

        # face models (landmarker + MagFace) are loaded in the background
        self._loader = ModelLoadThread(self)
        self._loader.loaded.connect(self._on_models_loaded)
        self._loader.failed.connect(self._on_models_failed)
        self._loader.start()
        self._update_buttons()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 6)
        outer.setSpacing(10)

        self.view = ImageView()
        outer.addWidget(self.view, 1)

        panel = QWidget()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(8)

        self.banner = QLabel()
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.banner.hide()
        layout.addWidget(self.banner)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_recognition_group())
        layout.addWidget(self._build_training_group())
        layout.addWidget(self._build_evaluation_group())
        layout.addWidget(self._build_misc_group())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(370)
        scroll.setMaximumWidth(440)
        outer.addWidget(scroll)

        # status bar
        bar = self.statusBar()
        self.lbl_status = QLabel("Starting...")
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_detect = QLabel("Landmarks: -- ms")
        self.lbl_recog = QLabel("MagFace: -- ms")
        self.lbl_faces = QLabel("Faces: 0")
        self.lbl_size = QLabel("")
        bar.addWidget(self.lbl_status, 1)
        for label in (self.lbl_fps, self.lbl_detect, self.lbl_recog, self.lbl_faces, self.lbl_size):
            bar.addPermanentWidget(label)

    def _row(self, *widgets):
        row = QHBoxLayout()
        row.setSpacing(6)
        for w in widgets:
            row.addWidget(w)
        return row

    def _dim(self, text):
        label = QLabel(text)
        label.setObjectName("dim")
        label.setWordWrap(True)
        return label

    def _build_source_group(self):
        group = QGroupBox("Input source")
        v = QVBoxLayout(group)

        self.radio_webcam = QRadioButton("Webcam")
        self.radio_image = QRadioButton("Image")
        self.radio_video = QRadioButton("Video")
        self.source_group = QButtonGroup(self)
        for idx, radio in enumerate((self.radio_webcam, self.radio_image, self.radio_video)):
            self.source_group.addButton(radio, idx)
        self.radio_webcam.setChecked(True)
        self.source_group.idClicked.connect(self._on_source_clicked)
        v.addLayout(self._row(self.radio_webcam, self.radio_image, self.radio_video))

        self.spin_camera = QSpinBox()
        self.spin_camera.setRange(0, MAX_CAMERA_INDEX)
        self.spin_camera.valueChanged.connect(self._on_camera_changed)
        cam_row = self._row(QLabel("Camera index"), self.spin_camera)
        v.addLayout(cam_row)

        self.edit_source = QLineEdit()
        self.edit_source.setReadOnly(True)
        self.edit_source.setPlaceholderText("No file selected")
        self.btn_browse_source = QPushButton("Browse...")
        self.btn_browse_source.clicked.connect(self._on_browse_source)
        v.addLayout(self._row(self.edit_source, self.btn_browse_source))

        self.btn_pause = QPushButton(f"Pause ({SHORTCUT_PAUSE})")
        self.btn_pause.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_pause.clicked.connect(self._toggle_pause)
        v.addWidget(self.btn_pause)
        return group

    def _build_recognition_group(self):
        group = QGroupBox("Recognition")
        v = QVBoxLayout(group)

        self.lbl_threshold = QLabel()
        v.addWidget(self.lbl_threshold)
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(THRESHOLD_MIN, THRESHOLD_MAX)
        self.slider_threshold.valueChanged.connect(self._on_params_changed)
        v.addWidget(self.slider_threshold)
        v.addWidget(self._dim("Cosine similarity needed to accept a match. Too high: everybody "
                              "is Unknown. Too low: strangers get matched."))

        self.spin_faces = QSpinBox()
        self.spin_faces.setRange(MAX_FACES_MIN, MAX_FACES_MAX)
        self.spin_faces.setToolTip("Only the k largest faces are recognised (MagFace is slow).")
        self.spin_faces.valueChanged.connect(self._on_params_changed)
        v.addLayout(self._row(QLabel("Max faces (k)"), self.spin_faces))

        self.check_landmarks = QCheckBox("Show facial landmarks")
        self.check_landmarks.toggled.connect(self._on_params_changed)
        v.addWidget(self.check_landmarks)
        return group

    def _build_training_group(self):
        group = QGroupBox("SVM training")
        v = QVBoxLayout(group)

        self.edit_database = QLineEdit()
        self.edit_database.setReadOnly(True)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_folder(self.edit_database, "Select the database folder"))
        v.addWidget(QLabel("Database folder (one sub-folder per person)"))
        v.addLayout(self._row(self.edit_database, btn))

        self.combo_kernel = QComboBox()
        self.combo_kernel.addItems(SVM_KERNELS)
        self.combo_kernel.currentTextChanged.connect(self._on_svm_changed)
        self.spin_c = QDoubleSpinBox()
        self.spin_c.setDecimals(2)
        self.spin_c.setRange(SVM_C_MIN, SVM_C_MAX)
        self.spin_c.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.spin_c.valueChanged.connect(self._on_svm_changed)
        self.combo_gamma = QComboBox()
        self.combo_gamma.addItems(SVM_GAMMAS)
        self.combo_gamma.currentTextChanged.connect(self._on_svm_changed)
        v.addLayout(self._row(QLabel("Kernel"), self.combo_kernel))
        v.addLayout(self._row(QLabel("C"), self.spin_c))
        v.addLayout(self._row(QLabel("Gamma (rbf)"), self.combo_gamma))

        self.btn_train = QPushButton("Train")
        self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self._on_train)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.hide()
        v.addLayout(self._row(self.btn_train, self.btn_cancel))

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.hide()
        v.addWidget(self.progress)

        self.lbl_model = self._dim("")
        v.addWidget(self.lbl_model)
        return group

    def _build_evaluation_group(self):
        group = QGroupBox("Evaluation")
        v = QVBoxLayout(group)
        self.edit_test = QLineEdit()
        self.edit_test.setReadOnly(True)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_folder(self.edit_test, "Select the test folder"))
        v.addWidget(QLabel("Test folder (same structure as the database)"))
        v.addLayout(self._row(self.edit_test, btn))
        self.btn_eval = QPushButton("Evaluate on test set")
        self.btn_eval.clicked.connect(self._on_evaluate)
        v.addWidget(self.btn_eval)
        return group

    def _build_misc_group(self):
        group = QGroupBox("Appearance")
        v = QVBoxLayout(group)
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["dark", "light"])
        self.combo_theme.currentTextChanged.connect(self._on_theme_changed)
        v.addLayout(self._row(QLabel("Theme"), self.combo_theme))
        self.btn_fullscreen = QPushButton(f"Full screen ({SHORTCUT_FULLSCREEN})")
        self.btn_fullscreen.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        self.btn_reset = QPushButton(f"Reset defaults ({SHORTCUT_RESET})")
        self.btn_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_reset.clicked.connect(self._reset_defaults)
        v.addWidget(self.btn_fullscreen)
        v.addWidget(self.btn_reset)
        return group

    def _build_shortcuts(self):
        for key, handler in (
            (SHORTCUT_PAUSE, self._toggle_pause),
            (SHORTCUT_FULLSCREEN, self._toggle_fullscreen),
            (SHORTCUT_RESET, self._reset_defaults),
            (SHORTCUT_QUIT, self.close),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    def _load_settings(self):
        rp, sp = RecognitionParameters(), SvmParameters()
        s = self.settings
        self.slider_threshold.setValue(s.value("threshold", int(round(rp.threshold * 100)), type=int))
        self.spin_faces.setValue(s.value("max_faces", rp.max_faces, type=int))
        self.check_landmarks.setChecked(s.value("show_landmarks", rp.show_landmarks, type=bool))
        self.combo_kernel.setCurrentText(s.value("kernel", sp.kernel, type=str))
        self.spin_c.setValue(s.value("C", sp.C, type=float))
        self.combo_gamma.setCurrentText(s.value("gamma", sp.gamma, type=str))
        self.edit_database.setText(s.value("database_dir", str(DEFAULT_DATABASE_DIR), type=str))
        self.edit_test.setText(s.value("test_dir", str(DEFAULT_TEST_DIR), type=str))
        self.spin_camera.setValue(s.value("camera_index", DEFAULT_CAMERA_INDEX, type=int))
        self._image_path = s.value("image_path", "", type=str)
        self._video_path = s.value("video_path", "", type=str)
        self._theme = s.value("theme", DEFAULT_THEME, type=str)
        if self._theme not in THEMES:
            self._theme = DEFAULT_THEME
        self.combo_theme.setCurrentText(self._theme)
        self._refresh_param_labels()
        self._refresh_source_widgets()

    def _save_settings(self):
        if self._loading:
            return
        s = self.settings
        s.setValue("threshold", self.slider_threshold.value())
        s.setValue("max_faces", self.spin_faces.value())
        s.setValue("show_landmarks", self.check_landmarks.isChecked())
        s.setValue("kernel", self.combo_kernel.currentText())
        s.setValue("C", self.spin_c.value())
        s.setValue("gamma", self.combo_gamma.currentText())
        s.setValue("database_dir", self.edit_database.text())
        s.setValue("test_dir", self.edit_test.text())
        s.setValue("camera_index", self.spin_camera.value())
        s.setValue("image_path", self._image_path)
        s.setValue("video_path", self._video_path)
        s.setValue("theme", self._theme)

    def _reset_defaults(self):
        rp, sp = RecognitionParameters(), SvmParameters()
        self._loading = True
        self.slider_threshold.setValue(int(round(rp.threshold * 100)))
        self.spin_faces.setValue(rp.max_faces)
        self.check_landmarks.setChecked(rp.show_landmarks)
        self.combo_kernel.setCurrentText(sp.kernel)
        self.spin_c.setValue(sp.C)
        self.combo_gamma.setCurrentText(sp.gamma)
        self.edit_database.setText(str(DEFAULT_DATABASE_DIR))
        self.edit_test.setText(str(DEFAULT_TEST_DIR))
        self._loading = False
        self._on_params_changed()
        self._on_svm_changed()
        self._save_settings()
        self.lbl_status.setText("Defaults restored")

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    def _recognition_params(self):
        return RecognitionParameters(
            threshold=self.slider_threshold.value() / 100.0,
            max_faces=self.spin_faces.value(),
            show_landmarks=self.check_landmarks.isChecked(),
        )

    def _svm_params(self):
        return SvmParameters(
            kernel=self.combo_kernel.currentText(),
            C=self.spin_c.value(),
            gamma=self.combo_gamma.currentText(),
        )

    def _refresh_param_labels(self):
        self.lbl_threshold.setText(
            f"Match similarity threshold: <b>{self.slider_threshold.value() / 100.0:.2f}</b>"
        )
        self.combo_gamma.setEnabled(self.combo_kernel.currentText() == "rbf")

    def _on_params_changed(self, *_):
        self._refresh_param_labels()
        params = self._recognition_params()
        self.view.set_threshold(params.threshold)
        self.view.set_show_landmarks(params.show_landmarks)
        if hasattr(self, "video"):
            self.video.set_parameters(params)
        self._save_settings()

    def _on_svm_changed(self, *_):
        self._refresh_param_labels()
        self._save_settings()

    def _on_camera_changed(self, value):
        if hasattr(self, "video"):
            self.video.set_camera_index(value)
        self._save_settings()

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #
    def _apply_theme(self):
        theme = THEMES[self._theme]
        self.setStyleSheet(build_stylesheet(theme))
        self.view.set_colors(theme["view"], theme["text_dim"])

    def _on_theme_changed(self, name):
        if name in THEMES:
            self._theme = name
            self._apply_theme()
            self._save_settings()

    # ------------------------------------------------------------------ #
    # Input source
    # ------------------------------------------------------------------ #
    def _refresh_source_widgets(self):
        kind = self._source_kind
        {SOURCE_WEBCAM: self.radio_webcam, SOURCE_IMAGE: self.radio_image,
         SOURCE_VIDEO: self.radio_video}[kind].setChecked(True)
        self.spin_camera.setEnabled(kind == SOURCE_WEBCAM)
        self.btn_browse_source.setEnabled(kind != SOURCE_WEBCAM)
        if kind == SOURCE_IMAGE:
            self.edit_source.setText(self._image_path)
        elif kind == SOURCE_VIDEO:
            self.edit_source.setText(self._video_path)
        else:
            self.edit_source.setText("")

    def _browse_file(self, kind):
        if kind == SOURCE_IMAGE:
            title, exts, start = "Open image", IMAGE_EXTENSIONS, self._image_path
        else:
            title, exts, start = "Open video", VIDEO_EXTENSIONS, self._video_path
        pattern = " ".join(f"*{e}" for e in sorted(exts))
        path, _ = QFileDialog.getOpenFileName(
            self, title, str(Path(start).parent) if start else "", f"Files ({pattern});;All files (*)"
        )
        return path

    def _activate_source(self, kind):
        self._source_kind = kind
        if self._paused:                       # a new source always starts running
            self._toggle_pause()
        self._refresh_source_widgets()
        path = {SOURCE_WEBCAM: "", SOURCE_IMAGE: self._image_path, SOURCE_VIDEO: self._video_path}[kind]
        self.video.set_source(kind, path)
        self._save_settings()

    def _on_source_clicked(self, button_id):
        kind = (SOURCE_WEBCAM, SOURCE_IMAGE, SOURCE_VIDEO)[button_id]
        if kind != SOURCE_WEBCAM:
            current = self._image_path if kind == SOURCE_IMAGE else self._video_path
            if not current or not Path(current).is_file():
                path = self._browse_file(kind)
                if not path:                   # cancelled: go back to the previous source
                    self._refresh_source_widgets()
                    return
                if kind == SOURCE_IMAGE:
                    self._image_path = path
                else:
                    self._video_path = path
        self._activate_source(kind)

    def _on_browse_source(self):
        kind = self._source_kind
        if kind == SOURCE_WEBCAM:
            return
        path = self._browse_file(kind)
        if not path:
            return
        if kind == SOURCE_IMAGE:
            self._image_path = path
        else:
            self._video_path = path
        self._activate_source(kind)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.video.set_paused(self._paused)
        label = "Resume" if self._paused else "Pause"
        self.btn_pause.setText(f"{label} ({SHORTCUT_PAUSE})")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ------------------------------------------------------------------ #
    # Video thread signals
    # ------------------------------------------------------------------ #
    def _on_frame(self, frame, overlays, stats):
        if frame is None:
            self.view.set_frame(None, [])
            self.view.set_message(self.lbl_status.text() or "No input")
            return
        self.view.set_frame(frame, overlays)
        self.lbl_fps.setText(f"FPS: {stats.fps:.1f}" if stats.fps > 0 else "FPS: --")
        self.lbl_detect.setText(f"Landmarks: {stats.detect_ms:.0f} ms")
        self.lbl_recog.setText(f"MagFace: {stats.recognize_ms:.0f} ms")
        self.lbl_faces.setText(f"Faces: {stats.faces}")
        self.lbl_size.setText(f"{stats.width}x{stats.height}")

    def _on_source_status(self, message):
        self.lbl_status.setText(message)
        self.view.set_message(message)

    def _show_banner(self, text):
        self.banner.setText(text)
        self.banner.setVisible(bool(text))

    # ------------------------------------------------------------------ #
    # Models and training
    # ------------------------------------------------------------------ #
    def _on_models_loaded(self, extractor):
        self.extractor = extractor
        self.video.set_extractor(extractor)
        self.lbl_status.setText("Face models loaded")
        self._show_banner("")
        self._update_buttons()

    def _on_models_failed(self, message):
        self._show_banner(
            f"{message}\n\nPut the files into assets/models/ and restart. "
            f"MagFace download: {MAGFACE_URL}"
        )
        self.lbl_status.setText("Face models missing")
        self._update_buttons()

    def _set_model(self, model, meta):
        self.model, self.model_meta = model, meta or {}
        self.video.set_model(model)
        classes = len(model["class_names"])
        meta = self.model_meta
        if meta.get("num_images"):
            self.lbl_model.setText(
                f"Model ready: {classes} people, {meta['num_images']} images "
                f"({meta.get('kernel', '?')}, C={meta.get('C', '?')})"
            )
        else:
            self.lbl_model.setText(f"Model ready: {classes} people")
        self._update_buttons()

    def _update_buttons(self):
        busy = self._busy
        ready = self.extractor is not None
        self.btn_train.setEnabled(ready and not busy)
        self.btn_eval.setEnabled(ready and not busy and self.model is not None)
        self.btn_cancel.setVisible(busy)
        self.progress.setVisible(busy)
        if self.model is None and not self.lbl_model.text():
            self.lbl_model.setText("No model yet: press Train.")

    def _start_task(self, thread, text):
        thread.progress.connect(self._on_task_progress)
        thread.failed.connect(self._on_task_failed)
        self.progress.setRange(0, 0)
        self.progress.setFormat(text)
        self.lbl_status.setText(text)
        self._busy = True
        thread.start()
        self._update_buttons()

    def _on_task_progress(self, done, total, message):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{total}")
        self.lbl_status.setText(message)

    def _on_task_failed(self, message):
        self._busy = False
        self.lbl_status.setText(message)
        if "cancelled" not in message.lower():
            QMessageBox.warning(self, APP_NAME, message)
        self._update_buttons()

    def _check_folder(self, edit, what):
        folder = Path(edit.text())
        if not folder.is_dir():
            QMessageBox.warning(self, APP_NAME, f"The {what} folder does not exist:\n{folder}")
            return None
        return folder

    def _on_train(self):
        if self.extractor is None:
            return
        folder = self._check_folder(self.edit_database, "database")
        if folder is None:
            return
        self._train_thread = TrainThread(self.extractor, folder, self._svm_params(), self)
        self._train_thread.finished_training.connect(self._on_trained)
        self._train_thread.not_implemented.connect(self._on_not_implemented)
        self._start_task(self._train_thread, "Training...")

    def _on_trained(self, model, info):
        self._busy = False
        self._set_model(model, info)
        self._show_banner("")
        message = f"Trained on {info['num_images']} images of {info['num_classes']} people"
        if info["skipped"]:
            message += f" ({len(info['skipped'])} images without a usable face were skipped)"
        self.lbl_status.setText(message)

    def _on_not_implemented(self, message):
        self._busy = False
        self.lbl_status.setText("train_svm() is not implemented yet")
        QMessageBox.information(
            self, APP_NAME,
            f"{message}\n\nOpen svm.py and follow the TODO comments of train_svm() and predict_svm().",
        )
        self._update_buttons()

    def _on_cancel(self):
        for thread in (self._train_thread, self._eval_thread):
            if thread is not None and thread.isRunning():
                thread.cancel()

    def _on_evaluate(self):
        if self.extractor is None or self.model is None:
            return
        folder = self._check_folder(self.edit_test, "test")
        if folder is None:
            return
        self._eval_thread = EvalThread(self.extractor, self.model, folder, self)
        self._eval_thread.finished_evaluation.connect(self._on_evaluated)
        self._start_task(self._eval_thread, "Evaluating...")

    def _on_evaluated(self, records, skipped, known_classes):
        self._busy = False
        self._update_buttons()
        if not records:
            QMessageBox.information(self, APP_NAME, "No test image with a usable face was found.")
            return
        threshold = self.slider_threshold.value() / 100.0
        text = format_report(records, known_classes, threshold, skipped)
        self.lbl_status.setText("Evaluation finished")
        ReportDialog(text, self).exec()

    def _browse_folder(self, edit, title):
        start = edit.text() if Path(edit.text()).is_dir() else ""
        folder = QFileDialog.getExistingDirectory(self, title, start)
        if folder:
            edit.setText(folder)
            self._save_settings()

    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        self._save_settings()
        for thread in (self._train_thread, self._eval_thread):
            if thread is not None and thread.isRunning():
                thread.cancel()
                thread.wait()
        if self._loader is not None and self._loader.isRunning():
            self._loader.wait()
        self.video.stop()
        event.accept()
