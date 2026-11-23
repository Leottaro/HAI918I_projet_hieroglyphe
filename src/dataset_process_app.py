import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import SEED
from dataset_io import append_sample_metadata, find_image_paths, save_processed_image
from image_canvas import ImageCanvas
from pretreatment import (
    DEFAULT_OUTPUT_SIZE,
    create_sam_predictor,
    find_prompt_groups,
    mask_to_box,
    prepare_processed_image,
    segment_image,
)

""" 
ain window for this interface
help to explore, label, segement, pretreat and save images
"""
class DatasetProcessWindow(QMainWindow):

    def __init__(self, sam_device: str = "cpu") -> None:
        super().__init__()
        self._init_state(sam_device)
        self._build_ui()
        self._connect_signals()
        self.statusBar().showMessage("Please select an image or directory")

    def _init_state(self, sam_device: str) -> None:
        self.sam_device: str = sam_device
        self.sam_predictor: SAM2ImagePredictor | None = None

        # two rng, one for prompt points, one for gaussin texture
        self.prompt_rng: np.random.Generator = np.random.default_rng(SEED)
        self.background_rng: np.random.Generator = np.random.default_rng(SEED)

        # exploration state for current images in this path
        self.input_image_paths: list[Path] = []
        self.current_image_index: int = -1
        self.current_source_path: Path | None = None
        self.output_directory: Path | None = None

        self.segmentation_mask: np.ndarray | None = None
        # image processed, we save this image
        self.processed_image: Image.Image | None = None
        # preview of processed image
        self.preview_pixmap: QPixmap | None = None
        # original image shown currently
        self.current_original_image: Image.Image | None = None


    def _build_ui(self) -> None:
        self.setWindowTitle("Dataset process")
        self.resize(1200, 820)
        central_widget: QWidget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout: QHBoxLayout = QHBoxLayout(central_widget)
        splitter: QSplitter = QSplitter(Qt.Orientation.Horizontal, central_widget)
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_controls_panel())
        splitter.setSizes([820, 340])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)

    def _build_canvas_panel(self) -> QWidget:
        panel: QWidget = QWidget()
        layout: QVBoxLayout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.source_filename_label: QLabel = QLabel("No image selected", panel)
        self.source_filename_label.setWordWrap(True)
        self.image_canvas: ImageCanvas = ImageCanvas(panel)
        hint_label: QLabel = QLabel(
            "Shift + left: draw box | Left/right: positive/negative point\n"
            "Drag a point or border: move | Middle drag: pan | Wheel: zoom",
            panel,
        )
        hint_label.setWordWrap(True)
        layout.addWidget(self.source_filename_label)
        layout.addWidget(self.image_canvas, stretch=1)
        layout.addWidget(hint_label)
        return panel

    def _build_controls_panel(self) -> QWidget:
        container: QWidget = QWidget()
        container.setMinimumWidth(320)
        container_layout: QVBoxLayout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area: QScrollArea = QScrollArea(container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        panel: QWidget = QWidget()
        layout: QVBoxLayout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)
        self._build_file_controls(layout, panel)
        self._build_canvas_tools(layout, panel)
        self._build_size_controls(layout, panel)
        self._build_preview_controls(layout, panel)
        self._build_save_controls(layout, panel)
        layout.addStretch(1)
        scroll_area.setWidget(panel)
        container_layout.addWidget(scroll_area, stretch=1)
        self._build_navigation_controls(container_layout, container)
        return container

    def _build_file_controls(self, layout: QVBoxLayout, parent: QWidget) -> None:
        self.open_image_button: QPushButton = QPushButton("Open image", parent)
        self.source_path_label: QLabel = QLabel("Image path: —", parent)
        self.open_directory_button: QPushButton = QPushButton("Open directory", parent)
        self.input_directory_label: QLabel = QLabel("Input directory: —", parent)
        self.set_output_directory_button: QPushButton = QPushButton("Set output directory", parent)
        self.output_directory_label: QLabel = QLabel("Output directory: —", parent)
        for label in (
            self.source_path_label,
            self.input_directory_label,
            self.output_directory_label,
        ):
            label.setWordWrap(True)
        for widget in (
            self.open_image_button, self.source_path_label,
            self.open_directory_button, self.input_directory_label,
            self.set_output_directory_button, self.output_directory_label,
        ):
            layout.addWidget(widget)

    def _build_canvas_tools(self, layout: QVBoxLayout, parent: QWidget) -> None:
        tools_layout: QGridLayout = QGridLayout()
        self.fit_image_button: QPushButton = QPushButton("Fit image", parent)
        self.delete_selected_button: QPushButton = QPushButton("Delete selected", parent)
        self.clear_points_button: QPushButton = QPushButton("Clear prompt points", parent)
        self.clear_box_button: QPushButton = QPushButton("Clear box", parent)
        self.auto_prompts_button: QPushButton = QPushButton("Auto prompts", parent)
        self.segment_button: QPushButton = QPushButton("Auto segment", parent)
        tools_layout.addWidget(self.fit_image_button, 0, 0)
        tools_layout.addWidget(self.delete_selected_button, 0, 1)
        tools_layout.addWidget(self.clear_points_button, 1, 0)
        tools_layout.addWidget(self.clear_box_button, 1, 1)
        tools_layout.addWidget(self.auto_prompts_button, 2, 0)
        tools_layout.addWidget(self.segment_button, 2, 1)
        layout.addLayout(tools_layout)

    def _build_size_controls(self, layout: QVBoxLayout, parent: QWidget) -> None:
        layout.addWidget(QLabel("Output size", parent))
        self.output_width_spinbox: QSpinBox = QSpinBox(parent)
        self.output_height_spinbox: QSpinBox = QSpinBox(parent)
        for spinbox, default_value in (
            (self.output_width_spinbox, DEFAULT_OUTPUT_SIZE[0]),
            (self.output_height_spinbox, DEFAULT_OUTPUT_SIZE[1]),
        ):
            spinbox.setRange(32, 2048)
            spinbox.setValue(default_value)
            spinbox.setSuffix(" px")
            spinbox.setMinimumHeight(30)
        size_layout: QFormLayout = QFormLayout()
        size_layout.addRow("Width", self.output_width_spinbox)
        size_layout.addRow("Height", self.output_height_spinbox)
        layout.addLayout(size_layout)

    def _build_preview_controls(self, layout: QVBoxLayout, parent: QWidget) -> None:
        layout.addWidget(QLabel("Processed preview", parent))
        self.preview_label: QLabel = QLabel("Select a character to preview", parent)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_label.setMinimumSize(160, 160)
        self.preview_label.setMaximumHeight(220)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.preview_label.installEventFilter(self)
        self.update_preview_button: QPushButton = QPushButton("Update preview", parent)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.update_preview_button)

    def _build_save_controls(self, layout: QVBoxLayout, parent: QWidget) -> None:
        self.class_label_edit: QLineEdit = QLineEdit(parent)
        self.class_label_edit.setPlaceholderText("class1 / class2 / class3")
        self.filename_edit: QLineEdit = QLineEdit(parent)
        self.filename_edit.setPlaceholderText("output.png")
        for edit in (self.class_label_edit, self.filename_edit):
            edit.setMinimumWidth(240)
            edit.setFixedHeight(35)
        metadata_layout: QFormLayout = QFormLayout()
        metadata_layout.addRow("Class label", self.class_label_edit)
        metadata_layout.addRow("Filename", self.filename_edit)
        self.save_button: QPushButton = QPushButton("Save character", parent)
        layout.addLayout(metadata_layout)
        layout.addWidget(self.save_button)

    def _build_navigation_controls(self, layout: QVBoxLayout, parent: QWidget) -> None:
        self.progress_label: QLabel = QLabel("-- / --", parent)
        self.previous_button: QPushButton = QPushButton("Previous image", parent)
        self.next_button: QPushButton = QPushButton("Next image", parent)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        navigation_layout: QHBoxLayout = QHBoxLayout()
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.next_button)
        layout.addWidget(self.progress_label)
        layout.addLayout(navigation_layout)

    def _connect_signals(self) -> None:
        self.open_image_button.clicked.connect(self.open_image)
        self.open_directory_button.clicked.connect(self.open_directory)
        self.set_output_directory_button.clicked.connect(self.set_output_directory)
        self.fit_image_button.clicked.connect(self.image_canvas.fit_image)
        self.delete_selected_button.clicked.connect(self.image_canvas.delete_selected_items)
        self.clear_points_button.clicked.connect(self.image_canvas.clear_prompt_points)
        self.clear_box_button.clicked.connect(self.image_canvas.clear_box)
        self.auto_prompts_button.clicked.connect(self.generate_auto_prompts)
        self.segment_button.clicked.connect(self.segment_current_image)
        self.update_preview_button.clicked.connect(self.update_preview)
        self.previous_button.clicked.connect(self.load_previous_image)
        self.next_button.clicked.connect(self.load_next_image)
        self.save_button.clicked.connect(self.save_current_result)

    """ image IO and directory"""
    def open_image(self) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(
            self, "Please select image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All (*)"
        )
        if not selected_path:
            return
        image_path: Path = Path(selected_path)
        if self.load_source_image(image_path):
            self.input_image_paths = [image_path]
            self.current_image_index = 0
            self.input_directory_label.setText("No input directory selected")
            self._update_navigation()

    def open_directory(self) -> None:
        selected_path: str = QFileDialog.getExistingDirectory(
            self, "Please select an input directory", ""
        )
        if not selected_path:
            return
        directory: Path = Path(selected_path)
        try:
            image_paths: list[Path] = find_image_paths(directory)
        except OSError as error:
            QMessageBox.warning(self, "Cannot read directory", str(error))
            return
        if not image_paths:
            self.statusBar().showMessage("No supported images found in this directory")
            return
        if self.load_source_image(image_paths[0]):
            self.input_image_paths = image_paths
            self.current_image_index = 0
            self.input_directory_label.setText(str(directory))
            self._update_navigation()

    def load_source_image(self, image_path: Path) -> bool:
        if not self.image_canvas.load_image(image_path):
            QMessageBox.warning(self, "Failed to open", f"Cannot read image: {image_path.name}")
            return False
        self.current_source_path = image_path.resolve()
        self.segmentation_mask = None
        self._clear_processed_preview()
        self.source_filename_label.setText(image_path.name)
        self.source_path_label.setText(str(self.current_source_path))
        self.filename_edit.setText(f"{image_path.stem}.png")
        self.class_label_edit.clear()
        self.statusBar().showMessage(f"Opened image: {image_path.name}")
        self.current_original_image = self.image_canvas.get_original_image()
        return True

    def load_previous_image(self) -> None:
        self._load_image_at_index(self.current_image_index - 1)

    def load_next_image(self) -> None:
        self._load_image_at_index(self.current_image_index + 1)

    def _load_image_at_index(self, index: int) -> None:
        if not 0 <= index < len(self.input_image_paths):
            return
        if self.load_source_image(self.input_image_paths[index]):
            self.current_image_index = index
            self._update_navigation()

    def _update_navigation(self) -> None:
        total: int = len(self.input_image_paths)
        self.progress_label.setText(
            f"{self.current_image_index + 1} / {total}" if total else "-- / --"
        )
        self.previous_button.setEnabled(0 < self.current_image_index < total)
        self.next_button.setEnabled(0 <= self.current_image_index < total - 1)

    def set_output_directory(self) -> None:
        selected_path: str = QFileDialog.getExistingDirectory(
            self, "Please select an output directory", ""
        )
        if selected_path:
            self.output_directory = Path(selected_path).resolve()
            self.output_directory_label.setText(str(self.output_directory))

    """ prompt points and SAM"""
    def generate_auto_prompts(self) -> None:
        image: Image.Image | None = self.image_canvas.get_original_image()
        if image is None:
            self.statusBar().showMessage("Please open an image first")
            return
        rgb: np.ndarray = np.array(image)
        groups: list[np.ndarray] = find_prompt_groups(rgb, self.prompt_rng)[1]
        self.image_canvas.clear_prompt_points()
        for points in groups:
            for x, y in points:
                self.image_canvas.add_prompt(QPointF(float(x), float(y)), label=1)
        self.statusBar().showMessage(
            f"Generated {len(self.image_canvas.prompt_items)} positive candidate prompts"
        )

    def segment_current_image(self) -> None:
        image: Image.Image | None = self.image_canvas.get_original_image()
        if image is None:
            self.statusBar().showMessage("Please open an image first")
            return
        point_coords, point_labels = self.image_canvas.get_prompts()
        box: tuple[int, int, int, int] | None = self.image_canvas.get_crop_box()
        if len(point_coords) == 0 and box is None:
            self.statusBar().showMessage("Please add points or a box")
            return
        try:
            if self.sam_predictor is None:
                self.sam_predictor = create_sam_predictor(device_name=self.sam_device)
            mask: np.ndarray = segment_image(
                predictor=self.sam_predictor,
                rgb=np.array(image),
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
            )
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Segmentation failed", str(error))
            return
        self.segmentation_mask = mask
        self.image_canvas.show_segmentation(mask, mask_to_box(mask))
        if not mask.any():
            self.statusBar().showMessage("No foreground detected")

    def update_preview(self) -> None:
        crop_box: tuple[int, int, int, int] | None = self.image_canvas.get_crop_box()
        image: Image.Image | None = self.image_canvas.get_original_image()
        if image is None:
            QMessageBox.warning(self, "Current image is null")
            self._clear_processed_preview()
            return
        
        if crop_box is None:
            self.processed_image = self.current_original_image
        else:
            try:
                output_size: tuple[int, int] = (self.output_width_spinbox.value(), self.output_height_spinbox.value())
                self.processed_image = prepare_processed_image(
                    image, crop_box, output_size, self.background_rng
                )
            except (OSError, ValueError) as error:
                self._clear_processed_preview()
                QMessageBox.warning(self, "Processing failed", str(error))
                return

        self.preview_pixmap = QPixmap.fromImage(ImageQt(self.processed_image))
        self._show_preview()

    def _clear_processed_preview(self) -> None:
        self.processed_image = None
        self.preview_pixmap = None
        self._show_preview()

    def _show_preview(self) -> None:
        if self.preview_pixmap is None:
            self.preview_label.clear()
            self.preview_label.setText("Select a character to preview")
            return
        self.preview_label.setPixmap(self.preview_pixmap.scaled(
            self.preview_label.contentsRect().size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.preview_label and event.type() == QEvent.Type.Resize:
            self._show_preview()
        return super().eventFilter(watched, event)

    

    def save_current_result(self) -> None:
        if (self.processed_image is None or self.current_source_path is None) and self.current_original_image is None:
            self.statusBar().showMessage("Please generate a processed preview first")
            return
        elif (self.processed_image is None) and (self.current_original_image is not None):
            self.processed_image = self.current_original_image  # 设置当前处理后的图片为原图
        if self.output_directory is None:
            self.statusBar().showMessage("Please select an output directory")
            return
        filename: str = self.filename_edit.text().strip()
        try:
            output_path: Path = save_processed_image(
                self.processed_image, self.output_directory, filename
            )
        except ValueError as error:
            self.statusBar().showMessage(str(error))
            return
        except OSError as error:
            QMessageBox.warning(self, "Save failed", str(error))
            return
        try:
            append_sample_metadata(
                self.output_directory, output_path, self.current_source_path,
                self.class_label_edit.text().strip() or None, self.processed_image.size,
            )
        except OSError as error:
            QMessageBox.warning(
                self, "Metadata save failed",
                f"Image saved to:\n{output_path}\n\nCould not record its metadata:\n{error}",
            )
            return
        self.filename_edit.setText(Path(filename).with_suffix(".png").name)
        self.statusBar().showMessage(f"Saved: {output_path.name}")


def main() -> None:
    app: QApplication = QApplication(sys.argv)
    window: DatasetProcessWindow = DatasetProcessWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
