import sys
from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt, fromqpixmap
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from image_canvas import ImageCanvas
from pretreatment import *
from config import SEED

from PySide6.QtCore import (
    Qt,
    QRectF,
    QPointF
)

from PySide6.QtGui import (
    QPixmap,
    QImage,
)

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
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class DatasetProcessWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.sam_predictor: SAM2ImagePredictor | None = create_sam_predictor(device_name="cpu")  # initialize sam predictor when open window
        self.prompt_rng: np.random.Generator = np.random.default_rng(SEED)
        self.mask: np.ndarray | None = None

        self.image_paths: list[Path] = []
        self.current_index: int = -1

        self.output_directory: Path | None = None
        self.current_image_path: Path | None = None

        self.processed_image: Image.Image | None = None
        self.preview_pixmap: QPixmap | None = None

        self.setWindowTitle("Dataset process")
        self.resize(1200, 820)

        # -------- main center widget ------
        # set a child widget as central widget
        central_widget: QWidget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout: QHBoxLayout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal, central_widget)
        main_layout.addWidget(splitter)

        # --------left part--------
        left_widget: QWidget = QWidget()    # left part widget
        left_layout: QVBoxLayout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.source_label: QLabel = QLabel("No image selected",left_widget,)    # show original image label name
        self.source_label.setWordWrap(True)
        left_layout.addWidget(self.source_label)

        self.canvas: ImageCanvas = ImageCanvas(left_widget) # initial ImageCanvas, left widget as father
        left_layout.addWidget(self.canvas, stretch=1)

        hint_label: QLabel = QLabel("hint", left_widget)    # how to use this util
        hint_label.setWordWrap(True)
        left_layout.addWidget(hint_label)

        splitter.addWidget(left_widget)

        # ---------right part--------
        right_widget = QWidget(central_widget)
        right_widget.setMinimumWidth(320)

        # set right widget vertical layout and style
        right_layout: QVBoxLayout= QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        # oepn image or directory
        open_layout: QVBoxLayout = QVBoxLayout()

        open_button: QPushButton = QPushButton("Open image", right_widget)
        self.image_path_label = QLabel("image path:", right_widget)
        self.image_path_label.setWordWrap(True)

        self.open_directory_button: QPushButton = QPushButton("Open directory", right_widget)
        self.open_directory_path_label = QLabel("opened directory path:", right_widget)
        self.open_directory_path_label.setWordWrap(True)
        
        self.set_output_directory_button: QPushButton = QPushButton("Set output directory", right_widget)
        self.output_directory_path_label = QLabel("output directory path:", right_widget)
        self.output_directory_path_label.setWordWrap(True)
        
        open_layout.addWidget(open_button)
        open_layout.addWidget(self.image_path_label)
        open_layout.addWidget(self.open_directory_button)
        open_layout.addWidget(self.open_directory_path_label)
        open_layout.addWidget(self.set_output_directory_button)
        open_layout.addWidget(self.output_directory_path_label)
        
        right_layout.addLayout(open_layout)

        # operate image buttons
        tools_layout: QGridLayout = QGridLayout()
        fit_button: QPushButton = QPushButton("Fit image",right_widget,)
        delete_button: QPushButton = QPushButton("Delete selected",right_widget,)
        clear_points_button: QPushButton = QPushButton("Clear prompt points",right_widget,)
        clear_box_button: QPushButton = QPushButton("Clear box",right_widget,)
        self.auto_prompts_button: QPushButton = QPushButton("Auto prompts",right_widget,)
        self.auto_segment_button: QPushButton = QPushButton("Auto segement",right_widget,)
        tools_layout.addWidget(fit_button, 0, 0)
        tools_layout.addWidget(delete_button, 0, 1)
        tools_layout.addWidget(clear_points_button, 1, 0)
        tools_layout.addWidget(clear_box_button, 1, 1)
        tools_layout.addWidget(self.auto_prompts_button, 2, 0)
        tools_layout.addWidget(self.auto_segment_button, 2, 1)
        right_layout.addLayout(tools_layout)

        # output setting
        right_layout.addWidget(QLabel("Output size", right_widget))
        self.output_width: QSpinBox = QSpinBox(right_widget)
        self.output_height: QSpinBox = QSpinBox(right_widget)
        for spin_box in (self.output_width, self.output_height):
            spin_box.setRange(32, 2048)
            spin_box.setValue(160)
            spin_box.setSuffix(" px")
            spin_box.setMinimumHeight(30)

        size_layout: QFormLayout = QFormLayout()
        size_layout.addRow("Width", self.output_width)
        size_layout.addRow("Height", self.output_height)
        right_layout.addLayout(size_layout)

        # preview part
        right_layout.addWidget(QLabel("Processed preview", right_widget))
        self.preview_label: QLabel = QLabel("result preview",right_widget,)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_label.setMinimumSize(160, 160)
        self.preview_label.setMaximumHeight(220)
        right_layout.addWidget(self.preview_label)
        self.update_preview_button: QPushButton = QPushButton("Update preview", right_widget,)
        right_layout.addWidget(self.update_preview_button)

        # label and filename setting
        self.label_edit: QLineEdit = QLineEdit(right_widget)
        self.label_edit.setPlaceholderText("class1 / class2 / class3")
        self.label_edit.setMinimumWidth(240)
        self.label_edit.setFixedHeight(35)
        self.filename_edit: QLineEdit = QLineEdit(right_widget)
        self.filename_edit.setPlaceholderText("output.png")
        self.filename_edit.setMinimumWidth(240)
        self.filename_edit.setFixedHeight(35)
        metadata_layout: QFormLayout = QFormLayout()
        metadata_layout.addRow("Class label", self.label_edit)
        metadata_layout.addRow("Filename", self.filename_edit)
        right_layout.addLayout(metadata_layout)

        # save result button
        self.save_button: QPushButton = QPushButton("Save character",right_widget,)
        right_layout.addWidget(self.save_button)

        right_layout.addStretch(1)
        self.progress_label: QLabel = QLabel("-- / --",right_widget,)
        self.previous_button: QPushButton = QPushButton("Previous image", right_widget)
        self.next_button: QPushButton = QPushButton("Next image",right_widget,)
        # 尚未加载图片时，两个导航按钮都不可用。
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)

        # 上一张与下一张并排显示，进度放在它们上方。
        navigation_layout: QHBoxLayout = QHBoxLayout()
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.next_button)
        right_layout.addWidget(self.progress_label)
        right_layout.addLayout(navigation_layout)

        splitter.addWidget(right_widget)

        # splitter settings
        splitter.setSizes([820, 340])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)

        # connect signal for buttons
        open_button.clicked.connect(self.open_image)
        self.open_directory_button.clicked.connect(self.open_directory)
        self.set_output_directory_button.clicked.connect(self.set_output_directory)
        fit_button.clicked.connect(self.canvas.fit_image)
        delete_button.clicked.connect(self.canvas.delete_selected_items)
        clear_points_button.clicked.connect(self.canvas.clear_prompt_points)
        clear_box_button.clicked.connect(self.canvas.clear_box)
        self.update_preview_button.clicked.connect(self.update_preview)
        self.auto_prompts_button.clicked.connect(self.generate_auto_prompts)
        self.auto_segment_button.clicked.connect(self.segment_current_image)
        self.previous_button.clicked.connect(self.get_previous_image)
        self.next_button.clicked.connect(self.get_next_image)
        self.save_button.clicked.connect(self.save_current_result)

        self.statusBar().showMessage("please select a image")


    def open_image(self) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Please select image",
            "",
            "image (*.png *.jpg *.jpeg *.bmp);;all (*)",
        )

        if not selected_path:
            return

        image_path: Path = Path(selected_path)

        if not self.load_source_image(image_path):
            return

        self.image_paths = [image_path]
        self.current_index = 0

        self.open_directory_path_label.setText(
            "No input directory selected"
        )
        self.update_navigation()


    def load_source_image(self, image_path: Path) -> bool:
        if not self.canvas.load_image(image_path):
            return False
        
        self.canvas.clear_mask()

        self.mask = None
        self.processed_image = None
        self.preview_pixmap = None
        self.show_preview()

        self.current_image_path = image_path.resolve()
        self.source_label.setText(image_path.name)
        self.image_path_label.setText(f"{image_path.stem}.png")
        self.filename_edit.setText(f"{image_path.stem}.png")
        self.label_edit.clear()

        self.statusBar().showMessage(f"Opened image: {image_path.name}")
        return True

    def open_directory(self) -> None:
        
        selected_path: str = QFileDialog.getExistingDirectory(
            self,
            "Please select an input directory",
            "",
        )

        if not selected_path:
            return

        directory: Path = Path(selected_path)
        supported_suffixes: set[str] = {
            ".png", ".jpg", ".jpeg", ".bmp",
        }
        image_paths: list[Path] = []
        for path in directory.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in supported_suffixes
            ):
                image_paths.append(path)
        image_paths.sort(key=lambda path: path.name.casefold())

        if not image_paths:
            return
        
        if not self.load_source_image(image_paths[0]):
            return

        self.image_paths = image_paths
        self.current_index = 0

        self.open_directory_path_label.setText(str(directory))
        self.update_navigation()


    def set_output_directory(self) -> None:
        selected_path: str = QFileDialog.getExistingDirectory(
            self,
            "Please select an output directory",
            "",
        )

        if not selected_path:
            return

        self.output_directory = Path(selected_path).resolve()
        self.output_directory_path_label.setText(self.output_directory) 


    def get_previous_image(self) -> None:
        # 没有图片或已经位于第一张时，不允许使用负索引回到末尾。
        if not self.image_paths or self.current_index <= 0:
            return

        previous_index: int = self.current_index - 1
        previous_path: Path = self.image_paths[previous_index]

        # 复用现有加载流程，成功后才更新索引和导航按钮。
        if not self.load_source_image(previous_path):
            return

        self.current_index = previous_index
        self.update_navigation()


    def get_next_image(self) -> None:
        if not self.image_paths:
            return
        next_index: int = self.current_index + 1
        if next_index >= len(self.image_paths):
            self.statusBar().showMessage("Already at the last image")
            return

        next_path: Path = self.image_paths[next_index]
        if not self.load_source_image(next_path):
            return
        
        self.current_index = next_index
        self.update_navigation()


    def update_navigation(self) -> None:
        total: int = len(self.image_paths)

        self.progress_label.setText(
            f"{self.current_index + 1} / {total}"
        )

        # 第一张没有上一张；仅有一张图片时同样禁用。
        has_previous: bool = 0 < self.current_index < total
        self.previous_button.setEnabled(has_previous)

        has_next: bool = (
            0 <= self.current_index < total - 1
        )

        self.next_button.setEnabled(has_next)


    # update preview image
    def update_preview(self) -> None:
        crop_box: tuple[int, int, int, int] | None = self.canvas.get_corp_box()

        if crop_box is None:
            self.processed_image = None
            self.preview_pixmap = None
            self.show_preview()
            return

        original: Image.Image = fromqpixmap(self.canvas.image_item.pixmap()).convert("RGB")

        self.processed_image = self.prepare_processed_image(original)
        qt_image: QImage = ImageQt(self.processed_image)
        self.preview_pixmap = QPixmap.fromImage(qt_image)

        self.show_preview()

    # show preview label content
    def show_preview(self) -> None:
        if self.preview_pixmap is None:
            self.preview_label.clear()
            self.preview_label.setText(
                "Select a character to preview"
            )
            return

        self.preview_label.setPixmap(
            self.preview_pixmap.scaled(
                self.preview_label.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


    # process image automatically by prompts points and selection box
    def prepare_processed_image(
            self,
            image: Image.Image,
        ) -> Image.Image:

            target_width, target_height = (
                self.output_width.value(),
                self.output_height.value(),
            )

            mask_box: tuple[int, int, int, int] | None = self.canvas.get_corp_box()
            crop: Image.Image = image.convert("RGB").crop(mask_box)

            scale: float = min(
                target_width / crop.width,
                target_height / crop.height,
            )

            resized_width: int = min(
                target_width,
                max(1, round(crop.width * scale)),
            )
            resized_height: int = min(
                target_height,
                max(1, round(crop.height * scale)),
            )

            resized_image: Image.Image = crop.resize(
                (resized_width, resized_height),
                resample=Image.Resampling.LANCZOS,
            )

            # get statistic information
            pixels: np.ndarray = np.array(crop)

            border_pixels: np.ndarray = np.concatenate(
                (
                    pixels[0, :, :],     # 上边
                    pixels[-1, :, :],    # 下边
                    pixels[:, 0, :],     # 左边
                    pixels[:, -1, :],    # 右边
                ),
                axis=0,
            )

            background_color: np.ndarray = np.median(
                border_pixels,
                axis=0,
            )

            # generate gaussian background
            noise: np.ndarray = self.prompt_rng.normal(
                loc=0.0,
                scale=6.0,
                size=(target_height, target_width, 1),
            )

            background_pixels: np.ndarray = np.clip(
                background_color + noise,
                0,
                255,
            ).astype(np.uint8)

            result: Image.Image = Image.fromarray(background_pixels)

            offset_x: int = (target_width - resized_width) // 2
            offset_y: int = (target_height - resized_height) // 2

            result.paste(resized_image, (offset_x, offset_y))

            return result


    # generate prompt points automatically
    def generate_auto_prompts(self) -> None:
        pixmap: QPixmap = self.canvas.image_item.pixmap()
        if pixmap.isNull():
            self.statusBar().showMessage("Please open an image first")
            return
        rgb_image: Image.Image = fromqpixmap(pixmap).convert("RGB") # convert to pillow image
        rgb: np.ndarray = np.array(rgb_image)   # convert to np array

        groups: list[np.ndarray] = find_prompt_groups(rgb,self.prompt_rng,)[1]    # get first group

        self.canvas.clear_prompt_points()

        for points in groups:
            for x, y in points:
                position: QPointF = QPointF(float(x), float(y))
                self.canvas.add_prompt(position, label=1)

        count: int = len(self.canvas.prompt_items)
        self.statusBar().showMessage(
            f"Generated {count} positive candidate prompts"
        )


    # segement current image and get mask as return
    def segment_current_image(self) -> None:
        self.mask = None
        pixmap: QPixmap = self.canvas.image_item.pixmap()
        if pixmap.isNull():
            self.statusBar().showMessage("Please open an image first")
            return None
        image: Image.Image = fromqpixmap(pixmap).convert("RGB")
        rgb: np.ndarray = np.array(image)   # convert to np array

        point_coords: np.ndarray
        point_labels: np.ndarray
        point_coords, point_labels = self.canvas.get_prompts()
        box: tuple[int, int, int, int] | None = (self.canvas.get_corp_box())

        if len(point_coords) == 0 and box is None:
            self.statusBar().showMessage("Please add points or a box")
            return None
        
        self.mask: np.ndarray = segment_image(
            predictor=self.sam_predictor,
            rgb=rgb,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
        )

        mask_box: tuple[int, int, int, int] | None = mask_to_box(self.mask)
        self.canvas.show_segmentation(self.mask, mask_box)   # update mask area

    def save_current_result(self) -> None:
        # no image processed
        if (
            self.processed_image is None
            or self.current_image_path is None
        ):
            self.statusBar().showMessage(
                "Please generate a processed preview first"
            )
            return

        # no output directory selected
        if self.output_directory is None:
            self.statusBar().showMessage(
                "Please select an output directory"
            )
            return

        filename: str = self.filename_edit.text().strip()

        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
        ):
            self.statusBar().showMessage(
                "Please enter a valid filename without a directory"
            )
            return

        base_path: Path = (
            self.output_directory / filename
        ).with_suffix(".png")
        output_path: Path = base_path
        number: int = 1

        # 先检查候选路径，存在重名时依次尝试 _001、_002 等编号。
        # 始终基于原始名称生成，避免编号不断嵌套。
        while output_path.exists():
            output_path = base_path.with_name(
                f"{base_path.stem}_{number:03d}{base_path.suffix}"
            )
            number += 1

        try:
            # 保留独占创建，确保实际写入时也不会覆盖已有文件。
            with output_path.open("xb") as file:
                self.processed_image.save(file, format="PNG")
        except FileExistsError:
            self.statusBar().showMessage(
                f"Filename already exists: {output_path.name}"
            )
            return
        except OSError as error:
            QMessageBox.warning(
                self,
                "Save failed",
                str(error),
            )
            return

        # record relations with original dataset
        record: dict[str, object] = {
            "output_file": output_path.name,
            "source_file": self.current_image_path.name,
            "label": self.label_edit.text().strip() or None,
            "image_size": list(self.processed_image.size),
        }

        metadata_path: Path = (
            self.output_directory / "metadata.jsonl"
        )

        try:
             with metadata_path.open("a", encoding="utf-8") as file:
                 file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
        except OSError as error:
            QMessageBox.warning(
                self,
                "Metadata save failed",
                f"Image saved to:\n{output_path}\n\n"
                f"Could not record its metadata:\n{error}",
            )
            return
        # 连续保存时保留基础名称；状态栏和元数据使用实际保存名称。
        self.filename_edit.setText(base_path.name)
        self.statusBar().showMessage(
            f"Saved: {output_path.name}"
        )



def main() -> None:
    app: QApplication = QApplication(sys.argv)

    window: DatasetProcessWindow = DatasetProcessWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


