import sys
from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt, fromqpixmap

from PySide6.QtCore import (
    Qt,
    QRectF
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

from image_canvas import ImageCanvas
from pretreatment import *


class DatasetProcessWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

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
        open_layout: QHBoxLayout = QHBoxLayout()
        open_button: QPushButton = QPushButton("Open image", right_widget)
        self.open_directory_button: QPushButton = QPushButton("Open directory", right_widget)
        self.open_output_directory_button: QPushButton = QPushButton("Open output directory", right_widget)
        open_layout.addWidget(open_button)
        open_layout.addWidget(self.open_directory_button)
        open_layout.addWidget(self.open_output_directory_button)
        right_layout.addLayout(open_layout)

        # operate image buttons
        tools_layout: QGridLayout = QGridLayout()
        fit_button: QPushButton = QPushButton("Fit image",right_widget,)
        delete_button: QPushButton = QPushButton("Delete selected",right_widget,)
        clear_points_button: QPushButton = QPushButton("Clear prompt points",right_widget,)
        clear_box_button: QPushButton = QPushButton("Clear box",right_widget,)
        tools_layout.addWidget(fit_button, 0, 0)
        tools_layout.addWidget(delete_button, 0, 1)
        tools_layout.addWidget(clear_points_button, 1, 0)
        tools_layout.addWidget(clear_box_button, 1, 1)
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
        self.next_button: QPushButton = QPushButton("Next image",right_widget,)
        right_layout.addWidget(self.progress_label)
        right_layout.addWidget(self.next_button)

        splitter.addWidget(right_widget)

        # splitter settings
        splitter.setSizes([820, 340])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)

        # connect signal for buttons
        open_button.clicked.connect(self.open_image)
        fit_button.clicked.connect(self.canvas.fit_image)
        delete_button.clicked.connect(self.canvas.delete_selected_items)
        clear_points_button.clicked.connect(self.canvas.clear_prompt_points)
        clear_box_button.clicked.connect(self.canvas.clear_box)
        self.update_preview_button.clicked.connect(self.update_preview)

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

        # load image into canvas
        if not self.canvas.load_image(image_path):
            QMessageBox.warning(
                self,
                "Failed to open",
                f"can not read image: {image_path.name}"
            )
            return

        # update widgets
        self.source_label.setText(image_path.name)
        self.filename_edit.setText(f"{image_path.stem}.png")
        self.label_edit.clear()

        self.statusBar().showMessage(f"Opened image: {image_path.name}")

    # update preview image
    def update_preview(self) -> None:
        crop_box: tuple[int, int, int, int] | None = self.canvas.get_corp_box()

        if crop_box is None:
            self.processed_image = None
            self.preview_pixmap = None
            self.show_preview()
            return

        original: Image.Image = fromqpixmap(self.canvas.image_item.pixmap()).convert("RGB")
        output_size: tuple[int, int] = (
            self.output_width.value(),
            self.output_height.value(),
        )

        self.processed_image = self.prepare_processed_image(original, crop_box, output_size)
        qt_image: QImage = ImageQt(self.processed_image)
        self.preview_pixmap = QPixmap.fromImage(qt_image)

        self.show_preview()


    def prepare_processed_image(
        self,
        image: Image.Image,
        crop_box: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> Image.Image:
        left, top, right, bottom = crop_box
        crop: Image.Image = image.convert("RGB").crop(crop_box)
        return crop

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

        


def main() -> None:
    app: QApplication = QApplication(sys.argv)

    window: DatasetProcessWindow = DatasetProcessWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


