import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from image_canvas import ImageCanvas


class DatasetProcessWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Dataset process")
        self.resize(1100, 750)

        # set a child widget as central widget
        central_widget: QWidget = QWidget(self)
        self.setCentralWidget(central_widget)

        # set central widget vertical layout 
        layout: QVBoxLayout= QVBoxLayout(central_widget)

        # initial open button, central widget as father
        open_button: QPushButton = QPushButton("Open image", central_widget)

        # initial ImageCanvas, central widget as father
        self.canvas: ImageCanvas = ImageCanvas(central_widget)

        # add open button and ImageCanvas into layout
        layout.addWidget(open_button)
        layout.addWidget(self.canvas, stretch=1)    # strech=1: it will take all the rest place

        # connect signal for open button
        open_button.clicked.connect(self.open_image)

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

        self.statusBar().showMessage(f"Opened image: {image_path.name}")
        


def main() -> None:
    app: QApplication = QApplication(sys.argv)

    window: DatasetProcessWindow = DatasetProcessWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


