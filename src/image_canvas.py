from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

class ImageCanvas(QGraphicsView):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # set secne
        self.graphics_scene: QGraphicsScene = QGraphicsScene(self)
        self.setScene(self.graphics_scene)

        # set image item
        self.image_item: QGraphicsPixmapItem = self.graphics_scene.addPixmap(QPixmap())

        # set transformation mode 
        self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)

        # set scroll mode
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


    def load_image(self, image_path: Path) -> bool:
        pixmap: QPixmap = QPixmap(str(image_path))

        if pixmap.isNull():
            return False

        self.image_item.setPixmap(pixmap)

        # set scene boundry (0,0) to image size (x, y)
        self.setSceneRect(self.image_item.boundingRect())

        # show all scene rect in view
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        return True


    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

        if not self.image_item.pixmap().isNull():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

