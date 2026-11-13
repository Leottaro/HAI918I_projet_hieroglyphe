from pathlib import Path

from PySide6.QtCore import QPoint,Qt
from PySide6.QtGui import (
    QPixmap,
    QMouseEvent,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QScrollBar,
    QWidget,
)

class ImageCanvas(QGraphicsView):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self.auto_fit: bool = True  # auto fit status
        self.fit_scale: float = 1.0
        self.mouse_start_position:QPoint | None = None

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

        # set scale anchor as mouse position
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)


    def load_image(self, image_path: Path) -> bool:
        pixmap: QPixmap = QPixmap(str(image_path))

        if pixmap.isNull():
            return False

        self.image_item.setPixmap(pixmap)

        # set scene boundry (0,0) to image size (x, y)
        self.setSceneRect(self.image_item.boundingRect())

        self.fit_image()

        return True


    def fit_image(self) -> None:
        if self.image_item.pixmap().isNull():
            return
        
        self.resetTransform()   # clean old transformation
        self.fitInView(
            self.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        self.fit_scale = self.transform().m11()
        self.auto_fit = True


    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

        if self.auto_fit:
            self.fit_image()


    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.image_item.pixmap().isNull():
            event.ignore()
            return

        steps: float = event.angleDelta().y() / 120.0   # 120 is sensitivity for mouse

        if steps == 0:
            steps = event.pixelDelta().y() / 60.0   # 60 is sensitivity for trackpad

        if steps == 0:
            event.accept()
            return

        factor: float = 1.2 ** steps
        current_scale: float = self.transform().m11()

        # limite target_scale [fit_scale, fit_scale * 20]
        target_scale: float = max(
            self.fit_scale,
            min(current_scale * factor, self.fit_scale * 20.0),
        )

        self.auto_fit = False
        self.scale(
            target_scale / current_scale,
            target_scale / current_scale,
        )   # scale() take parameter is reference scale for current scale

        event.accept()
        return


    def mousePressEvent(self, event: QMouseEvent) -> None:

        # move: mouse middle button or (shift + left button)
        wants_move: bool = (
             event.button() == Qt.MouseButton.MiddleButton
             or (
                 event.button() == Qt.MouseButton.LeftButton
                 and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
             )
        )

        if wants_move and not self.image_item.pixmap().isNull():
            # record mouse start move position
            self.mouse_start_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        
        super().mousePressEvent(event)


    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mouse_start_position is not None:
            position: QPoint = event.position().toPoint()
            delta: QPoint = position - self.mouse_start_position
            self.mouse_start_position = position    # update new mouse start position

            horizontal_bar: QScrollBar = self.horizontalScrollBar()
            vertical_bar: QScrollBar = self.verticalScrollBar()

            # move image view by changing scroll bar
            horizontal_bar.setValue(
                horizontal_bar.value() - delta.x()
            )
            vertical_bar.setValue(
                vertical_bar.value() - delta.y()
            )

            event.accept()
            return

        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.mouse_start_position is not None and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self.mouse_start_position = None
            self.unsetCursor()
            event.accept()
            return

        super().mouseReleaseEvent(event)






