import numpy as np

from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QPoint,
    QPointF,
    QRectF,
)

from PySide6.QtGui import (
    QPen,
    QColor,
    QPixmap,
    QKeyEvent,
    QMouseEvent,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QWidget,
    QScrollBar,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
)

class PromptPoint(QGraphicsEllipseItem):
    def __init__(
        self,
        position: QPointF,
        label: int,
        bounds: QRectF,
    ) -> None:
        # initialize an ellipse, left top position (-5, -5), diameter (10, 10), center (0,0)
        super().__init__(-5,-5,10,10)

        self.label = label
        self.bounds = bounds

        # set color according labels
        color: QColor =  QColor("#24b266") if label == 1 else QColor("#e34b57")

        # set point styles
        self.setBrush(color)
        self.setPen(QPen(QColor("white"), 1.5)) # white edge
        self.setZValue(10)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations) # keep display size fix
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)   # when position changed, emit signal

        self.setPos(position)


    # when point position changed, return new position
    def itemChange( 
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and isinstance(value, QPointF)
        ):
            # clip x into bounds
            x: float = max(
                self.bounds.left(),
                min(value.x(), self.bounds.right() - 1),
            )

            # clip y into bounds
            y: float = max(
                self.bounds.top(),
                min(value.y(), self.bounds.bottom() - 1),
            )

            return QPointF(x, y)
        
        return super().itemChange(change, value)






class ImageCanvas(QGraphicsView):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.prompt_items: list[PromptPoint] = []   # we save prompts points list for SAM
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # set canvas can recept keyboard input
        
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
        self.clear_prompts()    # clean old prompt points

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

        # when move canvas: mouse middle button or (shift + left button)
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

        # when operate prompts point (add, select)
        if self.image_item.pixmap().isNull():
            super().mousePressEvent(event)
            return

        clicked_item: QGraphicsItem | None = self.itemAt(
            event.position().toPoint()
        )   # get clicked item

        if isinstance(clicked_item, PromptPoint):   # if we click a prompt point
            super().mousePressEvent(event)  # left qt handle select and move point action
            return

        # we only handle left or right click to operate points
        if event.button() in (
            Qt.MouseButton.LeftButton,  # left click to add positive prompt point (label = 1)
            Qt.MouseButton.RightButton, # right click to add negative prompt point  (label = 0)
        ):
            position: QPointF = self.mapToScene(event.position().toPoint()) # position on image, because image is at (0,0) in the scene
            if self.image_item.boundingRect().contains(position):
                label: int = (1 if event.button() == Qt.MouseButton.LeftButton else 0)

            point: PromptPoint = PromptPoint(
                position,
                label,
                self.image_item.boundingRect(),
            )

            self.graphics_scene.clearSelection()
            self.graphics_scene.addItem(point)
            point.setSelected(True)    # select the point that we just added
            self.prompt_items.append(point) # add this point into prompt point list
            self.setFocus(Qt.FocusReason.MouseFocusReason)

            event.accept()
            return

        # left super handle event by defult
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


    def delete_selected_prompts(self) -> None:
        for point in self.prompt_items.copy():
            if point.isSelected():
                self.graphics_scene.removeItem(point)
                self.prompt_items.remove(point)

    
    def clear_prompts(self) -> None:
        # delete from scene first
        for point in self.prompt_items:
            self.graphics_scene.removeItem(point)

        # clean list after
        self.prompt_items.clear()


    def keyPressEvent(self, event: QKeyEvent) -> None:
        # delete prompt point selected when press Delete or Backspace
        if event.key() in (
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        ):
            self.delete_selected_prompts()
            event.accept()
            return

        super().keyPressEvent(event)


    # return all prompt point saved in list for SAM
    def get_prompts(self) -> tuple[np.ndarray, np.ndarray]:
        coordinates_list: list[tuple[float, float]] = []
        labels_list: list[int] = []

        # extract positions and lables into list
        for point in self.prompt_items:
            position: QPointF = point.pos()
            coordinates_list.append(
                (position.x(), position.y())
            )
            labels_list.append(point.label)

        # transform list data to np array
        coordinates: np.ndarray = np.array(
            coordinates_list,
            dtype=np.float32,
        ).reshape(-1, 2)    # 2 collumns, compute automatically arrays

        labels: np.ndarray = np.array(
            labels_list,
            dtype=np.int32,
        )

        return coordinates, labels




