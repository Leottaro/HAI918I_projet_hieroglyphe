import numpy as np

from pathlib import Path
from math import ceil, floor

from PySide6.QtCore import (
    Qt,
    QPoint,
    QPointF,
    QRectF,
    Signal,
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
    QGraphicsRectItem,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
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


class SelectionBox(QGraphicsRectItem):
    def __init__(self, bounds: QRectF) -> None:
        super().__init__()

        self.bounds: QRectF = bounds    # Rectangle boundry on image original

        pen: QPen = QPen(QColor("#ffd54f"), 6)
        pen.setCosmetic(True)   # keep boundry width
        self.setPen(pen)
        self.setZValue(5)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)   # when position changed, call itemChange function

    # when position changed, we want to limite the selection box in the boundry of image
    def itemChange(
            self,
            change: QGraphicsItem.GraphicsItemChange,
            value: object,
    ) -> object:
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and isinstance(value, QPointF)
        ):
            rect: QRectF = self.rect()

            # X and y is the origianl point of rect
            x: float = max(
                self.bounds.left() - rect.left(),
                min(value.x(), self.bounds.right() - rect.right()),
            )
            y: float = max(
                self.bounds.top() - rect.top(),
                min(value.y(), self.bounds.bottom() - rect.bottom()),
            )

            return QPointF(x, y)    # return new position of rect
        
        return super().itemChange(change, value)


class ImageCanvas(QGraphicsView):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.prompt_items: list[PromptPoint] = []   # we save prompts points list for SAM
        self.box_item: SelectionBox | None = None   # current box item selected

        self.draft_box: SelectionBox | None = None  # temproray box when we draw selection box
        self.box_start: QPointF | None = None   # when draw selection box, we record box start position
        self.mouse_start_position: QPoint | None = None # when we want to move canvas
        self.dragged_item: PromptPoint | SelectionBox | None = None # item that we can drag
        self.drag_offset: QPointF = QPointF()

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
        self.clear_prompt_points()    # clean old prompt points
        self.clear_draft_box()
        self.clear_box()

        self.mouse_start_position = None
        self.dragged_item = None
        self.unsetCursor()

        # load new image
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
        if self.image_item.pixmap().isNull():
            event.accept()
            return

        self.setFocus(Qt.FocusReason.MouseFocusReason)

        view_position: QPoint = event.position().toPoint()  # mouse position on screen
        position: QPointF = self.mapToScene(view_position)  # mouse position in scene

        # move canvas: middle button or (ctrl + left button)
        wants_move: bool = (
             event.button() == Qt.MouseButton.MiddleButton
             or (
                 event.button() == Qt.MouseButton.LeftButton
                 and bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
             )
        )

        if wants_move:
            # record mouse start move position
            self.mouse_start_position = view_position
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:

             #  draw selection box: shift + left button
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                if self.image_item.boundingRect().contains(position):
                    self.clear_draft_box()
                    self.box_start = position

                    self.draft_box = SelectionBox(
                        self.image_item.boundingRect()
                    )
                    self.draft_box.setRect(
                        QRectF(position, position)
                    )
                    self.draft_box.setZValue(6)
                    self.graphics_scene.addItem(self.draft_box)

                event.accept()
                return
            
             # try to select an item (point or box)
            item: PromptPoint | SelectionBox | None = (self.find_editable_item(view_position))
            if item is not None:
                self.graphics_scene.clearSelection()
                item.setSelected(True)

                self.dragged_item = item
                self.drag_offset = position - item.pos()

            # else add a new positive prompt point
            else:
                self.add_prompt(position, label=1)

            event.accept()
            return

        # add new negative point
        if event.button() == Qt.MouseButton.RightButton:
            self.add_prompt(position, label=0)
            event.accept()
            return

        event.accept()


    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # move canvas
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

        # move selection box
        if self.box_start is not None and self.draft_box is not None:
            position: QPointF = self.mapToScene(
                event.position().toPoint()
            )
            bounds: QRectF = self.image_item.boundingRect()

            position.setX(
                max(bounds.left(), min(position.x(), bounds.right()))
            )
            position.setY(
                max(bounds.top(), min(position.y(), bounds.bottom()))
            )

            self.draft_box.setRect(
                QRectF(self.box_start, position).normalized()
            )

            event.accept()
            return

        # move prompt point item
        if self.dragged_item is not None:
            position: QPointF = self.mapToScene(
                event.position().toPoint()
            )

            self.dragged_item.setPos(
                position - self.drag_offset
            )

            event.accept()
            return

        super().mouseMoveEvent(event)



    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # finish move canvas 
        if self.mouse_start_position is not None and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self.mouse_start_position = None
            self.unsetCursor()
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            event.accept()
            return

        # finsh to move selection box
        if self.draft_box is not None:
            rect: QRectF = self.draft_box.rect()

            if rect.width() >= 1 and rect.height() >= 1:
                self.clear_box()
                self.box_item = self.draft_box
                self.draft_box = None

                self.box_item.setZValue(5)
                self.graphics_scene.clearSelection()
                self.box_item.setSelected(True)

            self.clear_draft_box()
            event.accept()
            return

        # finish move prompt point item
        self.dragged_item = None
        event.accept()


    def delete_selected_prompts(self) -> None:
        for point in self.prompt_items.copy():
            if point.isSelected():
                self.graphics_scene.removeItem(point)
                self.prompt_items.remove(point)

        if self.prompt_items:   # select last prompt point
            last_point: PromptPoint = self.prompt_items[-1]
            last_point.setSelected(True)
        
    
    def clear_prompt_points(self) -> None:
        # delete from scene first
        for point in self.prompt_items:
            self.graphics_scene.removeItem(point)

        # clean list after
        self.prompt_items.clear()

    def clear_draft_box(self) -> None:
        self.box_start = None

        if self.draft_box is not None:
            self.graphics_scene.removeItem(self.draft_box)
            self.draft_box = None

    def clear_box(self) -> None:
        if self.box_item is not None:
            if self.dragged_item is self.box_item:
                self.dragged_item = None

            self.graphics_scene.removeItem(self.box_item)
            self.box_item = None

    def delete_selected_items(self) -> None:
        if self.box_item is not None and self.box_item.isSelected():
            self.clear_box()
        else:
            self.delete_selected_prompts()


    def keyPressEvent(self, event: QKeyEvent) -> None:
        # delete selected item when press Delete or Backspace
        if event.key() in (
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        ):
            self.delete_selected_items()
            event.accept()
            return

        super().keyPressEvent(event)


    def add_prompt(self, position: QPointF, label: int) -> None:
        bounds: QRectF = self.image_item.boundingRect()

        # can not add a prompt points out of image
        if not (
            bounds.left() <= position.x() < bounds.right()
            and bounds.top() <= position.y() < bounds.bottom()
        ):
            return

        point: PromptPoint = PromptPoint(position, label, bounds)

        self.graphics_scene.clearSelection()
        self.graphics_scene.addItem(point)
        point.setSelected(True)  # select directly this point after added
        self.prompt_items.append(point)


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


    def find_editable_item(
        self,
        view_position: QPoint,
    ) -> PromptPoint | SelectionBox | None:
        # check points first
        for item in self.items(view_position):
            if isinstance(item, PromptPoint):
                return item

        if self.box_item is None:
            return None

        # change rect local position to scene coordinate
        scene_rect: QRectF = self.box_item.mapRectToScene(self.box_item.rect())

        # chage rect scene position to screen coordinate
        screen_rect: QRectF = self.viewportTransform().mapRect(scene_rect)

        outer: QRectF = screen_rect.adjusted(-5, -5, 5, 5)
        inner: QRectF = screen_rect.adjusted(5, 5, -5, -5)
        cursor: QPointF = QPointF(view_position)    # mouse position en screen

        inside_inner: bool = (inner.isValid() and inner.contains(cursor))

        if outer.contains(cursor) and not inside_inner:
            return self.box_item

        return None


    # return box_item boundry
    def get_corp_box(self) -> tuple[int, int, int, int] | None:
        if self.box_item is None:
            return None
        # get box item boundary
        rect: QRectF = self.box_item.mapRectToScene(
            self.box_item.rect()
        ).intersected(self.image_item.boundingRect())

        if rect.isEmpty():
            return None

        # return boundary [left, top, right, bottom]
        return (
            floor(rect.left()),
            floor(rect.top()),
            ceil(rect.right()),
            ceil(rect.bottom()),
        )

