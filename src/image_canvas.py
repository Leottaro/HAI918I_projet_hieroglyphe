from math import ceil, floor
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt, fromqpixmap
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPen, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)


"""a guide point for SAM, it could be positive or negative"""
class PromptPoint(QGraphicsEllipseItem):

    def __init__(self, position: QPointF, label: int, bounds: QRectF) -> None:
        super().__init__(-5, -5, 10, 10)
        self.label: int = label
        self.bounds: QRectF = bounds
        color: QColor = QColor("#24b266" if label == 1 else "#e34b57")
        self.setBrush(color)
        self.setPen(QPen(QColor("white"), 1.5))
        self.setZValue(10)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(position)

    # limite its position on canvas
    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and isinstance(value, QPointF)
        ):
            return QPointF(
                max(self.bounds.left(), min(value.x(), self.bounds.right() - 1)),
                max(self.bounds.top(), min(value.y(), self.bounds.bottom() - 1)),
            )
        return super().itemChange(change, value)


""" selection box moveable for SAM or cut image"""
class SelectionBox(QGraphicsRectItem):

    def __init__(self, bounds: QRectF) -> None:
        super().__init__()
        self.bounds: QRectF = bounds
        pen: QPen = QPen(QColor("#ffd54f"), 6)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(5)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    # limite box position and range
    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and isinstance(value, QPointF)
        ):
            rect: QRectF = self.rect()
            return QPointF(
                max(
                    self.bounds.left() - rect.left(),
                    min(value.x(), self.bounds.right() - rect.right()),
                ),
                max(
                    self.bounds.top() - rect.top(),
                    min(value.y(), self.bounds.bottom() - rect.bottom()),
                ),
            )
        return super().itemChange(change, value)


""" canvas where we can manipulate prompt points, selection box on image"""
class ImageCanvas(QGraphicsView):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.prompt_items: list[PromptPoint] = []   # a list to stock all prompt points
        self.box_item: SelectionBox | None = None   # only one selection box can be create

        # mouse operation state
        self.draft_box_item: SelectionBox | None = None
        self.box_start_scene_position: QPointF | None = None
        self.pan_last_view_position: QPoint | None = None    # mouse position in view when move the canvas
        self.dragged_item: PromptPoint | SelectionBox | None = None
        self.drag_offset_scene: QPointF = QPointF()
        self.auto_fit_enabled: bool = True
        self.fit_scale: float = 1.0

        self.graphics_scene: QGraphicsScene = QGraphicsScene(self)
        self.setScene(self.graphics_scene)
        # image shown currently
        self.image_item: QGraphicsPixmapItem = self.graphics_scene.addPixmap(QPixmap())
        self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # mask segement by SAM overlay on image
        self.mask_overlay_item: QGraphicsPixmapItem = QGraphicsPixmapItem(self.image_item)
        self.mask_overlay_item.setZValue(1)
        self.mask_overlay_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    

    """load a new image to show, clean old image"""
    def load_image(self, image_path: Path) -> bool:
        pixmap: QPixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return False
        pixmap.setDevicePixelRatio(1.0)
        self.clear_prompt_points()
        self.clear_draft_box()
        self.clear_box()
        self.pan_last_view_position = None
        self.dragged_item = None
        self.unsetCursor()
        self.image_item.setPixmap(pixmap)
        self.setSceneRect(self.image_item.boundingRect())
        self.fit_image()
        return True

    """ return image not processed in format Python Imaging"""
    def get_original_image(self) -> Image.Image | None:
        pixmap: QPixmap = self.image_item.pixmap()
        if pixmap.isNull():
            return None
        return fromqpixmap(pixmap).convert("RGB")

    """ fit image into window"""
    def fit_image(self) -> None:
        if self.image_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.fit_scale = self.transform().m11()
        self.auto_fit_enabled = True

    """ add a new prompt point and draw it """
    def add_prompt(self, position: QPointF, label: int) -> None:
        bounds: QRectF = self.image_item.boundingRect()
        if not (
            bounds.left() <= position.x() < bounds.right()
            and bounds.top() <= position.y() < bounds.bottom()
        ):
            return
        point: PromptPoint = PromptPoint(position, label, bounds)
        self.graphics_scene.clearSelection()
        self.graphics_scene.addItem(point)
        point.setSelected(True)
        self.prompt_items.append(point)

    """ retutn all prompt points position and lables for SAM"""
    def get_prompts(self) -> tuple[np.ndarray, np.ndarray]:
        coordinates: list[tuple[float, float]] = []
        labels: list[int] = []
        for point in self.prompt_items:
            position: QPointF = point.scenePos()
            coordinates.append((position.x(), position.y()))
            labels.append(point.label)
        return (
            np.array(coordinates, dtype=np.float32).reshape(-1, 2),
            np.array(labels, dtype=np.int32),
        )

    def clear_prompt_points(self) -> None:
        for point in self.prompt_items.copy():
            self._remove_prompt(point)

    def delete_selected_prompts(self) -> None:
        for point in self.prompt_items.copy():
            if point.isSelected():
                self._remove_prompt(point)
        if self.prompt_items:
            self.prompt_items[-1].setSelected(True)

    def clear_prompts_outside_box(self) -> None:
        if self.box_item is None:
            return
        scene_rect: QRectF = self.box_item.mapRectToScene(self.box_item.rect())
        for point in self.prompt_items.copy():
            if not scene_rect.contains(point.scenePos()):
                self._remove_prompt(point)

    def _remove_prompt(self, point: PromptPoint) -> None:
        if self.dragged_item is point:
            self.dragged_item = None
        self.graphics_scene.removeItem(point)
        self.prompt_items.remove(point)


    """ return selection box range on image (left, top, right, bottom)"""
    def get_crop_box(self) -> tuple[int, int, int, int] | None:
        if self.box_item is None:
            return None
        rect: QRectF = self.box_item.mapRectToScene(self.box_item.rect()).intersected(
            self.image_item.boundingRect()
        )
        if rect.isEmpty():
            return None
        return floor(rect.left()), floor(rect.top()), ceil(rect.right()), ceil(rect.bottom())

    """ clean old selection box and set a new box"""
    def set_crop_box(self, box: tuple[int, int, int, int] | None) -> None:
        self.clear_draft_box()
        self.clear_box()
        if box is None:
            return
        left, top, right, bottom = box
        self.box_item = SelectionBox(self.image_item.boundingRect())
        self.box_item.setRect(QRectF(left, top, right - left, bottom - top))
        self.graphics_scene.addItem(self.box_item)
        self.graphics_scene.clearSelection()
        self.box_item.setSelected(True)

    def clear_draft_box(self) -> None:
        self.box_start_scene_position = None
        if self.draft_box_item is not None:
            self.graphics_scene.removeItem(self.draft_box_item)
            self.draft_box_item = None

    def clear_box(self) -> None:
        if self.box_item is not None:
            if self.dragged_item is self.box_item:
                self.dragged_item = None
            self.graphics_scene.removeItem(self.box_item)
            self.box_item = None
        self.clear_mask()

    def clear_mask(self) -> None:
        self.mask_overlay_item.setPixmap(QPixmap())

    def show_segmentation(self, mask: np.ndarray, box: tuple[int, int, int, int] | None) -> None:
        pixmap: QPixmap = self.image_item.pixmap()
        if mask.shape != (pixmap.height(), pixmap.width()):
            raise ValueError("Mask size must match the original image")
        self.set_crop_box(box)
        if box is None:
            return
        height, width = mask.shape
        rgba: np.ndarray = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[mask.astype(bool)] = (0, 120, 255, 90)
        overlay: Image.Image = Image.fromarray(rgba)
        self.mask_overlay_item.setPixmap(QPixmap.fromImage(ImageQt(overlay)))

    def delete_selected_items(self) -> None:
        if self.box_item is not None and self.box_item.isSelected():
            self.clear_box()
        else:
            self.delete_selected_prompts()

    """ mouse press event call back"""
    # if left click a prompt point, select it and can move it
    # if left click a selection boc, select it and can move it
    # if right click, add a negative prompt point
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.image_item.pixmap().isNull():
            event.accept()
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        view_position: QPoint = event.position().toPoint()
        scene_position: QPointF = self.mapToScene(view_position)
        wants_pan: bool = event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        )
        if wants_pan:
            self.pan_last_view_position = view_position
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                if self.image_item.boundingRect().contains(scene_position):
                    self._begin_box(scene_position)
            else:
                item: PromptPoint | SelectionBox | None = self._find_editable_item(view_position)
                if item is None:
                    self.add_prompt(scene_position, label=1)
                else:
                    self.graphics_scene.clearSelection()
                    item.setSelected(True)
                    self.dragged_item = item
                    self.drag_offset_scene = scene_position - item.pos()
        elif event.button() == Qt.MouseButton.RightButton:
            self.add_prompt(scene_position, label=0)
        event.accept()

    """ move canvas or prompt point or selection box """
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        view_position: QPoint = event.position().toPoint()
        scene_position: QPointF = self.mapToScene(view_position)
        if self.pan_last_view_position is not None:
            self._pan_view(view_position)
        elif self.draft_box_item is not None:
            self._update_draft_box(scene_position)
        elif self.dragged_item is not None:
            self.dragged_item.setPos(scene_position - self.drag_offset_scene)
        else:
            super().mouseMoveEvent(event)
            return
        event.accept()

    """ record mouse position in view when moving canvas / create the selection box """
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.pan_last_view_position is not None and event.button() in (
            Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton
        ):
            self.pan_last_view_position = None
            self.unsetCursor()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.draft_box_item is not None:
                self._update_draft_box(self.mapToScene(event.position().toPoint()))
                self._commit_draft_box()
            self.dragged_item = None
        event.accept()

    """ scale the canvas"""
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.image_item.pixmap().isNull():
            event.ignore()
            return
        steps: float = event.angleDelta().y() / 120.0
        if steps == 0:
            steps = event.pixelDelta().y() / 60.0
        if steps == 0:
            event.accept()
            return
        current_scale: float = self.transform().m11()
        target_scale: float = max(
            self.fit_scale, min(current_scale * 1.2 ** steps, self.fit_scale * 20.0)
        )
        self.auto_fit_enabled = False
        self.scale(target_scale / current_scale, target_scale / current_scale)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_items()
            event.accept()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.auto_fit_enabled:
            self.fit_image()

    def _begin_box(self, scene_position: QPointF) -> None:
        self.clear_draft_box()
        self.box_start_scene_position = scene_position
        self.draft_box_item = SelectionBox(self.image_item.boundingRect())
        self.draft_box_item.setRect(QRectF(scene_position, scene_position))
        self.draft_box_item.setZValue(6)
        self.graphics_scene.addItem(self.draft_box_item)

    def _update_draft_box(self, scene_position: QPointF) -> None:
        if self.draft_box_item is None or self.box_start_scene_position is None:
            return
        bounds: QRectF = self.image_item.boundingRect()
        clamped_position: QPointF = QPointF(
            max(bounds.left(), min(scene_position.x(), bounds.right())),
            max(bounds.top(), min(scene_position.y(), bounds.bottom())),
        )
        self.draft_box_item.setRect(
            QRectF(self.box_start_scene_position, clamped_position).normalized()
        )

    def _commit_draft_box(self) -> None:
        if self.draft_box_item is None:
            return
        rect: QRectF = self.draft_box_item.rect()
        if rect.width() >= 1 and rect.height() >= 1:
            # only new box is created, we clean old box
            self.clear_box()
            self.box_item = self.draft_box_item
            self.draft_box_item = None
            self.box_item.setZValue(5)
            self.graphics_scene.clearSelection()
            self.box_item.setSelected(True)
            self.clear_prompts_outside_box()
        self.clear_draft_box()

    def _pan_view(self, view_position: QPoint) -> None:
        if self.pan_last_view_position is None:
            return
        delta: QPoint = view_position - self.pan_last_view_position
        self.pan_last_view_position = view_position
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    """ give a mouse position, return if in this position there is a objet can be edit"""
    def _find_editable_item(self, view_position: QPoint) -> PromptPoint | SelectionBox | None:
        for item in self.items(view_position):
            if isinstance(item, PromptPoint):
                return item
        if self.box_item is None:
            return None
        scene_rect: QRectF = self.box_item.mapRectToScene(self.box_item.rect())
        view_rect: QRectF = self.viewportTransform().mapRect(scene_rect)
        outer: QRectF = view_rect.adjusted(-5, -5, 5, 5)
        inner: QRectF = view_rect.adjusted(5, 5, -5, -5)
        cursor: QPointF = QPointF(view_position)
        inside_inner: bool = inner.isValid() and inner.contains(cursor)
        if outer.contains(cursor) and not inside_inner:
            return self.box_item
        return None
