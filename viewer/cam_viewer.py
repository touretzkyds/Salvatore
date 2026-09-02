"""QtQuick-based camera viewer mirroring the legacy OpenGL interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QGuiApplication, QImage
from PyQt6.QtQuick import QQuickView

from aim_fsm.camera import AIVISION_RESOLUTION_SCALE

from .camera_provider import CameraImageProvider
from .help_texts import CAMERA_HELP_TEXT
from .lifecycle import stop_timer_if_view_hidden
from .snapshot_service import SnapshotService


class CamViewer(QObject):
    """Drop-in replacement for the legacy OpenGL camera viewer."""

    def __init__(
        self,
        robot: Any,
        width: int = 640,
        height: int = 480,
        user_annotate_function: Optional[Any] = None,
        windowName: str = "Robot View",
    ) -> None:
        super().__init__(parent=None)
        if robot is None:
            raise ValueError("robot instance is required")

        self._robot = robot
        self._width = int(width)
        self._height = int(height)
        self._user_hook = user_annotate_function
        self._active_user_hook = user_annotate_function
        self._active_user_hook_sig = self._hook_signature(user_annotate_function)
        self._window_title = windowName

        self._app = QGuiApplication.instance() or QGuiApplication([])
        self._crosshairs = False

        self._provider = CameraImageProvider()
        self._provider.set_robot_ref(robot)
        self._provider.set_status(self._resolve_status())
        self._provider.set_aruco_detector(getattr(robot, "aruco_detector", None))
        if user_annotate_function is not None:
            self._provider.set_user_hook(user_annotate_function)
        self._provider.set_crosshair_enabled(self._crosshairs)
        self._provider.register_notifier(self._queue_frame_bump)

        self._snapshot_service = SnapshotService(base_name="robot")
        self._frame_counter = 0
        self._last_frame_id: Optional[int] = None

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._poll_robot)

        self._view = QQuickView()
        self._view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        self._view.setTitle(self._window_title)
        self._bind_visibility_handler()

        self._context = self._initialise_qml_context()
        self._sync_crosshair_to_qml()

    # ------------------------------------------------------------------
    # Legacy-compatible API

    def start(self) -> None:
        """Show the viewer window and begin polling for robot frames."""

        if not self._timer.isActive():
            self._timer.start()
        self._view.setWidth(self._width)
        self._view.setHeight(self._height)
        self._view.show()
        self._focus_root()
        self._sync_crosshair_to_qml()

    def stop(self) -> None:
        self._timer.stop()
        self._view.close()

    def is_running(self) -> bool:
        return self._timer.isActive()

    def is_visible(self) -> bool:
        return self._view.isVisible()

    def capture_raw(self, name: str = "robot_snap") -> Optional[str]:
        """Persist the latest raw frame using the snapshot service."""

        image = self._provider.current_image("live")
        path = self._snapshot_service.capture(image, annotated=False)
        if path is None:
            return None
        return str(path)

    def capture_annotated(self, name: str = "robot_asnap") -> Optional[str]:
        image = self._provider.current_image("annotated")
        if image is None:
            image = self._provider.current_image("live")
        path = self._snapshot_service.capture(image, annotated=True)
        if path is None:
            return None
        return str(path)

    @property
    def crosshairs(self) -> bool:
        return self._crosshairs

    @crosshairs.setter
    def crosshairs(self, value: bool) -> None:
        self.setCrosshairEnabled(bool(value))

    # ------------------------------------------------------------------
    # Slots exposed to QML

    @pyqtSlot(result=str)
    def captureRawSnapshot(self) -> str:
        return self.capture_raw() or ""

    @pyqtSlot(result=str)
    def captureAnnotatedSnapshot(self) -> str:
        return self.capture_annotated() or ""

    @pyqtSlot(bool)
    def setCrosshairEnabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._crosshairs = enabled
        self._provider.set_crosshair_enabled(enabled)
        self._sync_crosshair_to_qml()

    @pyqtSlot()
    def requestQuit(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers

    def _initialise_qml_context(self):
        repo_root = Path(__file__).resolve().parents[1]
        qml_path = (repo_root / "qml" / "CamView.qml").resolve()

        engine = self._view.engine()
        engine.addImportPath(str(repo_root / "qml"))
        engine.addImageProvider("camera", self._provider)

        context = self._view.rootContext()
        context.setContextProperty("AIVISION_RESOLUTION_SCALE", float(AIVISION_RESOLUTION_SCALE))
        context.setContextProperty("cameraFrameId", self._frame_counter)
        context.setContextProperty("viewerApp", self)
        context.setContextProperty("cameraHelpText", CAMERA_HELP_TEXT)

        self._view.setSource(QUrl.fromLocalFile(str(qml_path)))
        return context

    def _bind_visibility_handler(self) -> None:
        signal = getattr(self._view, "visibleChanged", None)
        if signal is None:
            signal = getattr(self._view, "visibilityChanged", None)
        if signal is not None:
            signal.connect(self._handle_visibility_changed)

    def _handle_visibility_changed(self, *args) -> None:
        stop_timer_if_view_hidden(self._view, self._timer)

    def _focus_root(self) -> None:
        try:
            root = self._view.rootObject()
        except Exception:
            return
        if root is not None and hasattr(root, "forceActiveFocus"):
            root.forceActiveFocus()

    def _poll_robot(self) -> None:
        image = getattr(self._robot, "camera_image", None)
        frame_id = getattr(self._robot, "frame_count", None)
        if image is None:
            return
        if frame_id is not None and frame_id == self._last_frame_id:
            return

        self._provider.set_status(self._resolve_status())
        self._provider.set_aruco_detector(getattr(self._robot, "aruco_detector", None))
        user_hook = self._resolve_user_hook()
        hook_sig = self._hook_signature(user_hook)
        if hook_sig != self._active_user_hook_sig:
            self._provider.set_user_hook(user_hook)
            self._active_user_hook = user_hook
            self._active_user_hook_sig = hook_sig

        self._provider.update_live_frame(image)
        self._last_frame_id = frame_id

    def _resolve_user_hook(self):
        try:
            from aim_fsm import program as _program
        except Exception:
            _program = None
        if _program is not None:
            running = getattr(_program, "running_fsm", None)
            if running is not None:
                return getattr(running, "user_annotate", None)
        return self._user_hook

    @staticmethod
    def _hook_signature(hook):
        if hook is None:
            return None
        func = getattr(hook, "__func__", None)
        owner = getattr(hook, "__self__", None)
        if func is not None and owner is not None:
            return (owner, func)
        return hook

    def _resolve_status(self):
        status = getattr(self._robot, "status", None)
        if self._status_has_aivision(status):
            return status

        robot0 = getattr(self._robot, "robot0", None)
        if robot0 is not None:
            try:
                status = getattr(robot0, "status", None)
            except Exception:
                status = None
        return status

    @staticmethod
    def _status_has_aivision(status) -> bool:
        try:
            return status is not None and "aivision" in status
        except TypeError:
            return False

    def _queue_frame_bump(self) -> None:
        if QGuiApplication.instance() is None:
            self._increment_frame()
        else:
            QTimer.singleShot(0, self._increment_frame)

    @pyqtSlot()
    def _increment_frame(self) -> None:
        self._frame_counter += 1
        if self._context is not None:
            self._context.setContextProperty("cameraFrameId", self._frame_counter)

    def _sync_crosshair_to_qml(self) -> None:
        try:
            root = self._view.rootObject()
        except Exception:
            root = None
        if root is not None:
            try:
                root.setProperty("showCrosshair", self._crosshairs)
            except Exception:
                pass


__all__ = ["CamViewer"]
