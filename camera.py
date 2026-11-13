"""
SHIELD FSL — Shared Camera (camera.py)
High-speed threaded camera capture extracted from main.py.
"""

import sys
import threading
import time
import collections

import cv2


class CameraCapture:
    """Threaded camera grabber that continuously fetches frames at 30-60 FPS without blocking ML/GUI loops."""
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.fps_tracker = collections.deque(maxlen=30)
        self.cam_fps = 0.0
        self.thread = None
        self.start()

    def _init_camera(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_index, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(self.camera_index)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 60)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        with self.lock:
            self.cap = cap

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _capture_loop(self):
        self._init_camera()
        while self.running:
            with self.lock:
                cap_ref = self.cap
            if cap_ref is None or not cap_ref.isOpened():
                time.sleep(0.04)
                continue
            ret, frame = cap_ref.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            self.fps_tracker.append(now)
            if len(self.fps_tracker) > 1:
                self.cam_fps = (len(self.fps_tracker) - 1) / max(self.fps_tracker[-1] - self.fps_tracker[0], 1e-4)

            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def switch_camera(self, new_index):
        with self.lock:
            self.camera_index = new_index
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        threading.Thread(target=self._init_camera, daemon=True).start()

    def release(self):
        self.running = False
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def stop(self):
        self.release()
