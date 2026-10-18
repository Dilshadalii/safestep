import cv2
import threading
import time


class VideoSource:
    def __init__(self, source=0, width=640, height=480, loop=False):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.is_file = isinstance(source, str)
        self.loop = loop
        self.width = width
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/source: {source}")

        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        if self.is_file:
            self.delay = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
            self._next = time.time()
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.running = False
                break
            with self.lock:
                self.frame = frame

    def _read_file(self):
        now = time.time()
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(self._next, now) + self.delay

        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            self.running = False
            return None

        h, w = frame.shape[:2]
        if w != self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)))
        return frame

    def read(self):
        if self.is_file:
            return self._read_file()
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()