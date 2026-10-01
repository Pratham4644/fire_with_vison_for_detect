
"""
Real-time fire & smoke detection on 4 CCTV streams (3x RTSP + 1x IP-webcam MJPEG).

Design:
  - One reader thread per camera (keeps ONLY the newest frame -> no lag build-up)
  - Per-camera inference rate limit (CPU friendly: fire doesn't need 25 FPS analysis)
  - One batched inference call for all cameras that are due
  - Temporal voting per camera (alarm only if enough recent detections agree)
  - Per-camera alarm cooldown + snapshot saving
  - 2x2 live grid with NO SIGNAL tiles when a camera is down (press q to quit)

Install:  pip install ultralytics opencv-python numpy
"""

import os
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# Force RTSP over TCP (fewer corrupted/gray frames than UDP). Must be set before VideoCapture.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


# ----------------------------- CONFIG ---------------------------------------
CAMERAS = {
    # Use your WEB IPs please
}

MODEL_PATH = "fire_pretrained.pt"   # pretrained for now; swap to your fine-tuned best.pt later
IMGSZ = 640                         # model was trained at 640; raise (960/1280) after fine-tuning on GPU
DEVICE = "cpu"                      # "cpu" on this laptop; 0 for an NVIDIA GPU
HALF = False                        # FP16 only works on GPU

# Per-class confidence thresholds (class names are read from the model: smoke / fire)
CLASS_CONF = {"fire": 0.35, "smoke": 0.40}

# CPU budget: analyse each camera at most this many times per second
INFER_FPS_PER_CAM = 2.0

# Temporal voting (counted in ANALYSED frames, not camera frames):
# alarm if >= MIN_HITS of the last WINDOW analysed frames had a detection.
# At 2 FPS, WINDOW=6 is about 3 seconds of evidence.
WINDOW = 6
MIN_HITS = 4

ALARM_COOLDOWN_S = 60         # don't re-alarm the same camera within this time
SAVE_DIR = "alerts"
SHOW = True                   # False for headless servers
TILE_W, TILE_H = 640, 360     # size of each camera tile in the display grid
# -----------------------------------------------------------------------------


class CameraStream(threading.Thread):
    """Continuously reads a camera and keeps only the latest frame. Auto-reconnects."""

    def __init__(self, name, url):
        super().__init__(daemon=True)
        self.name, self.url = name, url
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.connected = False

    def run(self):
        while self.running:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                print(f"[{self.name}] cannot open stream, retrying in 3s")
                time.sleep(3)
                continue

            self.connected = True
            print(f"[{self.name}] connected")
            fails = 0
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:  # stream really dropped -> reconnect
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
            cap.release()
            self.connected = False
            print(f"[{self.name}] stream lost, reconnecting")
            time.sleep(2)

    def get_new_frame(self, last_id):
        """Return (frame, id) only if a newer frame than last_id exists, else (None, last_id)."""
        with self.lock:
            if self.frame is None or self.frame_id == last_id:
                return None, last_id
            return self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False


class CameraState:
    """Per-camera memory: voting history, timers, last annotated tile."""

    def __init__(self):
        self.last_id = -1
        self.last_infer = 0.0                 # time of last analysis (for FPS limit)
        self.history = deque(maxlen=WINDOW)   # 1 = detection in that analysed frame, 0 = none
        self.last_alarm = 0.0
        self.display = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        self.alarm_active = False


def send_alert(cam, kinds, frame):
    """Hook: replace/extend with SMS, WhatsApp, email, MQTT, siren relay, etc."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{cam}_{'-'.join(sorted(kinds))}_{stamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"!!! ALARM [{cam}] {sorted(kinds)} -> {path}")


def parse_detections(result, names):
    """Keep only boxes that pass the per-class threshold. Returns list of (cls_name, conf, xyxy)."""
    dets = []
    if result.boxes is None:
        return dets
    for box in result.boxes:
        cls_name = names[int(box.cls[0])].lower()
        conf = float(box.conf[0])
        if conf >= CLASS_CONF.get(cls_name, 0.5):
            dets.append((cls_name, conf, box.xyxy[0].cpu().numpy().astype(int)))
    return dets


def draw(frame, dets, cam, alarm):
    colors = {"fire": (0, 0, 255), "smoke": (200, 200, 200)}
    for cls_name, conf, (x1, y1, x2, y2) in dets:
        c = colors.get(cls_name, (0, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    label = f"{cam}  ALARM" if alarm else cam
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 180) if alarm else (40, 40, 40), -1)
    cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def no_signal_tile(cam):
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    cv2.putText(tile, f"{cam}: NO SIGNAL", (TILE_W // 2 - 110, TILE_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    return tile


def main():
    model = YOLO(MODEL_PATH)
    names = model.names
    print("model classes:", names)
    min_conf = min(CLASS_CONF.values())
    min_interval = 1.0 / INFER_FPS_PER_CAM

    streams = {n: CameraStream(n, u) for n, u in CAMERAS.items()}
    states = {n: CameraState() for n in CAMERAS}
    for s in streams.values():
        s.start()

    # Warm-up so the first real frame isn't slow
    model.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, device=DEVICE,
                  half=HALF, verbose=False)

    fps_t, fps_n, fps = time.time(), 0, 0.0
    try:
        while True:
            now = time.time()

            # 1) Collect a fresh frame from each camera that is due for analysis
            batch_names, batch_frames = [], []
            for n, s in streams.items():
                st = states[n]
                if now - st.last_infer < min_interval:
                    continue
                frame, fid = s.get_new_frame(st.last_id)
                if frame is not None:
                    st.last_id = fid
                    st.last_infer = now
                    batch_names.append(n)
                    batch_frames.append(frame)

            if not batch_frames:
                time.sleep(0.005)
            else:
                # 2) One batched inference call for all due cameras
                results = model.predict(batch_frames, imgsz=IMGSZ, conf=min_conf,
                                        device=DEVICE, half=HALF, verbose=False)

                for n, frame, res in zip(batch_names, batch_frames, results):
                    st = states[n]
                    dets = parse_detections(res, names)

                    # 3) Temporal voting
                    st.history.append(1 if dets else 0)
                    confirmed = (len(st.history) == WINDOW and sum(st.history) >= MIN_HITS)
                    st.alarm_active = confirmed

                    if confirmed and (time.time() - st.last_alarm) > ALARM_COOLDOWN_S:
                        st.last_alarm = time.time()
                        annotated_full = draw(frame.copy(), dets, n, True)
                        send_alert(n, {d[0] for d in dets}, annotated_full)

                    if SHOW:
                        st.display = cv2.resize(draw(frame, dets, n, st.alarm_active),
                                                (TILE_W, TILE_H))

            # FPS counter (analysed frames per second, all cameras combined)
            fps_n += len(batch_frames)
            if time.time() - fps_t >= 2:
                fps = fps_n / (time.time() - fps_t)
                fps_t, fps_n = time.time(), 0

            # 4) Show 2x2 grid
            if SHOW:
                tiles = []
                for n in CAMERAS:
                    tiles.append(states[n].display if streams[n].connected else no_signal_tile(n))
                while len(tiles) < 4:
                    tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
                grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
                cv2.putText(grid, f"{fps:.1f} analysed frames/s total", (8, grid.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Fire & Smoke Monitor", grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for s in streams.values():
            s.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()