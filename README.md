# Fire & Smoke Detection on CCTV Streams
 
Real-time fire and smoke detection across **4 camera streams** (3x RTSP + 1x IP-webcam MJPEG) using a YOLO model. It is built to run on a CPU-only laptop, and it raises an alarm only when several recent detections agree, which cuts down false alarms.
 
## Features
 
- **One reader thread per camera.** Only the newest frame is kept, so lag never builds up.
- **Auto-reconnect.** A dropped stream is retried automatically.
- **Per-camera inference rate limit.** Fire doesn't need 25 FPS analysis, so each camera is analysed about 2 times per second.
- **Batched inference.** All cameras that are due are processed in one model call.
- **Per-class confidence thresholds.** Fire and smoke each have their own threshold.
- **Temporal voting.** An alarm fires only if enough of the last few analysed frames contain a detection.
- **Alarm cooldown and snapshots.** The same camera won't re-alarm within the cooldown, and an annotated image is saved for each alarm.
- **Live 2x2 grid.** Cameras that are down show a `NO SIGNAL` tile.
## How It Works
 
```
Camera 1..4 ──► reader threads (latest frame only)
                      │
                      ▼
        main loop: pick cameras that are due (<= 2 FPS each)
                      │
                      ▼
          one batched YOLO inference call
                      │
                      ▼
     per-class confidence filter (fire / smoke)
                      │
                      ▼
   temporal voting: >= MIN_HITS of last WINDOW frames
                      │
                      ▼
     alarm (cooldown) ──► save snapshot to alerts/
                      │
                      ▼
              2x2 live display
```
 
## Requirements
 
- Python 3.8+
- A trained YOLO model file (`.pt`)
- Network access to your cameras
```bash
pip install ultralytics opencv-python numpy
```
 
## Setup
 
1. Put your model weights in the project folder (default name: `fire_pretrained.pt`).
2. Add your cameras to the `CAMERAS` dictionary in the script:
```python
CAMERAS = {
  use your web IP's
}
```
 
3. Run it:
```bash
python main.py
```
 
Press **q** in the video window to quit.
 
> Replace `main.py` with the actual name of your script file.
 
## Configuration
 
All settings are at the top of the script.
 
| Setting | Default | Description |
|---|---|---|
| `MODEL_PATH` | `fire_pretrained.pt` | Path to the YOLO weights. Swap in your fine-tuned `best.pt` later. |
| `IMGSZ` | `640` | Inference image size. Raise to 960/1280 after fine-tuning on a GPU. |
| `DEVICE` | `"cpu"` | Use `0` for an NVIDIA GPU. |
| `HALF` | `False` | FP16 inference. Works on GPU only. |
| `CLASS_CONF` | fire `0.35`, smoke `0.40` | Per-class confidence thresholds. |
| `INFER_FPS_PER_CAM` | `2.0` | Max analyses per camera per second. |
| `WINDOW` | `6` | Number of recent analysed frames used for voting. |
| `MIN_HITS` | `4` | Detections needed within the window to trigger an alarm. |
| `ALARM_COOLDOWN_S` | `60` | Seconds before the same camera can alarm again. |
| `SAVE_DIR` | `alerts` | Folder where alarm snapshots are saved. |
| `SHOW` | `True` | Set to `False` for headless servers. |
| `TILE_W`, `TILE_H` | `640`, `360` | Size of each camera tile in the grid. |
 
With the defaults, 4 hits in 6 analysed frames at 2 FPS means roughly 3 seconds of agreeing evidence before an alarm.
 
## Alerts
 
When an alarm triggers, the annotated frame is saved as:
 
```
alerts/<camera>_<classes>_<YYYYMMDD_HHMMSS>.jpg
```
 
The `send_alert()` function is a hook. Extend it to send SMS, WhatsApp, email, MQTT messages, or to trigger a siren relay.
 
## Tuning Tips
 
- **Too many false alarms:** raise `CLASS_CONF`, or increase `MIN_HITS`.
- **Alarms too slow:** lower `MIN_HITS` or `WINDOW`.
- **CPU too busy:** lower `INFER_FPS_PER_CAM` or `IMGSZ`.
- **Gray or corrupted RTSP frames:** the script already forces RTSP over TCP, so also check network quality and camera bitrate.
## Project Structure
 
```
fire_with_vision/
├── main.py                # detection script
├── fire_pretrained.pt     # model weights (not committed)
├── alerts/                # saved alarm snapshots (not committed)
└── README.md
```
 
## Notes
 
- Do not commit camera credentials. Use environment variables or a config file listed in `.gitignore`.
- Model weights and the `alerts/` folder are best left out of the repository.
## Roadmap
 
- Fine-tune the model on your own fire/smoke footage.
- Move inference to a GPU for higher resolution and frame rate.
- Add real notification channels (SMS, WhatsApp, MQTT) in `send_alert()`.
## License
 
Add a license of your choice (for example MIT).
 
