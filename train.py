from ultralytics import YOLO

if __name__ == "__main__":   # required on Windows, or the dataloader crashes
    model = YOLO("yolo26s.pt")   # pretrained weights, auto-downloaded
    model.train(
        data="dataset.yaml",
        imgsz=960,
        epochs=100,
        batch=8,          # lower to 4 if you run out of GPU memory
        patience=25,      # stop early if val stops improving
        device=0,
        workers=4,
    )