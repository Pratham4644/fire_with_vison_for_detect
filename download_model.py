from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
shutil.copy(path, "fire_pretrained.pt")
print("saved fire_pretrained.pt")