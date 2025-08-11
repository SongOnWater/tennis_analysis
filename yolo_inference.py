import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Defer YOLO import to avoid circular import
from ultralytics import YOLO
model = YOLO('yolov8x')

result = model.track('input_videos/input_video.mp4', conf=0.2, save=True, device=device)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)