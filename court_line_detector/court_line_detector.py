import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np
import logging

class CourtLineDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logging.getLogger()
        self.logger.info(f"CourtLineDetector initialized with device: {self.device}")
        
        self.model = None
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self):
        if self.model is None:
            try:
                self.model = models.resnet50(weights='DEFAULT')
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"CourtLineDetector model loaded on device: {self.device}")
            except Exception as e:
                self.logger.error(f"Error loading CourtLineDetector model: {e}")
                raise
            
    def predict(self, frame):
        self.logger.info("开始球场关键点预测...")
        if self.model is None:
            self._load_model()
            
        # 转换帧为模型输入格式
        transformed_frame = self.transform(frame)
        transformed_frame = transformed_frame.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(transformed_frame)
            
        keypoints = output.squeeze().cpu().numpy()
        original_h, original_w = frame.shape[:2]
        
        # 调整关键点坐标以匹配原始帧尺寸
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        
        self.logger.info("完成球场关键点预测")
        return keypoints

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames

    def draw_keypoints(self, frame, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(frame, str(i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)
        return frame