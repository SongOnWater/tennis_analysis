import cv2
import pickle
import pandas as pd
import torch

class BallTracker:
    def __init__(self,model_path):
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"BallTracker initialized with device: {self.device}")
        
    def _load_model(self):
        if self.model is None:
            try:
                # 深度清理模块环境
                import sys
                for module in ['charset_normalizer', 'chardet', 'requests', 'urllib3', 'ultralytics']:
                    if module in sys.modules:
                        del sys.modules[module]
                
                # 确保从干净环境导入
                from importlib import reload
                import ultralytics
                reload(ultralytics)
                
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"BallTracker model successfully loaded on device: {self.device}")
            except Exception as e:
                print(f"CRITICAL ERROR loading YOLO model: {e}")
                raise

    def interpolate_ball_positions(self, ball_positions):
        ball_data = []
        for x in ball_positions:
            bbox = x.get(1, [])
            # Handle both cases: with and without confidence value
            if len(bbox) >= 4:
                # Take only the first 4 elements (x1, y1, x2, y2)
                ball_data.append(bbox[:4])
            else:
                # If less than 4 elements, pad with zeros
                ball_data.append(bbox + [0] * (4 - len(bbox)))
                
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_data, columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_data = []
        for x in ball_positions:
            bbox = x.get(1, [])
            # Handle both cases: with and without confidence value
            if len(bbox) >= 4:
                # Take only the first 4 elements (x1, y1, x2, y2)
                ball_data.append(bbox[:4])
            else:
                # If less than 4 elements, pad with zeros
                ball_data.append(bbox + [0] * (4 - len(bbox)))
                
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_data, columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    # Use .loc instead of chained assignment to avoid pandas warnings
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        total_frames = len(frames)
        from tqdm import tqdm
        for i, frame in enumerate(tqdm(frames, desc="检测网球", unit="帧")):
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        # Load model if not already loaded
        if self.model is None:
            self._load_model()
            
        results = self.model.predict(frame, conf=0.15, device=self.device)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            confidence = box.conf.tolist()[0]
            ball_dict[1] = result + [confidence]
            print(f"Ball ID: 1, Confidence: {confidence:.2f}")
        
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox  # bbox 只包含坐标值
                cv2.putText(frame, f"Ball ID: {track_id}",(int(x1),int(y1 -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    