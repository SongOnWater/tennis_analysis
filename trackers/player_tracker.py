import cv2
import pickle
import sys
import torch
import logging
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"PlayerTracker initialized with device: {self.device}")
        self.fallback_id_counter = 0  # Counter for fallback IDs
        
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
                self.logger.info(f"PlayerTracker model successfully loaded on device: {self.device}")
            except Exception as e:
                self.logger.error(f"CRITICAL ERROR loading YOLO model: {e}")
                raise
                try:
                    # 备选方案：使用 stub 数据
                    self.logger.warning("无法加载模型，将使用缓存数据。请确保 tracker_stubs/player_detections.pkl 文件存在。")
                    self.model = None
                    return
                except Exception as backup_error:
                    self.logger.error(f"备选方案也失败: {backup_error}")
                    raise

    def choose_and_filter_players(self, court_keypoints, player_detections):
        self.logger.info("选择并过滤球员...")
        
        # 检查是否有检测结果
        if not player_detections or len(player_detections) == 0:
            self.logger.warning("警告: 没有球员检测结果")
            self.logger.debug(f"调试信息: player_detections={player_detections}")
            return []
            
        player_detections_first_frame = player_detections[0]
        self.logger.debug(f"第一帧球员检测结果: {player_detections_first_frame}")
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        self.logger.debug(f"选择的球员: {chosen_player}")
        
        # 如果没有选中任何球员，返回原始检测结果
        if not chosen_player:
            self.logger.warning("警告: 无法选择球员，返回所有检测结果")
            self.logger.debug(f"调试信息: player_detections={player_detections}")
            return player_detections
            
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        self.logger.debug(f"过滤后的球员检测结果: {filtered_player_detections}")
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        self.logger.debug(f"球员检测输入: court_keypoints={court_keypoints}, player_dict={player_dict}")
        
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            self.logger.debug(f"球员 {track_id} 的中心点: {player_center}")

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                self.logger.debug(f"球员 {track_id} 与球场关键点 {court_keypoint} 的距离: {distance}")
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
            self.logger.debug(f"球员 {track_id} 的最小距离: {min_distance}")
        
        # 检查是否有足够的球员被检测到
        if len(distances) == 0:
            self.logger.warning("警告: 没有检测到任何球员")
            self.logger.debug(f"调试信息: player_dict={player_dict}, court_keypoints={court_keypoints}")
            return []
        
        # 按距离升序排序
        distances.sort(key = lambda x: x[1])
        self.logger.debug(f"球员距离排序结果: {distances}")
        
        # 选择最近的球员（最多两个）
        chosen_players = []
        if len(distances) >= 1:
            chosen_players.append(distances[0][0])
        if len(distances) >= 2:
            chosen_players.append(distances[1][0])
            
        self.logger.info(f"最终选择的球员: {chosen_players}")
        return chosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        self.logger.info(f"检测帧输入: frames={len(frames)} 帧, read_from_stub={read_from_stub}, stub_path={stub_path}")

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            self.logger.info(f"从 stub 文件加载的球员检测结果: {player_detections}")
            return player_detections

        # Removed the early return that was preventing actual detection
        total_frames = len(frames)
        from tqdm import tqdm
        for i, frame in enumerate(tqdm(frames, desc="检测球员", unit="帧")):
            player_dict = self.detect_frame(frame)
            self.logger.debug(f"Frame {i}: Player dict content: {player_dict}")  # 调试输出
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        # Load model if not already loaded
        if self.model is None:
            self._load_model()

        results = self.model.track(frame, persist=False, device=self.device)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            # Null-safe ID extraction
            if box.id is not None:
                try:
                    track_id = int(box.id.tolist()[0])
                except Exception as e:
                    self.logger.warning(f"Failed to extract track_id from box.id: {e}. Using fallback ID.")
                    track_id = self.fallback_id_counter
                    self.fallback_id_counter += 1
                    self.logger.debug(f"Assigned fallback track_id: {track_id}")
            else:
                self.logger.warning("box.id is None. Using fallback ID.")
                track_id = self.fallback_id_counter
                self.fallback_id_counter += 1
                self.logger.debug(f"Assigned fallback track_id: {track_id}")

            # Null checks for other attributes
            if box.xyxy is not None:
                try:
                    result = box.xyxy.tolist()[0]
                except Exception as e:
                    self.logger.warning(f"Failed to extract xyxy from box.xyxy: {e}. Skipping box.")
                    continue
            else:
                self.logger.warning("box.xyxy is None. Skipping box.")
                continue

            if box.conf is not None:
                try:
                    confidence = box.conf.tolist()[0]
                except Exception as e:
                    self.logger.warning(f"Failed to extract conf from box.conf: {e}. Skipping box.")
                    continue
            else:
                self.logger.warning("box.conf is None. Skipping box.")
                continue

            if box.cls is not None:
                try:
                    object_cls_id = box.cls.tolist()[0]
                except Exception as e:
                    self.logger.warning(f"Failed to extract cls from box.cls: {e}. Skipping box.")
                    continue
            else:
                self.logger.warning("box.cls is None. Skipping box.")
                continue

            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result + [confidence]
                self.logger.debug(f"Player ID: {track_id}, Confidence: {confidence:.2f}")

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        confidence_threshold = 0.5
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                if len(bbox) >= 5:  # Check if confidence is included
                    x1, y1, x2, y2, confidence = bbox[:5]  # Extract first 5 values
                    self.logger.debug(f"Player ID: {track_id}, Confidence: {confidence:.2f}")
                    if confidence > confidence_threshold:
                        cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                else:  # Fallback if no confidence value
                    x1, y1, x2, y2 = bbox
                    cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
