import cv2
import logging

def read_video(video_path):
    logger = logging.getLogger()
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频总帧数: {total_frames}")
    
    # 导入进度条库
    from tqdm import tqdm
    
    # 创建进度条
    with tqdm(total=total_frames, desc="读取视频", unit="帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            pbar.update(1)
    
    cap.release()
    logger.info(f"成功读取 {len(frames)} 帧视频")
    return frames

def save_video(output_video_frames, output_video_path):
    logger = logging.getLogger()
    # 检查是否有帧可以保存
    if not output_video_frames:
        logger.warning(f"警告：没有视频帧可以保存到 {output_video_path}")
        return
        
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    # 导入进度条库
    from tqdm import tqdm
    
    # 创建进度条
    logger.info("保存视频中...")
    for frame in tqdm(output_video_frames, desc="保存视频", unit="帧"):
        out.write(frame)
    
    out.release()
    logger.info(f"视频已保存到 {output_video_path}")