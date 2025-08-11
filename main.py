from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy
import argparse
import os
import logging
from datetime import datetime
import sys
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='网球分析工具')
    parser.add_argument('--input', '-i', type=str, default="input_videos/input_video.mp4",
                        help='输入视频文件路径')
    parser.add_argument('--output', '-o', type=str, default="output_videos/output_video.avi",
                        help='输出视频文件路径')
    parser.add_argument('--use-stub', type=lambda x: (str(x).lower() in ('true', 't', 'yes', 'y', '1')), default=True,
                        help='是否使用缓存的检测结果 (True/False)')
    return parser.parse_args()

def setup_logging(input_video_path):
    """设置日志记录，将输出同时写入控制台和日志文件"""
    # 提取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件名，包含输入文件名和时间戳
    log_filename = f"tennis_analysis_{input_filename}_{timestamp}.log"
    
    # 确保日志目录存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 完整的日志文件路径
    log_file_path = os.path.join(log_dir, log_filename)
    
    # 创建自定义的日志处理器，同时输出到文件和控制台
    class ConsoleAndFileHandler(logging.StreamHandler):
        def __init__(self, file_stream):
            super().__init__()
            self.file_stream = file_stream
            
        def emit(self, record):
            # 输出到控制台（除了进度条）
            if "进度" not in record.getMessage() and "frame" not in record.getMessage().lower():
                super().emit(record)
            # 始终输出到文件
            msg = self.format(record)
            self.file_stream.write(msg + '\n')
            self.file_stream.flush()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    logger.info(f"日志文件: {log_file_path}")
    return logger, log_file_path

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志记录
    logger, log_file_path = setup_logging(args.input)
    
    logger.info(f"命令行参数: input={args.input}, output={args.output}, use_stub={args.use_stub} (类型: {type(args.use_stub)})")
    print(f"命令行参数: input={args.input}, output={args.output}, use_stub={args.use_stub} (类型: {type(args.use_stub)})")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
        print(f"创建输出目录: {output_dir}")
    
    # Read Video
    input_video_path = args.input
    logger.info(f"读取输入视频: {input_video_path}")
    print(f"读取输入视频: {input_video_path}")
    video_frames = read_video(input_video_path)

    # Check if CUDA is available
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Main process using device: {device}")
    print(f"Main process using device: {device}")

    # Detect Players and Ball
    logger.info("初始化 PlayerTracker...")
    player_tracker = PlayerTracker(model_path='yolov8x')
    logger.info("初始化 BallTracker...")
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    try:
        logger.info("开始检测球员...")
        player_detections = player_tracker.detect_frames(video_frames,
                                                        read_from_stub=args.use_stub,
                                                        stub_path="tracker_stubs/player_detections.pkl"
                                                        )
        if not player_detections:
            if args.use_stub:
                logger.warning("警告: 球员检测失败，尝试使用缓存数据...")
                print("警告: 球员检测失败，尝试使用缓存数据...")
                try:
                    with open("tracker_stubs/player_detections.pkl", 'rb') as f:
                        player_detections = pickle.load(f)
                    logger.info("成功加载球员检测缓存数据")
                    print("成功加载球员检测缓存数据")
                except Exception as e:
                    logger.error(f"无法加载球员检测缓存数据: {e}")
                    print(f"无法加载球员检测缓存数据: {e}")
                    logger.warning("将继续处理，但可能影响分析结果")
                    print("将继续处理，但可能影响分析结果")
            else:
                logger.warning("警告: 球员检测失败，且未启用缓存数据模式")
                print("警告: 球员检测失败，且未启用缓存数据模式")
                logger.warning("将继续处理，但可能影响分析结果")
                print("将继续处理，但可能影响分析结果")
    except Exception as e:
        logger.error(f"球员检测过程中出错: {e}")
        print(f"球员检测过程中出错: {e}")
        logger.info("尝试使用缓存数据...")
        print("尝试使用缓存数据...")
        try:
            with open("tracker_stubs/player_detections.pkl", 'rb') as f:
                player_detections = pickle.load(f)
            logger.info("成功加载球员检测缓存数据")
            print("成功加载球员检测缓存数据")
        except Exception as backup_error:
            logger.error(f"无法加载球员检测缓存数据: {backup_error}")
            print(f"无法加载球员检测缓存数据: {backup_error}")
            player_detections = []
    try:
        logger.info("开始检测球...")
        ball_detections = ball_tracker.detect_frames(video_frames,
                                                    read_from_stub=args.use_stub,
                                                    stub_path="tracker_stubs/ball_detections.pkl"
                                                    )
        if not ball_detections:
            if args.use_stub:
                logger.warning("警告: 球检测失败，尝试使用缓存数据...")
                print("警告: 球检测失败，尝试使用缓存数据...")
                try:
                    with open("tracker_stubs/ball_detections.pkl", 'rb') as f:
                        ball_detections = pickle.load(f)
                    logger.info("成功加载球检测缓存数据")
                    print("成功加载球检测缓存数据")
                except Exception as e:
                    logger.error(f"无法加载球检测缓存数据: {e}")
                    print(f"无法加载球检测缓存数据: {e}")
                    logger.warning("将继续处理，但可能影响分析结果")
                    print("将继续处理，但可能影响分析结果")
            else:
                logger.warning("警告: 球检测失败，且未启用缓存数据模式")
                print("警告: 球检测失败，且未启用缓存数据模式")
                logger.warning("将继续处理，但可能影响分析结果")
                print("将继续处理，但可能影响分析结果")
                ball_detections = []  # 确保 ball_detections 不为 None
    except Exception as e:
        logger.error(f"球检测过程中出错: {e}")
        print(f"球检测过程中出错: {e}")
        logger.info("尝试使用缓存数据...")
        print("尝试使用缓存数据...")
        try:
            with open("tracker_stubs/ball_detections.pkl", 'rb') as f:
                ball_detections = pickle.load(f)
            logger.info("成功加载球检测缓存数据")
            print("成功加载球检测缓存数据")
        except Exception as backup_error:
            logger.error(f"无法加载球检测缓存数据: {backup_error}")
            print(f"无法加载球检测缓存数据: {backup_error}")
            ball_detections = []
            logger.warning("将继续处理，但可能影响分析结果")
            print("将继续处理，但可能影响分析结果")
    
    # 只有在球检测数据不为空时才进行插值
    if ball_detections:
        try:
            logger.info("开始插值球轨迹...")
            ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
        except Exception as e:
            logger.error(f"球轨迹插值过程中出错: {e}")
            print(f"球轨迹插值过程中出错: {e}")
            logger.info("将使用原始球检测数据")
            print("将使用原始球检测数据")
    
    
    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    logger.info(f"加载球场关键点检测模型: {court_model_path}")
    print(f"加载球场关键点检测模型: {court_model_path}")
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    logger.info("完成球场关键点检测")
    print("完成球场关键点检测")

    # choose players
    logger.info("选择并过滤球员...")
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    logger.info("完成球员选择和过滤")

    # MiniCourt
    logger.info("初始化迷你球场...")
    mini_court = MiniCourt(video_frames[0]) 
    logger.info("完成迷你球场初始化")

    # Detect ball shots
    logger.info("检测击球帧...")
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)
    logger.info(f"检测到 {len(ball_shot_frames)} 个击球帧")

    # Convert positions to mini court positions
    logger.info("转换边界框到迷你球场坐标...")
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)
    logger.info("完成坐标转换")

    # 获取所有球员ID
    first_frame_with_players = 0
    player_ids = []  # 初始化为空列表，确保即使没有检测到球员也有定义
    
    for i, frame_data in enumerate(player_mini_court_detections):
        if frame_data:  # 如果有球员数据
            first_frame_with_players = i
            player_ids = list(frame_data.keys())
            break
    
    # 初始化球员统计数据
    player_stats_data = [{
        'frame_num': 0,
    }]
    
    # 为每个球员添加统计字段
    if player_ids:  # 只有在有球员ID时才添加统计字段
        for player_id in player_ids:
            player_stats_data[0][f'player_{player_id}_number_of_shots'] = 0
            player_stats_data[0][f'player_{player_id}_total_shot_speed'] = 0
            player_stats_data[0][f'player_{player_id}_last_shot_speed'] = 0
            player_stats_data[0][f'player_{player_id}_total_player_speed'] = 0
            player_stats_data[0][f'player_{player_id}_last_player_speed'] = 0
    
    logger.info(f"开始处理 {len(ball_shot_frames)-1} 个击球间隔")
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        
        # 检查帧索引是否有效
        if (start_frame >= len(player_mini_court_detections) or 
            end_frame >= len(player_mini_court_detections) or
            start_frame >= len(ball_mini_court_detections) or 
            end_frame >= len(ball_mini_court_detections)):
            logger.warning(f"跳过无效帧范围: start_frame={start_frame}, end_frame={end_frame}")
            continue
            
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        
        # 获取所有球员ID
        player_ids = list(player_mini_court_detections[start_frame].keys())
        
        # 检查是否有球员
        if not player_ids:
            logger.warning(f"警告：在帧 {start_frame} 中没有检测到球员，跳过此帧")
            print(f"警告：在帧 {start_frame} 中没有检测到球员，跳过此帧")
            continue
            
        # 确定击球的球员
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                               ball_mini_court_detections[start_frame][1]))
        
        # 检查球员数量
        if len(player_ids) < 2:
            logger.warning(f"警告：在帧 {start_frame} 中检测到的球员少于2个，跳过此帧")
            print(f"警告：在帧 {start_frame} 中检测到的球员少于2个，跳过此帧")
            continue
            
        # 确保我们有两个球员ID
        if player_shot_ball not in player_ids:
            logger.warning(f"警告：在帧 {start_frame} 中未找到击球球员ID {player_shot_ball}，使用第一个检测到的球员")
            print(f"警告：在帧 {start_frame} 中未找到击球球员ID {player_shot_ball}，使用第一个检测到的球员")
            player_shot_ball = player_ids[0]
            
        # 找到对手ID（不是击球球员的另一个球员）
        opponent_player_ids = [pid for pid in player_ids if pid != player_shot_ball]
        if not opponent_player_ids:
            logger.warning(f"警告：在帧 {start_frame} 中未找到对手球员，跳过此帧")
            print(f"警告：在帧 {start_frame} 中未找到对手球员，跳过此帧")
            continue
            
        opponent_player_id = opponent_player_ids[0]
        
        # 确保在结束帧中也有这两个球员
        if (end_frame >= len(player_mini_court_detections) or 
            opponent_player_id not in player_mini_court_detections[end_frame]):
            logger.warning(f"警告：在帧 {end_frame} 中未找到对手球员ID {opponent_player_id}，跳过此帧")
            print(f"警告：在帧 {end_frame} 中未找到对手球员ID {opponent_player_id}，跳过此帧")
            continue
            
        # opponent player speed
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)
        logger.info(f"处理击球间隔 {ball_shot_ind+1}/{len(ball_shot_frames)-1}: 击球球员 {player_shot_ball}, 球速 {speed_of_ball_shot:.2f} km/h, 对手速度 {speed_of_opponent:.2f} km/h")

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # 为每个球员计算平均速度
    if player_ids:  # 只有在有球员ID时才计算
        for player_id in player_ids:
            # 避免除以零错误
            shots_column = f'player_{player_id}_number_of_shots'
            if shots_column in player_stats_data_df.columns and player_stats_data_df[shots_column].max() > 0:
                player_stats_data_df[f'player_{player_id}_average_shot_speed'] = player_stats_data_df[f'player_{player_id}_total_shot_speed'] / player_stats_data_df[shots_column]
            else:
                player_stats_data_df[f'player_{player_id}_average_shot_speed'] = 0
                
        # 计算球员平均移动速度
        if len(player_ids) > 0:  # 确保有球员
            for i, player_id in enumerate(player_ids):
                other_player_id = player_ids[(i+1) % len(player_ids)]  # 获取另一个球员的ID
                shots_column = f'player_{other_player_id}_number_of_shots'
                if shots_column in player_stats_data_df.columns and player_stats_data_df[shots_column].max() > 0:
                    player_stats_data_df[f'player_{player_id}_average_player_speed'] = player_stats_data_df[f'player_{player_id}_total_player_speed'] / player_stats_data_df[shots_column]
                else:
                    player_stats_data_df[f'player_{player_id}_average_player_speed'] = 0
    logger.info("完成球员统计数据计算")


    # Draw output
    from tqdm import tqdm
    logger.info("开始绘制输出视频...")
    print("开始绘制输出视频...")
    
    # 检查是否有视频帧
    if not video_frames:
        logger.warning("警告：没有输入视频帧，无法生成输出视频")
        print("警告：没有输入视频帧，无法生成输出视频")
        return
        
    # 创建输出视频帧的副本
    output_video_frames = video_frames.copy()
    
    ## Draw Player Bounding Boxes
    logger.info("绘制球员边界框...")
    print("绘制球员边界框...")
    if player_detections:
        output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
    else:
        logger.warning("警告：没有球员检测结果，跳过绘制球员边界框")
        print("警告：没有球员检测结果，跳过绘制球员边界框")
    
    logger.info("绘制网球边界框...")
    print("绘制网球边界框...")
    if ball_detections:
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    else:
        logger.warning("没有网球检测结果，跳过绘制网球边界框")
        print("警告：没有网球检测结果，跳过绘制网球边界框")

    ## Draw court Keypoints
    logger.info("绘制球场关键点...")
    if court_keypoints is not None and len(court_keypoints) > 0:
        output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    else:
        logger.warning("没有球场关键点检测结果，跳过绘制球场关键点")
        print("警告：没有球场关键点检测结果，跳过绘制球场关键点")

    # Draw Mini Court
    logger.info("绘制迷你球场...")
    print("绘制迷你球场...")
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    
    # 绘制球员在迷你球场上的位置
    logger.info("绘制球员在迷你球场上的位置...")
    if player_mini_court_detections:
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    else:
        logger.warning("警告：没有球员在迷你球场上的位置数据，跳过绘制")
        print("警告：没有球员在迷你球场上的位置数据，跳过绘制")
        
    # 绘制球在迷你球场上的位置
    logger.info("绘制球在迷你球场上的位置...")
    if ball_mini_court_detections:
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))
    else:
        logger.warning("警告：没有球在迷你球场上的位置数据，跳过绘制")
        print("警告：没有球在迷你球场上的位置数据，跳过绘制")

    # 确保有输出视频帧
    if not output_video_frames:
        logger.warning("警告：没有输出视频帧，无法继续处理")
        print("警告：没有输出视频帧，无法继续处理")
        return
        
    # 确保 player_stats_data_df 的行数与 output_video_frames 的长度一致
    if len(output_video_frames) != len(player_stats_data_df):
        logger.warning(f"警告：视频帧数 ({len(output_video_frames)}) 与统计数据行数 ({len(player_stats_data_df)}) 不匹配，将截断或填充统计数据")
        print(f"警告：视频帧数 ({len(output_video_frames)}) 与统计数据行数 ({len(player_stats_data_df)}) 不匹配，将截断或填充统计数据")
        min_length = min(len(output_video_frames), len(player_stats_data_df))
        player_stats_data_df = player_stats_data_df.iloc[:min_length]
        output_video_frames = output_video_frames[:min_length]
    
    # Draw Player Stats
    logger.info("绘制球员统计数据...")
    print("绘制球员统计数据...")
    if not player_stats_data_df.empty and len(output_video_frames) > 0:
        output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)
    else:
        logger.warning("警告：没有球员统计数据或没有视频帧，跳过绘制球员统计数据")
        print("警告：没有球员统计数据或没有视频帧，跳过绘制球员统计数据")

    ## Draw frame number on top left corner
    logger.info("添加帧编号...")
    print("添加帧编号...")
    if output_video_frames:
        for i, frame in enumerate(tqdm(output_video_frames, desc="处理视频帧", unit="帧")):
            cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        logger.warning("警告：没有视频帧，跳过添加帧编号")
        print("警告：没有视频帧，跳过添加帧编号")

    logger.info(f"保存视频到: {args.output}")
    save_video(output_video_frames, args.output)
    logger.info("视频处理完成")

if __name__ == "__main__":
    main()