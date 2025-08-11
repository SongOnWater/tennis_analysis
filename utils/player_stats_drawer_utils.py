import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats):
    # 获取所有球员ID
    player_columns = [col for col in player_stats.columns if col.startswith('player_') and '_last_shot_speed' in col]
    player_ids = [col.split('_')[1] for col in player_columns]
    
    for index, row in player_stats.iterrows():
        # 动态获取球员数据
        player_data = {}
        if player_ids:  # 只有在有球员ID时才获取数据
            for player_id in player_ids:
                player_data[f'player_{player_id}_last_shot_speed'] = row.get(f'player_{player_id}_last_shot_speed', 0)
                player_data[f'player_{player_id}_last_player_speed'] = row.get(f'player_{player_id}_last_player_speed', 0)
                player_data[f'player_{player_id}_average_shot_speed'] = row.get(f'player_{player_id}_average_shot_speed', 0)
                player_data[f'player_{player_id}_average_player_speed'] = row.get(f'player_{player_id}_average_player_speed', 0)

        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width = 350
        height = 230

        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        # 动态生成球员标题
        if player_ids:  # 只有在有球员ID时才生成标题
            player_title = "    ".join([f"Player {player_id}" for player_id in player_ids])
            output_video_frames[index] = cv2.putText(output_video_frames[index], player_title, (start_x + 80, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            output_video_frames[index] = cv2.putText(output_video_frames[index], "No Players Detected", (start_x + 80, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 动态生成统计数据
        if player_ids:  # 只有在有球员ID时才生成统计数据
            stats = [
                ("Shot Speed", [player_data[f'player_{player_id}_last_shot_speed'] for player_id in player_ids]),
                ("Player Speed", [player_data[f'player_{player_id}_last_player_speed'] for player_id in player_ids]),
                ("avg. S. Speed", [player_data[f'player_{player_id}_average_shot_speed'] for player_id in player_ids]),
                ("avg. P. Speed", [player_data[f'player_{player_id}_average_player_speed'] for player_id in player_ids])
            ]
        else:
            stats = []  # 没有球员时，不显示统计数据

        y_offset = 80
        for stat_name, stat_values in stats:
            output_video_frames[index] = cv2.putText(output_video_frames[index], stat_name, (start_x + 10, start_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            stat_text = "    ".join([f"{value:.1f} km/h" for value in stat_values])
            output_video_frames[index] = cv2.putText(output_video_frames[index], stat_text, (start_x + 130, start_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 40
    
    return output_video_frames
