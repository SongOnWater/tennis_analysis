import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging

from utils import (
    measure_distance,
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position
)

class MiniCourt:
    def __init__(self, frame, logger=None):
        self.logger = logger or logging.getLogger()
        self.logger.info("初始化迷你球场...")
        
        # 初始化参数
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        self.logger.info("完成迷你球场初始化")
        
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, court_keypoints):
        self.logger.info("转换边界框到迷你球场坐标...")
        output_player_boxes = []
        output_ball_boxes = []

        for i in range(len(player_boxes)):
            player_height = {1: court_keypoints[1]-court_keypoints[5]}  # 默认包含键1
            for player_id in player_boxes[i].keys():
                player_height[player_id] = court_keypoints[1]-court_keypoints[5]
            output_player_bboxes_dict = {}
            for player_id in player_boxes[i].keys():
                # 获取球员脚部位置
                foot_position = get_foot_position(player_boxes[i][player_id][:4])  # 只取前4个元素作为坐标
                
                # 获取最近的球场关键点
                closest_key_point_index = self.get_closest_keypoint_index(foot_position, court_keypoints, [0,2,12,13])
                
                # 获取最近的关键点索引
                closest_key_point = (court_keypoints[closest_key_point_index*2], court_keypoints[closest_key_point_index*2+1])

                # 计算相对于最近关键点的偏移量
                relative_position = self.get_court_keyboard_distance(foot_position, closest_key_point, player_height[player_id])

                # 确保输出关键点在球场范围内
                output_key_point_index = self.get_closest_keypoint_index(closest_key_point, self.drawing_key_points, [0,2,12,13])
                output_key_point = self.drawing_key_points[output_key_point_index*2], self.drawing_key_points[output_key_point_index*2+1]
                
                # 计算迷你球场上的最终位置
                output_player_bboxes_dict[player_id] = self.get_basket_position(output_key_point, relative_position)

            output_player_boxes.append(output_player_bboxes_dict)

        # 处理球的位置
        for i in range(len(ball_boxes)):
            output_ball_bboxes_dict = {}
            for ball_id in ball_boxes[i].keys():
                # 获取球的位置
                ball_position = get_foot_position(ball_boxes[i][ball_id][:4])  # 只取前4个元素作为坐标
                
                # 获取最近的球场关键点
                closest_key_point_index = self.get_closest_keypoint_index(ball_position, court_keypoints, [0,2,12,13])
                closest_key_point = (court_keypoints[closest_key_point_index*2], court_keypoints[closest_key_point_index*2+1])

                # 计算相对于最近关键点的偏移量
                relative_position = self.get_court_keyboard_distance(ball_position, closest_key_point, player_height[1])  # 使用球员1的高度作为参考

                # 确保输出关键点在球场范围内
                output_key_point_index = self.get_closest_keypoint_index(closest_key_point, self.drawing_key_points, [0,2,12,13])
                output_key_point = self.drawing_key_points[output_key_point_index*2], self.drawing_key_points[output_key_point_index*2+1]
                
                # 计算迷你球场上的最终位置
                output_ball_bboxes_dict[ball_id] = self.get_basket_position(output_key_point, relative_position)

            output_ball_boxes.append(output_ball_bboxes_dict)
            
        self.logger.info("完成边界框到迷你球场坐标的转换")
        return output_player_boxes, output_ball_boxes

    def draw_mini_court(self, frames):
        self.logger.info("绘制迷你球场...")
        output_frames = []
        for frame in frames:
            frame = self.draw_court(frame)
            output_frames.append(frame)
        self.logger.info("完成迷你球场绘制")
        return output_frames

    def draw_points_on_mini_court(self, frames, positions, color=(0,255,0)):
        self.logger.info("在迷你球场上绘制点...")
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame_copy = frame.copy()

            # 绘制球员位置
            if frame_num < len(positions):
                for player_id in positions[frame_num].keys():
                    x, y = positions[frame_num][player_id]
                    x = int(x)
                    y = int(y)
                    cv2.circle(frame_copy, (x,y), 5, color, -1)
            
            output_frames.append(frame_copy)
        self.logger.info("完成在迷你球场上绘制点")
        return output_frames

    def set_canvas_background_box_position(self, frame):
        frame_height, frame_width = frame.shape[:2]
        self.canvas_background_pos_x = frame_width - self.drawing_rectangle_width - self.buffer
        self.canvas_background_pos_y = self.buffer

    def set_mini_court_position(self):
        self.court_drawing_x_start = self.canvas_background_pos_x + self.padding_court
        self.court_drawing_x_end = self.canvas_background_pos_x + self.drawing_rectangle_width - self.padding_court
        self.court_drawing_y_start = self.canvas_background_pos_y + self.padding_court
        self.court_drawing_y_end = self.canvas_background_pos_y + self.drawing_rectangle_height - self.padding_court

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # 点 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_drawing_x_start), int(self.court_drawing_y_start)
        # 点 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_drawing_x_end), int(self.court_drawing_y_start)
        # 点 2
        drawing_key_points[4], drawing_key_points[5] = int(self.court_drawing_x_start), int(self.court_drawing_y_end)
        # 点 3
        drawing_key_points[6], drawing_key_points[7] = int(self.court_drawing_x_end), int(self.court_drawing_y_end)

        # 中线
        # 点 4
        drawing_key_points[8], drawing_key_points[9] = int(self.court_drawing_x_start), int((self.court_drawing_y_start + self.court_drawing_y_end)/2)
        # 点 5
        drawing_key_points[10], drawing_key_points[11] = int(self.court_drawing_x_end), int((self.court_drawing_y_start + self.court_drawing_y_end)/2)

        # 网点 1
        drawing_key_points[12], drawing_key_points[13] = int((self.court_drawing_x_start + self.court_drawing_x_end)/2), int(self.court_drawing_y_start)
        # 网点 2
        drawing_key_points[14], drawing_key_points[15] = int((self.court_drawing_x_start + self.court_drawing_x_end)/2), int(self.court_drawing_y_end)

        # 网点 3
        drawing_key_points[16], drawing_key_points[17] = int((self.court_drawing_x_start + self.court_drawing_x_end)/2), int((self.court_drawing_y_start + self.court_drawing_y_end)/2)

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        # 确定球场线
        self.lines = [
            (0, 2),  # 左侧边线
            (1, 3),  # 右侧边线
            (0, 1),  # 上侧边线
            (2, 3),  # 下侧边线

            (4, 5),  # 中线

            (6, 7),  # 网顶线
            (8, 9),  # 网底线
            (10, 11), # 网中线

            (0, 4),  # 左侧发球线
            (1, 5),  # 右侧发球线
        ]

    def draw_court(self, frame):
        # 绘制背景矩形
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (self.canvas_background_pos_x, self.canvas_background_pos_y), 
                      (self.canvas_background_pos_x + self.drawing_rectangle_width, self.canvas_background_pos_y + self.drawing_rectangle_height),
                      (255, 255, 255), thickness=-1)

        # 绘制球场线
        for i in range(len(self.drawing_key_points)//2):
            x = int(self.drawing_key_points[i*2])
            y = int(self.drawing_key_points[i*2+1])
            cv2.circle(frame_copy, (x,y), 5, (0, 0, 255), -1)

        # 绘制连接线
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame_copy, start_point, end_point, (0, 0, 0), 2)

        # 绘制中点
        center_circle_radius = 20  # 增加半径使圆更清晰
        center_point = (int(self.drawing_key_points[16]), int(self.drawing_key_points[17]))
        cv2.circle(frame_copy, center_point, center_circle_radius, (0, 0, 255), 2)

        return frame_copy

    def get_closest_keypoint_index(self, point, court_keypoints, candidate_key_points_ids):
        closest_key_point_index = candidate_key_points_ids[0]
        closest_key_point_distance = measure_distance(point, (court_keypoints[candidate_key_points_ids[0]*2], court_keypoints[candidate_key_points_ids[0]*2+1]))

        for key_point_id in candidate_key_points_ids:
            key_point_distance = measure_distance(point, (court_keypoints[key_point_id*2], court_keypoints[key_point_id*2+1]))
            if key_point_distance < closest_key_point_distance:
                closest_key_point_distance = key_point_distance
                closest_key_point_index = key_point_id
                
        return closest_key_point_index

    def get_court_keyboard_distance(self, point1, point2, player_height_in_pixels):
        distance_x = point1[0] - point2[0]
        distance_y = point1[1] - point2[1]
        # Convert pixel distances to meters
        distance_x_in_meters = convert_pixel_distance_to_meters(distance_x, player_height_in_pixels, 1.8)
        distance_y_in_meters = convert_pixel_distance_to_meters(distance_y, player_height_in_pixels, 1.8)
        return (distance_x_in_meters, distance_y_in_meters)

    def get_basket_position(self, closest_key_point, relative_position):
        # relative_position should contain separate x and y distances
        x = closest_key_point[0] + relative_position[0]
        y = closest_key_point[1] + relative_position[1]
        return (x, y)
    
    def get_width_of_mini_court(self):
        return self.drawing_rectangle_width - 2 * self.padding_court
    