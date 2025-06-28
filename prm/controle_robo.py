#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Point, Quaternion, Pose

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import heapq
from collections import defaultdict # Para g_score e f_score no A*

from scipy.spatial.transform import Rotation

class ControleRobo(Node):
    def __init__(self):
        super().__init__('controle_robo')

        self.bridge = CvBridge()
        self.imagem = None 

        self.flag_pixel_centroid: tuple[int, int] | None = None
        self.flag_detected_in_current_image: bool = False
        self.flagpole_base_pixel_centroid: tuple[int, int] | None = None
        
        self.raw_estimated_flag_positions: list[Point] = [] 
        self.filtered_estimated_flag_world_position: Point | None = None 
        self.FLAG_ESTIMATION_HISTORY_SIZE = 5 
        self.flag_estimation_weights = np.linspace(1.0, self.FLAG_ESTIMATION_HISTORY_SIZE, 
                                                   self.FLAG_ESTIMATION_HISTORY_SIZE)
        if np.sum(self.flag_estimation_weights) > 0:
            self.flag_estimation_weights /= np.sum(self.flag_estimation_weights)
        else:
            self.flag_estimation_weights = np.array([1.0])

        self.flag_estimation_timer: rclpy.timer.Timer | None = None
        
        self.camera_hfov = 1.57; self.camera_image_width = 320; self.camera_image_height = 240
        self.camera_focal_length_x = (self.camera_image_width / 2.0) / np.tan(self.camera_hfov / 2.0)
        
        base_length = 0.42; base_height = 0.18
        z_lidar_offset = base_height / 2.0 + 0.055 / 2.0
        self.P_L_B = np.array([0.0, 0.0, z_lidar_offset])
        self.R_L_B = Rotation.identity()
        euler_CF_CL = np.array([-np.pi / 2.0, 0.0, -np.pi / 2.0])
        self.R_CF_CL = Rotation.from_euler('xyz', euler_CF_CL)
        x_cam_link_offset = base_length / 2.0 - 0.015 / 2.0
        z_cam_link_offset = base_height / 2.0 - 0.022 / 2.0
        self.P_CL_B = np.array([x_cam_link_offset, 0.0, z_cam_link_offset])
        self.R_CL_B = Rotation.identity()
        self.R_CF_B = self.R_CF_CL; self.P_CF_B = self.P_CL_B
        self.R_CF_L = self.R_L_B.inv() * self.R_CF_B

        self.lidar_ranges: list[float] = []; self.lidar_angle_min: float = 0.0
        self.lidar_angle_increment: float = 0.0174532925199 
        self.lidar_range_min: float = 0.12; self.lidar_range_max: float = 3.5
        
        self.posicao_atual: Point | None = None; self.orientacao_quat_atual: Quaternion | None = None
        self.robot_x: float | None = None; self.robot_y: float | None = None; self.robot_yaw: float | None = None

        self.map_resolution = 0.05; self.map_width_meters = 10.0; self.map_height_meters = 10.0
        self.map_num_cells_width = int(self.map_width_meters/self.map_resolution)
        self.map_num_cells_height = int(self.map_height_meters/self.map_resolution)
        self.map_origin_x = -self.map_width_meters/2.0; self.map_origin_y = -self.map_height_meters/2.0
        self.FREE_CELL_VALUE = 0; self.MAX_OCCUPANCY_STRENGTH = 5; self.OCCUPANCY_DECAY_RATE = 1
        self.OCCUPANCY_VISUALIZATION_THRESHOLD = 1; self.PLANNING_OBSTACLE_THRESHOLD = 3
        self.occupancy_map = np.full((self.map_num_cells_height, self.map_num_cells_width), self.FREE_CELL_VALUE, dtype=np.int8)
        self.map_display_window_width = 600; self.map_display_window_height = 600
        
        self.planned_path: list[tuple[int, int]] | None = None
        self.goal_for_current_path: Point | None = None 
        self.current_path_segment_index: int = 0
        self.WAYPOINT_REACHED_THRESHOLD_METERS = 0.25 
        self.SAFETY_RADIUS_METERS = 0.3 
        self.safety_radius_cells = int(self.SAFETY_RADIUS_METERS / self.map_resolution)
        self._astar_open_set_counter = 0 

        self.TARGET_STOPPING_DISTANCE_FROM_BASE = 0.25
        self.CENTERING_THRESHOLD_PIXELS = 10
        self.KP_ANGULAR_CENTERING = 0.008
        self.K_LINEAR_APPROACH_BASE = 0.08
        self.MAX_ANGULAR_VEL_CENTERING = 0.35
        self.MAX_ANGULAR_VEL_APPROACHING = 0.25
        
        self.final_approach_target_world_pos: Point | None = None

        self.centering_stuck_timer_start: float | None = None
        self.MAX_CENTERING_DURATION_S: float = 7.0
        self.last_centering_error_x: float | None = None
        self.no_progress_centering_counter: int = 0
        self.NO_PROGRESS_CENTERING_COUNT_THRESHOLD: int = 30 
        self.ignore_base_detection_until_ts: float = 0.0 
        self.BASE_IGNORE_COOLDOWN_S: float = 10.0

        # New parameters for forced exploration when stuck in final approach
        self.APROXIMANDO_FINAL_STUCK_THRESHOLD = 5 # Number of consecutive planning failures
        self.FORCED_EXPLORATION_DURATION_S = 15.0  # Duration of forced exploration in seconds
        self.aproximando_final_stuck_counter = 0   # Counter for planning failures in final approach
        self.forced_exploration_end_time = 0.0     # Timestamp for when forced exploration ends

        self.estado_atual = self.explorando

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Pose, '/model/prm_robot/pose', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam/labels_map', self.camera_callback, 10)

        self.get_logger().info("Robô inicializado com lógica de detecção de emperramento e exploração forçada.")
        self.timer_estado = self.create_timer(0.1, self.run_current_state) 
        self.timer_mapa = self.create_timer(0.5, self.atualizar_mapa)

    def run_current_state(self): 
        if self.estado_atual:
            self.estado_atual()

    def mudar_estado(self, novo_estado):
        estado_anterior_nome = self.estado_atual.__name__ if self.estado_atual else "Nenhum"
        self.get_logger().info(f"Mudando estado de '{estado_anterior_nome}' para '{novo_estado.__name__}'")
        if self.timer_estado is not None: self.timer_estado.destroy()
        
        if estado_anterior_nome == "navegando_para_bandeira_geral" and self.flag_estimation_timer:
            self.flag_estimation_timer.destroy(); self.flag_estimation_timer = None
            self.get_logger().info("Timer de estimação da bandeira (geral) parado.")
            
        self.estado_atual = novo_estado
        self.timer_estado = self.create_timer(0.1, self.run_current_state)

        if self.estado_atual == self.navegando_para_bandeira_geral:
            self.planned_path = None; self.current_path_segment_index = 0; self.goal_for_current_path = None
            self.final_approach_target_world_pos = None 
            if self.flag_estimation_timer is None:
                self.get_logger().info("Iniciando estimação da bandeira (geral) ao entrar em NAVEGANDO_PARA_BANDEIRA_GERAL.")
                if self.flag_detected_in_current_image and self.flag_pixel_centroid:
                    self.estimate_flag_position() 
                self.flag_estimation_timer = self.create_timer(1.0, self.estimate_flag_position)
                self.get_logger().info("Timer de estimação da bandeira (geral) iniciado.")
        elif self.estado_atual == self.centralizando_base:
            self.planned_path = None 
            self.current_path_segment_index = 0
            self.final_approach_target_world_pos = None 
            self.centering_stuck_timer_start = self.get_clock().now().nanoseconds / 1e9
            self.last_centering_error_x = None
            self.no_progress_centering_counter = 0
            self.get_logger().info("Contadores de falha de centralização reiniciados.")
        elif self.estado_atual == self.aproximando_final_base:
            self.get_logger().info("Entrando em APROXIMANDO_FINAL_BASE. O caminho será (re)planejado continuamente.")
            self.aproximando_final_stuck_counter = 0 # Reset counter on entering state
        elif self.estado_atual == self.posicionado_para_coleta:
            self.final_approach_target_world_pos = None 
            self.planned_path = None
        elif self.estado_atual == self.explorando:
            self.get_logger().info("Entrando no estado EXPLORANDO.")
            # If not already in forced exploration, ensure the end time is in the past
            # This handles transitions from other states to explorando, not just from being stuck.
            # current_time_s = self.get_clock().now().nanoseconds / 1e9 # Get current time
            # if self.forced_exploration_end_time < current_time_s : # Check if not already in forced exploration
            #    pass # No need to reset if not actively in forced exploration from APROX_FINAL
    
    def world_to_map_coords(self, world_x: float, world_y: float) -> tuple[int | None, int | None]:
        if world_x is None or world_y is None: return None, None
        map_col = int(np.floor((world_x - self.map_origin_x) / self.map_resolution))
        map_row = int(np.floor((world_y - self.map_origin_y) / self.map_resolution))
        return map_col, map_row

    def map_to_world_coords(self, map_col: int, map_row: int) -> tuple[float, float]:
        world_x = (map_col + 0.5) * self.map_resolution + self.map_origin_x
        world_y = (map_row + 0.5) * self.map_resolution + self.map_origin_y
        return world_x, world_y
        
    def get_line_cells(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        cells = []; dx = abs(x1 - x0); dy = abs(y1 - y0); sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1; err = dx - dy; curr_x, curr_y = x0, y0
        while True:
            cells.append((curr_x, curr_y))
            if curr_x == x1 and curr_y == y1: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; curr_x += sx
            if e2 < dx: err += dx; curr_y += sy
        return cells

    def scan_callback(self, msg: LaserScan):
        self.lidar_ranges = list(msg.ranges); self.lidar_angle_min = msg.angle_min
        self.lidar_angle_increment = msg.angle_increment; self.lidar_range_min = msg.range_min
        self.lidar_range_max = msg.range_max

    def imu_callback(self, msg: Imu): 
        pass # Não utilizado ativamente no código fornecido

    def odom_callback(self, msg: Pose):
        self.posicao_atual = msg.position; self.orientacao_quat_atual = msg.orientation
        if self.orientacao_quat_atual:
            rot_obj = Rotation.from_quat([
                self.orientacao_quat_atual.x, self.orientacao_quat_atual.y,
                self.orientacao_quat_atual.z, self.orientacao_quat_atual.w])
            _, _, self.robot_yaw = rot_obj.as_euler('xyz', degrees=False) 
        if self.posicao_atual:
            self.robot_x = self.posicao_atual.x; self.robot_y = self.posicao_atual.y

    def camera_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.imagem = cv_image # Store the raw image if needed elsewhere
            
            label_bandeira = 40 
            target_color_bgr = np.array([label_bandeira, label_bandeira, label_bandeira], dtype=np.uint8)
            label_chao = 10
            target_ground_color_bgr = np.array([label_chao, label_chao, label_chao], dtype=np.uint8)
            min_area_necessaria = 30 
            
            # Create mask for the flag color
            mask = cv2.inRange(cv_image, target_color_bgr, target_color_bgr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.flag_detected_in_current_image = False 
            largest_flag_contour = None
            max_area = 0
            # self.flag_pixel_centroid = None # Reset before detection
            # self.flagpole_base_pixel_centroid = None # Reset before detection
            
            for contour_candidate in contours:
                area = cv2.contourArea(contour_candidate)
                if area > min_area_necessaria and area > max_area:
                    max_area = area
                    largest_flag_contour = contour_candidate
            
            current_frame_flagpole_base_candidates = []
            if largest_flag_contour is not None:
                self.flag_detected_in_current_image = True
                M = cv2.moments(largest_flag_contour)
                if M["m00"] != 0:
                    self.flag_pixel_centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                else: 
                    self.flag_detected_in_current_image = False # Centroid calculation failed
                
                # If flag is detected, try to find its base
                if self.flag_detected_in_current_image and self.flag_pixel_centroid is not None:
                    # Iterate through points at the bottom of the contour
                    # This assumes the flagpole is somewhat vertical and the base is below the flag
                    for point_arr in largest_flag_contour:
                        x, y = point_arr[0]
                        # Check pixels directly below or slightly around the lowest points of the flag contour
                        # This logic might need refinement based on typical flag appearance
                        y_below = y + 1 # Check pixel immediately below
                        if 0 <= y_below < self.camera_image_height: # Ensure y_below is within image bounds
                            pixel_below_color = cv_image[y_below, x]
                            if np.array_equal(pixel_below_color, target_ground_color_bgr):
                                current_frame_flagpole_base_candidates.append((x,y)) # Use (x,y) of the flag point whose below is ground
                    
                    if current_frame_flagpole_base_candidates:
                        # Calculate the average of these candidate base points
                        avg_x = int(np.mean([p[0] for p in current_frame_flagpole_base_candidates]))
                        avg_y = int(np.mean([p[1] for p in current_frame_flagpole_base_candidates]))
                        self.flagpole_base_pixel_centroid = (avg_x, avg_y)
                    else:
                        self.flagpole_base_pixel_centroid = None # No base found for this flag
            
            # If no flag contour was found or centroid calculation failed
            if not self.flag_detected_in_current_image: # or self.flag_pixel_centroid is None: (already covered by flag_detected_in_current_image)
                self.flag_pixel_centroid = None
                self.flagpole_base_pixel_centroid = None


            # Visualization (optional, can be commented out for performance)
            vis_frame = cv_image.copy()
            if self.flag_detected_in_current_image and largest_flag_contour is not None:
                cv2.drawContours(vis_frame, [largest_flag_contour], -1, (0, 255, 0), 2) # Green for flag
            if self.flag_pixel_centroid:
                cv2.circle(vis_frame, self.flag_pixel_centroid, 7, (255, 100, 0), -1) # Blue for flag centroid
                cv2.circle(vis_frame, self.flag_pixel_centroid, 7, (255,255,255), 1) # White border
            if self.flagpole_base_pixel_centroid:
                cv2.circle(vis_frame, self.flagpole_base_pixel_centroid, 7, (0, 0, 255), -1) # Red for flagpole base
                cv2.circle(vis_frame, self.flagpole_base_pixel_centroid, 7, (255,255,255), 1) # White border

            cv2.imshow("Flag Detection Details", vis_frame)
            cv2.waitKey(1)

        except CvBridgeError as e: self.get_logger().error(f"Erro no CvBridge: {e}")
        except Exception as e: self.get_logger().error(f"Erro geral no camera_callback: {e}, Imagem: {self.imagem is not None}")


    def estimate_flag_position(self):
        if not self.flag_detected_in_current_image or self.flag_pixel_centroid is None: return
        if not self.lidar_ranges or self.robot_x is None or self.robot_yaw is None or self.lidar_angle_increment == 0: return
        
        px, _ = self.flag_pixel_centroid
        pixel_offset_x = px - (self.camera_image_width / 2.0)
        theta_cam_x_rad = np.arctan2(pixel_offset_x, self.camera_focal_length_x)
        V_cf = np.array([np.sin(theta_cam_x_rad), 0.0, np.cos(theta_cam_x_rad)]) # Assume flag is in front, y_cam=0
        V_L = self.R_CF_L.apply(V_cf) # Transform vector from Camera Frame to Lidar Frame
        target_angle_in_lidar_frame_rad = np.arctan2(V_L[1], V_L[0]) 
        
        best_idx = -1; min_angle_diff = float('inf')
        for idx_loop in range(len(self.lidar_ranges)):
            current_beam_angle = self.lidar_angle_min + idx_loop * self.lidar_angle_increment
            angle_diff = abs(current_beam_angle - target_angle_in_lidar_frame_rad)
            # Normalize angle difference to be within [0, pi]
            if angle_diff > np.pi: angle_diff = 2 * np.pi - angle_diff
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff; best_idx = idx_loop
        
        angle_match_tolerance = self.lidar_angle_increment * 3.0 # Allow some tolerance
        if best_idx == -1 or min_angle_diff > angle_match_tolerance: return 
            
        flag_dist_lidar = self.lidar_ranges[best_idx]
        if not (np.isfinite(flag_dist_lidar) and self.lidar_range_min < flag_dist_lidar < self.lidar_range_max): return

        # Position of the flag in Lidar frame (assuming Z_L=0 for the detected point)
        actual_beam_angle_lidar = self.lidar_angle_min + best_idx * self.lidar_angle_increment
        P_flag_L = np.array([flag_dist_lidar*np.cos(actual_beam_angle_lidar), 
                             flag_dist_lidar*np.sin(actual_beam_angle_lidar), 0.0])
        # Position of the flag in Robot Base frame
        P_flag_B = self.R_L_B.apply(P_flag_L) + self.P_L_B
        
        # Position of the flag in World frame
        flag_x_W = self.robot_x + (P_flag_B[0]*np.cos(self.robot_yaw) - P_flag_B[1]*np.sin(self.robot_yaw))
        flag_y_W = self.robot_y + (P_flag_B[0]*np.sin(self.robot_yaw) + P_flag_B[1]*np.cos(self.robot_yaw))
        # Z coordinate: robot's current Z + Z offset of flag in base frame
        flag_z_W = (self.posicao_atual.z if self.posicao_atual and self.posicao_atual.z is not None else 0.0) + P_flag_B[2]

        current_raw_estimate = Point(x=flag_x_W, y=flag_y_W, z=flag_z_W)
        self.raw_estimated_flag_positions.append(current_raw_estimate)
        if len(self.raw_estimated_flag_positions) > self.FLAG_ESTIMATION_HISTORY_SIZE: self.raw_estimated_flag_positions.pop(0)
        
        if self.raw_estimated_flag_positions:
            sum_w_x, sum_w_y, sum_w_z = 0.0, 0.0, 0.0; total_w_applied = 0.0
            num_est = len(self.raw_estimated_flag_positions)
            # Use weights corresponding to the number of available estimates
            effective_weights = self.flag_estimation_weights[:num_est] if num_est <= len(self.flag_estimation_weights) else self.flag_estimation_weights[-num_est:]
            current_total_weight_sum = np.sum(effective_weights) # Sum of weights actually being used

            if current_total_weight_sum > 1e-6: # Avoid division by zero if all weights are zero
                normalized_current_weights = effective_weights / current_total_weight_sum
            else: # Fallback to equal weighting if sum is too small (e.g. all weights zero or too few points)
                normalized_current_weights = np.ones(num_est) / num_est if num_est > 0 else []


            for i, pos in enumerate(self.raw_estimated_flag_positions):
                weight = normalized_current_weights[i] if i < len(normalized_current_weights) else (1.0/num_est if num_est > 0 else 0)
                sum_w_x += pos.x * weight; sum_w_y += pos.y * weight; sum_w_z += pos.z * weight
                total_w_applied += weight # Should be close to 1.0 if normalized_current_weights sum to 1
            
            if total_w_applied > 1e-6 : # Ensure some weight was applied
                 avg_x = sum_w_x / total_w_applied; avg_y = sum_w_y / total_w_applied; avg_z = sum_w_z / total_w_applied
                 self.filtered_estimated_flag_world_position = Point(x=avg_x, y=avg_y, z=avg_z)
            elif self.raw_estimated_flag_positions: # Fallback if weights somehow failed but positions exist
                 self.filtered_estimated_flag_world_position = self.raw_estimated_flag_positions[-1]


    def create_inflated_map(self) -> np.ndarray:
        binary_obstacle_map = (self.occupancy_map >= self.PLANNING_OBSTACLE_THRESHOLD).astype(np.uint8)
        if self.safety_radius_cells > 0:
            kernel_size = 2 * self.safety_radius_cells + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.dilate(binary_obstacle_map, kernel, iterations=1)
        return binary_obstacle_map

    def is_valid_map_coords(self, map_col: int, map_row: int) -> bool:
        return 0 <= map_col < self.map_num_cells_width and 0 <= map_row < self.map_num_cells_height

    def heuristic(self, current_coords: tuple[int,int], goal_coords: tuple[int,int]) -> float:
        # Euclidean distance
        return np.sqrt((current_coords[0] - goal_coords[0])**2 + (current_coords[1] - goal_coords[1])**2)

    def _reconstruct_astar_path(self, came_from_map: dict, current_target_node: tuple[int,int], start_node: tuple[int,int]) -> list:
        path = [current_target_node]
        node = current_target_node
        max_path_len = self.map_num_cells_width * self.map_num_cells_height # Safety break
        count = 0
        while node in came_from_map and count < max_path_len :
            node = came_from_map[node]
            path.append(node)
            if node == start_node: break
            count +=1
        if path[-1] != start_node :
             # Allow path of length 1 if start is goal
             if not (current_target_node == start_node and len(path) == 1):
                self.get_logger().error(f"A* Reconstrução: Início {start_node} não encontrado no caminho para {current_target_node}. Caminho reconstruído: {path}")
                return [] # Path is invalid if start not found (unless start is goal)
        return path[::-1] # Reverse to get start -> goal

    def is_line_collision_free(self, s_coords: tuple[int,int], e_coords: tuple[int,int], i_map: np.ndarray) -> bool:
        # Bresenham's line algorithm to get cells, then check collision
        for c, r in self.get_line_cells(s_coords[0], s_coords[1], e_coords[0], e_coords[1]):
            if not self.is_valid_map_coords(c,r) or i_map[r,c] == 1: # 1 means obstacle in inflated map
                return False 
        return True

    def smooth_path(self, path: list[tuple[int,int]], i_map: np.ndarray) -> list[tuple[int,int]]:
        if not path or len(path) < 3: return path # Not enough points to smooth
        smoothed_path = [path[0]] # Start with the first waypoint
        i = 0
        while i < len(path) - 1:
            current_waypoint_in_original_path = path[i]
            best_j = i + 1 # Default next point
            # Try to connect current_waypoint to the furthest possible waypoint j in the original path
            for j_loop in range(len(path) - 1, i + 1, -1): # Iterate backwards from end of path
                if self.is_line_collision_free(current_waypoint_in_original_path, path[j_loop], i_map):
                    best_j = j_loop # Found a direct, collision-free path
                    break 
            smoothed_path.append(path[best_j])
            i = best_j # Move to the new waypoint in the smoothed path
        return smoothed_path
        
    def plan_path_astar(self, start_map_coords: tuple[int, int], original_goal_map_coords: tuple[int, int]) -> list[tuple[int, int]] | None:
        self.get_logger().debug(f"A*: Planejando de {start_map_coords} para {original_goal_map_coords}")
        inflated_map = self.create_inflated_map()
        start_col, start_row = start_map_coords
        
        if not self.is_valid_map_coords(start_col, start_row) or \
           not self.is_valid_map_coords(original_goal_map_coords[0], original_goal_map_coords[1]):
            self.get_logger().error("A*: Coordenadas de início ou fim original fora do mapa.")
            return None
        
        if inflated_map[start_row, start_col] == 1: # Start is in an obstacle
            self.get_logger().error(f"A*: Início ({start_col},{start_row}) em obstáculo no mapa inflado.")
            return None

        open_set = []; self._astar_open_set_counter = 0 # Priority queue (min-heap)
        came_from = {} # Stores the predecessor of each node
        g_score = defaultdict(lambda: float('inf')) # Cost from start to node
        g_score[start_map_coords] = 0.0
        
        h_initial = self.heuristic(start_map_coords, original_goal_map_coords)
        f_initial = g_score[start_map_coords] + h_initial # Estimated total cost from start to goal through node
        heapq.heappush(open_set, (f_initial, self._astar_open_set_counter, start_map_coords))
        open_set_hash = {start_map_coords} # To check for presence in open_set efficiently
        path_to_return = None

        while open_set:
            _, _, current_coords = heapq.heappop(open_set) # Node with the lowest f_score
            open_set_hash.remove(current_coords)
            current_col, current_row = current_coords

            # Goal reached?
            if current_coords == original_goal_map_coords:
                self.get_logger().debug("A*: Caminho direto para o objetivo original encontrado.")
                path_to_return = self._reconstruct_astar_path(came_from, current_coords, start_map_coords)
                # If goal itself is an obstacle (but we reached it, meaning it was not in inflated_map during search for *itself*),
                # but if we want to stop *before* an obstacle goal, we might pop the last element.
                # Current logic: if goal is obstacle, path might be empty if it's the only point.
                if inflated_map[original_goal_map_coords[1], original_goal_map_coords[0]] == 1 and path_to_return and len(path_to_return) > 1:
                    self.get_logger().info("A*: Objetivo original é obstáculo, planejando para célula adjacente (removendo último ponto).")
                    path_to_return.pop() # Remove the goal itself if it's an obstacle
                if not path_to_return: self.get_logger().warn("A*: Caminho para objetivo original ficou vazio após ajuste.")
                break # Path found

            # Explore neighbors
            for d_col, d_row in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]: # 8-connectivity
                neighbor_col, neighbor_row = current_col + d_col, current_row + d_row
                neighbor_coords = (neighbor_col, neighbor_row)

                if not self.is_valid_map_coords(neighbor_col, neighbor_row): continue

                is_neighbor_the_goal = (neighbor_coords == original_goal_map_coords)
                # If neighbor is an obstacle AND it's not the goal itself, skip it.
                # (We can plan *to* an obstacle, but not *through* one unless it's the goal)
                if inflated_map[neighbor_row, neighbor_col] == 1 and not is_neighbor_the_goal: continue 
                
                cost = 1.0 if abs(d_col) + abs(d_row) == 1 else np.sqrt(2) # Cost for diagonal/straight
                tentative_g_score = g_score[current_coords] + cost

                if tentative_g_score < g_score[neighbor_coords]:
                    came_from[neighbor_coords] = current_coords
                    g_score[neighbor_coords] = tentative_g_score
                    h_val = self.heuristic(neighbor_coords, original_goal_map_coords)
                    f_val = tentative_g_score + h_val
                    if neighbor_coords not in open_set_hash:
                        self._astar_open_set_counter += 1 # Tie-breaker for heapq
                        heapq.heappush(open_set, (f_val, self._astar_open_set_counter, neighbor_coords))
                        open_set_hash.add(neighbor_coords)
        
        if path_to_return:
            self.get_logger().debug(f"A*: Caminho final para objetivo original com {len(path_to_return)} pontos.")
            return path_to_return

        # If no direct path to original_goal_map_coords was found
        self.get_logger().warn(f"A*: Nenhum caminho direto para {original_goal_map_coords}. Tentando ponto mais próximo.")
        # Fallback: find the explored node closest to the original goal
        reachable_nodes_with_finite_g = {node: score for node, score in g_score.items() if score != float('inf')}
        if not reachable_nodes_with_finite_g:
             self.get_logger().error("A*: Nenhum nó alcançável com g_score finito para fallback.")
             return None

        closest_node = None
        min_h = float('inf')
        # Find node in came_from (i.e. explored) that is closest to original_goal_map_coords
        for node, current_g_score_node in reachable_nodes_with_finite_g.items():
            # Ensure the node itself is not an obstacle (unless it's the start, which was checked)
            if inflated_map[node[1], node[0]] == 1 and node != start_map_coords:
                continue

            h = self.heuristic(node, original_goal_map_coords)
            if h < min_h:
                min_h = h
                closest_node = node
            elif h == min_h: # Tie-breaking: prefer shorter g_score
                if current_g_score_node < g_score.get(closest_node, float('inf')): # Ensure closest_node is in g_score
                    closest_node = node
        
        if closest_node is None:
            self.get_logger().warn("A*: Não foi possível encontrar um nó alcançável alternativo (fallback) que não seja obstáculo."); return None
        
        # Avoid returning only the start node if it's the "closest" but no actual path was made
        if closest_node == start_map_coords and start_map_coords != original_goal_map_coords : # and len(came_from)==0:
             self.get_logger().warn(f"A*: Ponto mais próximo {closest_node} é o início. Objetivo {original_goal_map_coords} inacessível."); return None


        self.get_logger().info(f"A*: Redirecionando para o ponto alcançável mais próximo: {closest_node} (dist heurística: {min_h:.2f})")
        path_to_closest = self._reconstruct_astar_path(came_from, closest_node, start_map_coords)
        
        if not path_to_closest:
            self.get_logger().error(f"A*: Falha ao reconstruir caminho para o ponto mais próximo {closest_node}."); return None
            
        self.get_logger().info(f"Caminho A* para ponto mais próximo com {len(path_to_closest)} pontos.")
        return path_to_closest


    def set_goal_and_plan_path_world_coords(self, goal_world_x: float, goal_world_y: float, goal_world_z: float = 0.0):
        goal_map_col, goal_map_row = self.world_to_map_coords(goal_world_x, goal_world_y)
        if goal_map_col is None or goal_map_row is None: 
            self.get_logger().error(f"Objetivo ({goal_world_x},{goal_world_y}) fora do mapa."); return
        
        intended_world_goal = Point(x=goal_world_x, y=goal_world_y, z=goal_world_z)
        self.set_goal_and_plan_path_map_coords(goal_map_col, goal_map_row, intended_world_target_for_replan_check=intended_world_goal)

    def set_goal_and_plan_path_map_coords(self, goal_map_col: int, goal_map_row: int, intended_world_target_for_replan_check: Point | None = None):
        if self.robot_x is None or self.robot_y is None: self.get_logger().warn("Posição robô desconhecida."); return
        start_map_col, start_map_row = self.world_to_map_coords(self.robot_x, self.robot_y)
        if start_map_col is None or start_map_row is None: self.get_logger().warn("Robô fora do mapa."); return
        
        # Attempt to plan path using A*
        self.planned_path = self.plan_path_astar((start_map_col, start_map_row), (goal_map_col, goal_map_row))
        
        if self.planned_path and len(self.planned_path) > 0:
            inflated_map_for_smoothing = self.create_inflated_map() # Get fresh inflated map for smoothing
            self.planned_path = self.smooth_path(self.planned_path, inflated_map_for_smoothing)
            self.get_logger().info(f"Caminho planejado e suavizado com {len(self.planned_path)} pontos.")
            self.current_path_segment_index = 0 # Reset path segment index for new path
            
            # Set the goal_for_current_path based on the intended world target or the end of the planned path
            if (self.estado_atual == self.navegando_para_bandeira_geral or self.estado_atual == self.aproximando_final_base) \
               and intended_world_target_for_replan_check is not None:
                 self.goal_for_current_path = intended_world_target_for_replan_check
            elif self.planned_path : # Ensure planned_path is still valid after smoothing
                 final_goal_map_coords_of_path = self.planned_path[-1]
                 wx, wy = self.map_to_world_coords(final_goal_map_coords_of_path[0], final_goal_map_coords_of_path[1])
                 current_z = intended_world_target_for_replan_check.z if intended_world_target_for_replan_check else 0.0
                 self.goal_for_current_path = Point(x=wx, y=wy, z=current_z)
            else: # Should not happen if planned_path was valid before
                 self.goal_for_current_path = None
        else: 
            self.get_logger().warn("Falha ao planejar caminho ou caminho vazio.")
            self.planned_path = None # Ensure it's None if planning failed
            self.goal_for_current_path = None


    def atualizar_mapa(self):
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None or not self.lidar_ranges: return
        robot_map_col, robot_map_row = self.world_to_map_coords(self.robot_x, self.robot_y)
        if robot_map_col is None or robot_map_row is None: return # Robot is off the map
        
        # Update occupancy map based on LiDAR scans
        for i_loop, distance in enumerate(self.lidar_ranges): 
            beam_angle_robot_frame = self.lidar_angle_min + i_loop * self.lidar_angle_increment
            beam_angle_world = self.robot_yaw + beam_angle_robot_frame # Angle in world frame
            effective_distance = self.lidar_range_max # Assume max range if no hit or invalid
            is_obstacle_hit = False

            if np.isfinite(distance) and distance >= self.lidar_range_min:
                if distance < self.lidar_range_max: # Valid hit within range
                    effective_distance = distance
                    is_obstacle_hit = True
            
            # Calculate endpoint of the beam in world coordinates
            endpoint_x_world = self.robot_x + effective_distance * np.cos(beam_angle_world)
            endpoint_y_world = self.robot_y + effective_distance * np.sin(beam_angle_world)
            endpoint_map_col, endpoint_map_row = self.world_to_map_coords(endpoint_x_world, endpoint_y_world)
            
            if endpoint_map_col is not None and endpoint_map_row is not None:
                # Get all cells along the line from robot to endpoint (Bresenham's)
                line_cells = self.get_line_cells(robot_map_col, robot_map_row, endpoint_map_col, endpoint_map_row)
                for cell_idx, (cell_col, cell_row) in enumerate(line_cells):
                    if not self.is_valid_map_coords(cell_col, cell_row): continue

                    is_this_cell_the_obstacle_endpoint = (is_obstacle_hit and cell_idx == len(line_cells) -1)

                    if is_this_cell_the_obstacle_endpoint: # This is the cell where an obstacle was hit
                        self.occupancy_map[cell_row, cell_col] = min(self.MAX_OCCUPANCY_STRENGTH, self.occupancy_map[cell_row, cell_col] + 2) # Increase occupancy
                    else: # This cell is part of free space along the beam
                        self.occupancy_map[cell_row, cell_col] = max(self.FREE_CELL_VALUE, self.occupancy_map[cell_row, cell_col] - self.OCCUPANCY_DECAY_RATE) # Decrease occupancy (or keep free)
        
        # Ensure occupancy values are within bounds
        self.occupancy_map = np.clip(self.occupancy_map, self.FREE_CELL_VALUE, self.MAX_OCCUPANCY_STRENGTH)

        # --- OpenCV Map Visualization ---
        if hasattr(self, 'occupancy_map') and self.occupancy_map.size > 0:
            # Create a displayable version of the map
            temp_map_for_display = np.copy(self.occupancy_map)
            display_map_content = np.zeros_like(temp_map_for_display, dtype=np.uint8)
            
            # Free cells are white
            display_map_content[temp_map_for_display == self.FREE_CELL_VALUE] = 255 
            
            # Occupied cells are shades of gray (darker for higher occupancy)
            occupied_mask = temp_map_for_display > self.FREE_CELL_VALUE
            if self.MAX_OCCUPANCY_STRENGTH > self.FREE_CELL_VALUE: # Avoid division by zero
                normalized_strength = (temp_map_for_display[occupied_mask] - self.FREE_CELL_VALUE) / \
                                      float(self.MAX_OCCUPANCY_STRENGTH - self.FREE_CELL_VALUE)
                display_map_content[occupied_mask] = 200 * (1 - normalized_strength) # Darker for higher values
            
            # Flip map for display (OpenCV origin is top-left, map origin is bottom-left)
            display_map_content_flipped = np.flipud(display_map_content)
            
            map_h, map_w = display_map_content_flipped.shape
            # Create a canvas for display, scaled to window size
            canvas = np.full((self.map_display_window_height, self.map_display_window_width), 128, dtype=np.uint8) # Gray background
            
            # Calculate scaling factor to fit map in window while preserving aspect ratio
            scale_h = self.map_display_window_height/map_h if map_h > 0 else 1
            scale_w = self.map_display_window_width/map_w if map_w > 0 else 1
            scale = min(scale_h, scale_w)
            new_content_h = int(map_h * scale)
            new_content_w = int(map_w * scale)
            start_x, start_y = 0,0 # Top-left corner for placing the scaled map on canvas
            
            if new_content_w > 0 and new_content_h > 0:
                resized_map_content = cv2.resize(display_map_content_flipped, (new_content_w, new_content_h), interpolation=cv2.INTER_NEAREST)
                # Center the map on the canvas
                start_x = (self.map_display_window_width - new_content_w) // 2
                start_y = (self.map_display_window_height - new_content_h) // 2
                canvas[start_y:start_y+new_content_h, start_x:start_x+new_content_w] = resized_map_content
            
            # Convert to BGR for drawing colored elements
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

            # Draw robot position
            if self.robot_x is not None and self.robot_y is not None: 
                robot_map_col_viz, robot_map_row_viz_unflipped = self.world_to_map_coords(self.robot_x, self.robot_y)
                if robot_map_col_viz is not None and robot_map_row_viz_unflipped is not None:
                    robot_map_row_viz_flipped = map_h - 1 - robot_map_row_viz_unflipped # Account for y-flip
                    # Scale and shift to canvas coordinates
                    robot_canvas_x = int((robot_map_col_viz/map_w)*new_content_w + start_x)
                    robot_canvas_y = int((robot_map_row_viz_flipped/map_h)*new_content_h + start_y)
                    if 0 <= robot_canvas_x < self.map_display_window_width and \
                       0 <= robot_canvas_y < self.map_display_window_height:
                        cv2.circle(canvas_bgr, (robot_canvas_x, robot_canvas_y), 5, (0,0,255), -1) # Red circle for robot
            
            # Draw planned path
            if self.planned_path: 
                for i_path in range(len(self.planned_path) - 1): 
                    p1_map_col, p1_map_row_unflipped = self.planned_path[i_path]
                    p2_map_col, p2_map_row_unflipped = self.planned_path[i_path+1]
                    
                    p1_map_row_flipped = map_h - 1 - p1_map_row_unflipped
                    p2_map_row_flipped = map_h - 1 - p2_map_row_unflipped
                    
                    p1_c_x = int((p1_map_col/map_w)*new_content_w+start_x); p1_c_y = int((p1_map_row_flipped/map_h)*new_content_h+start_y)
                    p2_c_x = int((p2_map_col/map_w)*new_content_w+start_x); p2_c_y = int((p2_map_row_flipped/map_h)*new_content_h+start_y)
                    cv2.line(canvas_bgr, (p1_c_x,p1_c_y), (p2_c_x,p2_c_y), (0,255,0), 2) # Green line for path
            
            # Draw estimated flag position
            if self.filtered_estimated_flag_world_position: 
                f_map_c, f_map_ru = self.world_to_map_coords(self.filtered_estimated_flag_world_position.x, self.filtered_estimated_flag_world_position.y)
                if f_map_c is not None and f_map_ru is not None:
                    f_map_rf = map_h - 1 - f_map_ru # Account for y-flip
                    f_c_x = int((f_map_c/map_w)*new_content_w+start_x); f_c_y = int((f_map_rf/map_h)*new_content_h+start_y)
                    if 0 <= f_c_x < self.map_display_window_width and \
                       0 <= f_c_y < self.map_display_window_height:
                        cv2.drawMarker(canvas_bgr, (f_c_x,f_c_y), (255,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2) # Magenta cross for flag
            
            cv2.imshow("Mapa de Ocupacao (OpenCV)", canvas_bgr)
            key = cv2.waitKey(1) & 0xFF
            # Manual goal setting via terminal (for debugging)
            if key == ord('p'): 
                try:
                    gx_str = input("X do objetivo (mundo): "); gy_str = input("Y do objetivo (mundo): ")
                    manual_goal_z = self.posicao_atual.z if self.posicao_atual else 0.0
                    self.set_goal_and_plan_path_world_coords(float(gx_str), float(gy_str), manual_goal_z)
                except ValueError: self.get_logger().error("Entrada inválida para objetivo manual.")

    def explorando(self):
        current_time_s = self.get_clock().now().nanoseconds / 1e9
        log_prefix = "EXPLORANDO: "

        if current_time_s < self.forced_exploration_end_time:
            self.get_logger().info(f"{log_prefix}Em modo de exploração forçada. Restam {self.forced_exploration_end_time - current_time_s:.1f}s. Ignorando detecção de bandeira.")
            # Flag detection is skipped during forced exploration
        elif self.flag_detected_in_current_image and self.flag_pixel_centroid:
            self.get_logger().info(f"{log_prefix}Bandeira detectada. Transicionando para NAVEGANDO_PARA_BANDEIRA_GERAL.")
            self.mudar_estado(self.navegando_para_bandeira_geral)
            return 
        
        # Standard exploration behavior
        if not self.lidar_ranges: 
            self.cmd_vel_pub.publish(Twist()) # Stop if no lidar
            return

        distancias_frontais = []
        if len(self.lidar_ranges) > 0:
            angulo_cone_frontal_rad = np.deg2rad(60) # Reduced cone for more direct obstacle check
            # Calculate indices for the frontal cone more precisely
            center_index = len(self.lidar_ranges) // 2
            angle_span_indices = int(round( (angulo_cone_frontal_rad / 2.0) / self.lidar_angle_increment )) if self.lidar_angle_increment > 1e-6 else 0
            start_idx = max(0, center_index - angle_span_indices)
            end_idx = min(len(self.lidar_ranges), center_index + angle_span_indices +1)

            for i in range(start_idx, end_idx):
                r = self.lidar_ranges[i]
                # angle = self.lidar_angle_min + i * self.lidar_angle_increment # Not strictly needed if using indices correctly
                if np.isfinite(r) and self.lidar_range_min < r < self.lidar_range_max:
                    distancias_frontais.append(r)
        
        obstaculo_a_frente = False
        distancia_seguranca_expl = 0.5 # Slightly reduced for tighter spaces
        if distancias_frontais and min(distancias_frontais) < distancia_seguranca_expl: 
            obstaculo_a_frente = True
            self.get_logger().debug(f"{log_prefix}Obstáculo frontal detectado a {min(distancias_frontais):.2f}m.")
        
        twist = Twist()
        if not obstaculo_a_frente: 
            twist.linear.x = 0.15 # Reduced speed for more careful exploration
            twist.angular.z = 0.0
            self.get_logger().debug(f"{log_prefix}Movendo para frente.")
        else: 
            twist.linear.x = 0.0; 
            twist.angular.z = -0.40 # Consistent turn direction
            self.get_logger().debug(f"{log_prefix}Obstáculo! Girando.")
        self.cmd_vel_pub.publish(twist)


    def navegando_para_bandeira_geral(self):
        log_prefix = "NAV_GERAL: "
        twist = Twist()
        current_time_s = self.get_clock().now().nanoseconds / 1e9

        # Check for flagpole base detection (if not in cooldown)
        if self.flagpole_base_pixel_centroid is not None and current_time_s > self.ignore_base_detection_until_ts:
            self.get_logger().info(f"{log_prefix}Base da haste detectada. Transicionando para CENTRALIZANDO_BASE.")
            self.mudar_estado(self.centralizando_base)
            self.cmd_vel_pub.publish(twist) # Stop robot before state change
            return
        elif self.flagpole_base_pixel_centroid is not None and current_time_s <= self.ignore_base_detection_until_ts:
            self.get_logger().debug(f"{log_prefix}Base da haste detectada, mas em cooldown de centralização. Ignorando por agora.")
        
        # Check if flag is still visible
        if not self.flag_detected_in_current_image or self.flag_pixel_centroid is None:
            self.get_logger().warn(f"{log_prefix}Bandeira (geral) perdida de vista! Voltando a explorar.")
            self.mudar_estado(self.explorando)
            self.cmd_vel_pub.publish(twist) # Stop robot
            return

        # Check if filtered flag position is available
        if self.filtered_estimated_flag_world_position is None:
            self.get_logger().warn(f"{log_prefix}Posição da bandeira (filtrada) desconhecida. Aguardando estimativa (girando lentamente).")
            twist.angular.z = 0.1 # Gentle turn to try and re-acquire
            self.cmd_vel_pub.publish(twist)
            return

        # Determine if re-planning is needed
        needs_replan = False
        if self.planned_path is None or not self.planned_path: 
            needs_replan = True
            self.get_logger().info(f"{log_prefix}Sem caminho planejado. Necessário replanejar.")
        elif self.goal_for_current_path is None: 
            needs_replan = True
            self.get_logger().info(f"{log_prefix}Sem objetivo para o caminho atual. Necessário replanejar.")
        else: 
            # Check if the estimated flag position has moved significantly from the current path's goal
            dist_sq = (self.filtered_estimated_flag_world_position.x - self.goal_for_current_path.x)**2 + \
                      (self.filtered_estimated_flag_world_position.y - self.goal_for_current_path.y)**2
            # Replan if flag moved more than, e.g., 3 map cells away from path goal
            if dist_sq > (self.map_resolution * 5)**2: # Increased threshold for replan
                needs_replan = True
                self.get_logger().info(f"{log_prefix}Estimativa da bandeira mudou significativamente (dist_sq={dist_sq:.2f}). Necessário replanejar.")

        if needs_replan:
            self.get_logger().info(f"{log_prefix}Re-planejando para posição atualizada da bandeira (geral): "
                                   f"X={self.filtered_estimated_flag_world_position.x:.2f}, "
                                   f"Y={self.filtered_estimated_flag_world_position.y:.2f}")
            flag_z = self.filtered_estimated_flag_world_position.z if self.filtered_estimated_flag_world_position.z is not None else 0.0
            self.set_goal_and_plan_path_world_coords(
                self.filtered_estimated_flag_world_position.x,
                self.filtered_estimated_flag_world_position.y,
                flag_z
            )
            if not self.planned_path: # Planning failed
                self.get_logger().warn(f"{log_prefix}Falha no re-planejamento. Robô parado.")
                self.cmd_vel_pub.publish(twist) # Stop robot
                return
        
        # Follow the planned path
        if self.planned_path and self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None:
            if self.current_path_segment_index >= len(self.planned_path):
                self.get_logger().info(f"{log_prefix}Chegou ao final do caminho para a bandeira (geral). Aguardando detecção da base ou novo replanejamento.")
                # Robot will stop here if path ends, and replan if flag moves
                self.cmd_vel_pub.publish(twist) # Stop robot
                return

            # Get current waypoint
            target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
            target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
            
            dx = target_world_x - self.robot_x; dy = target_world_y - self.robot_y
            distance_to_waypoint = np.sqrt(dx*dx + dy*dy)

            # Check if waypoint reached
            if distance_to_waypoint < self.WAYPOINT_REACHED_THRESHOLD_METERS:
                self.get_logger().info(f"{log_prefix}Waypoint {self.current_path_segment_index}/{len(self.planned_path)-1} alcançado.")
                self.current_path_segment_index += 1
                if self.current_path_segment_index >= len(self.planned_path): 
                    self.get_logger().info(f"{log_prefix}Fim do caminho A* alcançado (perto da bandeira geral).")
                    self.cmd_vel_pub.publish(twist); return # Stop and wait for next cycle/replan
                # Update target to next waypoint
                target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
                target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
                dx = target_world_x - self.robot_x; dy = target_world_y - self.robot_y # Recalculate dx, dy
            
            # Proportional controller for navigation
            angle_to_waypoint = np.arctan2(dy, dx)
            angle_diff = angle_to_waypoint - self.robot_yaw
            # Normalize angle_diff to [-pi, pi]
            while angle_diff > np.pi: angle_diff -= 2*np.pi
            while angle_diff < -np.pi: angle_diff += 2*np.pi
            
            K_angular = 0.8; K_linear_max_nav = 0.15 # Slightly adjusted gains
            
            twist.angular.z = K_angular * angle_diff
            
            # Reduce linear speed if not aligned
            if abs(angle_diff) < np.deg2rad(25): # Stricter alignment for full speed
                reduction_factor = max(0.2, 1 - abs(angle_diff)/(np.pi/4)) # Smoother reduction
                twist.linear.x = K_linear_max_nav * reduction_factor
            else: 
                twist.linear.x = 0.0 # Turn in place if too misaligned
            
            # Clip velocities
            twist.angular.z = np.clip(twist.angular.z, -0.45, 0.45) # Slightly higher max angular
            twist.linear.x = np.clip(twist.linear.x, 0.0, K_linear_max_nav) # Ensure non-negative linear
            self.get_logger().debug(f"{log_prefix}Seguindo caminho. Waypoint {self.current_path_segment_index}. LinVel: {twist.linear.x:.2f}, AngVel: {twist.angular.z:.2f}")
        else: 
            self.get_logger().info(f"{log_prefix}Sem caminho A* ou pose do robô. Parando e tentando girar para localizar.")
            twist.angular.z = 0.15 # Gentle turn
        self.cmd_vel_pub.publish(twist)


    def centralizando_base(self):
        log_prefix = "CENTR_BASE: "
        twist = Twist()
        current_time_s = self.get_clock().now().nanoseconds / 1e9

        if self.flagpole_base_pixel_centroid is None:
            self.get_logger().warn(f"{log_prefix}Base da haste perdida. Voltando para NAV_GERAL.")
            self.mudar_estado(self.navegando_para_bandeira_geral)
            self.cmd_vel_pub.publish(twist)
            return

        if self.centering_stuck_timer_start is None: # Should be set on state entry
            self.get_logger().error(f"{log_prefix}Timer de início da centralização não configurado! Resetando agora.")
            self.centering_stuck_timer_start = current_time_s
            self.last_centering_error_x = None
            self.no_progress_centering_counter = 0
            
        time_elapsed_centering = current_time_s - self.centering_stuck_timer_start
        center_x_image = self.camera_image_width / 2.0
        error_x = center_x_image - self.flagpole_base_pixel_centroid[0]

        # Stuck detection logic
        if self.last_centering_error_x is not None:
            # Progress is made if error reduces significantly, or changes sign (overshot)
            if abs(error_x) < abs(self.last_centering_error_x) - 1.0 or \
               (error_x * self.last_centering_error_x < 0 and abs(error_x) > 1.0) : # Made progress or overshot
                self.no_progress_centering_counter = 0
            else:
                self.no_progress_centering_counter += 1
        else: # First iteration with error
            self.no_progress_centering_counter = 0
        self.last_centering_error_x = error_x

        stuck_by_timeout = time_elapsed_centering > self.MAX_CENTERING_DURATION_S
        stuck_by_no_progress = self.no_progress_centering_counter > self.NO_PROGRESS_CENTERING_COUNT_THRESHOLD

        if stuck_by_timeout or stuck_by_no_progress:
            reason = "timeout" if stuck_by_timeout else "sem progresso"
            self.get_logger().warn(f"{log_prefix}Falha ao centralizar base ({reason}). T: {time_elapsed_centering:.1f}s, S/P Cnt: {self.no_progress_centering_counter}. "
                                   f"Voltando para NAV_GERAL e ativando cooldown para detecção de base.")
            self.ignore_base_detection_until_ts = current_time_s + self.BASE_IGNORE_COOLDOWN_S
            self.mudar_estado(self.navegando_para_bandeira_geral)
            self.cmd_vel_pub.publish(twist) # Stop robot
            return

        # If centered, calculate final approach target and change state
        if abs(error_x) < self.CENTERING_THRESHOLD_PIXELS:
            self.get_logger().info(f"{log_prefix}Base da haste centralizada (erro {error_x:.2f} px). Calculando Posição Final para Aproximação.")
            idx_zero_degree = -1; dist_to_base = -1.0
            if self.lidar_angle_increment > 1e-9 and len(self.lidar_ranges) > 0:
                # Find index of LiDAR beam closest to robot's forward direction (0 radians in robot frame)
                idx_zero_degree = int(round(-self.lidar_angle_min / self.lidar_angle_increment))
            
            if 0 <= idx_zero_degree < len(self.lidar_ranges):
                dist_to_base = self.lidar_ranges[idx_zero_degree]

            if np.isfinite(dist_to_base) and dist_to_base >= self.lidar_range_min and \
               self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None:
                
                # Calculate world coordinates of the point directly in front of the robot at dist_to_base
                # This assumes the flagpole base is what the LiDAR sees directly in front
                base_x_W = self.robot_x + dist_to_base * np.cos(self.robot_yaw)
                base_y_W = self.robot_y + dist_to_base * np.sin(self.robot_yaw)
                # Try to use Z from flag estimation, otherwise robot's current Z
                base_z_W = 0.0 
                if self.filtered_estimated_flag_world_position and self.filtered_estimated_flag_world_position.z is not None:
                    base_z_W = self.filtered_estimated_flag_world_position.z 
                elif self.posicao_atual and self.posicao_atual.z is not None: 
                    base_z_W = self.posicao_atual.z
                
                self.final_approach_target_world_pos = Point(x=base_x_W, y=base_y_W, z=base_z_W)
                self.get_logger().info(f"{log_prefix}Posição final da base calculada: X={base_x_W:.2f}, Y={base_y_W:.2f}, Z={base_z_W:.2f} (Dist LiDAR: {dist_to_base:.2f}m)")
                self.mudar_estado(self.aproximando_final_base)
            else:
                self.get_logger().warn(f"{log_prefix}Não foi possível obter distância LiDAR válida ({dist_to_base:.2f}) ou pose para calcular alvo final. Tentando recentralizar.")
                # Continue trying to center if LiDAR data is bad
                twist.angular.z = self.KP_ANGULAR_CENTERING * error_x 
                twist.angular.z = np.clip(twist.angular.z, -self.MAX_ANGULAR_VEL_CENTERING, self.MAX_ANGULAR_VEL_CENTERING)
        else: # Not centered yet, continue centering
            twist.angular.z = self.KP_ANGULAR_CENTERING * error_x
            twist.angular.z = np.clip(twist.angular.z, -self.MAX_ANGULAR_VEL_CENTERING, self.MAX_ANGULAR_VEL_CENTERING)
            self.get_logger().info(f"{log_prefix}Centralizando... Erro X: {error_x:.2f}, Ang Vel: {twist.angular.z:.3f} "
                                   f"(Decorrido: {time_elapsed_centering:.1f}s / {self.MAX_CENTERING_DURATION_S:.1f}s, "
                                   f"S/P Cnt: {self.no_progress_centering_counter}/{self.NO_PROGRESS_CENTERING_COUNT_THRESHOLD})")
        
        twist.linear.x = 0.0 # No linear movement while centering
        self.cmd_vel_pub.publish(twist)


    def aproximando_final_base(self):
        log_prefix = "APROX_FINAL_ASTAR: "
        twist = Twist()
        current_time_s = self.get_clock().now().nanoseconds / 1e9

        if self.flagpole_base_pixel_centroid is None:
            self.get_logger().warn(f"{log_prefix}Base da haste perdida visualmente durante aproximação A*. Continuando com o caminho planejado para o último alvo conhecido.")

        if self.final_approach_target_world_pos is None:
            self.get_logger().error(f"{log_prefix}Alvo final da base não definido. Voltando para NAV_GERAL.")
            self.mudar_estado(self.navegando_para_bandeira_geral)
            self.cmd_vel_pub.publish(twist)
            return

        self.get_logger().debug(f"{log_prefix}Tentando planejar/re-planejar caminho A* para o alvo final da base: "
                               f"X={self.final_approach_target_world_pos.x:.2f}, "
                               f"Y={self.final_approach_target_world_pos.y:.2f}, "
                               f"Z={self.final_approach_target_world_pos.z:.2f}")
        self.set_goal_and_plan_path_world_coords(
            self.final_approach_target_world_pos.x,
            self.final_approach_target_world_pos.y,
            self.final_approach_target_world_pos.z
        )

        if not self.planned_path:
            self.get_logger().warn(f"{log_prefix}Falha ao planejar/re-planejar caminho A* para o alvo final.")
            self.aproximando_final_stuck_counter += 1
            self.get_logger().info(f"{log_prefix}Contador de falhas de planejamento na aproximação final: {self.aproximando_final_stuck_counter}/{self.APROXIMANDO_FINAL_STUCK_THRESHOLD}")

            if self.aproximando_final_stuck_counter >= self.APROXIMANDO_FINAL_STUCK_THRESHOLD:
                self.get_logger().warn(f"{log_prefix}Muitas falhas consecutivas no planejamento para o alvo final. "
                                       f"Voltando para exploração forçada por {self.FORCED_EXPLORATION_DURATION_S}s.")
                self.forced_exploration_end_time = current_time_s + self.FORCED_EXPLORATION_DURATION_S
                self.aproximando_final_stuck_counter = 0 # Reset counter
                self.final_approach_target_world_pos = None # Limpar o alvo para não tentar voltar imediatamente
                self.planned_path = None                    # Limpar o caminho
                self.goal_for_current_path = None           # Limpar o objetivo do caminho
                self.mudar_estado(self.explorando)
                self.cmd_vel_pub.publish(Twist()) # Para o robô
                return
            
            self.cmd_vel_pub.publish(Twist()) # Publica Twist() que tem velocidades zero, aguarda próximo ciclo
            return
        else:
            # Se o planejamento foi bem-sucedido, resetar o contador de falhas
            if self.aproximando_final_stuck_counter > 0:
                 self.get_logger().info(f"{log_prefix}Planejamento para alvo final bem-sucedido. Resetando contador de falhas de aproximação.")
                 self.aproximando_final_stuck_counter = 0
        
        if self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None:
            if self.current_path_segment_index >= len(self.planned_path):
                self.get_logger().info(f"{log_prefix}Chegou ao final do caminho A* para a base da bandeira.")
                self.mudar_estado(self.posicionado_para_coleta)
                self.cmd_vel_pub.publish(twist) 
                return

            target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
            target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
            
            dx = target_world_x - self.robot_x
            dy = target_world_y - self.robot_y
            distance_to_waypoint = np.sqrt(dx*dx + dy*dy)

            if distance_to_waypoint < self.WAYPOINT_REACHED_THRESHOLD_METERS:
                self.get_logger().info(f"{log_prefix}Waypoint A* {self.current_path_segment_index} (de {len(self.planned_path)-1}) alcançado.")
                self.current_path_segment_index += 1
                if self.current_path_segment_index >= len(self.planned_path):
                    self.get_logger().info(f"{log_prefix}Fim do caminho A* para base alcançado após avançar waypoint.")
                    self.mudar_estado(self.posicionado_para_coleta)
                    self.cmd_vel_pub.publish(twist)
                    return
                target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
                target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
                dx = target_world_x - self.robot_x; dy = target_world_y - self.robot_y
            
            angle_to_waypoint = np.arctan2(dy, dx)
            angle_diff = angle_to_waypoint - self.robot_yaw
            while angle_diff > np.pi: angle_diff -= 2*np.pi
            while angle_diff < -np.pi: angle_diff += 2*np.pi
            
            K_angular_final_approach = 0.8 
            K_linear_final_approach = self.K_LINEAR_APPROACH_BASE
            
            twist.angular.z = K_angular_final_approach * angle_diff
            
            if abs(angle_diff) < np.deg2rad(20): 
                twist.linear.x = K_linear_final_approach
            elif abs(angle_diff) < np.deg2rad(45): 
                twist.linear.x = K_linear_final_approach * 0.5
            else: 
                twist.linear.x = 0.0
            
            twist.angular.z = np.clip(twist.angular.z, -self.MAX_ANGULAR_VEL_APPROACHING, self.MAX_ANGULAR_VEL_APPROACHING)
            twist.linear.x = np.clip(twist.linear.x, 0.0, K_linear_final_approach)
            
            self.get_logger().debug(f"{log_prefix}Seguindo A* para base. Waypoint {self.current_path_segment_index}/{len(self.planned_path)-1}. "
                                    f"Dist: {distance_to_waypoint:.2f}m, AngDiff: {np.rad2deg(angle_diff):.1f}deg. "
                                    f"LinVel: {twist.linear.x:.2f}, AngVel: {twist.angular.z:.2f}")
        else: 
            if not (self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None):
                 self.get_logger().warn(f"{log_prefix}Pose do robô desconhecida. Não é possível seguir o caminho A*.")
            self.cmd_vel_pub.publish(twist) 
            return
            
        self.cmd_vel_pub.publish(twist)

    def posicionado_para_coleta(self):
        log_prefix = "POS_COLETA: "
        self.get_logger().info(f"{log_prefix}Robô posicionado para coleta da bandeira. Nenhuma ação adicional implementada.")
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        pass

    def capturando_bandeira(self): 
        pass

    def retornando_pra_base(self): 
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ControleRobo()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        node.get_logger().info("Nó interrompido (Ctrl+C).")
    finally:
        node.get_logger().info("Parando robô...")
        stop_twist = Twist()
        if rclpy.ok() and hasattr(node, 'cmd_vel_pub') and node.cmd_vel_pub is not None and node.cmd_vel_pub.get_subscription_count() > 0:
             node.cmd_vel_pub.publish(stop_twist)
        cv2.destroyAllWindows()
        if node.is_valid(): 
            node.destroy_node()
        if rclpy.ok(): 
            rclpy.shutdown()
        node.get_logger().info("Nó encerrado.")

if __name__ == '__main__':
    main()
