#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import heapq
from collections import defaultdict, deque

from scipy.spatial.transform import Rotation

class ControleRobo(Node):
    def __init__(self):
        super().__init__('controle_robo')

        self.bridge = CvBridge()
        self.estado_atual = self.indo_para_zona_alvo 
        self.imagem = None 

        # Variáveis de deteção da bandeira
        self.flag_pixel_centroid: tuple[int, int] | None = None
        self.flagpole_base_pixel_centroid: tuple[int, int] | None = None
        self.flag_detected_in_current_image: bool = False
        
        # Variáveis para a sequência de captura e exploração
        self.capture_sequence_step = 0
        self.sequence_timer: rclpy.timer.Timer | None = None
        self.is_capturing: bool = False 
        self.CENTERING_TOLERANCE_PIXELS = 5.0
        
        # Parâmetros da câmera e transformações do robô
        self.camera_hfov = 1.57; self.camera_image_width = 320; self.camera_image_height = 240
        
        # Parâmetros e dados do LiDAR
        self.lidar_ranges: list[float] = []; self.lidar_angle_min: float = 0.0
        self.lidar_angle_increment: float = 0.0174532925199 
        self.lidar_range_min: float = 0.12; self.lidar_range_max: float = 3.5
        
        # Posição e orientação do robô
        self.posicao_atual: Point | None = None; self.orientacao_quat_atual: Quaternion | None = None
        self.robot_x: float | None = None; self.robot_y: float | None = None; self.robot_yaw: float | None = None

        # Configurações do mapa de ocupação
        self.map_resolution = 0.05; self.map_width_meters = 20.0; self.map_height_meters = 20.0
        self.map_num_cells_width = int(self.map_width_meters/self.map_resolution)
        self.map_num_cells_height = int(self.map_height_meters/self.map_resolution)
        self.map_origin_x = -self.map_width_meters/2.0; self.map_origin_y = -self.map_height_meters/2.0
        self.FREE_CELL_VALUE = 0; self.MAX_OCCUPANCY_STRENGTH = 5; self.OCCUPANCY_DECAY_RATE = 1
        self.OCCUPANCY_VISUALIZATION_THRESHOLD = 1; self.PLANNING_OBSTACLE_THRESHOLD = 3
        self.occupancy_map = np.full((self.map_num_cells_height, self.map_num_cells_width), self.FREE_CELL_VALUE, dtype=np.int8)
        self.map_display_window_width = 600; self.map_display_window_height = 600
        
        # Variáveis de planejamento de caminho
        self.planned_path: list[tuple[int, int]] | None = None
        self.goal_for_current_path: Point | None = None 
        self.current_path_segment_index: int = 0
        self.WAYPOINT_REACHED_THRESHOLD_METERS = 0.3
        self.SAFETY_RADIUS_METERS = 0.5
        self.safety_radius_cells = int(self.SAFETY_RADIUS_METERS / self.map_resolution)
        self._astar_open_set_counter = 0

        # Timer para replanejamento periódico
        self.navigation_replanning_timer: rclpy.timer.Timer | None = None
        self.NAVIGATION_REPLAN_INTERVAL = 5.0 # segundos

        # Publishers, Subscribers e Timers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_cmd_pub = self.create_publisher(Float64MultiArray, '/gripper_controller/commands', 10)
        
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Pose, '/model/prm_robot/pose', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam/labels_map', self.camera_callback, 10)

        self.get_logger().info("Robô inicializado corretamente")
        self.timer_estado = self.create_timer(0.1, self.run_current_state) 
        self.timer_mapa = self.create_timer(0.5, self.atualizar_mapa)

    def run_current_state(self): 
        if self.estado_atual:
            self.estado_atual()

    def mudar_estado(self, novo_estado):
        estado_anterior_nome = self.estado_atual.__name__ if self.estado_atual else "Nenhum"
        self.get_logger().info(f"Mudando estado de '{estado_anterior_nome}' para '{novo_estado.__name__}'")
        
        if self.timer_estado is not None: self.timer_estado.destroy()

        # Cancela timers específicos ao sair de seus estados
        if self.sequence_timer:
            self.sequence_timer.cancel()
            self.sequence_timer = None
        if self.navigation_replanning_timer:
            self.navigation_replanning_timer.cancel()
            self.navigation_replanning_timer = None
        
        if self.estado_atual == self.capturando_bandeira:
            self.is_capturing = False
        
        if self.estado_atual in [self.indo_para_zona_alvo, self.retornando_pra_base]:
            self.planned_path = None
            self.goal_for_current_path = None
        
        self.estado_atual = novo_estado
        self.timer_estado = self.create_timer(0.1, self.run_current_state)
    
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

    def imu_callback(self, msg: Imu): self.dados_imu = msg; pass

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
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.imagem = frame
            
            label_bandeira_bgr = np.array([25, 25, 25])
            label_chao_bgr = np.array([15, 15, 15]) 
            min_area_necessaria = 100 
            
            mask = cv2.inRange(self.imagem, label_bandeira_bgr, label_bandeira_bgr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.flag_detected_in_current_image = False 
            self.flag_pixel_centroid = None
            self.flagpole_base_pixel_centroid = None
            
            valid_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > min_area_necessaria:
                    valid_contours.append(contour)

            combined_contour = None
            if len(valid_contours) > 0:
                self.flag_detected_in_current_image = True
                combined_contour = np.vstack(valid_contours)
                
                M = cv2.moments(combined_contour)
                if M["m00"] != 0:
                    self.flag_pixel_centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                else:
                    self.flag_detected_in_current_image = False 

                if self.flag_detected_in_current_image:
                    idx_ponto_mais_baixo = combined_contour[:, 0, 1].argmax()
                    ponto_mais_baixo = combined_contour[idx_ponto_mais_baixo][0]
                    px_chao_y = ponto_mais_baixo[1] + 2
                    px_chao_x = ponto_mais_baixo[0]
                    if 0 <= px_chao_y < self.camera_image_height and 0 <= px_chao_x < self.camera_image_width:
                        cor_pixel_abaixo = self.imagem[px_chao_y, px_chao_x]
                        if np.array_equal(cor_pixel_abaixo, label_chao_bgr):
                            self.flagpole_base_pixel_centroid = tuple(ponto_mais_baixo)
            
            vis_frame = frame.copy()
            if self.flag_detected_in_current_image and combined_contour is not None:
                hull = cv2.convexHull(combined_contour)
                cv2.drawContours(vis_frame, [hull], -1, (0, 255, 255), 2)

            if self.flag_pixel_centroid:
                cv2.circle(vis_frame, self.flag_pixel_centroid, 7, (255, 100, 0), -1)
            if self.flagpole_base_pixel_centroid:
                cv2.circle(vis_frame, self.flagpole_base_pixel_centroid, 7, (0, 0, 255), -1)
            
            cv2.imshow("Flag Detection Details", vis_frame)
            cv2.waitKey(1)

        except CvBridgeError as e: self.get_logger().error(f"Erro no CvBridge: {e}")

    def is_flag_at_bottom(self) -> bool:
        if self.imagem is None:
            return False
        
        label_bandeira_bgr = np.array([25, 25, 25])
        bottom_row = self.imagem[-1, :] 

        return np.any(np.all(bottom_row == label_bandeira_bgr, axis=1))

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
        return np.sqrt((current_coords[0] - goal_coords[0])**2 + (current_coords[1] - goal_coords[1])**2)

    def _reconstruct_astar_path(self, came_from_map: dict, current_target_node: tuple[int,int], start_node: tuple[int,int]) -> list:
        path = [current_target_node]
        node = current_target_node
        max_path_len = self.map_num_cells_width * self.map_num_cells_height 
        count = 0
        while node in came_from_map and count < max_path_len :
            node = came_from_map[node]
            path.append(node)
            if node == start_node: break
            count +=1
        if path[-1] != start_node :
             if current_target_node == start_node and len(path) == 1:
                 pass
             else:
                self.get_logger().error(f"A* Reconstrução: Início {start_node} não encontrado no caminho para {current_target_node}.")
                return [] 
        return path[::-1]


    def is_line_collision_free(self, s_coords: tuple[int,int], e_coords: tuple[int,int], i_map: np.ndarray) -> bool:
        for c, r in self.get_line_cells(s_coords[0], s_coords[1], e_coords[0], e_coords[1]):
            if not self.is_valid_map_coords(c,r) or i_map[r,c] == 1:
                return False
        return True

    def smooth_path(self, path: list[tuple[int,int]], i_map: np.ndarray) -> list[tuple[int,int]]:
        if not path or len(path) < 3:
            return path
        
        smoothed_path = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            current_waypoint_in_original_path = path[i]
            best_j = i + 1
            
            for j_loop in range(len(path) - 1, i + 1, -1):
                if self.is_line_collision_free(current_waypoint_in_original_path, path[j_loop], i_map):
                    best_j = j_loop
                    break 
            
            smoothed_path.append(path[best_j])
            i = best_j
            
        return smoothed_path

    def plan_path_astar(self, start_map_coords: tuple[int, int], original_goal_map_coords: tuple[int, int]) -> list[tuple[int, int]] | None:
        self.get_logger().info(f"A*: Planejando de {start_map_coords} para {original_goal_map_coords}")
        inflated_map = self.create_inflated_map()
        
        if not self.is_valid_map_coords(start_map_coords[0], start_map_coords[1]) or \
           not self.is_valid_map_coords(original_goal_map_coords[0], original_goal_map_coords[1]):
            self.get_logger().error("A*: Coordenadas de início ou fim original fora do mapa.")
            return None

        if inflated_map[start_map_coords[1], start_map_coords[0]] == 1:
            self.get_logger().warn(f"A*: Início {start_map_coords} em obstáculo. Procurando ponto de partida válido mais próximo.")
            q = deque([start_map_coords])
            visited = {start_map_coords}
            found_valid_start = False
            while q:
                c, r = q.popleft()
                if inflated_map[r, c] == 0:
                    self.get_logger().info(f"A*: Ponto de partida válido encontrado em { (c, r) }. Re-planejando.")
                    start_map_coords = (c, r)
                    found_valid_start = True
                    break
                for dc, dr in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nc, nr = c + dc, r + dr
                    if self.is_valid_map_coords(nc, nr) and (nc, nr) not in visited:
                        visited.add((nc, nr))
                        q.append((nc, nr))
            if not found_valid_start:
                self.get_logger().error("A*: Não foi possível encontrar um ponto de partida válido próximo. Planejamento falhou.")
                return None
        
        open_set = []
        self._astar_open_set_counter = 0
        
        came_from = {} 
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_map_coords] = 0.0
        
        h_initial = self.heuristic(start_map_coords, original_goal_map_coords)
        f_initial = g_score[start_map_coords] + h_initial
        heapq.heappush(open_set, (f_initial, self._astar_open_set_counter, start_map_coords))
        open_set_hash = {start_map_coords}
        
        processed_nodes_for_fallback = {}

        path_found_to_original_goal = False
        path_to_return = None

        while open_set:
            _, _, current_coords = heapq.heappop(open_set)
            open_set_hash.remove(current_coords)
            
            current_col, current_row = current_coords
            processed_nodes_for_fallback[current_coords] = g_score[current_coords]


            if current_coords == original_goal_map_coords:
                self.get_logger().info("A*: Caminho direto para o objetivo original encontrado.")
                path_found_to_original_goal = True
                path_to_return = self._reconstruct_astar_path(came_from, current_coords, start_map_coords)
                if inflated_map[original_goal_map_coords[1], original_goal_map_coords[0]] == 1 and path_to_return and len(path_to_return) > 1:
                    self.get_logger().info("A*: Objetivo original é obstáculo, planejando para célula adjacente.")
                    path_to_return.pop()
                if not path_to_return:
                    self.get_logger().warn("A*: Caminho para objetivo original ficou vazio após ajuste.")
                else:
                    break

            for d_col, d_row in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor_col, neighbor_row = current_col + d_col, current_row + d_row
                neighbor_coords = (neighbor_col, neighbor_row)

                if not self.is_valid_map_coords(neighbor_col, neighbor_row): continue
                
                is_neighbor_the_goal = (neighbor_coords == original_goal_map_coords)
                if inflated_map[neighbor_row, neighbor_col] == 1 and not is_neighbor_the_goal:
                    continue 
                
                cost = 1.0 if abs(d_col) + abs(d_row) == 1 else np.sqrt(2)
                tentative_g_score = g_score[current_coords] + cost

                if tentative_g_score < g_score[neighbor_coords]:
                    came_from[neighbor_coords] = current_coords
                    g_score[neighbor_coords] = tentative_g_score
                    h_val = self.heuristic(neighbor_coords, original_goal_map_coords)
                    f_val = tentative_g_score + h_val
                    if neighbor_coords not in open_set_hash:
                        self._astar_open_set_counter += 1
                        heapq.heappush(open_set, (f_val, self._astar_open_set_counter, neighbor_coords))
                        open_set_hash.add(neighbor_coords)
        
        if path_to_return and path_found_to_original_goal:
            self.get_logger().info(f"A*: Caminho final para objetivo original com {len(path_to_return)} pontos.")
            return path_to_return

        self.get_logger().warn(f"A*: Nenhum caminho direto para {original_goal_map_coords}. Tentando ponto mais próximo.")
        if not processed_nodes_for_fallback:
            self.get_logger().error("A*: Nenhum nó explorado para fallback.")
            return None

        closest_node = None; min_h = float('inf')
        reachable_nodes_with_finite_g = {node: score for node, score in g_score.items() if score != float('inf')}

        if not reachable_nodes_with_finite_g:
             self.get_logger().error("A*: Nenhum nó alcançável com g_score finito para fallback.")
             return None

        for node, current_g_score in reachable_nodes_with_finite_g.items():
            h = self.heuristic(node, original_goal_map_coords)
            if h < min_h:
                min_h = h
                closest_node = node
            elif h == min_h:
                if current_g_score < g_score.get(closest_node, float('inf')):
                    closest_node = node
        
        if closest_node is None:
            self.get_logger().warn("A*: Não foi possível encontrar um nó alcançável alternativo (fallback).")
            return None
        if closest_node == start_map_coords and start_map_coords != original_goal_map_coords:
             self.get_logger().warn(f"A*: Ponto mais próximo {closest_node} é o início. Objetivo {original_goal_map_coords} inacessível.")
             return None

        self.get_logger().info(f"A*: Redirecionando para o ponto alcançável mais próximo: {closest_node}")
        path_to_closest = self._reconstruct_astar_path(came_from, closest_node, start_map_coords)
        
        if not path_to_closest:
            self.get_logger().error(f"A*: Falha ao reconstruir caminho para o ponto mais próximo {closest_node}.")
            return None
            
        self.get_logger().info(f"Caminho A* para ponto mais próximo com {len(path_to_closest)} pontos.")
        return path_to_closest

    def _force_replan(self):
        """Força o replanejamento invalidando o caminho atual."""
        self.get_logger().info("Timer de replanejamento disparado. Forçando novo cálculo de rota.")
        self.planned_path = None
        if self.navigation_replanning_timer:
            self.navigation_replanning_timer.cancel()
            self.navigation_replanning_timer = None

    def set_goal_and_plan_path_map_coords(self, goal_map_col: int, goal_map_row: int):
        if self.robot_x is None or self.robot_y is None: self.get_logger().warn("Posição robô desconhecida."); return
        start_map_col, start_map_row = self.world_to_map_coords(self.robot_x, self.robot_y)
        if start_map_col is None or start_map_row is None: self.get_logger().warn("Robô fora do mapa."); return
        
        # Cancela o timer de replanejamento anterior, se houver
        if self.navigation_replanning_timer:
            self.navigation_replanning_timer.cancel()

        self.planned_path = self.plan_path_astar((start_map_col, start_map_row), (goal_map_col, goal_map_row))
        
        if self.planned_path and len(self.planned_path) > 0:
            inflated_map_for_smoothing = self.create_inflated_map()
            self.planned_path = self.smooth_path(self.planned_path, inflated_map_for_smoothing)
            self.get_logger().info(f"Caminho planejado e suavizado com {len(self.planned_path)} pontos.")
            self.current_path_segment_index = 0
            # Cria um novo timer para o próximo replanejamento
            self.navigation_replanning_timer = self.create_timer(self.NAVIGATION_REPLAN_INTERVAL, self._force_replan)
        else: 
            self.get_logger().warn("Falha ao planejar caminho ou caminho vazio.")
            self.planned_path = None
        
        self.goal_for_current_path = Point(x=float(goal_map_col), y=float(goal_map_row)) # Apenas para o estado de retorno

    def set_goal_and_plan_path_world_coords(self, goal_world_x: float, goal_world_y: float):
        goal_map_col, goal_map_row = self.world_to_map_coords(goal_world_x, goal_world_y)
        if goal_map_col is None or goal_map_row is None: 
            self.get_logger().error(f"Objetivo ({goal_world_x},{goal_world_y}) fora do mapa."); return
        self.set_goal_and_plan_path_map_coords(goal_map_col, goal_map_row)


    def atualizar_mapa(self):
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None or not self.lidar_ranges: return
        robot_map_col, robot_map_row = self.world_to_map_coords(self.robot_x, self.robot_y)
        if robot_map_col is None or robot_map_row is None: return
        for i_loop, distance in enumerate(self.lidar_ranges): 
            beam_angle_robot_frame = self.lidar_angle_min + i_loop * self.lidar_angle_increment
            beam_angle_world = self.robot_yaw + beam_angle_robot_frame; effective_distance = self.lidar_range_max
            is_obstacle_hit = False
            if np.isfinite(distance) and distance >= self.lidar_range_min:
                if distance < self.lidar_range_max: effective_distance = distance; is_obstacle_hit = True
            endpoint_x_world = self.robot_x + effective_distance * np.cos(beam_angle_world)
            endpoint_y_world = self.robot_y + effective_distance * np.sin(beam_angle_world)
            endpoint_map_col, endpoint_map_row = self.world_to_map_coords(endpoint_x_world, endpoint_y_world)
            if endpoint_map_col is not None and endpoint_map_row is not None:
                line_cells = self.get_line_cells(robot_map_col, robot_map_row, endpoint_map_col, endpoint_map_row)
                for (cell_col, cell_row) in line_cells:
                    if not self.is_valid_map_coords(cell_col, cell_row): continue
                    is_this_cell_the_obstacle_endpoint = (is_obstacle_hit and cell_col == endpoint_map_col and cell_row == endpoint_map_row)
                    if is_this_cell_the_obstacle_endpoint: self.occupancy_map[cell_row, cell_col] = self.MAX_OCCUPANCY_STRENGTH
                    else:
                        if self.occupancy_map[cell_row, cell_col] > self.FREE_CELL_VALUE:
                            self.occupancy_map[cell_row, cell_col] = max(self.FREE_CELL_VALUE, self.occupancy_map[cell_row, cell_col] - self.OCCUPANCY_DECAY_RATE)
        
        if hasattr(self, 'occupancy_map') and self.occupancy_map.size > 0:
            temp_map_for_display = np.clip(self.occupancy_map, self.FREE_CELL_VALUE, self.MAX_OCCUPANCY_STRENGTH)
            if self.MAX_OCCUPANCY_STRENGTH > self.FREE_CELL_VALUE:
                 normalized_strength = temp_map_for_display.astype(np.float32) / self.MAX_OCCUPANCY_STRENGTH
            else: normalized_strength = np.where(temp_map_for_display > self.FREE_CELL_VALUE, 1.0, 0.0).astype(np.float32)
            display_map_content = (255 * (1 - normalized_strength)).astype(np.uint8)
            display_map_content[self.occupancy_map <= self.FREE_CELL_VALUE] = 255
            display_map_content_flipped = np.flipud(display_map_content)
            map_h, map_w = display_map_content_flipped.shape
            canvas = np.full((self.map_display_window_height, self.map_display_window_width), 128, dtype=np.uint8)
            scale_h = self.map_display_window_height/map_h if map_h > 0 else 1; scale_w = self.map_display_window_width/map_w if map_w > 0 else 1
            scale = min(scale_h, scale_w); new_content_h = int(map_h * scale); new_content_w = int(map_w * scale)
            start_x, start_y = 0,0
            if new_content_w > 0 and new_content_h > 0:
                resized_map_content = cv2.resize(display_map_content_flipped, (new_content_w, new_content_h), interpolation=cv2.INTER_NEAREST)
                start_x = (self.map_display_window_width - new_content_w) // 2; start_y = (self.map_display_window_height - new_content_h) // 2
                canvas[start_y:start_y+new_content_h, start_x:start_x+new_content_w] = resized_map_content
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if self.robot_x is not None and self.robot_y is not None: 
                robot_map_col_viz, robot_map_row_viz_flipped = self.world_to_map_coords(self.robot_x, self.robot_y)
                if robot_map_col_viz is not None and robot_map_row_viz_flipped is not None:
                    robot_map_row_viz = map_h - 1 - robot_map_row_viz_flipped
                    robot_canvas_x = int((robot_map_col_viz/map_w)*new_content_w + start_x); robot_canvas_y = int((robot_map_row_viz/map_h)*new_content_h + start_y)
                    if 0 <= robot_canvas_x < self.map_display_window_width and 0 <= robot_canvas_y < self.map_display_window_height:
                        cv2.circle(canvas_bgr, (robot_canvas_x, robot_canvas_y), 5, (0,0,255), -1)
            if self.planned_path: 
                for i_path in range(len(self.planned_path) - 1): 
                    p1_map_col, p1_map_row_f = self.planned_path[i_path]; p2_map_col, p2_map_row_f = self.planned_path[i_path+1]
                    p1_map_row = map_h - 1 - p1_map_row_f; p2_map_row = map_h - 1 - p2_map_row_f
                    p1_c_x = int((p1_map_col/map_w)*new_content_w+start_x); p1_c_y = int((p1_map_row/map_h)*new_content_h+start_y)
                    p2_c_x = int((p2_map_col/map_w)*new_content_w+start_x); p2_c_y = int((p2_map_row/map_h)*new_content_h+start_y)
                    cv2.line(canvas_bgr, (p1_c_x,p1_c_y), (p2_c_x,p2_c_y), (0,255,0), 2)
            
            cv2.imshow("Mapa de Ocupacao (OpenCV)", canvas_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'): 
                try:
                    gx_str = input("X do objetivo (mundo): "); gy_str = input("Y do objetivo (mundo): ")
                    self.set_goal_and_plan_path_world_coords(float(gx_str), float(gy_str))
                except ValueError: self.get_logger().error("Entrada inválida.")


    def indo_para_zona_alvo(self):
        log_prefix = "GOTO_ZONE:"
        # Se o caminho ainda não foi calculado ou falhou, tenta calcular.
        if self.planned_path is None:
            self.get_logger().info(f"{log_prefix} Tentando planejar o caminho para a zona alvo (3.5, 0).")
            self.set_goal_and_plan_path_world_coords(3.5, 0.0)
        
        twist = Twist()
        # Segue o caminho planejado
        if self.planned_path and self.robot_x is not None and self.robot_yaw is not None:
            if self.current_path_segment_index >= len(self.planned_path):
                self.get_logger().info(f"{log_prefix} Chegou à zona alvo. Iniciando exploração."); 
                self.cmd_vel_pub.publish(twist)
                self.mudar_estado(self.explorando)
                return

            target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
            target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
            dx = target_world_x - self.robot_x; dy = target_world_y - self.robot_y
            distance_to_waypoint = np.sqrt(dx*dx + dy*dy)
            
            if distance_to_waypoint < self.WAYPOINT_REACHED_THRESHOLD_METERS:
                self.get_logger().info(f"{log_prefix} Waypoint {self.current_path_segment_index} alcançado.")
                self.current_path_segment_index += 1
            
            if self.current_path_segment_index < len(self.planned_path):
                target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
                target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
                dx = target_world_x - self.robot_x; dy = target_world_y - self.robot_y
            
                angle_to_waypoint = np.arctan2(dy, dx)
                angle_diff = angle_to_waypoint - self.robot_yaw
                while angle_diff > np.pi: angle_diff -= 2*np.pi
                while angle_diff < -np.pi: angle_diff += 2*np.pi
                
                K_angular = 0.7; K_linear_max = 0.2
                twist.angular.z = K_angular * angle_diff
                if abs(angle_diff) < np.deg2rad(20): 
                    twist.linear.x = K_linear_max
                else: 
                    twist.linear.x = 0.0 
                twist.angular.z = np.clip(twist.angular.z, -0.5, 0.5)
                twist.linear.x = np.clip(twist.linear.x, 0.0, K_linear_max)
        else: 
            self.get_logger().warn(f"{log_prefix} Sem caminho válido ou pose do robô. Aguardando...")
        
        self.cmd_vel_pub.publish(twist)

    def find_safest_direction(self) -> float | None:
        """Encontra o ângulo do feixe de LiDAR com a maior distância livre."""
        if not self.lidar_ranges:
            return None
        
        # Substitui 'inf' por um valor alto para que não seja sempre o máximo
        processed_ranges = [r if np.isfinite(r) else 0.0 for r in self.lidar_ranges]
        
        # Encontra o índice do feixe com a maior distância
        best_index = np.argmax(processed_ranges)
        
        # Converte o índice de volta para um ângulo no referencial do robô
        safest_angle = self.lidar_angle_min + best_index * self.lidar_angle_increment
        
        return safest_angle

    def explorando(self):
        if self.flag_detected_in_current_image and self.flagpole_base_pixel_centroid:
            self.get_logger().info("Bandeira e base do mastro detectados. Mudando para aproximação.")
            self.mudar_estado(self.aproximando_da_bandeira)
            return

        if not self.lidar_ranges or self.robot_x is None: return
        
        twist = Twist()

        if self.robot_x < 3.2:
            self.get_logger().warn("Perto da fronteira (x=3)! Virando para a direita para permanecer na zona alvo.")
            twist.angular.z = -0.4 
            twist.linear.x = 0.05
            self.cmd_vel_pub.publish(twist)
            return

        # Verifica se há um obstáculo próximo na frente
        front_dist = float('inf')
        front_clearance = 0.8
        for i, r in enumerate(self.lidar_ranges):
            angle = self.lidar_angle_min + i * self.lidar_angle_increment
            if -0.2 < angle < 0.2: # Um cone estreito na frente
                if np.isfinite(r) and r < front_dist:
                    front_dist = r

        if front_dist > front_clearance:
            # Caminho livre, avança
            twist.linear.x = 0.15
            twist.angular.z = 0.0
        else:
            # Obstáculo à frente, encontra uma nova direção para ir
            self.get_logger().info("Obstáculo à frente. Procurando a direção mais segura...")
            safest_angle = self.find_safest_direction()
            
            if safest_angle is not None:
                # Gira em direção ao ângulo mais seguro
                twist.linear.x = 0.0
                # O sinal do ângulo já indica a direção da rotação
                twist.angular.z = 0.5 * np.sign(safest_angle) 
                self.get_logger().info(f"Virando para a direção segura: {safest_angle:.2f} rad.")
            else:
                # Caso de emergência: se não encontrar nenhuma saída, vira para trás
                self.get_logger().warn("Não foi encontrada nenhuma direção segura. Virando para trás.")
                twist.linear.x = 0.0
                twist.angular.z = -0.5
            
        self.cmd_vel_pub.publish(twist)


    def aproximando_da_bandeira(self):
        """Estado para centralizar e se aproximar da bandeira até que ela toque a base da câmera."""
        log_prefix = "APPROACH_FLAG: "
        twist = Twist()

        if not self.flag_detected_in_current_image:
            self.get_logger().warn(f"{log_prefix}Perdeu a visão da bandeira por completo. Voltando a explorar.")
            self.mudar_estado(self.explorando)
            self.cmd_vel_pub.publish(twist)
            return

        if self.is_flag_at_bottom():
            self.get_logger().info(f"{log_prefix}Bandeira na base da câmera. Posição de captura alcançada.")
            self.cmd_vel_pub.publish(Twist()) 
            self.mudar_estado(self.capturando_bandeira)
            return

        target_for_centering = self.flagpole_base_pixel_centroid or self.flag_pixel_centroid
        
        if target_for_centering is None:
            self.get_logger().warn(f"{log_prefix}Alvo de centralização perdido. Voltando a explorar.")
            self.mudar_estado(self.explorando)
            self.cmd_vel_pub.publish(twist)
            return

        camera_center_x = self.camera_image_width / 2.0
        pixel_offset = target_for_centering[0] - camera_center_x
        
        K_angular = 0.6
        twist.angular.z = -K_angular * (pixel_offset / camera_center_x)
        twist.angular.z = np.clip(twist.angular.z, -0.4, 0.4)

        if abs(pixel_offset) < self.CENTERING_TOLERANCE_PIXELS * 3:
            twist.linear.x = 0.05
        else:
            twist.linear.x = 0.0

        self.cmd_vel_pub.publish(twist)

    def send_gripper_command(self, extension, left_gripper, right_gripper):
        """Envia um comando de posição para as juntas da garra usando Float64MultiArray."""
        msg = Float64MultiArray()
        msg.data = [extension, right_gripper, left_gripper]
        self.gripper_cmd_pub.publish(msg)
        self.get_logger().info(f"Enviando comando para a garra: {msg.data}")
    
    def _execute_capture_step(self):
        """Executa um passo da sequência de captura baseada em timer."""
        if self.estado_atual != self.capturando_bandeira:
            return

        if self.capture_sequence_step == 0:
            self.get_logger().info("Passo de captura 1: Estendendo braço com garra aberta.")
            self.send_gripper_command(extension=-1.5, left_gripper=0.06, right_gripper=-0.06)
            self.capture_sequence_step = 1
            self.sequence_timer = self.create_timer(2.0, self._execute_capture_step)
        
        elif self.capture_sequence_step == 1:
            self.get_logger().info("Passo de captura 2: Fechando garra.")
            self.send_gripper_command(extension=-1.5, left_gripper=0.0, right_gripper=0.0)
            self.capture_sequence_step = 2
            self.sequence_timer = self.create_timer(2.0, self._execute_capture_step)

        elif self.capture_sequence_step == 2:
            self.get_logger().info("Passo de captura 3: Recolhendo braço.")
            self.send_gripper_command(extension=0.2, left_gripper=0.0, right_gripper=0.0)
            self.capture_sequence_step = 3
            self.sequence_timer = self.create_timer(2.0, self._execute_capture_step)

        elif self.capture_sequence_step == 3:
            self.get_logger().info("Sequência de captura completa.")
            self.sequence_timer = None
            self.mudar_estado(self.retornando_pra_base)

    def capturando_bandeira(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.cmd_vel_pub.publish(Twist()) 
            self.capture_sequence_step = 0
            self._execute_capture_step()

    def retornando_pra_base(self):
        log_prefix = "RETURN_HOME:"
        if self.goal_for_current_path is None or self.planned_path is None:
            self.get_logger().info(f"{log_prefix} Definindo meta para a base (-6,0) e planejando caminho.")
            self.set_goal_and_plan_path_world_coords(-6.0, 0.0)

        twist = Twist()
        if self.planned_path and self.robot_x is not None and self.robot_yaw is not None:
            if self.current_path_segment_index >= len(self.planned_path):
                self.get_logger().info(f"{log_prefix} Chegou à base! Missão cumprida."); 
                self.cmd_vel_pub.publish(twist) 
                return

            target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
            target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
            dx = target_world_x - self.robot_x; dy = target_world_y - self.robot_y
            distance_to_waypoint = np.sqrt(dx*dx + dy*dy)
            
            if distance_to_waypoint < self.WAYPOINT_REACHED_THRESHOLD_METERS:
                self.get_logger().info(f"{log_prefix} Waypoint {self.current_path_segment_index} alcançado.")
                self.current_path_segment_index += 1
            
            if self.current_path_segment_index < len(self.planned_path):
                target_map_col, target_map_row = self.planned_path[self.current_path_segment_index]
                target_world_x, target_world_y = self.map_to_world_coords(target_map_col, target_map_row)
                dx = target_world_x - self.robot_x; dy = target_world_y - self.robot_y
            
                angle_to_waypoint = np.arctan2(dy, dx)
                angle_diff = angle_to_waypoint - self.robot_yaw
                while angle_diff > np.pi: angle_diff -= 2*np.pi
                while angle_diff < -np.pi: angle_diff += 2*np.pi
                
                K_angular = 0.7; K_linear_max = 0.15
                twist.angular.z = K_angular * angle_diff
                if abs(angle_diff) < np.deg2rad(30): 
                    reduction_factor = max(0.2, 1 - abs(angle_diff)/(np.pi/4)) 
                    twist.linear.x = K_linear_max * reduction_factor
                else: twist.linear.x = 0.0 
                twist.angular.z = np.clip(twist.angular.z, -0.5, 0.5)
                twist.linear.x = np.clip(twist.linear.x, 0.0, K_linear_max)
        else: 
            self.get_logger().warn(f"{log_prefix} Sem caminho ou pose. Aguardando planejamento.")
        
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControleRobo()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info("Nó interrompido (Ctrl+C).")
    finally:
        node.get_logger().info("Parando robô..."); stop_twist = Twist()
        if rclpy.ok() and hasattr(node, 'cmd_vel_pub') and node.cmd_vel_pub.get_subscription_count() > 0:
             node.cmd_vel_pub.publish(stop_twist)
        if hasattr(node, 'map_display_window_width'): cv2.destroyAllWindows()
        node.destroy_node(); rclpy.shutdown()
        node.get_logger().info("Nó encerrado.")

if __name__ == '__main__':
    main()
