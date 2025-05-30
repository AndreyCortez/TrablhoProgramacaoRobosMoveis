#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Point, Quaternion, Pose

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError # Adicionado CvBridgeError

from scipy.spatial.transform import Rotation as R

class ControleRobo(Node):
    def __init__(self):
        super().__init__('controle_robo')

        self.bridge = CvBridge()
        self.estado_atual = self.explorando
        self.imagem = None

        self.lidar_ranges: list[float] = []
        self.lidar_angle_min: float = 0.0
        self.lidar_angle_increment: float = 0.0
        self.lidar_range_min: float = 0.0
        self.lidar_range_max: float = 0.0

        self.dados_imu = None

        self.posicao_atual: Point | None = None
        self.orientacao_quat_atual: Quaternion | None = None
        self.orientacao_euler_atual: tuple[float, float, float] | None = None
        self.robot_x: float | None = None
        self.robot_y: float | None = None
        self.robot_yaw: float | None = None

        self.posicao_bandeira = (0, 0)

        self.map_resolution = 0.05
        self.map_width_meters = 10.0
        self.map_height_meters = 10.0
        self.map_num_cells_width = int(self.map_width_meters / self.map_resolution)
        self.map_num_cells_height = int(self.map_height_meters / self.map_resolution)
        self.map_origin_x = -self.map_width_meters / 2.0
        self.map_origin_y = -self.map_height_meters / 2.0

        # Novas constantes para o mapa de ocupação com decaimento
        self.FREE_CELL_VALUE = 0
        self.MAX_OCCUPANCY_STRENGTH = 5  # Max "certeza" de obstáculo
        self.OCCUPANCY_DECAY_RATE = 1    # Quanto decai por observação "livre"
        # Limiar para visualização: células com força >= este valor são mostradas como ocupadas (preto)
        self.OCCUPANCY_VISUALIZATION_THRESHOLD = 1

        # Mapa agora armazena a "força" da ocupação
        self.occupancy_map = np.full((self.map_num_cells_height, self.map_num_cells_width),
                                     self.FREE_CELL_VALUE, dtype=np.int8)
        
        self.map_display_window_width = 600
        self.map_display_window_height = 600

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Pose, '/model/prm_robot/pose', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam/labels_map', self.camera_callback, 10)

        self.get_logger().info("Robô inicializado corretamente")
        self.get_logger().info(f"Mapa criado com {self.map_num_cells_height}x{self.map_num_cells_width} células.")

        self.timer_estado = self.create_timer(0.1, self.estado_atual)
        self.timer_mapa = self.create_timer(0.5, self.atualizar_mapa) # Mapa pode ser atualizado com menos frequência

    def mudar_estado(self, novo_estado):
        self.get_logger().info(f"Mudando estado de '{self.estado_atual.__name__}' para '{novo_estado.__name__}'")
        if self.timer_estado is not None:
            self.timer_estado.destroy()
        self.estado_atual = novo_estado
        self.timer_estado = self.create_timer(0.1, self.estado_atual)

    def world_to_map_coords(self, world_x: float, world_y: float) -> tuple[int | None, int | None]:
        if world_x is None or world_y is None:
            return None, None
        map_col = int(np.floor((world_x - self.map_origin_x) / self.map_resolution))
        map_row = int(np.floor((world_y - self.map_origin_y) / self.map_resolution))
        return map_col, map_row

    def get_line_cells(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        """ Retorna células da linha (x0,y0) para (x1,y1) usando Bresenham. """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        curr_x, curr_y = x0, y0
        while True:
            cells.append((curr_x, curr_y))
            if curr_x == x1 and curr_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy
        return cells

    def scan_callback(self, msg: LaserScan):
        self.lidar_ranges = list(msg.ranges)
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_increment = msg.angle_increment
        self.lidar_range_min = msg.range_min
        self.lidar_range_max = msg.range_max

    def imu_callback(self, msg: Imu):
        self.dados_imu = msg
        pass

    def odom_callback(self, msg: Pose):
        self.posicao_atual = msg.position
        self.orientacao_quat_atual = msg.orientation

        if self.orientacao_quat_atual:
            rot_obj = R.from_quat([
                self.orientacao_quat_atual.x, self.orientacao_quat_atual.y,
                self.orientacao_quat_atual.z, self.orientacao_quat_atual.w
            ])
            roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=False)
            self.orientacao_euler_atual = (roll, pitch, yaw)
            self.robot_yaw = yaw

        if self.posicao_atual:
            self.robot_x = self.posicao_atual.x
            self.robot_y = self.posicao_atual.y

        if self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None:
            print(f"Robô X: {self.robot_x:.3f}, Y: {self.robot_y:.3f}, Yaw: {self.robot_yaw:.3f}")

    def camera_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.imagem = frame
        except CvBridgeError as e:
            self.get_logger().error(f"Erro no CvBridge ao processar imagem: {e}")

    def atualizar_mapa(self):
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None or not self.lidar_ranges:
            return

        robot_map_col, robot_map_row = self.world_to_map_coords(self.robot_x, self.robot_y)
        if robot_map_col is None or robot_map_row is None:
            return

        for i, distance in enumerate(self.lidar_ranges):
            beam_angle_robot_frame = self.lidar_angle_min + i * self.lidar_angle_increment
            beam_angle_world = self.robot_yaw + beam_angle_robot_frame

            effective_distance = self.lidar_range_max
            is_obstacle_hit = False
            if np.isfinite(distance) and distance >= self.lidar_range_min:
                if distance < self.lidar_range_max:
                    effective_distance = distance
                    is_obstacle_hit = True

            endpoint_x_world = self.robot_x + effective_distance * np.cos(beam_angle_world)
            endpoint_y_world = self.robot_y + effective_distance * np.sin(beam_angle_world)
            endpoint_map_col, endpoint_map_row = self.world_to_map_coords(endpoint_x_world, endpoint_y_world)

            if endpoint_map_col is not None and endpoint_map_row is not None:
                line_cells = self.get_line_cells(robot_map_col, robot_map_row, endpoint_map_col, endpoint_map_row)

                for (cell_col, cell_row) in line_cells:
                    if not (0 <= cell_col < self.map_num_cells_width and 0 <= cell_row < self.map_num_cells_height):
                        continue

                    is_this_cell_the_obstacle_endpoint = (is_obstacle_hit and
                                                         cell_col == endpoint_map_col and
                                                         cell_row == endpoint_map_row)

                    if is_this_cell_the_obstacle_endpoint:
                        self.occupancy_map[cell_row, cell_col] = self.MAX_OCCUPANCY_STRENGTH
                    else:
                        if self.occupancy_map[cell_row, cell_col] > self.FREE_CELL_VALUE:
                            self.occupancy_map[cell_row, cell_col] = \
                                max(self.FREE_CELL_VALUE,
                                    self.occupancy_map[cell_row, cell_col] - self.OCCUPANCY_DECAY_RATE)
        
        # Visualização do Mapa
        if hasattr(self, 'occupancy_map') and self.occupancy_map.size > 0:
            # Opção 1: Visualização com limiar (preto/branco)
            # display_map_content = np.zeros_like(self.occupancy_map, dtype=np.uint8)
            # display_map_content[self.occupancy_map < self.OCCUPANCY_VISUALIZATION_THRESHOLD] = 255
            # display_map_content[self.occupancy_map >= self.OCCUPANCY_VISUALIZATION_THRESHOLD] = 0

            # Opção 2: Visualização em tons de cinza (mais força = mais escuro)
            temp_map_for_display = np.clip(self.occupancy_map, self.FREE_CELL_VALUE, self.MAX_OCCUPANCY_STRENGTH)
            if self.MAX_OCCUPANCY_STRENGTH > self.FREE_CELL_VALUE:
                 normalized_strength = temp_map_for_display.astype(np.float32) / self.MAX_OCCUPANCY_STRENGTH
            else: # Fallback
                 normalized_strength = np.where(temp_map_for_display > self.FREE_CELL_VALUE, 1.0, 0.0).astype(np.float32)
            display_map_content = (255 * (1 - normalized_strength)).astype(np.uint8)
            # Garante que células completamente livres sejam brancas
            display_map_content[self.occupancy_map <= self.FREE_CELL_VALUE] = 255


            display_map_content_flipped = np.flipud(display_map_content)
            map_h, map_w = display_map_content_flipped.shape
            canvas = np.full((self.map_display_window_height, self.map_display_window_width), 128, dtype=np.uint8)
            
            scale_h = self.map_display_window_height / map_h if map_h > 0 else 1
            scale_w = self.map_display_window_width / map_w if map_w > 0 else 1
            scale = min(scale_h, scale_w)
            new_content_h = int(map_h * scale)
            new_content_w = int(map_w * scale)
            
            start_x = 0
            start_y = 0
            if new_content_w > 0 and new_content_h > 0:
                resized_map_content = cv2.resize(display_map_content_flipped,
                                                (new_content_w, new_content_h),
                                                interpolation=cv2.INTER_NEAREST)
                start_x = (self.map_display_window_width - new_content_w) // 2
                start_y = (self.map_display_window_height - new_content_h) // 2
                canvas[start_y : start_y + new_content_h, start_x : start_x + new_content_w] = resized_map_content
            
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if self.robot_x is not None and self.robot_y is not None:
                robot_map_col_viz, robot_map_row_viz_flipped = self.world_to_map_coords(self.robot_x, self.robot_y)
                if robot_map_col_viz is not None and robot_map_row_viz_flipped is not None:
                    robot_map_row_viz = map_h - 1 - robot_map_row_viz_flipped
                    robot_canvas_x = int((robot_map_col_viz / map_w) * new_content_w + start_x)
                    robot_canvas_y = int((robot_map_row_viz / map_h) * new_content_h + start_y)
                    if 0 <= robot_canvas_x < self.map_display_window_width and \
                       0 <= robot_canvas_y < self.map_display_window_height:
                        cv2.circle(canvas_bgr, (robot_canvas_x, robot_canvas_y), 5, (0, 0, 255), -1)
            
            cv2.imshow("Mapa de Ocupacao (OpenCV)", canvas_bgr)
            cv2.waitKey(1)

    def explorando(self):
        if not self.lidar_ranges:
            return

        distancias_frontais = []
        if len(self.lidar_ranges) > 0: # Checa se a lista não está vazia
            angulo_cone_frontal_rad = 0.52
            for i, r in enumerate(self.lidar_ranges):
                angle = self.lidar_angle_min + i * self.lidar_angle_increment
                if -angulo_cone_frontal_rad < angle < angulo_cone_frontal_rad:
                    if np.isfinite(r) and self.lidar_range_min < r < self.lidar_range_max:
                        distancias_frontais.append(r)

        obstaculo_a_frente = False
        distancia_seguranca = 0.5
        if distancias_frontais and min(distancias_frontais) < distancia_seguranca:
            obstaculo_a_frente = True

        twist = Twist()
        if not obstaculo_a_frente:
            twist.linear.x = 0.15
            twist.angular.z = 0.0
        else:
            twist.linear.x = 0.0
            twist.angular.z = -0.5

        self.cmd_vel_pub.publish(twist)

        if self.imagem is not None:
            # Para '/robot_cam/labels_map' (câmera de segmentação):
            # 'label_bandeira' deve ser o ID numérico da sua bandeira.
            # Ajuste este valor conforme o label da bandeira na sua simulação.
            label_bandeira = 40 # !!! EXEMPLO - AJUSTE ESTE VALOR !!!
            target_color_bgr = np.array([label_bandeira, label_bandeira, label_bandeira])
            min_area_necessaria = 100

            mask = cv2.inRange(self.imagem, target_color_bgr, target_color_bgr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bandeira_detectada = False
            for contour in contours:
                if cv2.contourArea(contour) > min_area_necessaria:
                    bandeira_detectada = True
                    self.get_logger().info("BANDEIRA DETECTADA VISUALMENTE (LABEL)!")
                    break
            if bandeira_detectada:
                stop_twist = Twist()
                self.cmd_vel_pub.publish(stop_twist)
                self.mudar_estado(self.estado_bandeira_avistada)

    def estado_bandeira_avistada(self):
        self.get_logger().info("ESTADO: Bandeira Avistada.")
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def navegando_para_bandeira(self): pass
    def posicionado_para_coleta(self): pass
    def capturando_bandeira(self): pass
    def retornando_pra_base(self): pass

def main(args=None):
    rclpy.init(args=args)
    node = ControleRobo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Nó interrompido pelo usuário (Ctrl+C).")
    finally:
        node.get_logger().info("Parando o robô...")
        stop_twist = Twist()
        if rclpy.ok() and hasattr(node, 'cmd_vel_pub') and node.cmd_vel_pub.get_subscription_count() > 0:
             node.cmd_vel_pub.publish(stop_twist)
        if hasattr(node, 'map_display_window_width'):
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("Nó encerrado e ROS 2 desligado.")

if __name__ == '__main__':
    main()