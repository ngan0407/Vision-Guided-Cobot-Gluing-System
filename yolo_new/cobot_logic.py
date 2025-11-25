# #---------low latency (with soft stop button)
# import os
# import sys
# import time
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict
# import math
# import json
# from sklearn.cluster import KMeans
# import socket
# import struct
# import base64
# import threading

# # ============================================================================
# # Helper Functions for Web UI
# # ============================================================================

# def send_frame_to_ui(socketio, frame, event_name='video_frame'):
#     """Encodes a CV2 frame as a base64 JPEG and emits it."""
#     try:
#         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Lower quality for faster transfer
#         jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#         socketio.emit(event_name, {'image': jpg_as_text})
#     except Exception as e:
#         print(f"Error sending frame to UI: {e}")

# def wait_for_proceed(socketio, proceed_event, stop_event, frame, message):
#     """
#     Sends an image to the UI, emits a 'wait_for_proceed' status,
#     and then blocks until the 'proceed_event' is set (by the user).
#     """
#     socketio.emit('status_update', {'msg': f"WAITING: {message}"})
    
#     # Send the specific frame for the user to see (with good quality)
#     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
#     jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#     socketio.emit('show_image_and_wait', {'image': jpg_as_text})

#     # Wait for the proceed_event or stop_event
#     proceed_event.clear() # Ensure it's clear before waiting
#     while not proceed_event.is_set() and not stop_event.is_set():
#         socketio.sleep(0.1) # Non-blocking sleep

#     if stop_event.is_set():
#         socketio.emit('status_update', {'msg': 'Proceed cancelled by stop request.'})
#         return False # Stopped
        
#     socketio.emit('status_update', {'msg': 'User proceeded.'})
#     socketio.emit('hide_popup') # Tell UI to hide the popup
#     return True # Proceeded

# # ============================================================================
# # JAKA Cobot Communication Class (No changes from your original)
# # ============================================================================
# class JAKACobot:
#     """JAKA Cobot TCP/IP communication and control using the official JSON protocol."""

#     def __init__(self, ip="192.168.1.166", port=10001):
#         self.ip = ip
#         self.port = port
#         self.socket = None
#         self.connected = False

#     def connect(self):
#         try:
#             self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.socket.settimeout(15.0)
#             self.socket.connect((self.ip, self.port))
#             self.connected = True
#             print(f"[JAKA] Connected to cobot at {self.ip}:{self.port}")
#             return True
#         except Exception as e:
#             print(f"[JAKA ERROR] Connection failed: {e}")
#             self.connected = False
#             return False

#     def disconnect(self):
#         if self.socket:
#             self.socket.close()
#             self.connected = False
#             print("[JAKA] Disconnected from cobot")

#     def send_command(self, command_dict):
#         if not self.connected:
#             print("[JAKA ERROR] Not connected to cobot")
#             return None
#         try:
#             json_command = json.dumps(command_dict) + '\n'
#             self.socket.sendall(json_command.encode('utf-8'))
#             response_str = self.socket.recv(4096).decode('utf-8').strip()
#             if response_str:
#                 return json.loads(response_str)
#             return None
#         except (json.JSONDecodeError, socket.timeout, ConnectionResetError) as e:
#             print(f"[JAKA ERROR] Communication error: {e}")
#             return None

#     def _check_response(self, response_json, command_name):
#         if response_json and str(response_json.get("errorCode")) == "0":
#             return True
#         else:
#             msg = response_json.get('errorMsg', 'Unknown error') if response_json else 'No response'
#             print(f"[JAKA ERROR] {command_name} failed: {msg}")
#             return False

#     def power_on(self):
#         cmd = {"cmdName": "power_on"}
#         response = self.send_command(cmd)
#         return self._check_response(response, "power_on")

#     def enable_robot(self):
#         cmd = {"cmdName": "enable_robot"}
#         response = self.send_command(cmd)
#         return self._check_response(response, "enable_robot")

#     def get_tcp_position(self):
#         cmd = {"cmdName": "get_tcp_pos"}
#         response = self.send_command(cmd)
#         if response and str(response.get("errorCode")) == "0":
#             tcp_data = response.get("tcp_pos")
#             if tcp_data and len(tcp_data) == 6:
#                 return [float(p) for p in tcp_data]
#         print(f"[JAKA ERROR] Failed to get TCP position. Response: {response}")
#         return None

#     def get_joint_position(self):
#         cmd = {"cmdName": "get_joint_pos"}
#         response = self.send_command(cmd)
#         if response and str(response.get("errorCode")) == "0":
#             joint_data = response.get("joint_pos")
#             if joint_data and len(joint_data) == 6:
#                 return [float(p) for p in joint_data]
#         print(f"[JAKA ERROR] Failed to get Joint position. Response: {response}")
#         return None

#     def joint_move(self, joint_positions, speed=20, accel=50, tol=0.5, block=True):
#         cmd = {
#             "cmdName": "joint_move", "relFlag":0,
#             "jointPosition": [float(p) for p in joint_positions],
#             "speed": speed, "accel": accel, "tol": tol
#         }
#         response = self.send_command(cmd)
#         if not self._check_response(response, "joint_move (joint_move)"):
#             return False

#         print("[JAKA] moveJ command accepted.")
#         if block:
#             print("[JAKA] Waiting for joint move to complete...")
#             start_time = time.time()
#             while time.time() - start_time < 30: # 30 second timeout
#                 current_joints = self.get_joint_position()
#                 if current_joints:
#                     diff = np.array(joint_positions) - np.array(current_joints)
#                     dist = np.linalg.norm(diff)
#                     if dist < (tol + 0.5):
#                         print("[JAKA] Joint move complete.")
#                         return True
#                 time.sleep(0.1)
#             print("[JAKA ERROR] Joint move timed out.")
#             return False
#         return True

#     def linear_move(self, tcp_position, speed=20, accel=50, tol=0.5, relative=False, block=True):
#         cmd = {
#             "cmdName": "moveL", "relFlag": 1 if relative else 0,
#             "cartPosition": [float(p) for p in tcp_position],
#             "speed": speed, "accel": accel, "tol": tol
#         }
#         response = self.send_command(cmd)
#         if not self._check_response(response, "moveL (linear_move)"):
#             return False

#         print("[JAKA] moveL command accepted.")
#         if block and not relative:
#             print("[JAKA] Waiting for linear move to complete...")
#             start_time = time.time()
#             while time.time() - start_time < 30:
#                 current_pos = self.get_tcp_position()
#                 if current_pos:
#                     dist = math.hypot(
#                         tcp_position[0] - current_pos[0],
#                         tcp_position[1] - current_pos[1],
#                         tcp_position[2] - current_pos[2]
#                     )
#                     if dist < tol:
#                         print("[JAKA] Linear move complete.")
#                         return True
#                 time.sleep(0.1)
#             print("[JAKA ERROR] Linear move timed out.")
#             return False
#         elif block and relative:
#             time.sleep(1.0) # Small delay for relative moves to ensure completion
#         return True

#     def move_relative(self, dx=0, dy=0, dz=0, speed=10, block=True):
#         relative_pose = [dx, dy, dz, 0, 0, 0]
#         return self.linear_move(relative_pose, speed=speed, relative=True, block=block)

# # ============================================================================
# # Standalone Calibration Routine (MODIFIED FOR WEB UI)
# # ============================================================================
# def run_calibration_routine(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, settings):
    
#     needle_color = settings['last_needle_color']
#     move_dist = settings['last_move_mm']
#     standby_offset_mm = settings['last_offset_mm']
#     calibration_start_joints = settings.get('calibration_start_joints')

#     socketio.emit('status_update', {'msg': 'Calibration routine starting...'})
    
#     # Use resize info from settings if available, otherwise default
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
#     # resize = True # Assume we always resize to a known quantity - This should be handled by the camera thread if needed

#     calib_img_dir = "calibration_images"
#     os.makedirs(calib_img_dir, exist_ok=True)
#     socketio.emit('status_update', {'msg': f"Images will be saved in '{calib_img_dir}/'"})

#     robot_points_mm = np.float32([[0, 0], [move_dist, 0], [0, move_dist]])
#     pixel_points = []

#     # 1. Move to start position
#     if calibration_start_joints and sum(abs(j) for j in calibration_start_joints) > 0:
#         socketio.emit('status_update', {'msg': 'Moving to pre-defined calibration start position...'})
#         if not cobot.joint_move(calibration_start_joints, speed=20, block=True):
#              socketio.emit('status_update', {'msg': 'Failed to move to start position. Aborting.', 'error': True})
#              return False
#     else:
#         socketio.emit('status_update', {'msg': 'No start joints defined. Using current position.'})
        
#     socketio.emit('status_update', {'msg': "Getting robot's starting position..."})
#     start_pos = cobot.get_tcp_position()
#     if not start_pos:
#         socketio.emit('status_update', {'msg': 'Could not get robot start position. Aborting.', 'error': True})
#         return False

#     for i, (dx, dy) in enumerate(robot_points_mm):
#         if stop_event.is_set(): return False
        
#         tip_found = False
#         point_num = i + 1
#         socketio.emit('status_update', {'msg': f"Moving to relative point {point_num}/3 ({dx}mm, {dy}mm)..."})

#         target_pose = start_pos.copy()
#         target_pose[0] += dx
#         target_pose[1] += dy

#         if not cobot.linear_move(target_pose, speed=15, block=True):
#             socketio.emit('status_update', {'msg': f"Robot failed to move to point {point_num}. Aborting.", 'error': True})
#             return False
#         socketio.sleep(1.0) # Wait for vibrations to settle

#         socketio.emit('status_update', {'msg': f"Detecting tip at point {point_num}..."})
#         for attempt in range(5):
#             if stop_event.is_set(): return False
            
#             # --- NON-BLOCKING FRAME RETRIEVAL ---
#             with camera_lock:
#                 frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
#             if frame is None:
#                 socketio.sleep(0.1) # Wait for a frame if buffer is empty
#                 continue

#             # Original logic
#             tip_info = detect_needle_tip_rectangular(frame, needle_color, debug_frame=frame)
            
#             if tip_info and tip_info[0]:
#                 tip_point = tip_info[0]
#                 pixel_points.append(tip_point)
#                 socketio.emit('status_update', {'msg': f"Tip detected at pixel coordinate: {tip_point}"})

#                 cv2.circle(frame, tip_point, 10, (0, 0, 255), -1)
#                 cv2.putText(frame, f"Point {point_num}", (tip_point[0] + 15, tip_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 send_frame_to_ui(socketio, frame) # Show detection
#                 socketio.sleep(0.5)

#                 filepath = os.path.join(calib_img_dir, f"calibration_point_{point_num}.png")
#                 cv2.imwrite(filepath, frame)
#                 tip_found = True
#                 break
#             else:
#                 cv2.putText(frame, f"Detection Failed (Attempt {attempt+1})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 send_frame_to_ui(socketio, frame) # Show failure
#                 socketio.sleep(0.5)

#         if not tip_found:
#             socketio.emit('status_update', {'msg': f"Could not detect tip at point {point_num}. Aborting.", 'error': True})
#             return False

#     if len(pixel_points) == 3:
#         socketio.emit('status_update', {'msg': "Calculating transformation matrix..."})
#         pixel_points_np = np.float32(pixel_points)
#         pixel_to_robot_matrix = cv2.getAffineTransform(pixel_points_np, robot_points_mm)
#         np.save('calibration_matrix.npy', pixel_to_robot_matrix)

#         socketio.emit('status_update', {'msg': "CALIBRATION SUCCESSFUL! Matrix saved."})
#         print("Matrix values:\n", pixel_to_robot_matrix)
#         success = True
#     else:
#         socketio.emit('status_update', {'msg': "Did not collect enough points. Calibration failed.", 'error': True})
#         success = False

#     socketio.emit('status_update', {'msg': f"Moving {standby_offset_mm}mm sideways to standby position..."})
#     cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
#     socketio.emit('status_update', {'msg': "Cobot is in standby. Calibration finished."})
    
#     return success

# # ============================================================================
# # Vision Processing Functions (No changes from your original)
# # ============================================================================
# def detect_needle_tip_rectangular(image, color='red', debug_frame=None):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     if color == 'red':
#         lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
#         lower_red2, upper_red2 = np.array([170, 80, 80]), np.array([180, 255, 255])
#         mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
#     elif color == 'green':
#         mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
#     elif color == 'pink':
#         mask = cv2.inRange(hsv, np.array([140, 80, 80]), np.array([170, 255, 255]))
#     else: return None, None, None, None

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if debug_frame is not None:
#         cv2.drawContours(debug_frame, contours, -1, (255, 0, 0), 1)
#     if not contours: return None, None, None, None
#     needle_contour = max(contours, key=cv2.contourArea)
#     if len(needle_contour) < 5: return None, None, None, None
#     if debug_frame is not None:
#         cv2.drawContours(debug_frame, [needle_contour], -1, (0, 255, 0), 2)

#     rect = cv2.minAreaRect(needle_contour)
#     box = cv2.boxPoints(rect)
#     box = box.astype(int)
#     center, (width, height), angle = rect
#     if width < height:
#         width, height = height, width
#         angle = angle + 90
#     rect_points = box
#     angle_rad = np.deg2rad(angle)
#     direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
#     projections = [np.dot(point - center, direction) for point in rect_points]
#     projections = np.array(projections)
#     max_proj_idx = np.argmax(projections)
#     min_proj_idx = np.argmin(projections)
#     tip_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[max_proj_idx]) < width * 0.1]
#     base_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[min_proj_idx]) < width * 0.1]
#     def calculate_end_coverage(mask, center_point, direction_vec, distance):
#         end_point = center_point + direction_vec * distance
#         perp_vec = np.array([-direction_vec[1], direction_vec[0]])
#         coverage = 0
#         sample_range = int(height)
#         for offset in np.linspace(-sample_range / 2, sample_range / 2, 10):
#             sample_point = end_point + perp_vec * offset
#             x, y = int(sample_point[0]), int(sample_point[1])
#             if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
#                 coverage += 1
#         return coverage
#     coverage_max = calculate_end_coverage(mask, np.array(center), direction, projections[max_proj_idx])
#     coverage_min = calculate_end_coverage(mask, np.array(center), direction, projections[min_proj_idx])
#     frame_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
#     dist_max_to_center = np.linalg.norm(np.mean([rect_points[i] for i in tip_end_indices], axis=0) - frame_center)
#     dist_min_to_center = np.linalg.norm(np.mean([rect_points[i] for i in base_end_indices], axis=0) - frame_center)
#     use_coverage = abs(coverage_max - coverage_min) > 3
#     if (use_coverage and coverage_max < coverage_min) or (not use_coverage and dist_max_to_center < dist_min_to_center):
#         tip_end_points = [rect_points[i] for i in tip_end_indices]
#         tip_direction = direction
#     else:
#         tip_end_points = [rect_points[i] for i in base_end_indices]
#         tip_direction = -direction
#     if len(tip_end_points) >= 2:
#         midpoint_tip = np.mean(tip_end_points, axis=0)
#         line_start = np.array(center) + tip_direction * (min(projections) + (max(projections) - min(projections)) * 0.3)
#         midpoint_line = (tuple(line_start.astype(int)), tuple(midpoint_tip.astype(int)))
#         tip_point = tuple(midpoint_tip.astype(int))
#         orientation_angle = np.degrees(np.arctan2(tip_point[1] - midpoint_line[0][1], tip_point[0] - midpoint_line[0][0]))
#         if debug_frame is not None:
#              cv2.circle(debug_frame, tip_point, 10, (0, 0, 255), -1)
#         return tip_point, orientation_angle, (tuple(tip_end_points[0]), tuple(tip_end_points[1])), midpoint_line
#     return None, None, None, None

# def rotate_image(img, angle):
#     h, w = img.shape[:2]
#     center = (w // 2, h // 2)
#     rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
#     return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

# def compute_iou(mask1, mask2):
#     intersection = np.logical_and(mask1.astype(bool), mask2.astype(bool)).sum()
#     union = np.logical_or(mask1.astype(bool), mask2.astype(bool)).sum()
#     return intersection / union if union > 0 else 0

# def pixel_variance_score(dut_img, mask):
#     region = dut_img[mask == 1]
#     return float(np.var(region)) if len(region) > 0 else 0

# def align_with_iou_and_variance(cropped_img, golden_mask, min_variance=20):
#     if cropped_img is None or golden_mask is None: return None, None, 0, 0
#     dut_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img.copy()
#     golden_mask_resized = cv2.resize(golden_mask, (dut_img.shape[1], dut_img.shape[0]))
#     _, dut_mask = cv2.threshold(dut_img, 80, 1, cv2.THRESH_BINARY)
#     best_score, best_angle, best_rotated = -1, 0, None
#     for angle in range(0, 360, 5):
#         rotated = rotate_image(golden_mask_resized, angle)
#         if pixel_variance_score(dut_img, rotated) < min_variance: continue
#         score = compute_iou(dut_mask, rotated > 0)
#         if score > best_score:
#             best_score, best_angle, best_rotated = score, angle, rotated
#     return dut_img, best_rotated, best_angle, best_score

# def extract_waypoints(mask, num_points=1):
#     ys, xs = np.where(mask > 0)
#     points = np.column_stack((xs, ys))
#     if len(points) < num_points: return []
#     M = cv2.moments(mask.astype(np.uint8))
#     if M["m00"] == 0: return []
#     centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#     if num_points == 1: return [centroid]
#     kmeans = KMeans(n_clusters=num_points, n_init=10, random_state=0).fit(points)
#     return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

# def run_template_matching_dual(cropped_img, golden_mask_paths, object_id):
#     results = {}
#     golden_mask_GR1 = cv2.imread(golden_mask_paths["GR1"], cv2.IMREAD_GRAYSCALE)
#     if golden_mask_GR1 is None:
#         print(f"Error: GR1 mask missing at {golden_mask_paths['GR1']}!")
#         return None
#     _, golden_mask_GR1 = cv2.threshold(golden_mask_GR1, 127, 1, cv2.THRESH_BINARY)
#     dut_img, mask_GR1, angle_GR1, score_GR1 = align_with_iou_and_variance(cropped_img, golden_mask_GR1)
#     if mask_GR1 is None:
#         print("Failed to align with GR1")
#         return None
#     results["GR1"] = {"iou": float(score_GR1), "angle": int(angle_GR1), "waypoints": extract_waypoints(mask_GR1, num_points=1)}
#     overlay = cv2.cvtColor(dut_img, cv2.COLOR_GRAY2BGR)
#     overlay[mask_GR1 == 1] = [255, 0, 0] # Blue for GR1
#     for (x, y) in results["GR1"]["waypoints"]:
#         cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
#     golden_mask_GR2 = cv2.imread(golden_mask_paths["GR2"], cv2.IMREAD_GRAYSCALE) if golden_mask_paths.get("GR2") else None
#     if golden_mask_GR2 is not None:
#         _, golden_mask_GR2 = cv2.threshold(golden_mask_GR2, 127, 1, cv2.THRESH_BINARY)
#         mask_GR2 = rotate_image(cv2.resize(golden_mask_GR2, (dut_img.shape[1], dut_img.shape[0])), angle_GR1)
#         results["GR2"] = {"angle": int(angle_GR1), "waypoints": extract_waypoints(mask_GR2, num_points=1)}
#         overlay[mask_GR2 == 1] = [0, 0, 255] # Red for GR2
#         for (x, y) in results["GR2"]["waypoints"]:
#             cv2.circle(overlay, (x, y), 5, (0, 255, 255), -1)
#     results["overlay"] = overlay
#     return results

# def calculate_distance(box1, box2):
#     c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
#     c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
#     return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) # <-- Corrected bug from your code

# def point_in_box(point, bbox):
#     return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

# def reset_state():
#     return {'current_state': 0, 'dut_bbox': None, 'template_results': None, 'stationary_start_time': None, 'last_bbox': None, 'movement_initiated': False}

# # ============================================================================
# # Main Alignment Functions (MODIFIED FOR WEB UI)
# # ============================================================================
# def align_to_waypoints(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, state, settings):

#     needle_color = settings['last_needle_color']
#     pixel_to_robot_matrix = settings['calibration_matrix']
#     tolerance_mm = settings.get('alignment_tolerance', 0.1)
#     max_iterations = 15
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
#     # resize = True # Handled by camera thread if needed
    
#     socketio.emit('status_update', {'msg': "ALIGNMENT PROCESS STARTED"})
    
#     all_waypoints_512 = state['template_results']["GR1"]["waypoints"]
#     if state['template_results'].get("GR2"):
#         all_waypoints_512.extend(state['template_results']["GR2"]["waypoints"])
        
#     xmin, ymin, xmax, ymax = state['dut_bbox']
#     scale_x = (xmax - xmin) / 512.0
#     scale_y = (ymax - ymin) / 512.0
    
#     alignment_results = []
    
#     for wp_idx, wp_512 in enumerate(all_waypoints_512):
#         if stop_event.is_set(): return False

#         socketio.emit('status_update', {'msg': f"Targeting Waypoint {wp_idx+1}/{len(all_waypoints_512)}"})
#         target_wp_x_full = (wp_512[0] * scale_x) + xmin
#         target_wp_y_full = (wp_512[1] * scale_y) + ymin
#         target_point_display = (int(target_wp_x_full), int(target_wp_y_full))

#         for iteration in range(max_iterations):
#             if stop_event.is_set(): return False
            
#             # --- NON-BLOCKING FRAME RETRIEVAL ---
#             with camera_lock:
#                 frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
#             if frame is None:
#                 socketio.sleep(0.1) # Wait for a frame if buffer is empty
#                 continue
            
#             display_frame = frame.copy()
#             cv2.circle(display_frame, target_point_display, 12, (0, 255, 0), 2)
#             cv2.putText(display_frame, f"Target WP {wp_idx+1}", (target_point_display[0] + 15, target_point_display[1]),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             tip_info = detect_needle_tip_rectangular(frame, needle_color)
#             if not (tip_info and tip_info[0]):
#                 socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Cannot detect tip, retrying..."})
#                 cv2.putText(display_frame, "TIP NOT DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 send_frame_to_ui(socketio, display_frame) # Show failure
#                 socketio.sleep(0.5)
#                 continue
            
#             tip_point_full_frame = tip_info[0]
            
#             cv2.circle(display_frame, tip_point_full_frame, 12, (0, 0, 255), -1)
#             cv2.line(display_frame, tip_point_full_frame, target_point_display, (255, 255, 0), 2)

#             error_dx_full = target_wp_x_full - tip_point_full_frame[0]
#             error_dy_full = target_wp_y_full - tip_point_full_frame[1]
#             pixel_error_vector = np.array([error_dx_full, error_dy_full])
            
#             rotation_scale_matrix = pixel_to_robot_matrix[:, :2]
#             cobot_move_mm = rotation_scale_matrix @ pixel_error_vector
#             move_x_mm, move_y_mm = cobot_move_mm
#             error_magnitude_mm = np.linalg.norm(cobot_move_mm)
            
#             cv2.putText(display_frame, f"Iter: {iteration+1}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#             cv2.putText(display_frame, f"Error: {error_magnitude_mm:.2f} mm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#             send_frame_to_ui(socketio, display_frame) # Show progress
            
#             socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Tip at {tip_point_full_frame}. Error: {error_magnitude_mm:.3f} mm"})
            
#             if error_magnitude_mm < tolerance_mm:
#                 socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} reached!"})
#                 alignment_results.append({"waypoint_id": wp_idx, "success": True, "error_mm": error_magnitude_mm})
#                 socketio.sleep(1)
#                 break
            
#             damping = 0.7
#             move_x_damped, move_y_damped = move_x_mm * damping, move_y_mm * damping
            
#             if not cobot.move_relative(dx=move_x_damped, dy=move_y_damped, speed=8, block=True):
#                 socketio.emit('status_update', {'msg': "Robot movement failed.", 'error': True})
#                 alignment_results.append({"waypoint_id": wp_idx, "success": False, "error": "Movement failed"})
#                 break
#             socketio.sleep(0.3)
#         else:
#             socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} - Max iterations reached.", 'error': True})
#             alignment_results.append({"waypoint_id": wp_idx, "success": False, "error": "Max iterations"})

#         # Wait for user to press "Proceed"
#         msg = f"Waypoint {wp_idx+1} alignment finished. Press Proceed to continue."
#         if not wait_for_proceed(socketio, proceed_event, stop_event, display_frame, msg):
#              return False # Stop event was triggered

#     socketio.emit('status_update', {'msg': "ALIGNMENT COMPLETE"})
#     return alignment_results

# # ============================================================================
# # Main Process Loop (Replaces your old main())
# # ============================================================================
# def run_alignment_process(socketio, proceed_event, stop_event, request_process_stop_event, cobot, camera_frame_buffer, camera_lock, settings, model_path):
    
#     # Load settings from the passed dictionary
#     needle_color = settings['last_needle_color']
#     alignment_standby_joints = settings.get('alignment_standby_joints')
#     standby_offset_mm = settings.get('last_offset_mm', 50.0) # Use last_offset_mm as post-alignment move
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
#     try:
#         settings['calibration_matrix'] = np.load('calibration_matrix.npy')
#         socketio.emit('status_update', {'msg': "Loaded calibration matrix 'calibration_matrix.npy'."})
#     except FileNotFoundError:
#         socketio.emit('status_update', {'msg': "'calibration_matrix.npy' not found! Run calibration first.", 'error': True})
#         return

#     model = YOLO(model_path)
#     # Ensure correct paths for golden masks, relative to cobot_logic.py or server root
#     golden_masks = {
#         "GR1": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_mask_GR1.jpg'),
#         "GR2": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_mask_GR2.png')
#     }
#     # Check if files exist to provide better error messages
#     if not os.path.exists(golden_masks["GR1"]):
#         socketio.emit('status_update', {'msg': f"Error: golden_mask_GR1.jpg not found at {golden_masks['GR1']}", 'error': True})
#         return
#     if not os.path.exists(golden_masks["GR2"]):
#         socketio.emit('status_update', {'msg': f"Warning: golden_mask_GR2.png not found at {golden_masks['GR2']}", 'error': False})
#         # Not returning, as GR2 might be optional


#     STATE_DETECTING_DUT, STATE_MOVING_TO_POSITION, STATE_ALIGNING, STATE_COMPLETE = 0, 1, 2, 3
#     state = reset_state()
#     STATIONARY_SECONDS = 2
#     completion_time = None
#     auto_reset_delay = 5.0 # TODO: Make this a setting?

#     while not stop_event.is_set():
#         # --- NON-BLOCKING FRAME RETRIEVAL ---
#         with camera_lock:
#             frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None

#         if frame is None:
#             socketio.sleep(0.1) # Wait for a frame if buffer is empty
#             continue
            
#         display_frame = frame.copy()

#         if state['current_state'] == STATE_DETECTING_DUT:
            
#             # --- NEW "SOFT" STOP CHECK ---
#             # Check if a graceful stop has been requested
#             if request_process_stop_event.is_set():
#                 socketio.emit('status_update', {'msg': 'Process stop requested. Halting detection loop.'})
#                 break # Exit the main while loop gracefully
#             # --- END NEW CHECK ---

#             results = model(frame, verbose=False, conf=0.6)
            
#             if len(results[0].boxes) > 0:
#                 box = results[0].boxes[0]
#                 bbox = box.xyxy[0].cpu().numpy().astype(int)
                
#                 if state['last_bbox'] is None:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 if calculate_distance(bbox, state['last_bbox']) < 30: # Corrected variable here (c1[0] - c1[0]) -> (c1[0] - c2[0])
#                     if time.time() - state['stationary_start_time'] > STATIONARY_SECONDS:
#                         socketio.emit('status_update', {'msg': 'DUT is static. Performing template matching...'})
#                         state['dut_bbox'] = bbox
#                         dut_crop_resized = cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (512, 512))
                        
#                         state['template_results'] = run_template_matching_dual(dut_crop_resized, golden_masks, 0)
                        
#                         if state['template_results']:
#                             # This is the key change: wait for user proceed
#                             msg = "Template match found. Press Proceed to start alignment."
#                             if not wait_for_proceed(socketio, proceed_event, stop_event, state['template_results']['overlay'], msg):
#                                 state = reset_state() # User cancelled (stop)
#                                 continue

#                             # User proceeded
#                             if alignment_standby_joints and sum(abs(j) for j in alignment_standby_joints) > 0:
#                                 state['current_state'] = STATE_MOVING_TO_POSITION
#                             else:
#                                 socketio.emit('status_update', {'msg': 'No standby joints defined. Skipping move.'})
#                                 state['current_state'] = STATE_ALIGNING
#                         else:
#                             socketio.emit('status_update', {'msg': 'Template matching failed.', 'error': True})
#                             state = reset_state()
#                 else:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#             else:
#                 state['last_bbox'] = None
                
#             send_frame_to_ui(socketio, display_frame) # Send live feed

#         elif state['current_state'] == STATE_MOVING_TO_POSITION:
#             if not state['movement_initiated']:
#                 socketio.emit('status_update', {'msg': 'Moving to prefixed standby pose...'})
#                 state['movement_initiated'] = True
#                 if cobot.joint_move(alignment_standby_joints, speed=20, block=True):
#                     socketio.emit('status_update', {'msg': 'Standby position reached. Proceeding to alignment.'})
#                     state['current_state'] = STATE_ALIGNING
#                 else:
#                     socketio.emit('status_update', {'msg': 'Failed to move to standby position. Resetting.', 'error': True})
#                     state = reset_state()

#         elif state['current_state'] == STATE_ALIGNING:
#             socketio.emit('status_update', {'msg': 'Starting alignment procedure...'})
            
#             align_to_waypoints(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, state, settings)      
                    
#             socketio.emit('status_update', {'msg': f"Alignment finished. Moving {standby_offset_mm}mm to standby position."})
#             if cobot:
#                 cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
#                 socketio.emit('status_update', {'msg': 'Robot is in post-alignment standby.'})
                
#             state['current_state'] = STATE_COMPLETE
#             completion_time = time.time()

#         elif state['current_state'] == STATE_COMPLETE:
#             if time.time() - completion_time > auto_reset_delay:
#                 socketio.emit('status_update', {'msg': 'Cycle complete. Starting new detection cycle...'})
#                 state = reset_state()
#             else:
#                 cv2.putText(display_frame, "COMPLETE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
#                 send_frame_to_ui(socketio, display_frame) # Send live feed

#         socketio.sleep(0.02) # Main loop sleep to yield control
        
#     socketio.emit('status_update', {'msg': 'Alignment process stopped.'})




#improved template matching and bounding box
# import os
# import sys
# import time
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict
# import math
# import json
# from sklearn.cluster import KMeans
# import socket
# import struct
# import base64
# import threading

# # due to masks folder path change
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# # ============================================================================
# # Helper Functions for Web UI
# # ============================================================================

# def send_frame_to_ui(socketio, frame, event_name='video_frame'):
#     """Encodes a CV2 frame as a base64 JPEG and emits it."""
#     try:
#         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Lower quality for faster transfer
#         jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#         socketio.emit(event_name, {'image': jpg_as_text})
#     except Exception as e:
#         print(f"Error sending frame to UI: {e}")

# def wait_for_proceed(socketio, proceed_event, stop_event, frame, message):
#     """
#     Sends an image to the UI, emits a 'wait_for_proceed' status,
#     and then blocks until the 'proceed_event' is set (by the user).
#     """
#     socketio.emit('status_update', {'msg': f"WAITING: {message}"})
    
#     # Send the specific frame for the user to see (with good quality)
#     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
#     jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#     socketio.emit('show_image_and_wait', {'image': jpg_as_text})

#     # Wait for the proceed_event or stop_event
#     proceed_event.clear() # Ensure it's clear before waiting
#     while not proceed_event.is_set() and not stop_event.is_set():
#         socketio.sleep(0.1) # Non-blocking sleep

#     if stop_event.is_set():
#         socketio.emit('status_update', {'msg': 'Proceed cancelled by stop request.'})
#         return False # Stopped
        
#     socketio.emit('status_update', {'msg': 'User proceeded.'})
#     socketio.emit('hide_popup') # Tell UI to hide the popup
#     return True # Proceeded

# # ============================================================================
# # JAKA Cobot Communication Class (No changes)
# # ============================================================================
# class JAKACobot:
#     """JAKA Cobot TCP/IP communication and control using the official JSON protocol."""

#     def __init__(self, ip="192.168.1.166", port=10001):
#         self.ip = ip
#         self.port = port
#         self.socket = None
#         self.connected = False

#     def connect(self):
#         try:
#             self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.socket.settimeout(15.0)
#             self.socket.connect((self.ip, self.port))
#             self.connected = True
#             print(f"[JAKA] Connected to cobot at {self.ip}:{self.port}")
#             return True
#         except Exception as e:
#             print(f"[JAKA ERROR] Connection failed: {e}")
#             self.connected = False
#             return False

#     def disconnect(self):
#         if self.socket:
#             self.socket.close()
#             self.connected = False
#             print("[JAKA] Disconnected from cobot")

#     def send_command(self, command_dict):
#         if not self.connected:
#             print("[JAKA ERROR] Not connected to cobot")
#             return None
#         try:
#             json_command = json.dumps(command_dict) + '\n'
#             self.socket.sendall(json_command.encode('utf-8'))
#             response_str = self.socket.recv(4096).decode('utf-8').strip()
#             if response_str:
#                 return json.loads(response_str)
#             return None
#         except (json.JSONDecodeError, socket.timeout, ConnectionResetError) as e:
#             print(f"[JAKA ERROR] Communication error: {e}")
#             return None

#     def _check_response(self, response_json, command_name):
#         if response_json and str(response_json.get("errorCode")) == "0":
#             return True
#         else:
#             msg = response_json.get('errorMsg', 'Unknown error') if response_json else 'No response'
#             print(f"[JAKA ERROR] {command_name} failed: {msg}")
#             return False

#     def power_on(self):
#         cmd = {"cmdName": "power_on"}
#         response = self.send_command(cmd)
#         return self._check_response(response, "power_on")

#     def enable_robot(self):
#         cmd = {"cmdName": "enable_robot"}
#         response = self.send_command(cmd)
#         return self._check_response(response, "enable_robot")

#     def get_tcp_position(self):
#         cmd = {"cmdName": "get_tcp_pos"}
#         response = self.send_command(cmd)
#         if response and str(response.get("errorCode")) == "0":
#             tcp_data = response.get("tcp_pos")
#             if tcp_data and len(tcp_data) == 6:
#                 return [float(p) for p in tcp_data]
#         print(f"[JAKA ERROR] Failed to get TCP position. Response: {response}")
#         return None

#     def get_joint_position(self):
#         cmd = {"cmdName": "get_joint_pos"}
#         response = self.send_command(cmd)
#         if response and str(response.get("errorCode")) == "0":
#             joint_data = response.get("joint_pos")
#             if joint_data and len(joint_data) == 6:
#                 return [float(p) for p in joint_data]
#         print(f"[JAKA ERROR] Failed to get Joint position. Response: {response}")
#         return None

#     def joint_move(self, joint_positions, speed=20, accel=50, tol=0.5, block=True):
#         cmd = {
#             "cmdName": "joint_move", "relFlag":0,
#             "jointPosition": [float(p) for p in joint_positions],
#             "speed": speed, "accel": accel, "tol": tol
#         }
#         response = self.send_command(cmd)
#         if not self._check_response(response, "joint_move (joint_move)"):
#             return False

#         print("[JAKA] moveJ command accepted.")
#         if block:
#             print("[JAKA] Waiting for joint move to complete...")
#             start_time = time.time()
#             while time.time() - start_time < 30: # 30 second timeout
#                 current_joints = self.get_joint_position()
#                 if current_joints:
#                     diff = np.array(joint_positions) - np.array(current_joints)
#                     dist = np.linalg.norm(diff)
#                     if dist < (tol + 0.5):
#                         print("[JAKA] Joint move complete.")
#                         return True
#                 time.sleep(0.1)
#             print("[JAKA ERROR] Joint move timed out.")
#             return False
#         return True

#     def linear_move(self, tcp_position, speed=20, accel=50, tol=0.5, relative=False, block=True):
#         cmd = {
#             "cmdName": "moveL", "relFlag": 1 if relative else 0,
#             "cartPosition": [float(p) for p in tcp_position],
#             "speed": speed, "accel": accel, "tol": tol
#         }
#         response = self.send_command(cmd)
#         if not self._check_response(response, "moveL (linear_move)"):
#             return False

#         print("[JAKA] moveL command accepted.")
#         if block and not relative:
#             print("[JAKA] Waiting for linear move to complete...")
#             start_time = time.time()
#             while time.time() - start_time < 30:
#                 current_pos = self.get_tcp_position()
#                 if current_pos:
#                     dist = math.hypot(
#                         tcp_position[0] - current_pos[0],
#                         tcp_position[1] - current_pos[1],
#                         tcp_position[2] - current_pos[2]
#                     )
#                     if dist < tol:
#                         print("[JAKA] Linear move complete.")
#                         return True
#                 time.sleep(0.1)
#             print("[JAKA ERROR] Linear move timed out.")
#             return False
#         elif block and relative:
#             time.sleep(1.0) # Small delay for relative moves to ensure completion
#         return True

#     def move_relative(self, dx=0, dy=0, dz=0, speed=10, block=True):
#         relative_pose = [dx, dy, dz, 0, 0, 0]
#         return self.linear_move(relative_pose, speed=speed, relative=True, block=block)

# # ============================================================================
# # Standalone Calibration Routine (No changes)
# # ============================================================================
# def run_calibration_routine(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, settings):
    
#     needle_color = settings['last_needle_color']
#     move_dist = settings['last_move_mm']
#     standby_offset_mm = settings['last_offset_mm']
#     calibration_start_joints = settings.get('calibration_start_joints')

#     socketio.emit('status_update', {'msg': 'Calibration routine starting...'})
    
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)

#     calib_img_dir = "calibration_images"
#     os.makedirs(calib_img_dir, exist_ok=True)
#     socketio.emit('status_update', {'msg': f"Images will be saved in '{calib_img_dir}/'"})

#     robot_points_mm = np.float32([[0, 0], [move_dist, 0], [0, move_dist]])
#     pixel_points = []

#     # 1. Move to start position
#     if calibration_start_joints and sum(abs(j) for j in calibration_start_joints) > 0:
#         socketio.emit('status_update', {'msg': 'Moving to pre-defined calibration start position...'})
#         if not cobot.joint_move(calibration_start_joints, speed=20, block=True):
#              socketio.emit('status_update', {'msg': 'Failed to move to start position. Aborting.', 'error': True})
#              return False
#     else:
#         socketio.emit('status_update', {'msg': 'No start joints defined. Using current position.'})
        
#     socketio.emit('status_update', {'msg': "Getting robot's starting position..."})
#     start_pos = cobot.get_tcp_position()
#     if not start_pos:
#         socketio.emit('status_update', {'msg': 'Could not get robot start position. Aborting.', 'error': True})
#         return False

#     for i, (dx, dy) in enumerate(robot_points_mm):
#         if stop_event.is_set(): return False
        
#         tip_found = False
#         point_num = i + 1
#         socketio.emit('status_update', {'msg': f"Moving to relative point {point_num}/3 ({dx}mm, {dy}mm)..."})

#         target_pose = start_pos.copy()
#         target_pose[0] += dx
#         target_pose[1] += dy

#         if not cobot.linear_move(target_pose, speed=15, block=True):
#             socketio.emit('status_update', {'msg': f"Robot failed to move to point {point_num}. Aborting.", 'error': True})
#             return False
#         socketio.sleep(1.0) # Wait for vibrations to settle

#         socketio.emit('status_update', {'msg': f"Detecting tip at point {point_num}..."})
#         for attempt in range(5):
#             if stop_event.is_set(): return False
            
#             with camera_lock:
#                 frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
#             if frame is None:
#                 socketio.sleep(0.1)
#                 continue

#             tip_info = detect_needle_tip_rectangular(frame, needle_color, debug_frame=frame)
            
#             if tip_info and tip_info[0]:
#                 tip_point = tip_info[0]
#                 pixel_points.append(tip_point)
#                 socketio.emit('status_update', {'msg': f"Tip detected at pixel coordinate: {tip_point}"})

#                 cv2.circle(frame, tip_point, 10, (0, 0, 255), -1)
#                 cv2.putText(frame, f"Point {point_num}", (tip_point[0] + 15, tip_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 send_frame_to_ui(socketio, frame) # Show detection
#                 socketio.sleep(0.5)

#                 filepath = os.path.join(calib_img_dir, f"calibration_point_{point_num}.png")
#                 cv2.imwrite(filepath, frame)
#                 tip_found = True
#                 break
#             else:
#                 cv2.putText(frame, f"Detection Failed (Attempt {attempt+1})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 send_frame_to_ui(socketio, frame) # Show failure
#                 socketio.sleep(0.5)

#         if not tip_found:
#             socketio.emit('status_update', {'msg': f"Could not detect tip at point {point_num}. Aborting.", 'error': True})
#             return False

#     if len(pixel_points) == 3:
#         socketio.emit('status_update', {'msg': "Calculating transformation matrix..."})
#         pixel_points_np = np.float32(pixel_points)
#         pixel_to_robot_matrix = cv2.getAffineTransform(pixel_points_np, robot_points_mm)
#         np.save('calibration_matrix.npy', pixel_to_robot_matrix)

#         socketio.emit('status_update', {'msg': "CALIBRATION SUCCESSFUL! Matrix saved."})
#         print("Matrix values:\n", pixel_to_robot_matrix)
#         success = True
#     else:
#         socketio.emit('status_update', {'msg': "Did not collect enough points. Calibration failed.", 'error': True})
#         success = False

#     socketio.emit('status_update', {'msg': f"Moving {standby_offset_mm}mm sideways to standby position..."})
#     cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
#     socketio.emit('status_update', {'msg': "Cobot is in standby. Calibration finished."})
    
#     return success

# # ============================================================================
# # Vision Processing Functions (No changes)
# # ============================================================================
# def detect_needle_tip_rectangular(image, color='red', debug_frame=None):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     if color == 'red':
#         lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
#         lower_red2, upper_red2 = np.array([170, 80, 80]), np.array([180, 255, 255])
#         mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
#     elif color == 'green':
#         mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
#     elif color == 'pink':
#         mask = cv2.inRange(hsv, np.array([140, 80, 80]), np.array([170, 255, 255]))
#     else: return None, None, None, None

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if debug_frame is not None:
#         cv2.drawContours(debug_frame, contours, -1, (255, 0, 0), 1)
#     if not contours: return None, None, None, None
#     needle_contour = max(contours, key=cv2.contourArea)
#     if len(needle_contour) < 5: return None, None, None, None
#     if debug_frame is not None:
#         cv2.drawContours(debug_frame, [needle_contour], -1, (0, 255, 0), 2)

#     rect = cv2.minAreaRect(needle_contour)
#     box = cv2.boxPoints(rect)
#     box = box.astype(int)
#     center, (width, height), angle = rect
#     if width < height:
#         width, height = height, width
#         angle = angle + 90
#     rect_points = box
#     angle_rad = np.deg2rad(angle)
#     direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
#     projections = [np.dot(point - center, direction) for point in rect_points]
#     projections = np.array(projections)
#     max_proj_idx = np.argmax(projections)
#     min_proj_idx = np.argmin(projections)
#     tip_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[max_proj_idx]) < width * 0.1]
#     base_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[min_proj_idx]) < width * 0.1]
#     def calculate_end_coverage(mask, center_point, direction_vec, distance):
#         end_point = center_point + direction_vec * distance
#         perp_vec = np.array([-direction_vec[1], direction_vec[0]])
#         coverage = 0
#         sample_range = int(height)
#         for offset in np.linspace(-sample_range / 2, sample_range / 2, 10):
#             sample_point = end_point + perp_vec * offset
#             x, y = int(sample_point[0]), int(sample_point[1])
#             if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
#                 coverage += 1
#         return coverage
#     coverage_max = calculate_end_coverage(mask, np.array(center), direction, projections[max_proj_idx])
#     coverage_min = calculate_end_coverage(mask, np.array(center), direction, projections[min_proj_idx])
#     frame_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
#     dist_max_to_center = np.linalg.norm(np.mean([rect_points[i] for i in tip_end_indices], axis=0) - frame_center)
#     dist_min_to_center = np.linalg.norm(np.mean([rect_points[i] for i in base_end_indices], axis=0) - frame_center)
#     use_coverage = abs(coverage_max - coverage_min) > 3
#     if (use_coverage and coverage_max < coverage_min) or (not use_coverage and dist_max_to_center < dist_min_to_center):
#         tip_end_points = [rect_points[i] for i in tip_end_indices]
#         tip_direction = direction
#     else:
#         tip_end_points = [rect_points[i] for i in base_end_indices]
#         tip_direction = -direction
#     if len(tip_end_points) >= 2:
#         midpoint_tip = np.mean(tip_end_points, axis=0)
#         line_start = np.array(center) + tip_direction * (min(projections) + (max(projections) - min(projections)) * 0.3)
#         midpoint_line = (tuple(line_start.astype(int)), tuple(midpoint_tip.astype(int)))
#         tip_point = tuple(midpoint_tip.astype(int))
#         orientation_angle = np.degrees(np.arctan2(tip_point[1] - midpoint_line[0][1], tip_point[0] - midpoint_line[0][0]))
#         if debug_frame is not None:
#              cv2.circle(debug_frame, tip_point, 10, (0, 0, 255), -1)
#         return tip_point, orientation_angle, (tuple(tip_end_points[0]), tuple(tip_end_points[1])), midpoint_line
#     return None, None, None, None

# def rotate_image(img, angle):
#     h, w = img.shape[:2]
#     center = (w // 2, h // 2)
#     rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
#     return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

# def compute_iou(mask1, mask2):
#     intersection = np.logical_and(mask1.astype(bool), mask2.astype(bool)).sum()
#     union = np.logical_or(mask1.astype(bool), mask2.astype(bool)).sum()
#     return intersection / union if union > 0 else 0

# def pixel_variance_score(dut_img, mask):
#     region = dut_img[mask == 1]
#     return float(np.var(region)) if len(region) > 0 else 0

# def align_with_iou_and_variance(cropped_img, golden_mask, min_variance=20):
#     if cropped_img is None or golden_mask is None: return None, None, 0, 0
#     dut_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img.copy()
#     golden_mask_resized = cv2.resize(golden_mask, (dut_img.shape[1], dut_img.shape[0]))
#     _, dut_mask = cv2.threshold(dut_img, 80, 1, cv2.THRESH_BINARY)
#     best_score, best_angle, best_rotated = -1, 0, None
#     for angle in range(0, 360, 5):
#         rotated = rotate_image(golden_mask_resized, angle)
#         if pixel_variance_score(dut_img, rotated) < min_variance: continue
#         score = compute_iou(dut_mask, rotated > 0)
#         if score > best_score:
#             best_score, best_angle, best_rotated = score, angle, rotated
#     return dut_img, best_rotated, best_angle, best_score

# def extract_waypoints(mask, num_points=1):
#     ys, xs = np.where(mask > 0)
#     points = np.column_stack((xs, ys))
#     if len(points) < num_points: return []
#     M = cv2.moments(mask.astype(np.uint8))
#     if M["m00"] == 0: return []
#     centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#     if num_points == 1: return [centroid]
#     # This part is now unreachable if num_points is always 1, which is fine.
#     kmeans = KMeans(n_clusters=num_points, n_init=10, random_state=0).fit(points)
#     return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

# # ===================================================================
# # --- MODIFICATION 1: run_template_matching_dual ---
# # ===================================================================
# def run_template_matching_dual(cropped_img, golden_mask_paths, object_id):
#     """
#     Uses GR3 to find the best angle, then applies that angle
#     to GR1 and GR2.
#     """
#     results = {}
    
#     # --- 1. Load all masks ---
#     golden_mask_GR1 = cv2.imread(golden_mask_paths["GR1"], cv2.IMREAD_GRAYSCALE)
#     golden_mask_GR2 = cv2.imread(golden_mask_paths["GR2"], cv2.IMREAD_GRAYSCALE)
#     golden_mask_GR3 = cv2.imread(golden_mask_paths["GR3"], cv2.IMREAD_GRAYSCALE)

#     if golden_mask_GR1 is None or golden_mask_GR2 is None or golden_mask_GR3 is None:
#         print(f"Error: Could not load one or more masks (GR1, GR2, or GR3).")
#         return None
    
#     # Binarize all masks
#     _, golden_mask_GR1 = cv2.threshold(golden_mask_GR1, 127, 1, cv2.THRESH_BINARY)
#     _, golden_mask_GR2 = cv2.threshold(golden_mask_GR2, 127, 1, cv2.THRESH_BINARY)
#     _, golden_mask_GR3 = cv2.threshold(golden_mask_GR3, 127, 1, cv2.THRESH_BINARY)
    
#     # --- 2. Align using GR3 to find the best angle ---
#     # `align_with_iou_and_variance` returns: dut_img, best_rotated_mask, best_angle, best_score
#     dut_img, mask_GR3_aligned, best_angle, score_GR3 = align_with_iou_and_variance(cropped_img, golden_mask_GR3)
    
#     if mask_GR3_aligned is None:
#         print("Failed to align with GR3 (alignment mask).")
#         return None
    
#     print(f"Template Matching: Found best angle {best_angle} using GR3 (IoU: {score_GR3:.3f})")

#     # --- 3. Apply this angle to GR1 and GR2 ---
#     # Resize golden masks *before* rotation
#     h, w = dut_img.shape[:2]
#     golden_mask_GR1_resized = cv2.resize(golden_mask_GR1, (w, h), interpolation=cv2.INTER_NEAREST)
#     golden_mask_GR2_resized = cv2.resize(golden_mask_GR2, (w, h), interpolation=cv2.INTER_NEAREST)
    
#     mask_GR1_rotated = rotate_image(golden_mask_GR1_resized, best_angle)
#     mask_GR2_rotated = rotate_image(golden_mask_GR2_resized, best_angle)

#     # --- 4. Extract waypoints (num_points=1 as requested) ---
#     results["GR1"] = {"angle": int(best_angle), "waypoints": extract_waypoints(mask_GR1_rotated, num_points=1)}
#     results["GR2"] = {"angle": int(best_angle), "waypoints": extract_waypoints(mask_GR2_rotated, num_points=1)}

#     # --- 5. Create overlay for visualization ---
#     overlay = cv2.cvtColor(dut_img, cv2.COLOR_GRAY2BGR)
#     overlay[mask_GR1_rotated == 1] = [255, 0, 0] # GR1 = Blue
#     overlay[mask_GR2_rotated == 1] = [0, 0, 255] # GR2 = Red
    
#     # Draw waypoints
#     for (x, y) in results["GR1"]["waypoints"]:
#         cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1) # GR1 waypoint = Green
#     for (x, y) in results["GR2"]["waypoints"]:
#         cv2.circle(overlay, (x, y), 5, (0, 255, 255), -1) # GR2 waypoint = Yellow

#     results["overlay"] = overlay
#     return results

# # ===================================================================
# # --- END MODIFICATION 1 ---
# # ===================================================================

# def calculate_distance(box1, box2):
#     # --- MODIFICATION 3: Fixed typo ---
#     c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
#     c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
#     return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) # <-- Corrected bug
#     # --- END MODIFICATION 3 ---

# def point_in_box(point, bbox):
#     return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

# def reset_state():
#     return {'current_state': 0, 'dut_bbox': None, 'template_results': None, 'stationary_start_time': None, 'last_bbox': None, 'movement_initiated': False}

# # ============================================================================
# # Main Alignment Functions (MODIFIED FOR BUG)
# # ============================================================================
# def align_to_waypoints(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, state, settings):

#     needle_color = settings['last_needle_color']
#     pixel_to_robot_matrix = settings['calibration_matrix']
#     tolerance_mm = settings.get('alignment_tolerance', 0.1)
#     max_iterations = 15
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
#     socketio.emit('status_update', {'msg': "ALIGNMENT PROCESS STARTED"})
    
#     all_waypoints_512 = state['template_results']["GR1"]["waypoints"]
#     if state['template_results'].get("GR2"):
#         all_waypoints_512.extend(state['template_results']["GR2"]["waypoints"])
        
#     xmin, ymin, xmax, ymax = state['dut_bbox']
#     scale_x = (xmax - xmin) / 512.0
#     scale_y = (ymax - ymin) / 512.0
    
#     alignment_results = []
    
#     for wp_idx, wp_512 in enumerate(all_waypoints_512):
#         if stop_event.is_set(): return False # Return False if stopped

#         socketio.emit('status_update', {'msg': f"Targeting Waypoint {wp_idx+1}/{len(all_waypoints_512)}"})
#         target_wp_x_full = (wp_512[0] * scale_x) + xmin
#         target_wp_y_full = (wp_512[1] * scale_y) + ymin
#         target_point_display = (int(target_wp_x_full), int(target_wp_y_full))

#         # --- MODIFICATION 2: Initialize variables before loop ---
#         error_magnitude_mm = 0.0 # Initialize to 0
#         iteration = 0 # Initialize iteration count
#         # --- END MODIFICATION 2 ---

#         for iteration in range(max_iterations):
#             if stop_event.is_set(): return False
            
#             with camera_lock:
#                 frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
#             if frame is None:
#                 socketio.sleep(0.1)
#                 continue
            
#             display_frame = frame.copy()
#             cv2.circle(display_frame, target_point_display, 12, (0, 255, 0), 2)
#             cv2.putText(display_frame, f"Target WP {wp_idx+1}", (target_point_display[0] + 15, target_point_display[1]),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             tip_info = detect_needle_tip_rectangular(frame, needle_color)
#             if not (tip_info and tip_info[0]):
#                 socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Cannot detect tip, retrying..."})
#                 cv2.putText(display_frame, "TIP NOT DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 send_frame_to_ui(socketio, display_frame) # Show failure
#                 socketio.sleep(0.5)
#                 continue
            
#             tip_point_full_frame = tip_info[0]
            
#             cv2.circle(display_frame, tip_point_full_frame, 12, (0, 0, 255), -1)
#             cv2.line(display_frame, tip_point_full_frame, target_point_display, (255, 255, 0), 2)

#             error_dx_full = target_wp_x_full - tip_point_full_frame[0]
#             error_dy_full = target_wp_y_full - tip_point_full_frame[1]
#             pixel_error_vector = np.array([error_dx_full, error_dy_full])
            
#             rotation_scale_matrix = pixel_to_robot_matrix[:, :2]
#             cobot_move_mm = rotation_scale_matrix @ pixel_error_vector
#             move_x_mm, move_y_mm = cobot_move_mm
#             error_magnitude_mm = np.linalg.norm(cobot_move_mm)
            
#             cv2.putText(display_frame, f"Iter: {iteration+1}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#             cv2.putText(display_frame, f"Error: {error_magnitude_mm:.2f} mm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#             send_frame_to_ui(socketio, display_frame) # Show progress
            
#             socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Tip at {tip_point_full_frame}. Error: {error_magnitude_mm:.3f} mm"})
            
#             if error_magnitude_mm < tolerance_mm:
#                 socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} reached!"})
#                 alignment_results.append({"waypoint_id": wp_idx, "success": True, "error_mm": error_magnitude_mm, "iterations": iteration+1})
#                 socketio.sleep(1)
#                 break
            
#             damping = 0.7
#             move_x_damped, move_y_damped = move_x_mm * damping, move_y_mm * damping
            
#             if not cobot.move_relative(dx=move_x_damped, dy=move_y_damped, speed=8, block=True):
#                 socketio.emit('status_update', {'msg': "Robot movement failed.", 'error': True})
#                 alignment_results.append({"waypoint_id": wp_idx, "success": False, "error_mm": error_magnitude_mm, "iterations": iteration+1, "error_msg": "Movement failed"})
#                 break
#             socketio.sleep(0.3)
#         else:
#             # This 'else' block runs if the for loop completes (max_iterations reached)
#             socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} - Max iterations reached.", 'error': True})
#             alignment_results.append({
#                 "waypoint_id": wp_idx, 
#                 "success": False, 
#                 "error_mm": error_magnitude_mm, # This is now safe due to initialization
#                 "iterations": max_iterations,
#                 "error_msg": "Max iterations"
#             })

#         # Wait for user to press "Proceed"
#         msg = f"Waypoint {wp_idx+1} alignment finished. Press Proceed to continue."
#         if not wait_for_proceed(socketio, proceed_event, stop_event, display_frame, msg):
#              return alignment_results # Return data collected so far if stopped

#     socketio.emit('status_update', {'msg': "ALIGNMENT COMPLETE"})
#     return alignment_results

# # ============================================================================
# # Main Process Loop (Replaces your old main())
# # ============================================================================
# def run_alignment_process(socketio, proceed_event, stop_event, request_process_stop_event, cobot, camera_frame_buffer, camera_lock, settings, model_path):
    
#     # Load settings from the passed dictionary
#     needle_color = settings['last_needle_color']
#     alignment_standby_joints = settings.get('alignment_standby_joints')
#     standby_offset_mm = settings.get('last_offset_mm', 50.0)
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
#     try:
#         settings['calibration_matrix'] = np.load('calibration_matrix.npy')
#         socketio.emit('status_update', {'msg': "Loaded calibration matrix 'calibration_matrix.npy'."})
#     except FileNotFoundError:
#         socketio.emit('status_update', {'msg': "'calibration_matrix.npy' not found! Run calibration first.", 'error': True})
#         return

#     model = YOLO(model_path)
    
#     # --- MODIFICATION: Define mask paths for GR1, GR2, and GR3 ---
#     golden_masks = {
#         "GR1": os.path.join(PARENT_DIR, 'masks', 'mask_1.png'),
#         "GR2": os.path.join(PARENT_DIR, 'masks', 'mask_2.jpg'),
#         "GR3": os.path.join(PARENT_DIR, 'masks', 'mask_3.png')
#     }
    
#     # Check if files exist to provide better error messages
#     if not os.path.exists(golden_masks["GR1"]):
#         socketio.emit('status_update', {'msg': f"Error: mask_1.png not found at {golden_masks['GR1']}", 'error': True})
#         return
#     if not os.path.exists(golden_masks["GR2"]):
#         socketio.emit('status_update', {'msg': f"Warning: mask_2.jpg not found at {golden_masks['GR2']}", 'error': False})
#     if not os.path.exists(golden_masks["GR3"]):
#         socketio.emit('status_update', {'msg': f"Error: mask_3.png (Alignment Mask) not found at {golden_masks['GR3']}", 'error': True})
#         return
#     # --- END MODIFICATION ---


#     STATE_DETECTING_DUT, STATE_MOVING_TO_POSITION, STATE_ALIGNING, STATE_COMPLETE = 0, 1, 2, 3
#     state = reset_state()
#     STATIONARY_SECONDS = 2
#     completion_time = None
#     auto_reset_delay = 5.0

#     while not stop_event.is_set():
#         with camera_lock:
#             frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None

#         if frame is None:
#             socketio.sleep(0.1)
#             continue
            
#         display_frame = frame.copy()

#         if state['current_state'] == STATE_DETECTING_DUT:
            
#             if request_process_stop_event.is_set():
#                 socketio.emit('status_update', {'msg': 'Process stop requested. Halting detection loop.'})
#                 break
            
#             # --- MODIFICATION 2: Set conf=0.96 ---
#             results = model(frame, verbose=False, conf=0.96)
            
#             if len(results[0].boxes) > 0:
                
#                 # --- MODIFICATION 2: Find box with highest confidence ---
#                 all_boxes = results[0].boxes
#                 confs = all_boxes.conf.cpu().numpy()
#                 best_idx = np.argmax(confs)
#                 box = all_boxes[best_idx] # This is now the box with the highest confidence
#                 # --- END MODIFICATION ---
                
#                 bbox = box.xyxy[0].cpu().numpy().astype(int)
                
#                 if state['last_bbox'] is None:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 if calculate_distance(bbox, state['last_bbox']) < 30: 
#                     if time.time() - state['stationary_start_time'] > STATIONARY_SECONDS:
#                         socketio.emit('status_update', {'msg': 'DUT is static. Performing template matching...'})
#                         state['dut_bbox'] = bbox
#                         dut_crop_resized = cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (512, 512))
                        
#                         state['template_results'] = run_template_matching_dual(dut_crop_resized, golden_masks, 0)
                        
#                         if state['template_results']:
#                             msg = "Template match found. Press Proceed to start alignment."
#                             if not wait_for_proceed(socketio, proceed_event, stop_event, state['template_results']['overlay'], msg):
#                                 state = reset_state()
#                                 continue

#                             if alignment_standby_joints and sum(abs(j) for j in alignment_standby_joints) > 0:
#                                 state['current_state'] = STATE_MOVING_TO_POSITION
#                             else:
#                                 socketio.emit('status_update', {'msg': 'No standby joints defined. Skipping move.'})
#                                 state['current_state'] = STATE_ALIGNING
#                         else:
#                             socketio.emit('status_update', {'msg': 'Template matching failed.', 'error': True})
#                             state = reset_state()
#                 else:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#             else:
#                 state['last_bbox'] = None
                
#             send_frame_to_ui(socketio, display_frame) # Send live feed

#         elif state['current_state'] == STATE_MOVING_TO_POSITION:
#             if not state['movement_initiated']:
#                 socketio.emit('status_update', {'msg': 'Moving to prefixed standby pose...'})
#                 state['movement_initiated'] = True
#                 if cobot.joint_move(alignment_standby_joints, speed=20, block=True):
#                     socketio.emit('status_update', {'msg': 'Standby position reached. Proceeding to alignment.'})
#                     state['current_state'] = STATE_ALIGNING
#                 else:
#                     socketio.emit('status_update', {'msg': 'Failed to move to standby position. Resetting.', 'error': True})
#                     state = reset_state()

#         elif state['current_state'] == STATE_ALIGNING:
#             socketio.emit('status_update', {'msg': 'Starting alignment procedure...'})
            
#             # This function returns a list of results, which we will log
#             alignment_data = align_to_waypoints(
#                 socketio, proceed_event, stop_event, cobot, 
#                 camera_frame_buffer, camera_lock, state, settings
#             )
            
#             # --- REMOVED: CSV Logging Call ---
#             # --- You will log this manually based on the console output ---
                    
#             socketio.emit('status_update', {'msg': f"Alignment finished. Moving {standby_offset_mm}mm to standby position."})
#             if cobot:
#                 cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
#                 socketio.emit('status_update', {'msg': 'Robot is in post-alignment standby.'})
                
#             state['current_state'] = STATE_COMPLETE
#             completion_time = time.time()

#         elif state['current_state'] == STATE_COMPLETE:
#             if time.time() - completion_time > auto_reset_delay:
#                 socketio.emit('status_update', {'msg': 'Cycle complete. Starting new detection cycle...'})
#                 state = reset_state()
#             else:
#                 cv2.putText(display_frame, "COMPLETE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
#                 send_frame_to_ui(socketio, display_frame) # Send live feed

#         socketio.sleep(0.02) # Main loop sleep to yield control
        
#     socketio.emit('status_update', {'msg': 'Alignment process stopped.'})



#------------------with demo mode
# import os
# import sys
# import time
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict
# import math
# import json
# from sklearn.cluster import KMeans
# import socket
# import struct
# import base64
# import threading

# # due to masks folder path change
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # PARENT_DIR is E:\yolo_new
# PARENT_DIR = os.path.dirname(SCRIPT_DIR) 

# # ============================================================================
# # Helper Functions for Web UI
# # ============================================================================

# def send_frame_to_ui(socketio, frame, event_name='video_frame'):
#     """Encodes a CV2 frame as a base64 JPEG and emits it."""
#     try:
#         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Lower quality for faster transfer
#         jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#         socketio.emit(event_name, {'image': jpg_as_text})
#     except Exception as e:
#         print(f"Error sending frame to UI: {e}")

# def wait_for_proceed(socketio, proceed_event, stop_event, frame, message):
#     """
#     Sends an image to the UI, emits a 'wait_for_proceed' status,
#     and then blocks until the 'proceed_event' is set (by the user).
#     """
#     socketio.emit('status_update', {'msg': f"WAITING: {message}"})
    
#     # Send the specific frame for the user to see (with good quality)
#     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
#     jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#     socketio.emit('show_image_and_wait', {'image': jpg_as_text})

#     # Wait for the proceed_event or stop_event
#     proceed_event.clear() # Ensure it's clear before waiting
#     while not proceed_event.is_set() and not stop_event.is_set():
#         socketio.sleep(0.1) # Non-blocking sleep

#     if stop_event.is_set():
#         socketio.emit('status_update', {'msg': 'Proceed cancelled by stop request.'})
#         return False # Stopped
        
#     socketio.emit('status_update', {'msg': 'User proceeded.'})
#     socketio.emit('hide_popup') # Tell UI to hide the popup
#     return True # Proceeded

# # ============================================================================
# # JAKA Cobot Communication Class (No changes)
# # ============================================================================
# class JAKACobot:
#     """JAKA Cobot TCP/IP communication and control using the official JSON protocol."""

#     def __init__(self, ip="192.168.1.166", port=10001):
#         self.ip = ip
#         self.port = port
#         self.socket = None
#         self.connected = False

#     def connect(self):
#         try:
#             self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.socket.settimeout(15.0)
#             self.socket.connect((self.ip, self.port))
#             self.connected = True
#             print(f"[JAKA] Connected to cobot at {self.ip}:{self.port}")
#             return True
#         except Exception as e:
#             print(f"[JAKA ERROR] Connection failed: {e}")
#             self.connected = False
#             return False

#     def disconnect(self):
#         if self.socket:
#             self.socket.close()
#             self.connected = False
#             print("[JAKA] Disconnected from cobot")

#     def send_command(self, command_dict):
#         if not self.connected:
#             print("[JAKA ERROR] Not connected to cobot")
#             return None
#         try:
#             json_command = json.dumps(command_dict) + '\n'
#             self.socket.sendall(json_command.encode('utf-8'))
#             response_str = self.socket.recv(4096).decode('utf-8').strip()
#             if response_str:
#                 return json.loads(response_str)
#             return None
#         except (json.JSONDecodeError, socket.timeout, ConnectionResetError) as e:
#             print(f"[JAKA ERROR] Communication error: {e}")
#             return None

#     def _check_response(self, response_json, command_name):
#         if response_json and str(response_json.get("errorCode")) == "0":
#             return True
#         else:
#             msg = response_json.get('errorMsg', 'Unknown error') if response_json else 'No response'
#             print(f"[JAKA ERROR] {command_name} failed: {msg}")
#             return False

#     def power_on(self):
#         cmd = {"cmdName": "power_on"}
#         response = self.send_command(cmd)
#         return self._check_response(response, "power_on")

#     def enable_robot(self):
#         cmd = {"cmdName": "enable_robot"}
#         response = self.send_command(cmd)
#         return self._check_response(response, "enable_robot")

#     def get_tcp_position(self):
#         cmd = {"cmdName": "get_tcp_pos"}
#         response = self.send_command(cmd)
#         if response and str(response.get("errorCode")) == "0":
#             tcp_data = response.get("tcp_pos")
#             if tcp_data and len(tcp_data) == 6:
#                 return [float(p) for p in tcp_data]
#         print(f"[JAKA ERROR] Failed to get TCP position. Response: {response}")
#         return None

#     def get_joint_position(self):
#         cmd = {"cmdName": "get_joint_pos"}
#         response = self.send_command(cmd)
#         if response and str(response.get("errorCode")) == "0":
#             joint_data = response.get("joint_pos")
#             if joint_data and len(joint_data) == 6:
#                 return [float(p) for p in joint_data]
#         print(f"[JAKA ERROR] Failed to get Joint position. Response: {response}")
#         return None

#     def joint_move(self, joint_positions, speed=20, accel=50, tol=0.5, block=True):
#         cmd = {
#             "cmdName": "joint_move", "relFlag":0,
#             "jointPosition": [float(p) for p in joint_positions],
#             "speed": speed, "accel": accel, "tol": tol
#         }
#         response = self.send_command(cmd)
#         if not self._check_response(response, "joint_move (joint_move)"):
#             return False

#         print("[JAKA] moveJ command accepted.")
#         if block:
#             print("[JAKA] Waiting for joint move to complete...")
#             start_time = time.time()
#             while time.time() - start_time < 30: # 30 second timeout
#                 current_joints = self.get_joint_position()
#                 if current_joints:
#                     diff = np.array(joint_positions) - np.array(current_joints)
#                     dist = np.linalg.norm(diff)
#                     if dist < (tol + 0.5):
#                         print("[JAKA] Joint move complete.")
#                         return True
#                 time.sleep(0.1)
#             print("[JAKA ERROR] Joint move timed out.")
#             return False
#         return True

#     def linear_move(self, tcp_position, speed=20, accel=50, tol=0.5, relative=False, block=True):
#         cmd = {
#             "cmdName": "moveL", "relFlag": 1 if relative else 0,
#             "cartPosition": [float(p) for p in tcp_position],
#             "speed": speed, "accel": accel, "tol": tol
#         }
#         response = self.send_command(cmd)
#         if not self._check_response(response, "moveL (linear_move)"):
#             return False

#         print("[JAKA] moveL command accepted.")
#         if block and not relative:
#             print("[JAKA] Waiting for linear move to complete...")
#             start_time = time.time()
#             while time.time() - start_time < 30:
#                 current_pos = self.get_tcp_position()
#                 if current_pos:
#                     dist = math.hypot(
#                         tcp_position[0] - current_pos[0],
#                         tcp_position[1] - current_pos[1],
#                         tcp_position[2] - current_pos[2]
#                     )
#                     if dist < tol:
#                         print("[JAKA] Linear move complete.")
#                         return True
#                 time.sleep(0.1)
#             print("[JAKA ERROR] Linear move timed out.")
#             return False
#         elif block and relative:
#             time.sleep(1.0) # Small delay for relative moves to ensure completion
#         return True

#     def move_relative(self, dx=0, dy=0, dz=0, speed=10, block=True):
#         relative_pose = [dx, dy, dz, 0, 0, 0]
#         return self.linear_move(relative_pose, speed=speed, relative=True, block=block)

# # ============================================================================
# # Standalone Calibration Routine (No changes)
# # ============================================================================
# def run_calibration_routine(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, settings):
    
#     needle_color = settings['last_needle_color']
#     move_dist = settings['last_move_mm']
#     standby_offset_mm = settings['last_offset_mm']
#     calibration_start_joints = settings.get('calibration_start_joints')

#     socketio.emit('status_update', {'msg': 'Calibration routine starting...'})
    
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)

#     calib_img_dir = "calibration_images"
#     os.makedirs(calib_img_dir, exist_ok=True)
#     socketio.emit('status_update', {'msg': f"Images will be saved in '{calib_img_dir}/'"})

#     robot_points_mm = np.float32([[0, 0], [move_dist, 0], [0, move_dist]])
#     pixel_points = []

#     # 1. Move to start position
#     if calibration_start_joints and sum(abs(j) for j in calibration_start_joints) > 0:
#         socketio.emit('status_update', {'msg': 'Moving to pre-defined calibration start position...'})
#         if not cobot.joint_move(calibration_start_joints, speed=20, block=True):
#              socketio.emit('status_update', {'msg': 'Failed to move to start position. Aborting.', 'error': True})
#              return False
#     else:
#         socketio.emit('status_update', {'msg': 'No start joints defined. Using current position.'})
        
#     socketio.emit('status_update', {'msg': "Getting robot's starting position..."})
#     start_pos = cobot.get_tcp_position()
#     if not start_pos:
#         socketio.emit('status_update', {'msg': 'Could not get robot start position. Aborting.', 'error': True})
#         return False

#     for i, (dx, dy) in enumerate(robot_points_mm):
#         if stop_event.is_set(): return False
        
#         tip_found = False
#         point_num = i + 1
#         socketio.emit('status_update', {'msg': f"Moving to relative point {point_num}/3 ({dx}mm, {dy}mm)..."})

#         target_pose = start_pos.copy()
#         target_pose[0] += dx
#         target_pose[1] += dy

#         if not cobot.linear_move(target_pose, speed=15, block=True):
#             socketio.emit('status_update', {'msg': f"Robot failed to move to point {point_num}. Aborting.", 'error': True})
#             return False
#         socketio.sleep(1.0) # Wait for vibrations to settle

#         socketio.emit('status_update', {'msg': f"Detecting tip at point {point_num}..."})
#         for attempt in range(5):
#             if stop_event.is_set(): return False
            
#             with camera_lock:
#                 frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
#             if frame is None:
#                 socketio.sleep(0.1)
#                 continue

#             tip_info = detect_needle_tip_rectangular(frame, needle_color, debug_frame=frame)
            
#             if tip_info and tip_info[0]:
#                 tip_point = tip_info[0]
#                 pixel_points.append(tip_point)
#                 socketio.emit('status_update', {'msg': f"Tip detected at pixel coordinate: {tip_point}"})

#                 cv2.circle(frame, tip_point, 10, (0, 0, 255), -1)
#                 cv2.putText(frame, f"Point {point_num}", (tip_point[0] + 15, tip_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 send_frame_to_ui(socketio, frame) # Show detection
#                 socketio.sleep(0.5)

#                 filepath = os.path.join(calib_img_dir, f"calibration_point_{point_num}.png")
#                 cv2.imwrite(filepath, frame)
#                 tip_found = True
#                 break
#             else:
#                 cv2.putText(frame, f"Detection Failed (Attempt {attempt+1})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 send_frame_to_ui(socketio, frame) # Show failure
#                 socketio.sleep(0.5)

#         if not tip_found:
#             socketio.emit('status_update', {'msg': f"Could not detect tip at point {point_num}. Aborting.", 'error': True})
#             return False

#     if len(pixel_points) == 3:
#         socketio.emit('status_update', {'msg': "Calculating transformation matrix..."})
#         pixel_points_np = np.float32(pixel_points)
#         pixel_to_robot_matrix = cv2.getAffineTransform(pixel_points_np, robot_points_mm)
#         np.save('calibration_matrix.npy', pixel_to_robot_matrix)

#         socketio.emit('status_update', {'msg': "CALIBRATION SUCCESSFUL! Matrix saved."})
#         print("Matrix values:\n", pixel_to_robot_matrix)
#         success = True
#     else:
#         socketio.emit('status_update', {'msg': "Did not collect enough points. Calibration failed.", 'error': True})
#         success = False

#     socketio.emit('status_update', {'msg': f"Moving {standby_offset_mm}mm sideways to standby position..."})
#     cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
#     socketio.emit('status_update', {'msg': "Cobot is in standby. Calibration finished."})
    
#     return success

# # ============================================================================
# # Vision Processing Functions (No changes)
# # ============================================================================
# def detect_needle_tip_rectangular(image, color='red', debug_frame=None):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     if color == 'red':
#         lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
#         lower_red2, upper_red2 = np.array([170, 80, 80]), np.array([180, 255, 255])
#         mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
#     elif color == 'green':
#         mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
#     elif color == 'pink':
#         mask = cv2.inRange(hsv, np.array([140, 80, 80]), np.array([170, 255, 255]))
#     else: return None, None, None, None

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if debug_frame is not None:
#         cv2.drawContours(debug_frame, contours, -1, (255, 0, 0), 1)
#     if not contours: return None, None, None, None
#     needle_contour = max(contours, key=cv2.contourArea)
#     if len(needle_contour) < 5: return None, None, None, None
#     if debug_frame is not None:
#         cv2.drawContours(debug_frame, [needle_contour], -1, (0, 255, 0), 2)

#     rect = cv2.minAreaRect(needle_contour)
#     box = cv2.boxPoints(rect)
#     box = box.astype(int)
#     center, (width, height), angle = rect
#     if width < height:
#         width, height = height, width
#         angle = angle + 90
#     rect_points = box
#     angle_rad = np.deg2rad(angle)
#     direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
#     projections = [np.dot(point - center, direction) for point in rect_points]
#     projections = np.array(projections)
#     max_proj_idx = np.argmax(projections)
#     min_proj_idx = np.argmin(projections)
#     tip_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[max_proj_idx]) < width * 0.1]
#     base_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[min_proj_idx]) < width * 0.1]
#     def calculate_end_coverage(mask, center_point, direction_vec, distance):
#         end_point = center_point + direction_vec * distance
#         perp_vec = np.array([-direction_vec[1], direction_vec[0]])
#         coverage = 0
#         sample_range = int(height)
#         for offset in np.linspace(-sample_range / 2, sample_range / 2, 10):
#             sample_point = end_point + perp_vec * offset
#             x, y = int(sample_point[0]), int(sample_point[1])
#             if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
#                 coverage += 1
#         return coverage
#     coverage_max = calculate_end_coverage(mask, np.array(center), direction, projections[max_proj_idx])
#     coverage_min = calculate_end_coverage(mask, np.array(center), direction, projections[min_proj_idx])
#     frame_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
#     dist_max_to_center = np.linalg.norm(np.mean([rect_points[i] for i in tip_end_indices], axis=0) - frame_center)
#     dist_min_to_center = np.linalg.norm(np.mean([rect_points[i] for i in base_end_indices], axis=0) - frame_center)
#     use_coverage = abs(coverage_max - coverage_min) > 3
#     if (use_coverage and coverage_max < coverage_min) or (not use_coverage and dist_max_to_center < dist_min_to_center):
#         tip_end_points = [rect_points[i] for i in tip_end_indices]
#         tip_direction = direction
#     else:
#         tip_end_points = [rect_points[i] for i in base_end_indices]
#         tip_direction = -direction
#     if len(tip_end_points) >= 2:
#         midpoint_tip = np.mean(tip_end_points, axis=0)
#         line_start = np.array(center) + tip_direction * (min(projections) + (max(projections) - min(projections)) * 0.3)
#         midpoint_line = (tuple(line_start.astype(int)), tuple(midpoint_tip.astype(int)))
#         tip_point = tuple(midpoint_tip.astype(int))
#         orientation_angle = np.degrees(np.arctan2(tip_point[1] - midpoint_line[0][1], tip_point[0] - midpoint_line[0][0]))
#         if debug_frame is not None:
#              cv2.circle(debug_frame, tip_point, 10, (0, 0, 255), -1)
#         return tip_point, orientation_angle, (tuple(tip_end_points[0]), tuple(tip_end_points[1])), midpoint_line
#     return None, None, None, None

# def rotate_image(img, angle):
#     h, w = img.shape[:2]
#     center = (w // 2, h // 2)
#     rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
#     return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

# def compute_iou(mask1, mask2):
#     intersection = np.logical_and(mask1.astype(bool), mask2.astype(bool)).sum()
#     union = np.logical_or(mask1.astype(bool), mask2.astype(bool)).sum()
#     return intersection / union if union > 0 else 0

# def pixel_variance_score(dut_img, mask):
#     region = dut_img[mask == 1]
#     return float(np.var(region)) if len(region) > 0 else 0

# def align_with_iou_and_variance(cropped_img, golden_mask, min_variance=20):
#     if cropped_img is None or golden_mask is None: return None, None, 0, 0
#     dut_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img.copy()
#     golden_mask_resized = cv2.resize(golden_mask, (dut_img.shape[1], dut_img.shape[0]))
#     _, dut_mask = cv2.threshold(dut_img, 80, 1, cv2.THRESH_BINARY)
#     best_score, best_angle, best_rotated = -1, 0, None
#     for angle in range(0, 360, 5):
#         rotated = rotate_image(golden_mask_resized, angle)
#         if pixel_variance_score(dut_img, rotated) < min_variance: continue
#         score = compute_iou(dut_mask, rotated > 0)
#         if score > best_score:
#             best_score, best_angle, best_rotated = score, angle, rotated
#     return dut_img, best_rotated, best_angle, best_score

# def extract_waypoints(mask, num_points=1):
#     ys, xs = np.where(mask > 0)
#     points = np.column_stack((xs, ys))
#     if len(points) < num_points: return []
#     M = cv2.moments(mask.astype(np.uint8))
#     if M["m00"] == 0: return []
#     centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#     if num_points == 1: return [centroid]
#     # This part is now unreachable if num_points is always 1, which is fine.
#     kmeans = KMeans(n_clusters=num_points, n_init=10, random_state=0).fit(points)
#     return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

# # ===================================================================
# # --- MODIFICATION 1: run_template_matching_dual ---
# # ===================================================================
# def run_template_matching_dual(cropped_img, golden_mask_paths, object_id):
#     """
#     Uses GR3 to find the best angle, then applies that angle
#     to GR1 and GR2.
#     """
#     results = {}
    
#     # --- 1. Load all masks ---
#     golden_mask_GR1 = cv2.imread(golden_mask_paths["GR1"], cv2.IMREAD_GRAYSCALE)
#     golden_mask_GR2 = cv2.imread(golden_mask_paths["GR2"], cv2.IMREAD_GRAYSCALE)
#     golden_mask_GR3 = cv2.imread(golden_mask_paths["GR3"], cv2.IMREAD_GRAYSCALE)

#     if golden_mask_GR1 is None or golden_mask_GR2 is None or golden_mask_GR3 is None:
#         print(f"Error: Could not load one or more masks (GR1, GR2, or GR3).")
#         return None
    
#     # Binarize all masks
#     _, golden_mask_GR1 = cv2.threshold(golden_mask_GR1, 127, 1, cv2.THRESH_BINARY)
#     _, golden_mask_GR2 = cv2.threshold(golden_mask_GR2, 127, 1, cv2.THRESH_BINARY)
#     _, golden_mask_GR3 = cv2.threshold(golden_mask_GR3, 127, 1, cv2.THRESH_BINARY)
    
#     # --- 2. Align using GR3 to find the best angle ---
#     # `align_with_iou_and_variance` returns: dut_img, best_rotated_mask, best_angle, best_score
#     dut_img, mask_GR3_aligned, best_angle, score_GR3 = align_with_iou_and_variance(cropped_img, golden_mask_GR3)
    
#     if mask_GR3_aligned is None:
#         print("Failed to align with GR3 (alignment mask).")
#         return None
    
#     print(f"Template Matching: Found best angle {best_angle} using GR3 (IoU: {score_GR3:.3f})")

#     # --- 3. Apply this angle to GR1 and GR2 ---
#     # Resize golden masks *before* rotation
#     h, w = dut_img.shape[:2]
#     golden_mask_GR1_resized = cv2.resize(golden_mask_GR1, (w, h), interpolation=cv2.INTER_NEAREST)
#     golden_mask_GR2_resized = cv2.resize(golden_mask_GR2, (w, h), interpolation=cv2.INTER_NEAREST)
    
#     mask_GR1_rotated = rotate_image(golden_mask_GR1_resized, best_angle)
#     mask_GR2_rotated = rotate_image(golden_mask_GR2_resized, best_angle)

#     # --- 4. Extract waypoints (num_points=1 as requested) ---
#     results["GR1"] = {"angle": int(best_angle), "waypoints": extract_waypoints(mask_GR1_rotated, num_points=1)}
#     results["GR2"] = {"angle": int(best_angle), "waypoints": extract_waypoints(mask_GR2_rotated, num_points=1)}

#     # --- 5. Create overlay for visualization ---
#     overlay = cv2.cvtColor(dut_img, cv2.COLOR_GRAY2BGR)
#     overlay[mask_GR1_rotated == 1] = [255, 0, 0] # GR1 = Blue
#     overlay[mask_GR2_rotated == 1] = [0, 0, 255] # GR2 = Red
    
#     # Draw waypoints
#     for (x, y) in results["GR1"]["waypoints"]:
#         cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1) # GR1 waypoint = Green
#     for (x, y) in results["GR2"]["waypoints"]:
#         cv2.circle(overlay, (x, y), 5, (0, 255, 255), -1) # GR2 waypoint = Yellow

#     results["overlay"] = overlay
#     return results

# # ===================================================================
# # --- END MODIFICATION 1 ---
# # ===================================================================

# def calculate_distance(box1, box2):
#     # --- MODIFICATION 3: Fixed typo ---
#     c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
#     c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
#     return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) # <-- Corrected bug
#     # --- END MODIFICATION 3 ---

# def point_in_box(point, bbox):
#     return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

# def reset_state():
#     return {'current_state': 0, 'dut_bbox': None, 'template_results': None, 'stationary_start_time': None, 'last_bbox': None, 'movement_initiated': False}

# # ============================================================================
# # Main Alignment Functions (MODIFIED FOR BUG)
# # ============================================================================
# def align_to_waypoints(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, state, settings):

#     needle_color = settings['last_needle_color']
#     pixel_to_robot_matrix = settings['calibration_matrix']
#     tolerance_mm = settings.get('alignment_tolerance', 0.1)
#     max_iterations = 15
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
#     socketio.emit('status_update', {'msg': "ALIGNMENT PROCESS STARTED"})
    
#     all_waypoints_512 = state['template_results']["GR1"]["waypoints"]
#     if state['template_results'].get("GR2"):
#         all_waypoints_512.extend(state['template_results']["GR2"]["waypoints"])
        
#     xmin, ymin, xmax, ymax = state['dut_bbox']
#     scale_x = (xmax - xmin) / 512.0
#     scale_y = (ymax - ymin) / 512.0
    
#     alignment_results = []
    
#     for wp_idx, wp_512 in enumerate(all_waypoints_512):
#         if stop_event.is_set(): return False # Return False if stopped

#         socketio.emit('status_update', {'msg': f"Targeting Waypoint {wp_idx+1}/{len(all_waypoints_512)}"})
#         target_wp_x_full = (wp_512[0] * scale_x) + xmin
#         target_wp_y_full = (wp_512[1] * scale_y) + ymin
#         target_point_display = (int(target_wp_x_full), int(target_wp_y_full))

#         # --- MODIFICATION 2: Initialize variables before loop ---
#         error_magnitude_mm = 0.0 # Initialize to 0
#         iteration = 0 # Initialize iteration count
#         # --- END MODIFICATION 2 ---

#         for iteration in range(max_iterations):
#             if stop_event.is_set(): return False
            
#             with camera_lock:
#                 frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
#             if frame is None:
#                 socketio.sleep(0.1)
#                 continue
            
#             display_frame = frame.copy()
#             cv2.circle(display_frame, target_point_display, 12, (0, 255, 0), 2)
#             cv2.putText(display_frame, f"Target WP {wp_idx+1}", (target_point_display[0] + 15, target_point_display[1]),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             tip_info = detect_needle_tip_rectangular(frame, needle_color)
#             if not (tip_info and tip_info[0]):
#                 socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Cannot detect tip, retrying..."})
#                 cv2.putText(display_frame, "TIP NOT DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 send_frame_to_ui(socketio, display_frame) # Show failure
#                 socketio.sleep(0.5)
#                 continue
            
#             tip_point_full_frame = tip_info[0]
            
#             cv2.circle(display_frame, tip_point_full_frame, 12, (0, 0, 255), -1)
#             cv2.line(display_frame, tip_point_full_frame, target_point_display, (255, 255, 0), 2)

#             error_dx_full = target_wp_x_full - tip_point_full_frame[0]
#             error_dy_full = target_wp_y_full - tip_point_full_frame[1]
#             pixel_error_vector = np.array([error_dx_full, error_dy_full])
            
#             rotation_scale_matrix = pixel_to_robot_matrix[:, :2]
#             cobot_move_mm = rotation_scale_matrix @ pixel_error_vector
#             move_x_mm, move_y_mm = cobot_move_mm
#             error_magnitude_mm = np.linalg.norm(cobot_move_mm)
            
#             cv2.putText(display_frame, f"Iter: {iteration+1}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#             cv2.putText(display_frame, f"Error: {error_magnitude_mm:.2f} mm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#             send_frame_to_ui(socketio, display_frame) # Show progress
            
#             socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Tip at {tip_point_full_frame}. Error: {error_magnitude_mm:.3f} mm"})
            
#             if error_magnitude_mm < tolerance_mm:
#                 socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} reached!"})
#                 alignment_results.append({"waypoint_id": wp_idx, "success": True, "error_mm": error_magnitude_mm, "iterations": iteration+1})
#                 socketio.sleep(1)
#                 break
            
#             damping = 0.7
#             move_x_damped, move_y_damped = move_x_mm * damping, move_y_mm * damping
            
#             if not cobot.move_relative(dx=move_x_damped, dy=move_y_damped, speed=8, block=True):
#                 socketio.emit('status_update', {'msg': "Robot movement failed.", 'error': True})
#                 alignment_results.append({"waypoint_id": wp_idx, "success": False, "error_mm": error_magnitude_mm, "iterations": iteration+1, "error_msg": "Movement failed"})
#                 break
#             socketio.sleep(0.3)
#         else:
#             # This 'else' block runs if the for loop completes (max_iterations reached)
#             socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} - Max iterations reached.", 'error': True})
#             alignment_results.append({
#                 "waypoint_id": wp_idx, 
#                 "success": False, 
#                 "error_mm": error_magnitude_mm, # This is now safe due to initialization
#                 "iterations": max_iterations,
#                 "error_msg": "Max iterations"
#             })

#         # Wait for user to press "Proceed"
#         msg = f"Waypoint {wp_idx+1} alignment finished. Press Proceed to continue."
#         if not wait_for_proceed(socketio, proceed_event, stop_event, display_frame, msg):
#              return alignment_results # Return data collected so far if stopped

#     socketio.emit('status_update', {'msg': "ALIGNMENT COMPLETE"})
#     return alignment_results

# # ============================================================================
# # Main Process Loop (Original)
# # ============================================================================
# def run_alignment_process(socketio, proceed_event, stop_event, request_process_stop_event, cobot, camera_frame_buffer, camera_lock, settings, model_path):
    
#     # Load settings from the passed dictionary
#     needle_color = settings['last_needle_color']
#     alignment_standby_joints = settings.get('alignment_standby_joints')
#     standby_offset_mm = settings.get('last_offset_mm', 50.0)
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
#     try:
#         settings['calibration_matrix'] = np.load('calibration_matrix.npy')
#         socketio.emit('status_update', {'msg': "Loaded calibration matrix 'calibration_matrix.npy'."})
#     except FileNotFoundError:
#         socketio.emit('status_update', {'msg': "'calibration_matrix.npy' not found! Run calibration first.", 'error': True})
#         return

#     model = YOLO(model_path)
    
#     # --- FIXED PATH ---
#     # We use PARENT_DIR (E:\yolo_new) and add 'MY_MODEL' to it
#     golden_masks = {
#         "GR1": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_1.png'),
#         "GR2": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_2.jpg'),
#         "GR3": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_3.png')
#     }
#     # --- END FIX ---
    
#     # Check if files exist to provide better error messages
#     if not os.path.exists(golden_masks["GR1"]):
#         socketio.emit('status_update', {'msg': f"Error: mask_1.png not found at {golden_masks['GR1']}", 'error': True})
#         return
#     if not os.path.exists(golden_masks["GR2"]):
#         socketio.emit('status_update', {'msg': f"Warning: mask_2.jpg not found at {golden_masks['GR2']}", 'error': False})
#     if not os.path.exists(golden_masks["GR3"]):
#         socketio.emit('status_update', {'msg': f"Error: mask_3.png (Alignment Mask) not found at {golden_masks['GR3']}", 'error': True})
#         return
#     # --- END MODIFICATION ---


#     STATE_DETECTING_DUT, STATE_MOVING_TO_POSITION, STATE_ALIGNING, STATE_COMPLETE = 0, 1, 2, 3
#     state = reset_state()
#     STATIONARY_SECONDS = 2
#     completion_time = None
#     auto_reset_delay = 5.0

#     while not stop_event.is_set():
#         with camera_lock:
#             frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None

#         if frame is None:
#             socketio.sleep(0.1)
#             continue
            
#         display_frame = frame.copy()

#         if state['current_state'] == STATE_DETECTING_DUT:
            
#             if request_process_stop_event.is_set():
#                 socketio.emit('status_update', {'msg': 'Process stop requested. Halting detection loop.'})
#                 break
            
#             # --- MODIFICATION 2: Set conf=0.96 ---
#             results = model(frame, verbose=False, conf=0.96)
            
#             if len(results[0].boxes) > 0:
                
#                 # --- MODIFICATION 2: Find box with highest confidence ---
#                 all_boxes = results[0].boxes
#                 confs = all_boxes.conf.cpu().numpy()
#                 best_idx = np.argmax(confs)
#                 box = all_boxes[best_idx] # This is now the box with the highest confidence
#                 # --- END MODIFICATION ---
                
#                 bbox = box.xyxy[0].cpu().numpy().astype(int)
                
#                 if state['last_bbox'] is None:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 if calculate_distance(bbox, state['last_bbox']) < 30: 
#                     if time.time() - state['stationary_start_time'] > STATIONARY_SECONDS:
#                         socketio.emit('status_update', {'msg': 'DUT is static. Performing template matching...'})
#                         state['dut_bbox'] = bbox
#                         dut_crop_resized = cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (512, 512))
                        
#                         state['template_results'] = run_template_matching_dual(dut_crop_resized, golden_masks, 0)
                        
#                         if state['template_results']:
#                             msg = "Template match found. Press Proceed to start alignment."
#                             if not wait_for_proceed(socketio, proceed_event, stop_event, state['template_results']['overlay'], msg):
#                                 state = reset_state()
#                                 continue

#                             if alignment_standby_joints and sum(abs(j) for j in alignment_standby_joints) > 0:
#                                 state['current_state'] = STATE_MOVING_TO_POSITION
#                             else:
#                                 socketio.emit('status_update', {'msg': 'No standby joints defined. Skipping move.'})
#                                 state['current_state'] = STATE_ALIGNING
#                         else:
#                             socketio.emit('status_update', {'msg': 'Template matching failed.', 'error': True})
#                             state = reset_state()
#                 else:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#             else:
#                 state['last_bbox'] = None
                
#             send_frame_to_ui(socketio, display_frame) # Send live feed

#         elif state['current_state'] == STATE_MOVING_TO_POSITION:
#             if not state['movement_initiated']:
#                 socketio.emit('status_update', {'msg': 'Moving to prefixed standby pose...'})
#                 state['movement_initiated'] = True
#                 if cobot.joint_move(alignment_standby_joints, speed=20, block=True):
#                     socketio.emit('status_update', {'msg': 'Standby position reached. Proceeding to alignment.'})
#                     state['current_state'] = STATE_ALIGNING
#                 else:
#                     socketio.emit('status_update', {'msg': 'Failed to move to standby position. Resetting.', 'error': True})
#                     state = reset_state()

#         elif state['current_state'] == STATE_ALIGNING:
#             socketio.emit('status_update', {'msg': 'Starting alignment procedure...'})
            
#             # This function returns a list of results, which we will log
#             alignment_data = align_to_waypoints(
#                 socketio, proceed_event, stop_event, cobot, 
#                 camera_frame_buffer, camera_lock, state, settings
#             )
            
#             # --- REMOVED: CSV Logging Call ---
#             # --- You will log this manually based on the console output ---
                    
#             socketio.emit('status_update', {'msg': f"Alignment finished. Moving {standby_offset_mm}mm to standby position."})
#             if cobot:
#                 cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
#                 socketio.emit('status_update', {'msg': 'Robot is in post-alignment standby.'})
                
#             state['current_state'] = STATE_COMPLETE
#             completion_time = time.time()

#         elif state['current_state'] == STATE_COMPLETE:
#             if time.time() - completion_time > auto_reset_delay:
#                 socketio.emit('status_update', {'msg': 'Cycle complete. Starting new detection cycle...'})
#                 state = reset_state()
#             else:
#                 cv2.putText(display_frame, "COMPLETE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
#                 send_frame_to_ui(socketio, display_frame) # Send live feed

#         socketio.sleep(0.02) # Main loop sleep to yield control
        
#     socketio.emit('status_update', {'msg': 'Alignment process stopped.'})


# # ============================================================================
# # --- NEW: DEMO MODE Main Process Loop ---
# # ============================================================================
# def run_demo_alignment_process(socketio, proceed_event, stop_event, request_process_stop_event, camera_frame_buffer, camera_lock, settings, model_path):
#     """
#     Main process loop for DEMO MODE.
#     Runs Detection -> Template Match -> Continuous Tip Detection.
#     Stops when 'request_process_stop_event' is set.
#     """
    
#     # Load settings from the passed dictionary
#     needle_color = settings['last_needle_color']
#     resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
#     # --- No calibration matrix needed for demo ---
#     socketio.emit('status_update', {'msg': "Running in DEMO MODE. No cobot alignment."})

#     model = YOLO(model_path)
    
#     # --- THIS WAS THE MISSING PIECE ---
#     # We use PARENT_DIR (E:\yolo_new) and add 'MY_MODEL' to it
#     golden_masks = {
#         "GR1": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_1.png'),
#         "GR2": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_2.jpg'),
#         "GR3": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_3.png')
#     }
#     # --- END FIX ---
    
#     # Check if files exist to provide better error messages
#     if not os.path.exists(golden_masks["GR1"]):
#         socketio.emit('status_update', {'msg': f"Error: mask_1.png not found at {golden_masks['GR1']}", 'error': True})
#         return
#     if not os.path.exists(golden_masks["GR2"]):
#         socketio.emit('status_update', {'msg': f"Warning: mask_2.jpg not found at {golden_masks['GR2']}", 'error': False})
#     if not os.path.exists(golden_masks["GR3"]):
#         socketio.emit('status_update', {'msg': f"Error: mask_3.png (Alignment Mask) not found at {golden_masks['GR3']}", 'error': True})
#         return

#     # Define states for this demo loop
#     STATE_DETECTING_DUT = 0
#     STATE_DEMO_TIP_DETECT = 1 # New state for demo
    
#     state = reset_state() # Uses state 0 by default
#     STATIONARY_SECONDS = 2

#     while not stop_event.is_set():
#         with camera_lock:
#             frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None

#         if frame is None:
#             socketio.sleep(0.1)
#             continue
            
#         display_frame = frame.copy()

#         if state['current_state'] == STATE_DETECTING_DUT:
#             # This is the "Stop Process After Cycle" check
#             if request_process_stop_event.is_set():
#                 socketio.emit('status_update', {'msg': 'Process stop requested. Halting detection loop.'})
#                 break
            
#             results = model(frame, verbose=False, conf=0.96)
            
#             if len(results[0].boxes) > 0:
#                 all_boxes = results[0].boxes
#                 confs = all_boxes.conf.cpu().numpy()
#                 best_idx = np.argmax(confs)
#                 box = all_boxes[best_idx]
#                 bbox = box.xyxy[0].cpu().numpy().astype(int)
                
#                 if state['last_bbox'] is None:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 if calculate_distance(bbox, state['last_bbox']) < 30: 
#                     if time.time() - state['stationary_start_time'] > STATIONARY_SECONDS:
#                         socketio.emit('status_update', {'msg': 'DEMO: DUT static. Performing template matching...'})
#                         state['dut_bbox'] = bbox
#                         dut_crop_resized = cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (512, 512))
                        
#                         state['template_results'] = run_template_matching_dual(dut_crop_resized, golden_masks, 0)
                        
#                         if state['template_results']:
#                             msg = "DEMO: Template match found. Press Proceed to start tip detection."
#                             # Show the template match overlay and wait
#                             if not wait_for_proceed(socketio, proceed_event, stop_event, state['template_results']['overlay'], msg):
#                                 state = reset_state() # Reset if user cancels
#                                 continue

#                             # --- DEMO MODE CHANGE ---
#                             # Instead of moving cobot, go to tip detection state
#                             socketio.emit('status_update', {'msg': 'DEMO: Proceeding to continuous tip detection.'})
#                             state['current_state'] = STATE_DEMO_TIP_DETECT
#                             # --- END DEMO MODE CHANGE ---
                        
#                         else:
#                             socketio.emit('status_update', {'msg': 'DEMO: Template matching failed.', 'error': True})
#                             state = reset_state()
#                 else:
#                     state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
#                 cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#             else:
#                 state['last_bbox'] = None
                
#             send_frame_to_ui(socketio, display_frame) # Send live feed

#         # --- NEW DEMO STATE ---
#         elif state['current_state'] == STATE_DEMO_TIP_DETECT:
#             # This state runs continuously until "Stop Process" is pressed
#             if request_process_stop_event.is_set():
#                 socketio.emit('status_update', {'msg': 'DEMO: Stop requested. Halting tip detection.'})
#                 break # Exit the main while loop
            
#             # Continuously detect and display the needle tip
#             tip_info = detect_needle_tip_rectangular(frame, needle_color, debug_frame=display_frame)
            
#             if tip_info and tip_info[0]:
#                 tip_point = tip_info[0]
#                 # Draw the detected tip (already done by debug_frame)
#                 cv2.putText(display_frame, f"Tip Detected: {tip_point}", (30, 60), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             else:
#                 cv2.putText(display_frame, "Detecting Tip...", (30, 60), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
#             # Also draw the original DUT bounding box for context
#             if state['dut_bbox'] is not None:
#                 bbox = state['dut_bbox']
#                 cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

#             send_frame_to_ui(socketio, display_frame) # Send live feed
        
#         # --- END NEW DEMO STATE ---

#         socketio.sleep(0.02) # Main loop sleep to yield control
        
#     socketio.emit('status_update', {'msg': 'Demo process stopped.'})





#------with demo, improved template matching
import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import json
from sklearn.cluster import KMeans
import socket
import struct
import base64
import threading

# due to masks folder path change
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR is E:\yolo_new
PARENT_DIR = os.path.dirname(SCRIPT_DIR) 

# ============================================================================
# Helper Functions for Web UI
# ============================================================================

def send_frame_to_ui(socketio, frame, event_name='video_frame'):
    """Encodes a CV2 frame as a base64 JPEG and emits it."""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Lower quality for faster transfer
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        socketio.emit(event_name, {'image': jpg_as_text})
    except Exception as e:
        print(f"Error sending frame to UI: {e}")

def wait_for_proceed(socketio, proceed_event, stop_event, frame, message):
    """
    Sends an image to the UI, emits a 'wait_for_proceed' status,
    and then blocks until the 'proceed_event' is set (by the user).
    """
    socketio.emit('status_update', {'msg': f"WAITING: {message}"})
    
    # Send the specific frame for the user to see (with good quality)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('show_image_and_wait', {'image': jpg_as_text})

    # Wait for the proceed_event or stop_event
    proceed_event.clear() # Ensure it's clear before waiting
    while not proceed_event.is_set() and not stop_event.is_set():
        socketio.sleep(0.1) # Non-blocking sleep

    if stop_event.is_set():
        socketio.emit('status_update', {'msg': 'Proceed cancelled by stop request.'})
        return False # Stopped
        
    socketio.emit('status_update', {'msg': 'User proceeded.'})
    socketio.emit('hide_popup') # Tell UI to hide the popup
    return True # Proceeded

# ============================================================================
# JAKA Cobot Communication Class (No changes)
# ============================================================================
class JAKACobot:
    """JAKA Cobot TCP/IP communication and control using the official JSON protocol."""

    def __init__(self, ip="192.168.1.166", port=10001):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(15.0)
            self.socket.connect((self.ip, self.port))
            self.connected = True
            print(f"[JAKA] Connected to cobot at {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[JAKA ERROR] Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.connected = False
            print("[JAKA] Disconnected from cobot")

    def send_command(self, command_dict):
        if not self.connected:
            print("[JAKA ERROR] Not connected to cobot")
            return None
        try:
            json_command = json.dumps(command_dict) + '\n'
            self.socket.sendall(json_command.encode('utf-8'))
            response_str = self.socket.recv(4096).decode('utf-8').strip()
            if response_str:
                return json.loads(response_str)
            return None
        except (json.JSONDecodeError, socket.timeout, ConnectionResetError) as e:
            print(f"[JAKA ERROR] Communication error: {e}")
            return None

    def _check_response(self, response_json, command_name):
        if response_json and str(response_json.get("errorCode")) == "0":
            return True
        else:
            msg = response_json.get('errorMsg', 'Unknown error') if response_json else 'No response'
            print(f"[JAKA ERROR] {command_name} failed: {msg}")
            return False

    def power_on(self):
        cmd = {"cmdName": "power_on"}
        response = self.send_command(cmd)
        return self._check_response(response, "power_on")

    def enable_robot(self):
        cmd = {"cmdName": "enable_robot"}
        response = self.send_command(cmd)
        return self._check_response(response, "enable_robot")

    def get_tcp_position(self):
        cmd = {"cmdName": "get_tcp_pos"}
        response = self.send_command(cmd)
        if response and str(response.get("errorCode")) == "0":
            tcp_data = response.get("tcp_pos")
            if tcp_data and len(tcp_data) == 6:
                return [float(p) for p in tcp_data]
        print(f"[JAKA ERROR] Failed to get TCP position. Response: {response}")
        return None

    def get_joint_position(self):
        cmd = {"cmdName": "get_joint_pos"}
        response = self.send_command(cmd)
        if response and str(response.get("errorCode")) == "0":
            joint_data = response.get("joint_pos")
            if joint_data and len(joint_data) == 6:
                return [float(p) for p in joint_data]
        print(f"[JAKA ERROR] Failed to get Joint position. Response: {response}")
        return None

    def joint_move(self, joint_positions, speed=20, accel=50, tol=0.5, block=True):
        cmd = {
            "cmdName": "joint_move", "relFlag":0,
            "jointPosition": [float(p) for p in joint_positions],
            "speed": speed, "accel": accel, "tol": tol
        }
        response = self.send_command(cmd)
        if not self._check_response(response, "joint_move (joint_move)"):
            return False

        print("[JAKA] moveJ command accepted.")
        if block:
            print("[JAKA] Waiting for joint move to complete...")
            start_time = time.time()
            while time.time() - start_time < 30: # 30 second timeout
                current_joints = self.get_joint_position()
                if current_joints:
                    diff = np.array(joint_positions) - np.array(current_joints)
                    dist = np.linalg.norm(diff)
                    if dist < (tol + 0.5):
                        print("[JAKA] Joint move complete.")
                        return True
                time.sleep(0.1)
            print("[JAKA ERROR] Joint move timed out.")
            return False
        return True

    def linear_move(self, tcp_position, speed=20, accel=50, tol=0.5, relative=False, block=True):
        cmd = {
            "cmdName": "moveL", "relFlag": 1 if relative else 0,
            "cartPosition": [float(p) for p in tcp_position],
            "speed": speed, "accel": accel, "tol": tol
        }
        response = self.send_command(cmd)
        if not self._check_response(response, "moveL (linear_move)"):
            return False

        print("[JAKA] moveL command accepted.")
        if block and not relative:
            print("[JAKA] Waiting for linear move to complete...")
            start_time = time.time()
            while time.time() - start_time < 30:
                current_pos = self.get_tcp_position()
                if current_pos:
                    dist = math.hypot(
                        tcp_position[0] - current_pos[0],
                        tcp_position[1] - current_pos[1],
                        tcp_position[2] - current_pos[2]
                    )
                    if dist < tol:
                        print("[JAKA] Linear move complete.")
                        return True
                time.sleep(0.1)
            print("[JAKA ERROR] Linear move timed out.")
            return False
        elif block and relative:
            time.sleep(1.0) # Small delay for relative moves to ensure completion
        return True

    def move_relative(self, dx=0, dy=0, dz=0, speed=10, block=True):
        relative_pose = [dx, dy, dz, 0, 0, 0]
        return self.linear_move(relative_pose, speed=speed, relative=True, block=block)

# ============================================================================
# Standalone Calibration Routine (No changes)
# ============================================================================
def run_calibration_routine(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, settings):
    
    needle_color = settings['last_needle_color']
    move_dist = settings['last_move_mm']
    standby_offset_mm = settings['last_offset_mm']
    calibration_start_joints = settings.get('calibration_start_joints')

    socketio.emit('status_update', {'msg': 'Calibration routine starting...'})
    
    resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)

    calib_img_dir = "calibration_images"
    os.makedirs(calib_img_dir, exist_ok=True)
    socketio.emit('status_update', {'msg': f"Images will be saved in '{calib_img_dir}/'"})

    robot_points_mm = np.float32([[0, 0], [move_dist, 0], [0, move_dist]])
    pixel_points = []

    # 1. Move to start position
    if calibration_start_joints and sum(abs(j) for j in calibration_start_joints) > 0:
        socketio.emit('status_update', {'msg': 'Moving to pre-defined calibration start position...'})
        if not cobot.joint_move(calibration_start_joints, speed=20, block=True):
             socketio.emit('status_update', {'msg': 'Failed to move to start position. Aborting.', 'error': True})
             return False
    else:
        socketio.emit('status_update', {'msg': 'No start joints defined. Using current position.'})
        
    socketio.emit('status_update', {'msg': "Getting robot's starting position..."})
    start_pos = cobot.get_tcp_position()
    if not start_pos:
        socketio.emit('status_update', {'msg': 'Could not get robot start position. Aborting.', 'error': True})
        return False

    for i, (dx, dy) in enumerate(robot_points_mm):
        if stop_event.is_set(): return False
        
        tip_found = False
        point_num = i + 1
        socketio.emit('status_update', {'msg': f"Moving to relative point {point_num}/3 ({dx}mm, {dy}mm)..."})

        target_pose = start_pos.copy()
        target_pose[0] += dx
        target_pose[1] += dy

        if not cobot.linear_move(target_pose, speed=15, block=True):
            socketio.emit('status_update', {'msg': f"Robot failed to move to point {point_num}. Aborting.", 'error': True})
            return False
        socketio.sleep(1.0) # Wait for vibrations to settle

        socketio.emit('status_update', {'msg': f"Detecting tip at point {point_num}..."})
        for attempt in range(5):
            if stop_event.is_set(): return False
            
            with camera_lock:
                frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
            if frame is None:
                socketio.sleep(0.1)
                continue

            tip_info = detect_needle_tip_rectangular(frame, needle_color, debug_frame=frame)
            
            if tip_info and tip_info[0]:
                tip_point = tip_info[0]
                pixel_points.append(tip_point)
                socketio.emit('status_update', {'msg': f"Tip detected at pixel coordinate: {tip_point}"})

                cv2.circle(frame, tip_point, 10, (0, 0, 255), -1)
                cv2.putText(frame, f"Point {point_num}", (tip_point[0] + 15, tip_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                send_frame_to_ui(socketio, frame) # Show detection
                socketio.sleep(0.5)

                filepath = os.path.join(calib_img_dir, f"calibration_point_{point_num}.png")
                cv2.imwrite(filepath, frame)
                tip_found = True
                break
            else:
                cv2.putText(frame, f"Detection Failed (Attempt {attempt+1})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                send_frame_to_ui(socketio, frame) # Show failure
                socketio.sleep(0.5)

        if not tip_found:
            socketio.emit('status_update', {'msg': f"Could not detect tip at point {point_num}. Aborting.", 'error': True})
            return False

    if len(pixel_points) == 3:
        socketio.emit('status_update', {'msg': "Calculating transformation matrix..."})
        pixel_points_np = np.float32(pixel_points)
        pixel_to_robot_matrix = cv2.getAffineTransform(pixel_points_np, robot_points_mm)
        np.save('calibration_matrix.npy', pixel_to_robot_matrix)

        socketio.emit('status_update', {'msg': "CALIBRATION SUCCESSFUL! Matrix saved."})
        print("Matrix values:\n", pixel_to_robot_matrix)
        success = True
    else:
        socketio.emit('status_update', {'msg': "Did not collect enough points. Calibration failed.", 'error': True})
        success = False

    socketio.emit('status_update', {'msg': f"Moving {standby_offset_mm}mm sideways to standby position..."})
    cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
    socketio.emit('status_update', {'msg': "Cobot is in standby. Calibration finished."})
    
    return success

# ============================================================================
# Vision Processing Functions
# ============================================================================
def detect_needle_tip_rectangular(image, color='red', debug_frame=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == 'red':
        lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 80, 80]), np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    elif color == 'green':
        mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
    elif color == 'pink':
        mask = cv2.inRange(hsv, np.array([140, 80, 80]), np.array([170, 255, 255]))
    else: return None, None, None, None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug_frame is not None:
        cv2.drawContours(debug_frame, contours, -1, (255, 0, 0), 1)
    if not contours: return None, None, None, None
    needle_contour = max(contours, key=cv2.contourArea)
    if len(needle_contour) < 5: return None, None, None, None
    if debug_frame is not None:
        cv2.drawContours(debug_frame, [needle_contour], -1, (0, 255, 0), 2)

    rect = cv2.minAreaRect(needle_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    center, (width, height), angle = rect
    if width < height:
        width, height = height, width
        angle = angle + 90
    rect_points = box
    angle_rad = np.deg2rad(angle)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    projections = [np.dot(point - center, direction) for point in rect_points]
    projections = np.array(projections)
    max_proj_idx = np.argmax(projections)
    min_proj_idx = np.argmin(projections)
    tip_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[max_proj_idx]) < width * 0.1]
    base_end_indices = [idx for idx in range(4) if abs(projections[idx] - projections[min_proj_idx]) < width * 0.1]
    def calculate_end_coverage(mask, center_point, direction_vec, distance):
        end_point = center_point + direction_vec * distance
        perp_vec = np.array([-direction_vec[1], direction_vec[0]])
        coverage = 0
        sample_range = int(height)
        for offset in np.linspace(-sample_range / 2, sample_range / 2, 10):
            sample_point = end_point + perp_vec * offset
            x, y = int(sample_point[0]), int(sample_point[1])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                coverage += 1
        return coverage
    coverage_max = calculate_end_coverage(mask, np.array(center), direction, projections[max_proj_idx])
    coverage_min = calculate_end_coverage(mask, np.array(center), direction, projections[min_proj_idx])
    frame_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
    dist_max_to_center = np.linalg.norm(np.mean([rect_points[i] for i in tip_end_indices], axis=0) - frame_center)
    dist_min_to_center = np.linalg.norm(np.mean([rect_points[i] for i in base_end_indices], axis=0) - frame_center)
    use_coverage = abs(coverage_max - coverage_min) > 3
    if (use_coverage and coverage_max < coverage_min) or (not use_coverage and dist_max_to_center < dist_min_to_center):
        tip_end_points = [rect_points[i] for i in tip_end_indices]
        tip_direction = direction
    else:
        tip_end_points = [rect_points[i] for i in base_end_indices]
        tip_direction = -direction
    if len(tip_end_points) >= 2:
        midpoint_tip = np.mean(tip_end_points, axis=0)
        line_start = np.array(center) + tip_direction * (min(projections) + (max(projections) - min(projections)) * 0.3)
        midpoint_line = (tuple(line_start.astype(int)), tuple(midpoint_tip.astype(int)))
        tip_point = tuple(midpoint_tip.astype(int))
        orientation_angle = np.degrees(np.arctan2(tip_point[1] - midpoint_line[0][1], tip_point[0] - midpoint_line[0][0]))
        if debug_frame is not None:
             cv2.circle(debug_frame, tip_point, 10, (0, 0, 255), -1)
        return tip_point, orientation_angle, (tuple(tip_end_points[0]), tuple(tip_end_points[1])), midpoint_line
    return None, None, None, None

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

# ===================================================================
# --- NEW TEMPLATE MATCHING LOGIC ---
# ===================================================================

def compute_fill_score(dut_mask_bool, template_mask_bool):
    """
    Calculates what percentage of the template_mask's area
    is "filled" by the dut_mask.
    Score = Intersection / Area_of_Template
    """
    intersection = np.logical_and(dut_mask_bool, template_mask_bool).sum()
    template_area = template_mask_bool.sum()
    return intersection / template_area if template_area > 0 else 0

def extract_waypoints(mask, num_points=1):
    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys))
    if len(points) < num_points: return []
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0: return []
    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if num_points == 1: return [centroid]
    # This part is not used by your current logic but is good to keep
    kmeans = KMeans(n_clusters=num_points, n_init=10, random_state=0).fit(points)
    return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

def run_template_matching_dual(cropped_img, golden_mask_paths, object_id, settings):
    """
    Runs the "Positive-Negative Template Matching" using Fill Scores.
    Finds the best angle using Mask 3 and Mask 4, then applies that
    angle to Mask 1 and Mask 2 to get waypoints.
    """
    results = {}
    
    # --- 1. Load all masks ---
    try:
        golden_mask_GR1 = cv2.imread(golden_mask_paths["GR1"], cv2.IMREAD_GRAYSCALE)
        golden_mask_GR2 = cv2.imread(golden_mask_paths["GR2"], cv2.IMREAD_GRAYSCALE)
        golden_mask_GR3 = cv2.imread(golden_mask_paths["GR3"], cv2.IMREAD_GRAYSCALE)
        golden_mask_GR4 = cv2.imread(golden_mask_paths["GR4_Negative"], cv2.IMREAD_GRAYSCALE)
        
        if golden_mask_GR1 is None or golden_mask_GR2 is None or golden_mask_GR3 is None or golden_mask_GR4 is None:
            print("Error: Could not load one or more masks (GR1, GR2, GR3, or GR4).")
            return None
    except Exception as e:
        print(f"Error loading masks: {e}")
        return None

    # --- 2. Prepare DUT Image Masks ---
    # The input `cropped_img` is already 512x512
    dut_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # Use the hardcoded thresholds from the standalone script
    DUT_BINARY_THRESHOLD_LOWER = 80
    DUT_BINARY_THRESHOLD_UPPER = 255
    
    # Create Positive (bright) and Negative (dark) logic masks for the DUT
    dut_mask_pos_visual = cv2.inRange(dut_img_gray, DUT_BINARY_THRESHOLD_LOWER, DUT_BINARY_THRESHOLD_UPPER)
    dut_mask_pos_logic = (dut_mask_pos_visual > 0) 
    dut_mask_neg_logic = np.logical_not(dut_mask_pos_logic)

    # --- 3. Prepare Template Masks ---
    h, w = dut_img_gray.shape
    
    # Mask 3 (Positive)
    golden_mask_GR3_resized = cv2.resize(golden_mask_GR3, (w, h), interpolation=cv2.INTER_NEAREST)
    _, golden_mask_GR3_visual = cv2.threshold(golden_mask_GR3_resized, 127, 255, cv2.THRESH_BINARY)
    _, golden_mask_GR3_logic = cv2.threshold(golden_mask_GR3_resized, 127, 1, cv2.THRESH_BINARY)
    
    # Mask 4 (Negative)
    golden_mask_GR4_resized = cv2.resize(golden_mask_GR4, (w, h), interpolation=cv2.INTER_NEAREST)
    _, golden_mask_GR4_visual = cv2.threshold(golden_mask_GR4_resized, 127, 255, cv2.THRESH_BINARY)
    _, golden_mask_GR4_logic = cv2.threshold(golden_mask_GR4_resized, 127, 1, cv2.THRESH_BINARY)
    
    # Waypoint Masks (GR1, GR2)
    golden_mask_GR1_resized = cv2.resize(golden_mask_GR1, (w, h), interpolation=cv2.INTER_NEAREST)
    _, golden_mask_GR1_logic = cv2.threshold(golden_mask_GR1_resized, 127, 1, cv2.THRESH_BINARY)
    golden_mask_GR2_resized = cv2.resize(golden_mask_GR2, (w, h), interpolation=cv2.INTER_NEAREST)
    _, golden_mask_GR2_logic = cv2.threshold(golden_mask_GR2_resized, 127, 1, cv2.THRESH_BINARY)

    # --- 4. Run Alignment Loop ---
    best_score, best_angle = -1, 0
    best_score_pos, best_score_neg = 0, 0

    for angle in range(0, 360, 1):
        # Rotate logic masks
        rotated_mask_3_logic_bool = rotate_image(golden_mask_GR3_logic, angle) > 0
        rotated_mask_4_logic_bool = rotate_image(golden_mask_GR4_logic, angle) > 0
        
        # Score 1: How many BRIGHT DUT pixels fill the MASK 3 arc?
        score_positive = compute_fill_score(dut_mask_pos_logic, rotated_mask_3_logic_bool)
        
        # Score 2: How many DARK DUT pixels fill the MASK 4 circle?
        score_negative = compute_fill_score(dut_mask_neg_logic, rotated_mask_4_logic_bool)
        
        # Combined score
        final_score = score_positive + score_negative
        
        if final_score > best_score:
            best_score = final_score
            best_angle = angle
            best_score_pos = score_positive
            best_score_neg = score_negative
            
    if best_score < 0:
        print("Template Matching Failed: Could not find a suitable angle.")
        return None
        
    print(f"Template Matching: Best Angle {best_angle} (Score: {best_score:.4f}, Pos: {best_score_pos:.4f}, Neg: {best_score_neg:.4f})")

    # --- 5. Generate Waypoints and Overlay ---
    
    # Rotate GR1/GR2 masks by the best_angle
    mask_GR1_rotated_logic_bool = rotate_image(golden_mask_GR1_logic, best_angle) > 0
    mask_GR2_rotated_logic_bool = rotate_image(golden_mask_GR2_logic, best_angle) > 0
    
    results["GR1"] = {"angle": int(best_angle), "waypoints": extract_waypoints(mask_GR1_rotated_logic_bool, num_points=1)}
    results["GR2"] = {"angle": int(best_angle), "waypoints": extract_waypoints(mask_GR2_rotated_logic_bool, num_points=1)}

    # Create the final overlay image for wait_for_proceed
    overlay = cv2.cvtColor(dut_img_gray, cv2.COLOR_GRAY2BGR)
    overlay[mask_GR1_rotated_logic_bool] = [255, 0, 0] # GR1 = Blue (Solid)
    overlay[mask_GR2_rotated_logic_bool] = [0, 0, 255] # GR2 = Red (Solid)
    
    # Extract and draw waypoints
    for (x, y) in results["GR1"]["waypoints"]:
        cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1) # GR1 waypoint = Green fill
    for (x, y) in results["GR2"]["waypoints"]:
        cv2.circle(overlay, (x, y), 5, (0, 255, 255), -1) # GR2 waypoint = Yellow fill

    # Get and draw GR3 outline (Green)
    best_rotated_mask_3_visual = rotate_image(golden_mask_GR3_visual, best_angle)
    contours_gr3, _ = cv2.findContours(best_rotated_mask_3_visual.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_gr3, -1, (0, 255, 0), 2) # Green outline
    
    # Get and draw GR4 outline (Magenta)
    best_rotated_mask_4_visual = rotate_image(golden_mask_GR4_visual, best_angle)
    contours_gr4, _ = cv2.findContours(best_rotated_mask_4_visual.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_gr4, -1, (255, 0, 255), 2) # Magenta outline

    results["overlay"] = overlay
    return results

# ===================================================================
# --- END NEW TEMPLATE MATCHING LOGIC ---
# ===================================================================

def calculate_distance(box1, box2):
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

def point_in_box(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

def reset_state():
    return {'current_state': 0, 'dut_bbox': None, 'template_results': None, 'stationary_start_time': None, 'last_bbox': None, 'movement_initiated': False}

# ============================================================================
# Main Alignment Functions (MODIFIED FOR BUG)
# ============================================================================
def align_to_waypoints(socketio, proceed_event, stop_event, cobot, camera_frame_buffer, camera_lock, state, settings):

    needle_color = settings['last_needle_color']
    pixel_to_robot_matrix = settings['calibration_matrix']
    tolerance_mm = settings.get('alignment_tolerance', 0.1)
    max_iterations = 15
    resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
    socketio.emit('status_update', {'msg': "ALIGNMENT PROCESS STARTED"})
    
    all_waypoints_512 = state['template_results']["GR1"]["waypoints"]
    if state['template_results'].get("GR2"):
        all_waypoints_512.extend(state['template_results']["GR2"]["waypoints"])
        
    xmin, ymin, xmax, ymax = state['dut_bbox']
    scale_x = (xmax - xmin) / 512.0
    scale_y = (ymax - ymin) / 512.0
    
    alignment_results = []
    
    for wp_idx, wp_512 in enumerate(all_waypoints_512):
        if stop_event.is_set(): return False # Return False if stopped

        socketio.emit('status_update', {'msg': f"Targeting Waypoint {wp_idx+1}/{len(all_waypoints_512)}"})
        target_wp_x_full = (wp_512[0] * scale_x) + xmin
        target_wp_y_full = (wp_512[1] * scale_y) + ymin
        target_point_display = (int(target_wp_x_full), int(target_wp_y_full))

        error_magnitude_mm = 0.0 # Initialize to 0
        iteration = 0 # Initialize iteration count

        for iteration in range(max_iterations):
            if stop_event.is_set(): return False
            
            with camera_lock:
                frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None
            
            if frame is None:
                socketio.sleep(0.1)
                continue
            
            display_frame = frame.copy()
            cv2.circle(display_frame, target_point_display, 12, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Target WP {wp_idx+1}", (target_point_display[0] + 15, target_point_display[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            tip_info = detect_needle_tip_rectangular(frame, needle_color)
            if not (tip_info and tip_info[0]):
                socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Cannot detect tip, retrying..."})
                cv2.putText(display_frame, "TIP NOT DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                send_frame_to_ui(socketio, display_frame) # Show failure
                socketio.sleep(0.5)
                continue
            
            tip_point_full_frame = tip_info[0]
            
            cv2.circle(display_frame, tip_point_full_frame, 12, (0, 0, 255), -1)
            cv2.line(display_frame, tip_point_full_frame, target_point_display, (255, 255, 0), 2)

            error_dx_full = target_wp_x_full - tip_point_full_frame[0]
            error_dy_full = target_wp_y_full - tip_point_full_frame[1]
            pixel_error_vector = np.array([error_dx_full, error_dy_full])
            
            rotation_scale_matrix = pixel_to_robot_matrix[:, :2]
            cobot_move_mm = rotation_scale_matrix @ pixel_error_vector
            move_x_mm, move_y_mm = cobot_move_mm
            error_magnitude_mm = np.linalg.norm(cobot_move_mm)
            
            cv2.putText(display_frame, f"Iter: {iteration+1}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(display_frame, f"Error: {error_magnitude_mm:.2f} mm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            send_frame_to_ui(socketio, display_frame) # Show progress
            
            socketio.emit('status_update', {'msg': f"Iter {iteration+1}: Tip at {tip_point_full_frame}. Error: {error_magnitude_mm:.3f} mm"})
            
            if error_magnitude_mm < tolerance_mm:
                socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} reached!"})
                alignment_results.append({"waypoint_id": wp_idx, "success": True, "error_mm": error_magnitude_mm, "iterations": iteration+1})
                socketio.sleep(1)
                break
            
            damping = 0.7
            move_x_damped, move_y_damped = move_x_mm * damping, move_y_mm * damping
            
            if not cobot.move_relative(dx=move_x_damped, dy=move_y_damped, speed=8, block=True):
                socketio.emit('status_update', {'msg': "Robot movement failed.", 'error': True})
                alignment_results.append({"waypoint_id": wp_idx, "success": False, "error_mm": error_magnitude_mm, "iterations": iteration+1, "error_msg": "Movement failed"})
                break
            socketio.sleep(0.3)
        else:
            # This 'else' block runs if the for loop completes (max_iterations reached)
            socketio.emit('status_update', {'msg': f"Waypoint {wp_idx+1} - Max iterations reached.", 'error': True})
            alignment_results.append({
                "waypoint_id": wp_idx, 
                "success": False, 
                "error_mm": error_magnitude_mm, # This is now safe due to initialization
                "iterations": max_iterations,
                "error_msg": "Max iterations"
            })

        # Wait for user to press "Proceed"
        msg = f"Waypoint {wp_idx+1} alignment finished. Press Proceed to continue."
        if not wait_for_proceed(socketio, proceed_event, stop_event, display_frame, msg):
             return alignment_results # Return data collected so far if stopped

    socketio.emit('status_update', {'msg': "ALIGNMENT COMPLETE"})
    return alignment_results

# ============================================================================
# Main Process Loop (Original)
# ============================================================================
def run_alignment_process(socketio, proceed_event, stop_event, request_process_stop_event, cobot, camera_frame_buffer, camera_lock, settings, model_path):
    
    # Load settings from the passed dictionary
    needle_color = settings['last_needle_color']
    alignment_standby_joints = settings.get('alignment_standby_joints')
    standby_offset_mm = settings.get('last_offset_mm', 50.0)
    resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
    try:
        settings['calibration_matrix'] = np.load('calibration_matrix.npy')
        socketio.emit('status_update', {'msg': "Loaded calibration matrix 'calibration_matrix.npy'."})
    except FileNotFoundError:
        socketio.emit('status_update', {'msg': "'calibration_matrix.npy' not found! Run calibration first.", 'error': True})
        return

    model = YOLO(model_path)
    
    # --- UPDATED MASK DICTIONARY ---
    golden_masks = {
        "GR1": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_1.png'),
        "GR2": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_2.jpg'),
        "GR3": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_3.png'),
        "GR4_Negative": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_4.png')
    }
    # --- END UPDATE ---
    
    # Check if files exist to provide better error messages
    if not os.path.exists(golden_masks["GR1"]):
        socketio.emit('status_update', {'msg': f"Error: mask_1.png not found at {golden_masks['GR1']}", 'error': True})
        return
    if not os.path.exists(golden_masks["GR2"]):
        socketio.emit('status_update', {'msg': f"Warning: mask_2.jpg not found at {golden_masks['GR2']}", 'error': False})
    if not os.path.exists(golden_masks["GR3"]):
        socketio.emit('status_update', {'msg': f"Error: mask_3.png (Alignment Mask) not found at {golden_masks['GR3']}", 'error': True})
        return
    if not os.path.exists(golden_masks["GR4_Negative"]):
        socketio.emit('status_update', {'msg': f"Error: mask_4.png (Negative Mask) not found at {golden_masks['GR4_Negative']}", 'error': True})
        return


    STATE_DETECTING_DUT, STATE_MOVING_TO_POSITION, STATE_ALIGNING, STATE_COMPLETE = 0, 1, 2, 3
    state = reset_state()
    STATIONARY_SECONDS = 2
    completion_time = None
    auto_reset_delay = 5.0

    while not stop_event.is_set():
        with camera_lock:
            frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None

        if frame is None:
            socketio.sleep(0.1)
            continue
            
        display_frame = frame.copy()

        if state['current_state'] == STATE_DETECTING_DUT:
            
            if request_process_stop_event.is_set():
                socketio.emit('status_update', {'msg': 'Process stop requested. Halting detection loop.'})
                break
            
            results = model(frame, verbose=False, conf=0.8)
            
            if len(results[0].boxes) > 0:
                
                all_boxes = results[0].boxes
                confs = all_boxes.conf.cpu().numpy()
                best_idx = np.argmax(confs)
                box = all_boxes[best_idx]
                
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                
                if state['last_bbox'] is None:
                    state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
                if calculate_distance(bbox, state['last_bbox']) < 30: 
                    if time.time() - state['stationary_start_time'] > STATIONARY_SECONDS:
                        socketio.emit('status_update', {'msg': 'DUT is static. Performing template matching...'})
                        state['dut_bbox'] = bbox
                        dut_crop_resized = cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (512, 512))
                        
                        # This call now uses the new dual-score logic
                        state['template_results'] = run_template_matching_dual(dut_crop_resized, golden_masks, 0, settings)
                        
                        if state['template_results']:
                            msg = "Template match found. Press Proceed to start alignment."
                            if not wait_for_proceed(socketio, proceed_event, stop_event, state['template_results']['overlay'], msg):
                                state = reset_state()
                                continue

                            if alignment_standby_joints and sum(abs(j) for j in alignment_standby_joints) > 0:
                                state['current_state'] = STATE_MOVING_TO_POSITION
                            else:
                                socketio.emit('status_update', {'msg': 'No standby joints defined. Skipping move.'})
                                state['current_state'] = STATE_ALIGNING
                        else:
                            socketio.emit('status_update', {'msg': 'Template matching failed.', 'error': True})
                            state = reset_state()
                else:
                    state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            else:
                state['last_bbox'] = None
                
            send_frame_to_ui(socketio, display_frame) # Send live feed

        elif state['current_state'] == STATE_MOVING_TO_POSITION:
            if not state['movement_initiated']:
                socketio.emit('status_update', {'msg': 'Moving to prefixed standby pose...'})
                state['movement_initiated'] = True
                if cobot.joint_move(alignment_standby_joints, speed=20, block=True):
                    socketio.emit('status_update', {'msg': 'Standby position reached. Proceeding to alignment.'})
                    state['current_state'] = STATE_ALIGNING
                else:
                    socketio.emit('status_update', {'msg': 'Failed to move to standby position. Resetting.', 'error': True})
                    state = reset_state()

        elif state['current_state'] == STATE_ALIGNING:
            socketio.emit('status_update', {'msg': 'Starting alignment procedure...'})
            
            alignment_data = align_to_waypoints(
                socketio, proceed_event, stop_event, cobot, 
                camera_frame_buffer, camera_lock, state, settings
            )
                    
            socketio.emit('status_update', {'msg': f"Alignment finished. Moving {standby_offset_mm}mm to standby position."})
            if cobot:
                cobot.move_relative(dx=standby_offset_mm, dy=0, dz=0, speed=20, block=True)
                socketio.emit('status_update', {'msg': 'Robot is in post-alignment standby.'})
                
            state['current_state'] = STATE_COMPLETE
            completion_time = time.time()

        elif state['current_state'] == STATE_COMPLETE:
            if time.time() - completion_time > auto_reset_delay:
                socketio.emit('status_update', {'msg': 'Cycle complete. Starting new detection cycle...'})
                state = reset_state()
            else:
                cv2.putText(display_frame, "COMPLETE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                send_frame_to_ui(socketio, display_frame) # Send live feed

        socketio.sleep(0.02) # Main loop sleep to yield control
        
    socketio.emit('status_update', {'msg': 'Alignment process stopped.'})


# ============================================================================
# --- NEW: DEMO MODE Main Process Loop ---
# ============================================================================
def run_demo_alignment_process(socketio, proceed_event, stop_event, request_process_stop_event, camera_frame_buffer, camera_lock, settings, model_path):
    """
    Main process loop for DEMO MODE.
    Runs Detection -> Template Match -> Continuous Tip Detection.
    Stops when 'request_process_stop_event' is set.
    """
    
    # Load settings from the passed dictionary
    needle_color = settings['last_needle_color']
    resW, resH = settings.get('resolution_w', 640), settings.get('resolution_h', 480)
    
    # --- No calibration matrix needed for demo ---
    socketio.emit('status_update', {'msg': "Running in DEMO MODE. No cobot alignment."})

    model = YOLO(model_path)
    
    # --- UPDATED MASK DICTIONARY ---
    golden_masks = {
        "GR1": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_1.png'),
        "GR2": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_2.jpg'),
        "GR3": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_3.png'),
        "GR4_Negative": os.path.join(PARENT_DIR, 'MY_MODEL', 'masks', 'mask_4.png')
    }
    # --- END UPDATE ---
    
    # Check if files exist to provide better error messages
    if not os.path.exists(golden_masks["GR1"]):
        socketio.emit('status_update', {'msg': f"Error: mask_1.png not found at {golden_masks['GR1']}", 'error': True})
        return
    if not os.path.exists(golden_masks["GR2"]):
        socketio.emit('status_update', {'msg': f"Warning: mask_2.jpg not found at {golden_masks['GR2']}", 'error': False})
    if not os.path.exists(golden_masks["GR3"]):
        socketio.emit('status_update', {'msg': f"Error: mask_3.png (Alignment Mask) not found at {golden_masks['GR3']}", 'error': True})
        return
    if not os.path.exists(golden_masks["GR4_Negative"]):
        socketio.emit('status_update', {'msg': f"Error: mask_4.png (Negative Mask) not found at {golden_masks['GR4_Negative']}", 'error': True})
        return

    # Define states for this demo loop
    STATE_DETECTING_DUT = 0
    STATE_DEMO_TIP_DETECT = 1 # New state for demo
    
    state = reset_state() # Uses state 0 by default
    STATIONARY_SECONDS = 2

    while not stop_event.is_set():
        with camera_lock:
            frame = camera_frame_buffer[0].copy() if camera_frame_buffer[0] is not None else None

        if frame is None:
            socketio.sleep(0.1)
            continue
            
        display_frame = frame.copy()

        if state['current_state'] == STATE_DETECTING_DUT:
            # This is the "Stop Process After Cycle" check
            if request_process_stop_event.is_set():
                socketio.emit('status_update', {'msg': 'Process stop requested. Halting detection loop.'})
                break
            
            results = model(frame, verbose=False, conf=0.96)
            
            if len(results[0].boxes) > 0:
                all_boxes = results[0].boxes
                confs = all_boxes.conf.cpu().numpy()
                best_idx = np.argmax(confs)
                box = all_boxes[best_idx]
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                
                if state['last_bbox'] is None:
                    state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
                if calculate_distance(bbox, state['last_bbox']) < 30: 
                    if time.time() - state['stationary_start_time'] > STATIONARY_SECONDS:
                        socketio.emit('status_update', {'msg': 'DEMO: DUT static. Performing template matching...'})
                        state['dut_bbox'] = bbox
                        dut_crop_resized = cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (512, 512))
                        
                        # This call now uses the new dual-score logic
                        state['template_results'] = run_template_matching_dual(dut_crop_resized, golden_masks, 0, settings)
                        
                        if state['template_results']:
                            msg = "DEMO: Template match found. Press Proceed to start tip detection."
                            # Show the template match overlay and wait
                            if not wait_for_proceed(socketio, proceed_event, stop_event, state['template_results']['overlay'], msg):
                                state = reset_state() # Reset if user cancels
                                continue

                            # --- DEMO MODE CHANGE ---
                            # Instead of moving cobot, go to tip detection state
                            socketio.emit('status_update', {'msg': 'DEMO: Proceeding to continuous tip detection.'})
                            state['current_state'] = STATE_DEMO_TIP_DETECT
                            # --- END DEMO MODE CHANGE ---
                        
                        else:
                            socketio.emit('status_update', {'msg': 'DEMO: Template matching failed.', 'error': True})
                            state = reset_state()
                else:
                    state['last_bbox'], state['stationary_start_time'] = bbox, time.time()
                    
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            else:
                state['last_bbox'] = None
                
            send_frame_to_ui(socketio, display_frame) # Send live feed

        # --- NEW DEMO STATE ---
        elif state['current_state'] == STATE_DEMO_TIP_DETECT:
            # This state runs continuously until "Stop Process" is pressed
            if request_process_stop_event.is_set():
                socketio.emit('status_update', {'msg': 'DEMO: Stop requested. Halting tip detection.'})
                break # Exit the main while loop
            
            # Continuously detect and display the needle tip
            tip_info = detect_needle_tip_rectangular(frame, needle_color, debug_frame=display_frame)
            
            if tip_info and tip_info[0]:
                tip_point = tip_info[0]
                # Draw the detected tip (already done by debug_frame)
                cv2.putText(display_frame, f"Tip Detected: {tip_point}", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Detecting Tip...", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Also draw the original DUT bounding box for context
            if state['dut_bbox'] is not None:
                bbox = state['dut_bbox']
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            send_frame_to_ui(socketio, display_frame) # Send live feed
        
        # --- END NEW DEMO STATE ---

        socketio.sleep(0.02) # Main loop sleep to yield control
        
    socketio.emit('status_update', {'msg': 'Demo process stopped.'})