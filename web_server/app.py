# import subprocess
# from flask import Flask, jsonify, render_template, request
# from flask_socketio import SocketIO
# import os
# import json # <-- Import the json library

# # --- Paths (Settings file path added) ---
# WEB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(WEB_SERVER_DIR)
# YOLO_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'yolo_detect.py')
# MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'my_model_n_300.pt')
# SETTINGS_FILE_PATH = os.path.join(WEB_SERVER_DIR, 'settings.json') # <-- Path to settings

# # --- Initialize Flask and SocketIO ---
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key!' 
# socketio = SocketIO(app)

# # --- Helper function to save settings ---
# def save_settings(new_settings):
#     """Reads current settings, updates them, and saves back to the file."""
#     try:
#         # Read the existing settings first to not lose any keys
#         with open(SETTINGS_FILE_PATH, 'r') as f:
#             settings = json.load(f)
        
#         # Update the settings with the new values provided
#         settings.update(new_settings)

#         # Write the updated dictionary back to the file
#         with open(SETTINGS_FILE_PATH, 'w') as f:
#             json.dump(settings, f, indent=2)
#         print(f"Saved new settings: {new_settings}")
#     except FileNotFoundError:
#         print(f"Warning: settings.json not found. Creating a new one.")
#         with open(SETTINGS_FILE_PATH, 'w') as f:
#             json.dump(new_settings, f, indent=2)
#     except Exception as e:
#         print(f"Error saving settings: {e}")

# # --- Helper function to stream output (No changes here) ---
# def run_command_and_stream_output(command):
#     try:
#         process = subprocess.Popen(
#             command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#             text=True, bufsize=1
#         )
#         for line in process.stdout:
#             line = line.strip()
#             print(line)
#             if line.startswith("STATUS::"):
#                 message = line.replace("STATUS::", "", 1)
#                 socketio.emit('status_update', {'msg': message})
        
#         stderr_output = process.stderr.read()
#         if process.wait() != 0:
#             error_message = f"Script failed: {stderr_output}"
#             print(error_message) 
#             socketio.emit('status_update', {'msg': error_message, 'error': True})
#         else:
#             socketio.emit('status_update', {'msg': "Process completed successfully."})
#     except Exception as e:
#         print(f"Error: {e}")
#         socketio.emit('status_update', {'msg': f"Failed to start process: {str(e)}", 'error': True})

# @app.route('/')
# def index():
#     return render_template('index.html')

# # --- NEW: Route to get the last used settings ---
# @app.route('/get-settings', methods=['GET'])
# def get_settings():
#     try:
#         with open(SETTINGS_FILE_PATH, 'r') as f:
#             settings = json.load(f)
#             return jsonify(settings)
#     except FileNotFoundError:
#         # If the file doesn't exist, return default values
#         return jsonify({
#             "last_move_mm": 1.5, "last_offset_mm": 50, "last_needle_color": "green"
#         })

# @app.route('/run-calibrate', methods=['POST'])
# def run_calibrate():
#     data = request.get_json()
#     move_mm = data.get('move_mm', '1.5')
#     offset_mm = data.get('offset_mm', '50')

#     # --- Save the new settings before running ---
#     save_settings({'last_move_mm': float(move_mm), 'last_offset_mm': float(offset_mm)})
    
#     command = [
#         "python", YOLO_SCRIPT_PATH, "--source", "0", "--cobot_ip", "192.168.1.166",
#         "--needle_color", "green", "--calibrate", "--calibration_move_mm", str(move_mm),
#         "--standby_offset_mm", str(offset_mm)
#     ]
#     print(f"🚀 Executing command: {' '.join(command)}")
#     socketio.start_background_task(run_command_and_stream_output, command)
#     return jsonify({'status': 'success', 'message': 'Calibration process initiated.'})

# @app.route('/run-alignment', methods=['POST'])
# def run_alignment():
#     data = request.get_json()
#     needle_color = data.get('needle_color', 'green')
    
#     # --- Save the new setting before running ---
#     save_settings({'last_needle_color': needle_color})

#     command = [
#         "python", YOLO_SCRIPT_PATH, "--model", MODEL_PATH, "--source", "0",
#         "--cobot_ip", "192.168.1.166", "--needle_color", str(needle_color)
#     ]
#     print(f"🚀 Executing command: {' '.join(command)}") 
#     socketio.start_background_task(run_command_and_stream_output, command)
#     return jsonify({'status': 'success', 'message': 'Alignment process initiated.'})

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=5000, debug=True)



#-------------------testing only (without cobot)
# import subprocess
# from flask import Flask, jsonify, render_template, request
# from flask_socketio import SocketIO
# import os
# import json

# # --- Paths ---
# WEB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(WEB_SERVER_DIR)
# YOLO_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'yolo_detect.py')
# MODEL_PATH = os.path.join(PROJECT_ROOT, 'my_model_n_300.pt') # Ensure this model name is correct

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key!'
# socketio = SocketIO(app)

# # Global variable to hold the current running process
# current_process = None

# def run_command_and_stream_output(command):
#     """Runs a command, streams its output, and handles termination."""
#     global current_process
#     try:
#         process = subprocess.Popen(
#             command,
#             cwd=PROJECT_ROOT,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             stdin=subprocess.PIPE,
#             text=True,
#             bufsize=1,
#             universal_newlines=True
#         )
#         current_process = process

#         for line in process.stdout:
#             line = line.strip()
#             if not line.startswith("FRAME::"):
#                 print(line)
            
#             if line.startswith("STATUS::"):
#                 message = line.replace("STATUS::", "", 1)
#                 socketio.emit('status_update', {'msg': message})
#             elif line.startswith("FRAME::"):
#                 img_data = line.replace("FRAME::", "", 1)
#                 socketio.emit('video_frame', {'img_data': img_data})
#             elif line.startswith("PAUSE::"):
#                 message = line.replace("PAUSE::", "", 1)
#                 socketio.emit('status_update', {'msg': message, 'paused': True})
#             elif line.startswith("FPS::"):
#                 fps_value = line.replace("FPS::", "", 1)
#                 socketio.emit('fps_update', {'fps': fps_value})
        
#         return_code = process.wait()
#         stderr_output = process.stderr.read()
#         print(f"DEBUG: Subprocess finished with exit code {return_code}")
#         if return_code != 0:
#             error_message = f"Script exited with code {return_code}."
#             if stderr_output:
#                 error_message += f" Details: {stderr_output.strip()}"
#             socketio.emit('status_update', {'msg': error_message, 'error': True})
#         else:
#             socketio.emit('status_update', {'msg': "Process completed."})

#     except Exception as e:
#         socketio.emit('status_update', {'msg': f"Failed to start process: {str(e)}", 'error': True})
#     finally:
#         current_process = None

# @socketio.on('emergency_stop')
# def handle_emergency_stop():
#     global current_process
#     if current_process:
#         print("🚨 Received emergency stop signal. Terminating process...")
#         current_process.terminate()
#         socketio.emit('status_update', {'msg': "Process terminated by user.", 'error': True})
#     else:
#         socketio.emit('status_update', {'msg': "No process is currently running."})

# @socketio.on('continue_process')
# def handle_continue():
#     global current_process
#     if current_process:
#         try:
#             current_process.stdin.write('\n')
#             current_process.stdin.flush()
#             socketio.emit('status_update', {'msg': "Resuming process...", 'paused': False})
#         except Exception as e:
#             print(f"Error sending continue signal: {e}")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/run-test', methods=['POST'])
# def run_test():
#     command = [
#         "python", YOLO_SCRIPT_PATH, "--model", MODEL_PATH, "--source", "0", "--no_cobot"
#     ]
#     print(f"🚀 Executing TEST command: {' '.join(command)}")
#     socketio.start_background_task(run_command_and_stream_output, command)
#     return jsonify({'status': 'success', 'message': 'Integration test initiated.'})

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=5000, debug=True)



#---- real with cobot
# import subprocess
# from flask import Flask, jsonify, render_template, request
# from flask_socketio import SocketIO
# import os
# import json

# # ============================================================================
# # FILE PATHS
# # ============================================================================
# # These paths are dynamically calculated to ensure the app is portable.
# WEB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(WEB_SERVER_DIR)
# YOLO_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'yolo_detect.py')
# MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'my_model_n_300.pt') # Your chosen model
# SETTINGS_FILE_PATH = os.path.join(WEB_SERVER_DIR, 'settings.json') # For future use
# POSITIONS_FILE_PATH = os.path.join(PROJECT_ROOT, 'cobot_positions.json')

# # ============================================================================
# # FLASK & SOCKET.IO SETUP
# # ============================================================================
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'a_very_secret_key_that_should_be_changed!'
# socketio = SocketIO(app)

# # Global variable to hold the currently running subprocess
# current_process = None

# # ============================================================================
# # CORE PROCESS MANAGEMENT
# # ============================================================================
# def run_command_and_stream_output(command):
#     """
#     Runs a command in a subprocess, streams its stdout line by line,
#     and forwards formatted messages to the web browser via WebSockets.
#     """
#     global current_process
#     try:
#         process = subprocess.Popen(
#             command,
#             cwd=PROJECT_ROOT,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             stdin=subprocess.PIPE,
#             text=True,
#             bufsize=1,
#             universal_newlines=True
#         )
#         current_process = process

#         # Read the script's output line by line, in real-time
#         for line in process.stdout:
#             line = line.strip()
#             if not line:
#                 continue

#             # Don't clutter the server terminal with long image data
#             if not line.startswith("FRAME::"):
#                 print(line)
            
#             # Parse the line for special prefixes and emit corresponding events
#             if line.startswith("STATUS::"):
#                 socketio.emit('status_update', {'msg': line.replace("STATUS::", "", 1)})
#             elif line.startswith("FRAME::"):
#                 socketio.emit('video_frame', {'img_data': line.replace("FRAME::", "", 1)})
#             elif line.startswith("PAUSE::"):
#                 socketio.emit('status_update', {'msg': line.replace("PAUSE::", "", 1), 'paused': True})
#             elif line.startswith("FPS::"):
#                 socketio.emit('fps_update', {'fps': line.replace("FPS::", "", 1)})
#             elif line.startswith("ERROR::"):
#                 socketio.emit('status_update', {'msg': line.replace("ERROR::", "", 1), 'error': True})
        
#         # After the process ends, check for errors
#         return_code = process.wait()
#         stderr_output = process.stderr.read()
        
#         print(f"DEBUG: Subprocess finished with exit code {return_code}")
#         if return_code != 0:
#             error_message = f"Script Error: {stderr_output.strip()}" if stderr_output else f"Script exited with code {return_code}."
#             socketio.emit('status_update', {'msg': error_message, 'error': True})
#         else:
#             socketio.emit('status_update', {'msg': "Process completed successfully."})

#     except Exception as e:
#         socketio.emit('status_update', {'msg': f"Failed to start process: {str(e)}", 'error': True})
#     finally:
#         current_process = None # Clear the process reference

# # ============================================================================
# # WEBSOCKET EVENT HANDLERS
# # ============================================================================
# @socketio.on('emergency_stop')
# def handle_emergency_stop():
#     """Receives stop signal from browser and terminates the script."""
#     global current_process
#     if current_process:
#         print("🚨 Received emergency stop signal. Terminating process...")
#         current_process.terminate()
#         socketio.emit('status_update', {'msg': "Process terminated by user.", 'error': True})
#     else:
#         socketio.emit('status_update', {'msg': "No process is currently running."})

# @socketio.on('continue_process')
# def handle_continue():
#     """Receives 'continue' signal from browser and unpauses the script."""
#     global current_process
#     if current_process and current_process.stdin:
#         try:
#             current_process.stdin.write('\n')
#             current_process.stdin.flush()
#             socketio.emit('status_update', {'msg': "Resuming...", 'paused': False})
#         except Exception as e:
#             print(f"Error sending continue signal: {e}")

# # ============================================================================
# # HTTP ROUTES
# # ============================================================================
# @app.route('/')
# def index():
#     """Serves the main HTML page."""
#     return render_template('index.html')

# @app.route('/get-positions', methods=['GET'])
# def get_positions():
#     """Reads and returns the saved joint positions from the JSON file."""
#     try:
#         with open(POSITIONS_FILE_PATH, 'r') as f:
#             return jsonify(json.load(f))
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/save-positions', methods=['POST'])
# def save_positions():
#     """Receives joint positions from the browser and saves them to the JSON file."""
#     try:
#         with open(POSITIONS_FILE_PATH, 'w') as f:
#             json.dump(request.get_json(), f, indent=4)
#         return jsonify({"status": "success", "message": "Positions saved successfully."})
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/start-positioning-view', methods=['POST'])
# def start_positioning_view():
#     """Starts the yolo_detect.py script in 'positioning' mode."""
#     command = ["python", YOLO_SCRIPT_PATH, "--mode", "positioning", "--source", "0"]
#     socketio.start_background_task(run_command_and_stream_output, command)
#     return jsonify({'status': 'success'})

# @app.route('/run-calibrate', methods=['POST'])
# def run_calibrate():
#     """Starts the yolo_detect.py script in 'calibration' mode."""
#     data = request.get_json()
#     command = [
#         "python", YOLO_SCRIPT_PATH,
#         "--mode", "calibration",
#         "--source", "0",
#         "--cobot_ip", data.get('cobot_ip', '192.168.1.166'),
#         "--needle_color", data.get('needle_color', 'red'),
#         "--calibration_move_mm", str(data.get('move_mm', 10.0)),
#         "--standby_offset_mm", str(data.get('offset_mm', 50.0)),
#         "--positions_file", POSITIONS_FILE_PATH
#     ]
#     print(f"🚀 Executing CALIBRATION: {' '.join(command)}")
#     socketio.start_background_task(run_command_and_stream_output, command)
#     return jsonify({'status': 'success'})

# @app.route('/run-alignment', methods=['POST'])
# def run_alignment():
#     """Starts the yolo_detect.py script in 'alignment' mode."""
#     data = request.get_json()
#     command = [
#         "python", YOLO_SCRIPT_PATH,
#         "--mode", "alignment",
#         "--model", MODEL_PATH,
#         "--source", "0",
#         "--cobot_ip", data.get('cobot_ip', '192.168.1.166'),
#         "--needle_color", data.get('needle_color', 'red'),
#         "--positions_file", POSITIONS_FILE_PATH,
#         "--alignment_standby_offset_mm", str(data.get('offset_mm', 50.0))
#     ]
#     print(f"🚀 Executing ALIGNMENT: {' '.join(command)}")
#     socketio.start_background_task(run_command_and_stream_output, command)
#     return jsonify({'status': 'success'})

# # ============================================================================
# # SCRIPT ENTRY POINT
# # ============================================================================
# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=5000, debug=True)



#------for real integration (high latency)
# import os
# import json
# import base64
# import threading
# import cv2
# import sys

# # --- Global State Variables ---
# cap = None                  # OpenCV Camera object
# cobot = None                # JAKACobot object
# current_process_task = None # Background task for calibration/alignment
# video_thread = None         # Background task for streaming video
# stop_event = threading.Event()   # Flag to stop background tasks
# proceed_event = threading.Event() # Flag to signal user "proceed"

# # --- Paths ---
# WEB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(WEB_SERVER_DIR) # Assumes app.py is in a 'server' subdir
# sys.path.insert(0, PROJECT_ROOT)
# YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'my_model_n_300.pt')
# SETTINGS_FILE_PATH = os.path.join(WEB_SERVER_DIR, 'settings.json') 

# from flask import Flask, jsonify, render_template, request
# from flask_socketio import SocketIO
# # --- Import all logic from your refactored file ---
# import cobot_logic

# # --- Default Settings ---
# DEFAULT_SETTINGS = {
#     "last_move_mm": 1.5,
#     "last_offset_mm": 50.0,
#     "last_needle_color": "green",
#     "cobot_ip": "192.168.1.166",
#     "calibration_start_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     "alignment_standby_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# }

# # --- Initialize Flask and SocketIO ---
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key!' 
# socketio = SocketIO(app, async_mode='threading')

# # --- Helper function to read/save settings ---
# def get_all_settings():
#     """Reads all settings from the JSON file, using defaults if file not found."""
#     try:
#         with open(SETTINGS_FILE_PATH, 'r') as f:
#             settings = json.load(f)
#             # Ensure all default keys are present
#             for key, value in DEFAULT_SETTINGS.items():
#                 if key not in settings:
#                     settings[key] = value
#             return settings
#     except (FileNotFoundError, json.JSONDecodeError):
#         print(f"Warning: {SETTINGS_FILE_PATH} not found or invalid. Creating with defaults.")
#         save_all_settings(DEFAULT_SETTINGS)
#         return DEFAULT_SETTINGS

# def save_all_settings(new_settings):
#     """Saves the entire settings dictionary to the file."""
#     try:
#         with open(SETTINGS_FILE_PATH, 'w') as f:
#             json.dump(new_settings, f, indent=2)
#         print(f"Saved new settings: {new_settings}")
#         return True
#     except Exception as e:
#         print(f"Error saving settings: {e}")
#         return False

# # ============================================================================
# # Background Tasks (Video Stream, Calibration, Alignment)
# # ============================================================================

# def stream_video(stop_flag):
#     """
#     Runs in a background thread to stream video to the UI
#     when no other process is active.
#     """
#     global cap
#     print("Starting video stream thread...")
#     try:
#         if cap is None or not cap.isOpened():
#             cap = cv2.VideoCapture(0) # TODO: Make camera index configurable?
#             if not cap.isOpened():
#                 socketio.emit('status_update', {'msg': 'Error: Cannot open camera.', 'error': True})
#                 print("Error: Cannot open camera.")
#                 return
        
#         resW, resH = 640, 480 # Standard streaming resolution
#         cap.set(3, resW); cap.set(4, resH)

#         while not stop_flag.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 socketio.emit('status_update', {'msg': 'Camera disconnected?', 'error': True})
#                 break
            
#             # Draw the alignment box
#             cx, cy = resW // 2, resH // 2
#             box_size = 150 # Half-width/height of the box
#             cv2.rectangle(frame, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (0, 255, 255), 2)
#             cv2.putText(frame, "Align DUT and Needle Here", (cx - 100, cy - box_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#             cobot_logic.send_frame_to_ui(socketio, frame, 'video_frame')
#             socketio.sleep(0.05) # ~20 FPS
            
#     except Exception as e:
#         print(f"Error in video thread: {e}")
#     finally:
#         if cap:
#             cap.release()
#             cap = None
#         print("Video stream thread stopped.")

# def run_calibration_wrapper(settings_data):
#     """Wrapper to safely run calibration in a background task."""
#     global cap, cobot, stop_event, proceed_event
#     socketio.emit('status_update', {'msg': 'Starting calibration process...'})
    
#     try:
#         # 1. Get settings
#         current_settings = get_all_settings()
#         current_settings.update(settings_data) # Update with latest from UI
        
#         # 2. Initialize Hardware
#         if cap is None or not cap.isOpened():
#             cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             socketio.emit('status_update', {'msg': 'Cannot open camera for calibration.', 'error': True})
#             return

#         cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
#         if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
#             socketio.emit('status_update', {'msg': 'Failed to connect and enable the cobot.', 'error': True})
#             raise Exception("Cobot connection failed")

#         # 3. Run the logic
#         cobot_logic.run_calibration_routine(
#             socketio, proceed_event, stop_event,
#             cobot, cap, current_settings
#         )
        
#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Calibration error: {e}', 'error': True})
#     finally:
#         # 4. Cleanup
#         if cobot:
#             cobot.disconnect()
#             cobot = None
#         if cap:
#             cap.release()
#             cap = None
#         socketio.emit('status_update', {'msg': 'Calibration process finished.'})
#         socketio.emit('process_finished') # Tell UI process is done

# def run_alignment_wrapper(settings_data):
#     """Wrapper to safely run alignment in a background task."""
#     global cap, cobot, stop_event, proceed_event
#     socketio.emit('status_update', {'msg': 'Starting alignment process...'})
    
#     try:
#         # 1. Get settings
#         current_settings = get_all_settings()
#         current_settings.update(settings_data)
        
#         # 2. Initialize Hardware
#         if cap is None or not cap.isOpened():
#             cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             socketio.emit('status_update', {'msg': 'Cannot open camera for alignment.', 'error': True})
#             return
            
#         cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
#         if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
#             socketio.emit('status_update', {'msg': 'Failed to connect and enable the cobot.', 'error': True})
#             raise Exception("Cobot connection failed")

#         # 3. Run the logic
#         cobot_logic.run_alignment_process(
#             socketio, proceed_event, stop_event,
#             cobot, cap, current_settings, YOLO_MODEL_PATH
#         )

#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Alignment error: {e}', 'error': True})
#     finally:
#         # 4. Cleanup
#         if cobot:
#             cobot.disconnect()
#             cobot = None
#         if cap:
#             cap.release()
#             cap = None
#         socketio.emit('status_update', {'msg': 'Alignment process finished.'})
#         socketio.emit('process_finished') # Tell UI process is done

# # ============================================================================
# # HTTP Routes
# # ============================================================================
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/get-settings', methods=['GET'])
# def get_settings_route():
#     return jsonify(get_all_settings())

# @app.route('/save-settings', methods=['POST'])
# def save_settings_route():
#     data = request.get_json()
#     if save_all_settings(data):
#         return jsonify({'status': 'success', 'message': 'Settings saved.'})
#     else:
#         return jsonify({'status': 'error', 'message': 'Failed to save settings.'}), 500

# # ============================================================================
# # SocketIO Event Handlers
# # ============================================================================

# def stop_all_tasks():
#     """Helper function to stop all running tasks."""
#     global video_thread, current_process_task, cobot
    
#     # 1. Signal stop
#     stop_event.set()
#     proceed_event.set() # Unblock any waiting tasks
    
#     # 2. Wait for tasks to finish
#     if video_thread:
#         video_thread.join()
#         video_thread = None
#     if current_process_task:
#         current_process_task.join()
#         current_process_task = None
        
#     # 3. Disconnect cobot just in case
#     if cobot:
#         print("Force disconnecting cobot...")
#         cobot.disconnect()
#         cobot = None
    
#     print("All tasks stopped.")

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
#     socketio.emit('status_update', {'msg': 'Server connected. Requesting video stream...'})
    
# @socketio.on('request_video_stream')
# def handle_request_video_stream():
#     """Client requests the idle video stream."""
#     global video_thread, stop_event
#     stop_all_tasks() # Stop anything currently running
#     stop_event.clear() # Clear stop flag for the new task
#     video_thread = socketio.start_background_task(stream_video, stop_event)
    
# @socketio.on('start_calibration')
# def handle_start_calibration(data):
#     """Client requests to start the calibration process."""
#     global current_process_task, stop_event, proceed_event
#     stop_all_tasks() # Stop video stream
#     stop_event.clear()
#     proceed_event.clear()
#     current_process_task = socketio.start_background_task(run_calibration_wrapper, data)

# @socketio.on('start_alignment')
# def handle_start_alignment(data):
#     """Client requests to start the alignment process."""
#     global current_process_task, stop_event, proceed_event
#     stop_all_tasks()
#     stop_event.clear()
#     proceed_event.clear()
#     current_process_task = socketio.start_background_task(run_alignment_wrapper, data)

# @socketio.on('user_proceed')
# def handle_user_proceed():
#     """Client clicked the 'Proceed' button."""
#     print("User 'Proceed' signal received.")
#     proceed_event.set()

# @socketio.on('emergency_stop')
# def handle_emergency_stop():
#     """Client clicked the 'EMERGENCY STOP' button."""
#     print("EMERGENCY STOP received!")
#     socketio.emit('status_update', {'msg': 'EMERGENCY STOP requested!', 'error': True})
#     stop_all_tasks()
    
#     # Restart the idle video stream
#     socketio.emit('status_update', {'msg': 'System stopped. Restarting idle video stream.'})
#     handle_request_video_stream() # Re-request video

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected. Stopping all tasks.')
#     stop_all_tasks()

# # --- Main ---
# if __name__ == '__main__':
#     print("Starting server on http://0.0.0.0:5000")
#     socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)


#-------------------low latency final integration
# import os
# import json
# import base64
# import threading
# import cv2
# import sys
# import time # Import time for delays
# from flask import Flask, jsonify, render_template, request
# from flask_socketio import SocketIO

# # --- Add Project Root to Path ---
# WEB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(WEB_SERVER_DIR)
# sys.path.insert(0, PROJECT_ROOT)
# # --- End Path Addition ---

# # --- Import all logic from your refactored file ---
# import cobot_logic

# # --- Global State Variables ---
# cap = None                               # OpenCV Camera object
# cobot = None                             # JAKACobot object
# current_process_task = None              # Background task for calibration/alignment
# video_sender_thread = None               # Background task for sending video frames to UI
# camera_reader_thread = None              # Dedicated background task for reading from camera
# stop_event = threading.Event()           # Flag to stop background tasks (all tasks)
# proceed_event = threading.Event()        # Flag to signal user "proceed"
# send_video_to_ui_flag = threading.Event() # Flag to enable/disable sending video frames to UI

# # Camera frame buffer and lock for thread-safe access
# camera_frame_buffer = [None] # A list to hold the latest frame (mutable)
# camera_lock = threading.Lock()

# # --- Paths ---
# YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'my_model_n_300.pt')
# SETTINGS_FILE_PATH = os.path.join(WEB_SERVER_DIR, 'settings.json') 

# # --- Default Settings ---
# DEFAULT_SETTINGS = {
#     "camera_index": 0, # NEW: Camera index setting
#     "camera_res_w": 640, # NEW: Camera resolution setting
#     "camera_res_h": 480, # NEW: Camera resolution setting
#     "last_move_mm": 1.5,
#     "last_offset_mm": 50.0,
#     "last_needle_color": "green",
#     "cobot_ip": "192.168.1.166",
#     "calibration_start_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     "alignment_standby_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# }

# # --- Initialize Flask and SocketIO ---
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key!' 
# socketio = SocketIO(app, async_mode='threading')

# # --- Helper function to read/save settings ---
# def get_all_settings():
#     """Reads all settings from the JSON file, using defaults if file not found."""
#     try:
#         with open(SETTINGS_FILE_PATH, 'r') as f:
#             settings = json.load(f)
#             # Ensure all default keys are present
#             for key, value in DEFAULT_SETTINGS.items():
#                 if key not in settings:
#                     settings[key] = value
#             return settings
#     except (FileNotFoundError, json.JSONDecodeError):
#         print(f"Warning: {SETTINGS_FILE_PATH} not found or invalid. Creating with defaults.")
#         save_all_settings(DEFAULT_SETTINGS)
#         return DEFAULT_SETTINGS

# def save_all_settings(new_settings):
#     """Saves the entire settings dictionary to the file."""
#     try:
#         with open(SETTINGS_FILE_PATH, 'w') as f:
#             json.dump(new_settings, f, indent=2)
#         print(f"Saved new settings: {new_settings}")
#         return True
#     except Exception as e:
#         print(f"Error saving settings: {e}")
#         return False

# # ============================================================================
# # Background Camera Management Threads
# # ============================================================================

# def camera_reader(camera_index, res_w, res_h, stop_flag, frame_buffer, frame_lock):
#     """
#     Dedicated thread to continuously read frames from the camera.
#     This keeps the camera open and ensures frames are always available quickly.
#     """
#     global cap
#     print(f"Camera Reader: Starting for index {camera_index} at {res_w}x{res_h}...")
#     try:
#         cap = cv2.VideoCapture(camera_index)
#         if not cap.isOpened():
#             socketio.emit('status_update', {'msg': f'Error: Cannot open camera at index {camera_index}.', 'error': True})
#             print(f"Camera Reader ERROR: Cannot open camera at index {camera_index}.")
#             return

#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        
#         # Read a few frames to let camera auto-adjust
#         for _ in range(5):
#             _, _ = cap.read()
#             time.sleep(0.1)

#         print(f"Camera Reader: Successfully opened camera {camera_index}.")
#         socketio.emit('status_update', {'msg': f'Camera {camera_index} opened successfully.'})

#         while not stop_flag.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 socketio.emit('status_update', {'msg': 'Camera Reader: Failed to read frame. Disconnected?', 'error': True})
#                 print("Camera Reader ERROR: Failed to read frame.")
#                 break # Exit loop on read failure

#             with frame_lock:
#                 frame_buffer[0] = frame.copy() # Store the latest frame

#             time.sleep(0.01) # Read as fast as possible, but yield some CPU
            
#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Camera Reader critical error: {e}', 'error': True})
#         print(f"Camera Reader CRITICAL ERROR: {e}")
#     finally:
#         if cap:
#             cap.release()
#             print(f"Camera Reader: Camera {camera_index} released.")
#             cap = None
#         with frame_lock:
#             frame_buffer[0] = None # Clear buffer
#         print("Camera Reader: Thread stopped.")

# def video_sender(stop_flag, send_video_flag, frame_buffer, frame_lock):
#     """
#     Runs in a background thread to send the latest frame from the buffer to the UI
#     when `send_video_flag` is set.
#     """
#     print("Video Sender: Thread starting...")
#     while not stop_flag.is_set():
#         if send_video_flag.is_set():
#             frame = None
#             with frame_lock:
#                 if frame_buffer[0] is not None:
#                     frame = frame_buffer[0].copy()

#             if frame is not None:
#                 # Draw the alignment box on the stream frames
#                 resW, resH = frame.shape[1], frame.shape[0]
#                 cx, cy = resW // 2, resH // 2
#                 box_size = 150 # Half-width/height of the box
#                 cv2.rectangle(frame, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (0, 255, 255), 2)
#                 cv2.putText(frame, "Align DUT and Needle Here", (cx - 100, cy - box_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#                 cobot_logic.send_frame_to_ui(socketio, frame, 'video_frame')
        
#         socketio.sleep(0.05) # Send at ~20 FPS

#     print("Video Sender: Thread stopped.")

# # ============================================================================
# # Background Process Wrappers (Calibration, Alignment)
# # ============================================================================

# def run_calibration_wrapper(settings_data):
#     """Wrapper to safely run calibration in a background task."""
#     global cobot, stop_event, proceed_event, send_video_to_ui_flag
#     socketio.emit('status_update', {'msg': 'Starting calibration process...'})
    
#     # Temporarily stop sending idle video frames to UI
#     send_video_to_ui_flag.clear()
    
#     try:
#         # 1. Get settings
#         current_settings = get_all_settings()
#         current_settings.update(settings_data) # Update with latest from UI
        
#         # 2. Initialize Hardware (Cobot only, camera is globally managed)
#         cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
#         if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
#             socketio.emit('status_update', {'msg': 'Failed to connect and enable the cobot.', 'error': True})
#             raise Exception("Cobot connection failed")

#         # 3. Run the logic
#         cobot_logic.run_calibration_routine(
#             socketio, proceed_event, stop_event,
#             cobot, camera_frame_buffer, camera_lock, current_settings
#         )
        
#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Calibration error: {e}', 'error': True})
#     finally:
#         # 4. Cleanup
#         if cobot:
#             cobot.disconnect()
#             cobot = None
        
#         # Resume sending idle video frames
#         send_video_to_ui_flag.set()
#         socketio.emit('status_update', {'msg': 'Calibration process finished.'})
#         socketio.emit('process_finished') # Tell UI process is done

# def run_alignment_wrapper(settings_data):
#     """Wrapper to safely run alignment in a background task."""
#     global cobot, stop_event, proceed_event, send_video_to_ui_flag
#     socketio.emit('status_update', {'msg': 'Starting alignment process...'})

#     # Temporarily stop sending idle video frames to UI
#     send_video_to_ui_flag.clear()
    
#     try:
#         # 1. Get settings
#         current_settings = get_all_settings()
#         current_settings.update(settings_data)
        
#         # 2. Initialize Hardware (Cobot only, camera is globally managed)
#         cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
#         if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
#             socketio.emit('status_update', {'msg': 'Failed to connect and enable the cobot.', 'error': True})
#             raise Exception("Cobot connection failed")

#         # 3. Run the logic
#         cobot_logic.run_alignment_process(
#             socketio, proceed_event, stop_event,
#             cobot, camera_frame_buffer, camera_lock, current_settings, YOLO_MODEL_PATH
#         )

#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Alignment error: {e}', 'error': True})
#     finally:
#         # 4. Cleanup
#         if cobot:
#             cobot.disconnect()
#             cobot = None
        
#         # Resume sending idle video frames
#         send_video_to_ui_flag.set()
#         socketio.emit('status_update', {'msg': 'Alignment process finished.'})
#         socketio.emit('process_finished') # Tell UI process is done

# # ============================================================================
# # HTTP Routes
# # ============================================================================
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/get-settings', methods=['GET'])
# def get_settings_route():
#     return jsonify(get_all_settings())

# @app.route('/save-settings', methods=['POST'])
# def save_settings_route():
#     data = request.get_json()
#     if save_all_settings(data):
#         return jsonify({'status': 'success', 'message': 'Settings saved.'})
#     else:
#         return jsonify({'status': 'error', 'message': 'Failed to save settings.'}), 500

# # ============================================================================
# # SocketIO Event Handlers
# # ============================================================================

# def stop_all_tasks():
#     """Helper function to stop all running worker tasks (calibration/alignment).
#        The camera reader and video sender threads are managed globally."""
#     global current_process_task
    
#     stop_event.set() # Signal all current process tasks to stop
#     proceed_event.set() # Unblock any waiting prompts
    
#     if current_process_task:
#         print("Waiting for current process task to finish...")
#         current_process_task.join(timeout=5) # Wait for it to finish
#         if current_process_task.is_alive():
#             print("Warning: Process task did not terminate gracefully.")
#         current_process_task = None
    
#     stop_event.clear() # Clear for the next task
#     proceed_event.clear() # Clear for the next prompt
    
#     print("All worker tasks stopped.")

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
#     socketio.emit('status_update', {'msg': 'Server connected. Initializing camera...'})
    
#     # Start the idle video stream (if not already running)
#     handle_request_video_stream()

# @socketio.on('request_video_stream')
# def handle_request_video_stream():
#     """Client requests the idle video stream. This just enables the sender."""
#     global video_sender_thread, camera_reader_thread, stop_event, send_video_to_ui_flag
    
#     current_settings = get_all_settings()
#     cam_idx = current_settings['camera_index']
#     res_w = current_settings['camera_res_w']
#     res_h = current_settings['camera_res_h']

#     # Start camera reader if not already running
#     if camera_reader_thread is None or not camera_reader_thread.is_alive():
#         stop_event.clear() # Ensure stop event is clear for camera threads
#         camera_reader_thread = socketio.start_background_task(camera_reader, cam_idx, res_w, res_h, stop_event, camera_frame_buffer, camera_lock)
#         # Give a moment for the camera to open
#         time.sleep(1.0) 

#     # Start video sender if not already running
#     if video_sender_thread is None or not video_sender_thread.is_alive():
#         stop_event.clear() # Ensure stop event is clear for camera threads
#         video_sender_thread = socketio.start_background_task(video_sender, stop_event, send_video_to_ui_flag, camera_frame_buffer, camera_lock)
        
#     send_video_to_ui_flag.set() # Enable sending frames to UI
#     socketio.emit('status_update', {'msg': 'Idle video stream enabled.'})


# @socketio.on('start_calibration')
# def handle_start_calibration(data):
#     """Client requests to start the calibration process."""
#     global current_process_task, stop_event, proceed_event
#     stop_all_tasks() # Stop any previous worker task
#     stop_event.clear() # Clear stop flag for the new task
#     proceed_event.clear() # Clear proceed flag for the new task
    
#     # Start calibration in a new background thread
#     current_process_task = socketio.start_background_task(run_calibration_wrapper, data)

# @socketio.on('start_alignment')
# def handle_start_alignment(data):
#     """Client requests to start the alignment process."""
#     global current_process_task, stop_event, proceed_event
#     stop_all_tasks() # Stop any previous worker task
#     stop_event.clear()
#     proceed_event.clear()
    
#     # Start alignment in a new background thread
#     current_process_task = socketio.start_background_task(run_alignment_wrapper, data)

# @socketio.on('user_proceed')
# def handle_user_proceed():
#     """Client clicked the 'Proceed' button."""
#     print("User 'Proceed' signal received.")
#     proceed_event.set()

# @socketio.on('emergency_stop')
# def handle_emergency_stop():
#     """Client clicked the 'EMERGENCY STOP' button."""
#     print("EMERGENCY STOP received!")
#     socketio.emit('status_update', {'msg': 'EMERGENCY STOP requested!', 'error': True})
    
#     # Stop current worker task if any
#     stop_all_tasks()
    
#     # Force disconnect cobot if it was connected
#     if cobot:
#         print("Force disconnecting cobot...")
#         cobot.disconnect()
        
#     # Ensure idle video stream is re-enabled
#     send_video_to_ui_flag.set() 
#     socketio.emit('status_update', {'msg': 'System stopped. Idle video stream active.'})

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected. Stopping worker tasks.')
#     # Only stop worker tasks, camera reader and sender should remain active
#     # until server shutdown to allow quick reconnects.
#     stop_all_tasks()

# # --- Main ---
# if __name__ == '__main__':
#     print("Starting server on http://0.0.0.0:5000")
#     # This ensures that when the Flask app shuts down, the camera reader and sender threads are also stopped.
#     try:
#         # Start camera reader and sender threads immediately on server start
#         # Use a dummy settings to get camera_index, will be updated by client
#         initial_settings = get_all_settings()
#         initial_cam_idx = initial_settings.get('camera_index', DEFAULT_SETTINGS['camera_index'])
#         initial_res_w = initial_settings.get('camera_res_w', DEFAULT_SETTINGS['camera_res_w'])
#         initial_res_h = initial_settings.get('camera_res_h', DEFAULT_SETTINGS['camera_res_h'])
        
#         stop_event.clear() # Ensure stop event is clear for initial camera threads
#         camera_reader_thread = socketio.start_background_task(camera_reader, initial_cam_idx, initial_res_w, initial_res_h, stop_event, camera_frame_buffer, camera_lock)
#         time.sleep(1.5) # Give camera some time to initialize before starting sender
#         send_video_to_ui_flag.set() # Initially enable sending frames
#         video_sender_thread = socketio.start_background_task(video_sender, stop_event, send_video_to_ui_flag, camera_frame_buffer, camera_lock)

#         socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
#     finally:
#         print("Server shutting down. Stopping all camera threads...")
#         stop_event.set() # Signal all threads to stop
#         if camera_reader_thread and camera_reader_thread.is_alive():
#             camera_reader_thread.join(timeout=5)
#         if video_sender_thread and video_sender_thread.is_alive():
#             video_sender_thread.join(timeout=5)
#         print("All threads terminated.")


#-----low latency with soft stop button
# import os
# import json
# import base64
# import threading
# import cv2
# import sys
# import time # Import time for delays
# from flask import Flask, jsonify, render_template, request
# from flask_socketio import SocketIO

# # --- Add Project Root to Path ---
# WEB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(WEB_SERVER_DIR)
# sys.path.insert(0, PROJECT_ROOT)
# # --- End Path Addition ---

# # --- Import all logic from your refactored file ---
# import cobot_logic

# # --- Global State Variables ---
# cap = None                               # OpenCV Camera object
# cobot = None                             # JAKACobot object
# current_process_task = None              # Background task for calibration/alignment
# video_sender_thread = None               # Background task for sending video frames to UI
# camera_reader_thread = None              # Dedicated background task for reading from camera
# stop_event = threading.Event()           # Flag to stop background tasks (all tasks)
# proceed_event = threading.Event()        # Flag to signal user "proceed"
# send_video_to_ui_flag = threading.Event() # Flag to enable/disable sending video frames to UI

# # --- NEW: Flag for "soft" process stop ---
# request_process_stop_event = threading.Event()

# # Camera frame buffer and lock for thread-safe access
# camera_frame_buffer = [None] # A list to hold the latest frame (mutable)
# camera_lock = threading.Lock()

# # --- Paths ---
# YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'my_model_n_300.pt')
# SETTINGS_FILE_PATH = os.path.join(WEB_SERVER_DIR, 'settings.json') 

# # --- Default Settings ---
# DEFAULT_SETTINGS = {
#     "camera_index": 0, # NEW: Camera index setting
#     "camera_res_w": 640, # NEW: Camera resolution setting
#     "camera_res_h": 480, # NEW: Camera resolution setting
#     "last_move_mm": 1.5,
#     "last_offset_mm": 50.0,
#     "last_needle_color": "green",
#     "cobot_ip": "192.168.1.166",
#     "calibration_start_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     "alignment_standby_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# }

# # --- Initialize Flask and SocketIO ---
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key!' 
# socketio = SocketIO(app, async_mode='threading')

# # --- Helper function to read/save settings ---
# def get_all_settings():
#     """Reads all settings from the JSON file, using defaults if file not found."""
#     try:
#         with open(SETTINGS_FILE_PATH, 'r') as f:
#             settings = json.load(f)
#             # Ensure all default keys are present
#             for key, value in DEFAULT_SETTINGS.items():
#                 if key not in settings:
#                     settings[key] = value
#             return settings
#     except (FileNotFoundError, json.JSONDecodeError):
#         print(f"Warning: {SETTINGS_FILE_PATH} not found or invalid. Creating with defaults.")
#         save_all_settings(DEFAULT_SETTINGS)
#         return DEFAULT_SETTINGS

# def save_all_settings(new_settings):
#     """Saves the entire settings dictionary to the file."""
#     try:
#         with open(SETTINGS_FILE_PATH, 'w') as f:
#             json.dump(new_settings, f, indent=2)
#         print(f"Saved new settings: {new_settings}")
#         return True
#     except Exception as e:
#         print(f"Error saving settings: {e}")
#         return False

# # ============================================================================
# # Background Camera Management Threads
# # ============================================================================

# def camera_reader(camera_index, res_w, res_h, stop_flag, frame_buffer, frame_lock):
#     """
#     Dedicated thread to continuously read frames from the camera.
#     This keeps the camera open and ensures frames are always available quickly.
#     """
#     global cap
#     print(f"Camera Reader: Starting for index {camera_index} at {res_w}x{res_h}...")
#     try:
#         cap = cv2.VideoCapture(camera_index)
#         if not cap.isOpened():
#             socketio.emit('status_update', {'msg': f'Error: Cannot open camera at index {camera_index}.', 'error': True})
#             print(f"Camera Reader ERROR: Cannot open camera at index {camera_index}.")
#             return

#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        
#         # Read a few frames to let camera auto-adjust
#         for _ in range(5):
#             _, _ = cap.read()
#             time.sleep(0.1)

#         print(f"Camera Reader: Successfully opened camera {camera_index}.")
#         socketio.emit('status_update', {'msg': f'Camera {camera_index} opened successfully.'})

#         while not stop_flag.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 socketio.emit('status_update', {'msg': 'Camera Reader: Failed to read frame. Disconnected?', 'error': True})
#                 print("Camera Reader ERROR: Failed to read frame.")
#                 break # Exit loop on read failure

#             with frame_lock:
#                 frame_buffer[0] = frame.copy() # Store the latest frame

#             time.sleep(0.01) # Read as fast as possible, but yield some CPU
            
#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Camera Reader critical error: {e}', 'error': True})
#         print(f"Camera Reader CRITICAL ERROR: {e}")
#     finally:
#         if cap:
#             cap.release()
#             print(f"Camera Reader: Camera {camera_index} released.")
#             cap = None
#         with frame_lock:
#             frame_buffer[0] = None # Clear buffer
#         print("Camera Reader: Thread stopped.")

# def video_sender(stop_flag, send_video_flag, frame_buffer, frame_lock):
#     """
#     Runs in a background thread to send the latest frame from the buffer to the UI
#     when `send_video_flag` is set.
#     """
#     print("Video Sender: Thread starting...")
#     while not stop_flag.is_set():
#         if send_video_flag.is_set():
#             frame = None
#             with frame_lock:
#                 if frame_buffer[0] is not None:
#                     frame = frame_buffer[0].copy()

#             if frame is not None:
#                 # Draw the alignment box on the stream frames
#                 resW, resH = frame.shape[1], frame.shape[0]
#                 cx, cy = resW // 2, resH // 2
#                 box_size = 150 # Half-width/height of the box
#                 cv2.rectangle(frame, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (0, 255, 255), 2)
#                 cv2.putText(frame, "Align DUT and Needle Here", (cx - 100, cy - box_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#                 cobot_logic.send_frame_to_ui(socketio, frame, 'video_frame')
        
#         socketio.sleep(0.05) # Send at ~20 FPS

#     print("Video Sender: Thread stopped.")

# # ============================================================================
# # Background Process Wrappers (Calibration, Alignment)
# # ============================================================================

# def run_calibration_wrapper(settings_data):
#     """Wrapper to safely run calibration in a background task."""
#     global cobot, stop_event, proceed_event, send_video_to_ui_flag
#     socketio.emit('status_update', {'msg': 'Starting calibration process...'})
    
#     # Temporarily stop sending idle video frames to UI
#     send_video_to_ui_flag.clear()
    
#     try:
#         # 1. Get settings
#         current_settings = get_all_settings()
#         current_settings.update(settings_data) # Update with latest from UI
        
#         # 2. Initialize Hardware (Cobot only, camera is globally managed)
#         cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
#         if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
#             socketio.emit('status_update', {'msg': 'Failed to connect and enable the cobot.', 'error': True})
#             raise Exception("Cobot connection failed")

#         # 3. Run the logic
#         cobot_logic.run_calibration_routine(
#             socketio, proceed_event, stop_event,
#             cobot, camera_frame_buffer, camera_lock, current_settings
#         )
        
#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Calibration error: {e}', 'error': True})
#     finally:
#         # 4. Cleanup
#         if cobot:
#             cobot.disconnect()
#             cobot = None
        
#         # Resume sending idle video frames
#         send_video_to_ui_flag.set()
#         socketio.emit('status_update', {'msg': 'Calibration process finished.'})
#         socketio.emit('process_finished') # Tell UI process is done

# def run_alignment_wrapper(settings_data):
#     """Wrapper to safely run alignment in a background task."""
#     global cobot, stop_event, proceed_event, send_video_to_ui_flag, request_process_stop_event
#     socketio.emit('status_update', {'msg': 'Starting alignment process...'})

#     # Temporarily stop sending idle video frames to UI
#     send_video_to_ui_flag.clear()
    
#     try:
#         # 1. Get settings
#         current_settings = get_all_settings()
#         current_settings.update(settings_data)
        
#         # 2. Initialize Hardware (Cobot only, camera is globally managed)
#         cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
#         if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
#             socketio.emit('status_update', {'msg': 'Failed to connect and enable the cobot.', 'error': True})
#             raise Exception("Cobot connection failed")

#         # 3. Run the logic
#         cobot_logic.run_alignment_process(
#             socketio, proceed_event, stop_event, request_process_stop_event,
#             cobot, camera_frame_buffer, camera_lock, current_settings, YOLO_MODEL_PATH
#         )

#     except Exception as e:
#         socketio.emit('status_update', {'msg': f'Alignment error: {e}', 'error': True})
#     finally:
#         # 4. Cleanup
#         if cobot:
#             cobot.disconnect()
#             cobot = None
        
#         # Resume sending idle video frames
#         send_video_to_ui_flag.set()
#         socketio.emit('status_update', {'msg': 'Alignment process finished.'})
#         socketio.emit('process_finished') # Tell UI process is done

# # ============================================================================
# # HTTP Routes
# # ============================================================================
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/get-settings', methods=['GET'])
# def get_settings_route():
#     return jsonify(get_all_settings())

# @app.route('/save-settings', methods=['POST'])
# def save_settings_route():
#     data = request.get_json()
#     if save_all_settings(data):
#         return jsonify({'status': 'success', 'message': 'Settings saved.'})
#     else:
#         return jsonify({'status': 'error', 'message': 'Failed to save settings.'}), 500

# # ============================================================================
# # SocketIO Event Handlers
# # ============================================================================

# def stop_all_tasks():
#     """Helper function to stop all running worker tasks (calibration/alignment).
#        The camera reader and video sender threads are managed globally."""
#     global current_process_task
    
#     stop_event.set() # Signal all current process tasks to stop
#     proceed_event.set() # Unblock any waiting prompts
#     request_process_stop_event.set() # Also trigger the "soft" stop
    
#     if current_process_task:
#         print("Waiting for current process task to finish...")
#         current_process_task.join(timeout=5) # Wait for it to finish
#         if current_process_task.is_alive():
#             print("Warning: Process task did not terminate gracefully.")
#         current_process_task = None
    
#     stop_event.clear() # Clear for the next task
#     proceed_event.clear() # Clear for the next prompt
#     # DO NOT clear request_process_stop_event, it's cleared when a new process starts
    
#     print("All worker tasks stopped.")

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
#     socketio.emit('status_update', {'msg': 'Server connected. Initializing camera...'})
    
#     # Start the idle video stream (if not already running)
#     handle_request_video_stream()

# @socketio.on('request_video_stream')
# def handle_request_video_stream():
#     """Client requests the idle video stream. This just enables the sender."""
#     global video_sender_thread, camera_reader_thread, stop_event, send_video_to_ui_flag
    
#     current_settings = get_all_settings()
#     cam_idx = current_settings['camera_index']
#     res_w = current_settings['camera_res_w']
#     res_h = current_settings['camera_res_h']

#     # Start camera reader if not already running
#     if camera_reader_thread is None or not camera_reader_thread.is_alive():
#         stop_event.clear() # Ensure stop event is clear for camera threads
#         camera_reader_thread = socketio.start_background_task(camera_reader, cam_idx, res_w, res_h, stop_event, camera_frame_buffer, camera_lock)
#         # Give a moment for the camera to open
#         time.sleep(1.0) 

#     # Start video sender if not already running
#     if video_sender_thread is None or not video_sender_thread.is_alive():
#         stop_event.clear() # Ensure stop event is clear for camera threads
#         video_sender_thread = socketio.start_background_task(video_sender, stop_event, send_video_to_ui_flag, camera_frame_buffer, camera_lock)
        
#     send_video_to_ui_flag.set() # Enable sending frames to UI
#     socketio.emit('status_update', {'msg': 'Idle video stream enabled.'})


# @socketio.on('start_calibration')
# def handle_start_calibration(data):
#     """Client requests to start the calibration process."""
#     global current_process_task, stop_event, proceed_event, request_process_stop_event
#     stop_all_tasks() # Stop any previous worker task
#     stop_event.clear() # Clear stop flag for the new task
#     proceed_event.clear() # Clear proceed flag for the new task
#     request_process_stop_event.clear() # Clear soft stop flag
    
#     # Start calibration in a new background thread
#     current_process_task = socketio.start_background_task(run_calibration_wrapper, data)

# @socketio.on('start_alignment')
# def handle_start_alignment(data):
#     """Client requests to start the alignment process."""
#     global current_process_task, stop_event, proceed_event, request_process_stop_event
#     stop_all_tasks() # Stop any previous worker task
#     stop_event.clear()
#     proceed_event.clear()
#     request_process_stop_event.clear() # Clear soft stop flag
    
#     # Start alignment in a new background thread
#     current_process_task = socketio.start_background_task(run_alignment_wrapper, data)

# @socketio.on('user_proceed')
# def handle_user_proceed():
#     """Client clicked the 'Proceed' button."""
#     print("User 'Proceed' signal received.")
#     proceed_event.set()

# @socketio.on('stop_process')
# def handle_stop_process():
#     """Client clicked the 'Stop Process' (soft stop) button."""
#     print("SOFT STOP received!")
#     socketio.emit('status_update', {'msg': 'Stop requested. Process will halt after current cycle.'})
#     request_process_stop_event.set() # Set the soft stop flag

# @socketio.on('emergency_stop')
# def handle_emergency_stop():
#     """Client clicked the 'EMERGENCY STOP' button."""
#     print("EMERGENCY STOP received!")
#     socketio.emit('status_update', {'msg': 'EMERGENCY STOP requested!', 'error': True})
    
#     # Stop current worker task if any
#     stop_all_tasks()
    
#     # Force disconnect cobot if it was connected
#     if cobot:
#         print("Force disconnecting cobot...")
#         cobot.disconnect()
        
#     # Ensure idle video stream is re-enabled
#     send_video_to_ui_flag.set() 
#     socketio.emit('status_update', {'msg': 'System stopped. Idle video stream active.'})

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected. Stopping worker tasks.')
#     # Only stop worker tasks, camera reader and sender should remain active
#     # until server shutdown to allow quick reconnects.
#     stop_all_tasks()

# # --- Main ---
# if __name__ == '__main__':
#     print("Starting server on http://0.0.0.0:5000")
#     # This ensures that when the Flask app shuts down, the camera reader and sender threads are also stopped.
#     try:
#         # Start camera reader and sender threads immediately on server start
#         # Use a dummy settings to get camera_index, will be updated by client
#         initial_settings = get_all_settings()
#         initial_cam_idx = initial_settings.get('camera_index', DEFAULT_SETTINGS['camera_index'])
#         initial_res_w = initial_settings.get('camera_res_w', DEFAULT_SETTINGS['camera_res_w'])
#         initial_res_h = initial_settings.get('camera_res_h', DEFAULT_SETTINGS['camera_res_h'])
        
#         stop_event.clear() # Ensure stop event is clear for initial camera threads
#         camera_reader_thread = socketio.start_background_task(camera_reader, initial_cam_idx, initial_res_w, initial_res_h, stop_event, camera_frame_buffer, camera_lock)
#         time.sleep(1.5) # Give camera some time to initialize before starting sender
#         send_video_to_ui_flag.set() # Initially enable sending frames
#         video_sender_thread = socketio.start_background_task(video_sender, stop_event, send_video_to_ui_flag, camera_frame_buffer, camera_lock)

#         socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
#     finally:
#         print("Server shutting down. Stopping all camera threads...")
#         stop_event.set() # Signal all threads to stop
#         if camera_reader_thread and camera_reader_thread.is_alive():
#             camera_reader_thread.join(timeout=5)
#         if video_sender_thread and video_sender_thread.is_alive():
#             video_sender_thread.join(timeout=5)
#         print("All threads terminated.")




#------with demo mode
import os
import json
import base64
import threading
import cv2
import sys
import time # Import time for delays
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

# --- Add Project Root to Path ---
WEB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEB_SERVER_DIR)
sys.path.insert(0, PROJECT_ROOT)
# --- End Path Addition ---

# --- Import all logic from your refactored file ---
import cobot_logic

# --- Global State Variables ---
cap = None                      # OpenCV Camera object
cobot = None                    # JAKACobot object
current_process_task = None     # Background task for calibration/alignment
video_sender_thread = None      # Background task for sending video frames to UI
camera_reader_thread = None     # Dedicated background task for reading from camera
stop_event = threading.Event()      # Flag to stop background tasks (all tasks)
proceed_event = threading.Event()   # Flag to signal user "proceed"
send_video_to_ui_flag = threading.Event() # Flag to enable/disable sending video frames to UI

# --- NEW: Flag for "soft" process stop ---
request_process_stop_event = threading.Event()

# Camera frame buffer and lock for thread-safe access
camera_frame_buffer = [None] # A list to hold the latest frame (mutable)
camera_lock = threading.Lock()

# --- Paths ---
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'my_model_n_300.pt')
SETTINGS_FILE_PATH = os.path.join(WEB_SERVER_DIR, 'settings.json') 

# --- Default Settings ---
DEFAULT_SETTINGS = {
    "camera_index": 0, # NEW: Camera index setting
    "camera_res_w": 640, # NEW: Camera resolution setting
    "camera_res_h": 480, # NEW: Camera resolution setting
    "last_move_mm": 1.5,
    "last_offset_mm": 50.0,
    "last_needle_color": "green",
    "cobot_ip": "192.168.1.166",
    "calibration_start_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "alignment_standby_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# --- Initialize Flask and SocketIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!' 
socketio = SocketIO(app, async_mode='threading')

# --- Helper function to read/save settings ---
def get_all_settings():
    """Reads all settings from the JSON file, using defaults if file not found."""
    try:
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
            # Ensure all default keys are present
            for key, value in DEFAULT_SETTINGS.items():
                if key not in settings:
                    settings[key] = value
            return settings
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: {SETTINGS_FILE_PATH} not found or invalid. Creating with defaults.")
        save_all_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS

def save_all_settings(new_settings):
    """Saves the entire settings dictionary to the file."""
    try:
        with open(SETTINGS_FILE_PATH, 'w') as f:
            json.dump(new_settings, f, indent=2)
        print(f"Saved new settings: {new_settings}")
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# ============================================================================
# Background Camera Management Threads
# ============================================================================

def camera_reader(camera_index, res_w, res_h, stop_flag, frame_buffer, frame_lock):
    """
    Dedicated thread to continuously read frames from the camera.
    This keeps the camera open and ensures frames are always available quickly.
    """
    global cap
    print(f"Camera Reader: Starting for index {camera_index} at {res_w}x{res_h}...")
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            socketio.emit('status_update', {'msg': f'Error: Cannot open camera at index {camera_index}.', 'error': True})
            print(f"Camera Reader ERROR: Cannot open camera at index {camera_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        
        # Read a few frames to let camera auto-adjust
        for _ in range(5):
            _, _ = cap.read()
            time.sleep(0.1)

        print(f"Camera Reader: Successfully opened camera {camera_index}.")
        socketio.emit('status_update', {'msg': f'Camera {camera_index} opened successfully.'})

        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                socketio.emit('status_update', {'msg': 'Camera Reader: Failed to read frame. Disconnected?', 'error': True})
                print("Camera Reader ERROR: Failed to read frame.")
                break # Exit loop on read failure

            with frame_lock:
                frame_buffer[0] = frame.copy() # Store the latest frame

            time.sleep(0.01) # Read as fast as possible, but yield some CPU
            
    except Exception as e:
        socketio.emit('status_update', {'msg': f'Camera Reader critical error: {e}', 'error': True})
        print(f"Camera Reader CRITICAL ERROR: {e}")
    finally:
        if cap:
            cap.release()
            print(f"Camera Reader: Camera {camera_index} released.")
            cap = None
        with frame_lock:
            frame_buffer[0] = None # Clear buffer
        print("Camera Reader: Thread stopped.")

def video_sender(stop_flag, send_video_flag, frame_buffer, frame_lock):
    """
    Runs in a background thread to send the latest frame from the buffer to the UI
    when `send_video_flag` is set.
    """
    print("Video Sender: Thread starting...")
    while not stop_flag.is_set():
        if send_video_flag.is_set():
            frame = None
            with frame_lock:
                if frame_buffer[0] is not None:
                    frame = frame_buffer[0].copy()

            if frame is not None:
                # Draw the alignment box on the stream frames
                resW, resH = frame.shape[1], frame.shape[0]
                cx, cy = resW // 2, resH // 2
                box_size = 150 # Half-width/height of the box
                cv2.rectangle(frame, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (0, 255, 255), 2)
                cv2.putText(frame, "Align DUT and Needle Here", (cx - 100, cy - box_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cobot_logic.send_frame_to_ui(socketio, frame, 'video_frame')
        
        socketio.sleep(0.05) # Send at ~20 FPS

    print("Video Sender: Thread stopped.")

# ============================================================================
# Background Process Wrappers (Calibration, Alignment)
# ============================================================================

def run_calibration_wrapper(settings_data):
    """Wrapper to safely run calibration in a background task."""
    global cobot, stop_event, proceed_event, send_video_to_ui_flag
    socketio.emit('status_update', {'msg': 'Starting calibration process...'})
    
    # Temporarily stop sending idle video frames to UI
    send_video_to_ui_flag.clear()
    
    try:
        # 1. Get settings
        current_settings = get_all_settings()
        current_settings.update(settings_data) # Update with latest from UI
        
        # 2. Initialize Hardware (Cobot only, camera is globally managed)
        # This is handled by handle_start_calibration
        if not cobot or not cobot.connected:
             socketio.emit('status_update', {'msg': 'Cobot not connected. Aborting calibration.', 'error': True})
             raise Exception("Cobot not connected for calibration wrapper")

        # 3. Run the logic
        cobot_logic.run_calibration_routine(
            socketio, proceed_event, stop_event,
            cobot, camera_frame_buffer, camera_lock, current_settings
        )
        
    except Exception as e:
        socketio.emit('status_update', {'msg': f'Calibration error: {e}', 'error': True})
    finally:
        # 4. Cleanup
        if cobot:
            cobot.disconnect()
            cobot = None
        
        # Resume sending idle video frames
        send_video_to_ui_flag.set()
        socketio.emit('status_update', {'msg': 'Calibration process finished.'})
        socketio.emit('process_finished') # Tell UI process is done

def run_alignment_wrapper(settings_data, cobot_instance=None):
    """
    Wrapper to safely run alignment in a background task.
    Accepts a pre-connected cobot instance.
    """
    global cobot, stop_event, proceed_event, send_video_to_ui_flag, request_process_stop_event
    socketio.emit('status_update', {'msg': 'Starting alignment process...'})

    # Temporarily stop sending idle video frames to UI
    send_video_to_ui_flag.clear()
    
    try:
        # 1. Get settings
        current_settings = get_all_settings()
        current_settings.update(settings_data)
        
        # 2. Initialize Hardware
        if cobot_instance:
            cobot = cobot_instance # Use the pre-connected instance
        else:
            # Fallback if not pre-connected (shouldnt happen with new logic)
            socketio.emit('status_update', {'msg': 'Connecting to cobot...'})
            cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
            if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
                socketio.emit('status_update', {'msg': 'Failed to connect and enable the cobot.', 'error': True})
                raise Exception("Cobot connection failed")

        # 3. Run the logic
        cobot_logic.run_alignment_process(
            socketio, proceed_event, stop_event, request_process_stop_event,
            cobot, camera_frame_buffer, camera_lock, current_settings, YOLO_MODEL_PATH
        )

    except Exception as e:
        socketio.emit('status_update', {'msg': f'Alignment error: {e}', 'error': True})
    finally:
        # 4. Cleanup
        if cobot:
            cobot.disconnect()
            cobot = None
        
        # Resume sending idle video frames
        send_video_to_ui_flag.set()
        socketio.emit('status_update', {'msg': 'Alignment process finished.'})
        socketio.emit('process_finished') # Tell UI process is done

# ============================================================================
# --- NEW: DEMO MODE WRAPPER ---
# ============================================================================
def run_demo_alignment_wrapper(settings_data):
    """
    Wrapper to safely run alignment in DEMO MODE (no cobot).
    """
    global stop_event, proceed_event, send_video_to_ui_flag, request_process_stop_event
    socketio.emit('status_update', {'msg': 'Starting DEMO alignment process...'})

    # Temporarily stop sending idle video frames to UI
    send_video_to_ui_flag.clear()
    
    try:
        # 1. Get settings
        current_settings = get_all_settings()
        current_settings.update(settings_data)
        
        # 2. NO COBOT INITIALIZATION

        # 3. Run the DEMO logic
        cobot_logic.run_demo_alignment_process(
            socketio, proceed_event, stop_event, request_process_stop_event,
            camera_frame_buffer, camera_lock, current_settings, YOLO_MODEL_PATH
        )

    except Exception as e:
        socketio.emit('status_update', {'msg': f'Demo Alignment error: {e}', 'error': True})
    finally:
        # 4. NO COBOT CLEANUP
        
        # Resume sending idle video frames
        send_video_to_ui_flag.set()
        socketio.emit('status_update', {'msg': 'Demo Alignment process finished.'})
        socketio.emit('process_finished') # Tell UI process is done
# ============================================================================
# --- END NEW WRAPPER ---
# ============================================================================


# ============================================================================
# HTTP Routes
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-settings', methods=['GET'])
def get_settings_route():
    return jsonify(get_all_settings())

@app.route('/save-settings', methods=['POST'])
def save_settings_route():
    data = request.get_json()
    if save_all_settings(data):
        return jsonify({'status': 'success', 'message': 'Settings saved.'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to save settings.'}), 500

# ============================================================================
# SocketIO Event Handlers
# ============================================================================

def stop_all_tasks():
    """Helper function to stop all running worker tasks (calibration/alignment).
       The camera reader and video sender threads are managed globally."""
    global current_process_task
    
    stop_event.set() # Signal all current process tasks to stop
    proceed_event.set() # Unblock any waiting prompts
    request_process_stop_event.set() # Also trigger the "soft" stop
    
    if current_process_task:
        print("Waiting for current process task to finish...")
        current_process_task.join(timeout=5) # Wait for it to finish
        if current_process_task.is_alive():
            print("Warning: Process task did not terminate gracefully.")
        current_process_task = None
    
    stop_event.clear() # Clear for the next task
    proceed_event.clear() # Clear for the next prompt
    # DO NOT clear request_process_stop_event, it's cleared when a new process starts
    
    print("All worker tasks stopped.")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('status_update', {'msg': 'Server connected. Initializing camera...'})
    
    # Start the idle video stream (if not already running)
    handle_request_video_stream()

@socketio.on('request_video_stream')
def handle_request_video_stream():
    """Client requests the idle video stream. This just enables the sender."""
    global video_sender_thread, camera_reader_thread, stop_event, send_video_to_ui_flag
    
    current_settings = get_all_settings()
    cam_idx = current_settings['camera_index']
    res_w = current_settings['camera_res_w']
    res_h = current_settings['camera_res_h']

    # Start camera reader if not already running
    if camera_reader_thread is None or not camera_reader_thread.is_alive():
        stop_event.clear() # Ensure stop event is clear for camera threads
        camera_reader_thread = socketio.start_background_task(camera_reader, cam_idx, res_w, res_h, stop_event, camera_frame_buffer, camera_lock)
        # Give a moment for the camera to open
        time.sleep(1.0) 

    # Start video sender if not already running
    if video_sender_thread is None or not video_sender_thread.is_alive():
        stop_event.clear() # Ensure stop event is clear for camera threads
        video_sender_thread = socketio.start_background_task(video_sender, stop_event, send_video_to_ui_flag, camera_frame_buffer, camera_lock)
        
    send_video_to_ui_flag.set() # Enable sending frames to UI
    socketio.emit('status_update', {'msg': 'Idle video stream enabled.'})


@socketio.on('start_calibration')
def handle_start_calibration(data):
    """Client requests to start the calibration process."""
    global current_process_task, stop_event, proceed_event, request_process_stop_event, cobot
    stop_all_tasks() # Stop any previous worker task
    stop_event.clear() # Clear stop flag for the new task
    proceed_event.clear() # Clear proceed flag for the new task
    request_process_stop_event.clear() # Clear soft stop flag
    
    # --- MODIFICATION: Check connection first ---
    current_settings = get_all_settings()
    current_settings.update(data)
    
    cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
    if not (cobot.connect() and cobot.power_on() and cobot.enable_robot()):
        socketio.emit('status_update', {'msg': 'Cobot connection failed. Calibration unavailable in Demo Mode.', 'error': True})
        socketio.emit('process_finished') # Tell UI to re-enable buttons
        cobot = None
        return
    # --- END MODIFICATION ---

    socketio.emit('status_update', {'msg': 'Cobot connected. Starting calibration.'})
    # Start calibration in a new background thread (cobot is now connected and stored in global)
    current_process_task = socketio.start_background_task(run_calibration_wrapper, data)

@socketio.on('start_alignment')
def handle_start_alignment(data):
    """Client requests to start the alignment process."""
    global current_process_task, stop_event, proceed_event, request_process_stop_event
    stop_all_tasks() # Stop any previous worker task
    stop_event.clear()
    proceed_event.clear()
    request_process_stop_event.clear() # Clear soft stop flag
    
    # --- MODIFICATION: Check connection to decide mode ---
    current_settings = get_all_settings()
    current_settings.update(data)
    
    temp_cobot = cobot_logic.JAKACobot(ip=current_settings['cobot_ip'])
    
    if not (temp_cobot.connect() and temp_cobot.power_on() and temp_cobot.enable_robot()):
        # --- DEMO MODE ---
        socketio.emit('status_update', {'msg': 'Cobot connection failed. ENTERING DEMO MODE.', 'error': True})
        current_process_task = socketio.start_background_task(run_demo_alignment_wrapper, data)
    else:
        # --- REAL MODE ---
        socketio.emit('status_update', {'msg': 'Cobot connected successfully. Starting REAL alignment.'})
        # Start alignment and pass the already-connected instance
        current_process_task = socketio.start_background_task(run_alignment_wrapper, data, cobot_instance=temp_cobot)
    # --- END MODIFICATION ---


@socketio.on('user_proceed')
def handle_user_proceed():
    """Client clicked the 'Proceed' button."""
    print("User 'Proceed' signal received.")
    proceed_event.set()

@socketio.on('stop_process')
def handle_stop_process():
    """Client clicked the 'Stop Process' (soft stop) button."""
    print("SOFT STOP received!")
    socketio.emit('status_update', {'msg': 'Stop requested. Process will halt after current cycle.'})
    request_process_stop_event.set() # Set the soft stop flag

@socketio.on('emergency_stop')
def handle_emergency_stop():
    """Client clicked the 'EMERGENCY STOP' button."""
    print("EMERGENCY STOP received!")
    socketio.emit('status_update', {'msg': 'EMERGENCY STOP requested!', 'error': True})
    
    # Stop current worker task if any
    stop_all_tasks()
    
    # Force disconnect cobot if it was connected
    if cobot:
        print("Force disconnecting cobot...")
        cobot.disconnect()
        
    # Ensure idle video stream is re-enabled
    send_video_to_ui_flag.set() 
    socketio.emit('status_update', {'msg': 'System stopped. Idle video stream active.'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected. Stopping worker tasks.')
    # Only stop worker tasks, camera reader and sender should remain active
    # until server shutdown to allow quick reconnects.
    stop_all_tasks()

# --- Main ---
if __name__ == '__main__':
    print("Starting server on http://0.0.0.0:5000")
    # This ensures that when the Flask app shuts down, the camera reader and sender threads are also stopped.
    try:
        # Start camera reader and sender threads immediately on server start
        # Use a dummy settings to get camera_index, will be updated by client
        initial_settings = get_all_settings()
        initial_cam_idx = initial_settings.get('camera_index', DEFAULT_SETTINGS['camera_index'])
        initial_res_w = initial_settings.get('camera_res_w', DEFAULT_SETTINGS['camera_res_w'])
        initial_res_h = initial_settings.get('camera_res_h', DEFAULT_SETTINGS['camera_res_h'])
        
        stop_event.clear() # Ensure stop event is clear for initial camera threads
        camera_reader_thread = socketio.start_background_task(camera_reader, initial_cam_idx, initial_res_w, initial_res_h, stop_event, camera_frame_buffer, camera_lock)
        time.sleep(1.5) # Give camera some time to initialize before starting sender
        send_video_to_ui_flag.set() # Initially enable sending frames
        video_sender_thread = socketio.start_background_task(video_sender, stop_event, send_video_to_ui_flag, camera_frame_buffer, camera_lock)

        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    finally:
        print("Server shutting down. Stopping all camera threads...")
        stop_event.set() # Signal all threads to stop
        if camera_reader_thread and camera_reader_thread.is_alive():
            camera_reader_thread.join(timeout=5)
        if video_sender_thread and video_sender_thread.is_alive():
            video_sender_thread.join(timeout=5)
        print("All threads terminated.")