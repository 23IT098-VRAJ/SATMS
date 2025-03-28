import cv2
import numpy as np
import time
import os
import json
from threading import Thread, Lock
from datetime import datetime

class CameraStream:
    """Class to handle camera input streams for traffic monitoring"""
    
    def __init__(self, source=0, name="CameraStream", resolution=None, buffer_size=10):
        """Initialize camera stream
        
        Args:
            source: Camera index, video file path, or rtsp stream URL
            name: Name for the camera stream
            resolution: Optional tuple (width, height) to resize frames
            buffer_size: Number of frames to buffer for smoother processing
        """
        self.source = source
        self.stream = cv2.VideoCapture(source)
        self.name = name
        self.stopped = False
        self.frame = None
        self.resolution = resolution
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.frame_lock = Lock()  # Thread safety
        
        # Frame buffer for smoother processing
        self.buffer_size = buffer_size
        self.frame_buffer = []
        
        # Status tracking
        self.status = {
            'is_connected': False,
            'last_error': None,
            'reconnect_attempts': 0,
            'max_reconnect_attempts': 5,
            'last_frame_time': None
        }
        
        # Configuration
        self.config = {
            'source': source,
            'name': name,
            'resolution': resolution,
            'reconnect_delay': 5,  # seconds
            'frame_skip': 0  # Process every Nth frame (0 = process all)
        }
        
        # Try to load camera-specific configuration
        self._load_config()
        
        # Initialize camera
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera with proper settings"""
        # Set resolution if specified
        if self.resolution:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Check if camera opened successfully
        if self.stream.isOpened():
            self.status['is_connected'] = True
            # Read first frame
            (grabbed, frame) = self.stream.read()
            if grabbed:
                with self.frame_lock:
                    self.frame = frame
                self.status['last_frame_time'] = time.time()
        else:
            self.status['is_connected'] = False
            self.status['last_error'] = "Failed to open camera source"
    
    def _load_config(self):
        """Load camera-specific configuration if available"""
        config_dir = "config/cameras"
        config_file = f"{config_dir}/{self.name}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Update configuration
                    self.config.update(loaded_config)
                    
                    # Apply configuration settings
                    if 'resolution' in loaded_config and loaded_config['resolution']:
                        self.resolution = tuple(loaded_config['resolution'])
                    
                    print(f"Loaded configuration for camera {self.name}")
            except Exception as e:
                print(f"Error loading camera configuration: {e}")
    
    def save_config(self):
        """Save current camera configuration"""
        config_dir = "config/cameras"
        config_file = f"{config_dir}/{self.name}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Saved configuration for camera {self.name}")
        except Exception as e:
            print(f"Error saving camera configuration: {e}")
    
    def start(self):
        """Start the thread to read frames from the video stream"""
        Thread(target=self.update, args=(), daemon=True).start()
        return self
    
    def update(self):
        """Update frames from stream continuously"""
        while not self.stopped:
            # Check if camera is connected
            if not self.stream.isOpened():
                self.status['is_connected'] = False
                self.status['last_error'] = "Camera disconnected"
                
                # Attempt to reconnect
                if self.status['reconnect_attempts'] < self.status['max_reconnect_attempts']:
                    self.status['reconnect_attempts'] += 1
                    print(f"Attempting to reconnect to {self.name} (Attempt {self.status['reconnect_attempts']})")
                    
                    # Release current stream
                    self.stream.release()
                    
                    # Wait before reconnecting
                    time.sleep(self.config['reconnect_delay'])
                    
                    # Try to reconnect
                    self.stream = cv2.VideoCapture(self.source)
                    if self.stream.isOpened():
                        self.status['is_connected'] = True
                        self.status['reconnect_attempts'] = 0
                        print(f"Successfully reconnected to {self.name}")
                    else:
                        print(f"Failed to reconnect to {self.name}")
                else:
                    print(f"Maximum reconnection attempts reached for {self.name}. Stopping stream.")
                    self.stop()
                    return
            
            # Read frame
            (grabbed, frame) = self.stream.read()
            
            if not grabbed:
                # Frame could not be grabbed, but connection might still be active
                continue
            
            # Update FPS calculation
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.fps = 1.0 / time_diff if time_diff > 0 else 0
            self.frame_count += 1
            
            # Apply frame skip if configured
            if self.config['frame_skip'] > 0 and (self.frame_count % (self.config['frame_skip'] + 1)) != 0:
                continue
            
            # Resize frame if resolution is specified
            if self.resolution:
                frame = cv2.resize(frame, self.resolution)
            
            # Update frame with thread safety
            with self.frame_lock:
                self.frame = frame
                
                # Update frame buffer
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
            
            # Update status
            self.status['last_frame_time'] = current_time
            self.status['is_connected'] = True
    
    def read(self):
        """Return the most recent frame"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def read_buffer(self):
        """Return the entire frame buffer"""
        with self.frame_lock:
            return self.frame_buffer.copy()
    
    def stop(self):
        """Stop the camera stream"""
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
    
    def get_frame_dimensions(self):
        """Get dimensions of the video frames"""
        if self.resolution:
            return self.resolution
        else:
            width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
    
    def get_status(self):
        """Get current status of the camera stream"""
        status = self.status.copy()
        status['fps'] = self.fps
        status['frame_count'] = self.frame_count
        status['buffer_size'] = len(self.frame_buffer)
        status['dimensions'] = self.get_frame_dimensions()
        return status
    
    def set_config(self, config):
        """Update camera configuration
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config.update(config)
        
        # Apply configuration changes
        if 'resolution' in config and config['resolution']:
            self.resolution = tuple(config['resolution'])
            # Only apply to stream if it's open
            if self.stream.isOpened():
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Save updated configuration
        self.save_config()
    
    def save_snapshot(self, output_dir="static/snapshots"):
        """Save a snapshot from the camera
        
        Args:
            output_dir: Directory to save snapshots
            
        Returns:
            Path to saved snapshot or None if failed
        """
        frame = self.read()
        if frame is None:
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{self.name}_{timestamp}.jpg"
        
        # Save image
        try:
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None
