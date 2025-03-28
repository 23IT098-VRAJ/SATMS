import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
from datetime import datetime

class VehicleDetector:
    """Class for detecting vehicles in traffic camera feeds"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5, config_path=None):
        """Initialize the vehicle detector
        
        Args:
            model_path: Path to the pre-trained detection model
            confidence_threshold: Minimum confidence for detection
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set parameters from config or arguments
        self.confidence_threshold = self.config.get('confidence_threshold', confidence_threshold)
        model_path = model_path or self.config.get('model_path')
        
        # Initialize model
        self.model = None
        if model_path:
            self.model = self._load_model(model_path)
        
        # Load class definitions
        self.classes = self._load_classes()
        
        # Performance tracking
        self.performance = {
            'inference_times': [],
            'max_inference_times': 100,  # Keep last 100 inference times
            'detection_counts': {},
            'frame_count': 0,
            'last_detection_time': None
        }
        
        # Object tracking for consistent IDs
        self.enable_tracking = self.config.get('enable_tracking', True)
        self.trackers = {}
        self.next_track_id = 1
        self.track_history = {}
        self.max_track_age = self.config.get('max_track_age', 30)  # frames
        
        # Initialize tracker if enabled
        if self.enable_tracking:
            self._setup_trackers()
    
    def _load_config(self, config_path=None):
        """Load detector configuration"""
        default_config = {
            'model_path': 'models/saved_models/vehicle_detection_model',
            'confidence_threshold': 0.5,
            'enable_tracking': True,
            'max_track_age': 30,
            'nms_threshold': 0.45,
            'tracker_type': 'KCF',
            'class_colors': {
                'car': (0, 255, 0),
                'truck': (0, 0, 255),
                'bus': (255, 0, 0),
                'motorcycle': (255, 255, 0),
                'bicycle': (0, 255, 255)
            },
            'detection_frequency': 5  # Process detection every N frames
        }
        
        # If no config path provided, use default
        if config_path is None:
            config_path = 'config/vehicle_detection_config.json'
        
        # Try to load configuration
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with default config
                    config = default_config.copy()
                    config.update(loaded_config)
                    return config
            except Exception as e:
                print(f"Error loading vehicle detection configuration: {e}")
                return default_config
        else:
            # Create default configuration file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                print(f"Created default vehicle detection configuration at {config_path}")
            except Exception as e:
                print(f"Error creating default configuration: {e}")
            
            return default_config
    
    def _setup_trackers(self):
        """Initialize object trackers"""
        self.tracker_type = self.config.get('tracker_type', 'KCF')
        
        # Dictionary of available trackers
        tracker_types = {
            'BOOSTING': cv2.legacy.TrackerBoosting_create,
            'MIL': cv2.legacy.TrackerMIL_create,
            'KCF': cv2.legacy.TrackerKCF_create,
            'TLD': cv2.legacy.TrackerTLD_create,
            'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create,
            'CSRT': cv2.legacy.TrackerCSRT_create
        }
        
        # Set tracker creator function
        if self.tracker_type in tracker_types:
            self.create_tracker = tracker_types[self.tracker_type]
        else:
            print(f"Warning: Tracker type {self.tracker_type} not available. Using KCF.")
            self.create_tracker = cv2.legacy.TrackerKCF_create
    
    def _load_model(self, model_path):
        """Load the detection model from disk"""
        # Using TF SavedModel format
        try:
            print(f"Loading model from {model_path}...")
            model = tf.saved_model.load(model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _load_classes(self):
        """Load class names for detection"""
        # Check if classes are defined in config
        class_dict = self.config.get('classes', None)
        if class_dict:
            return class_dict
        
        # Default classes
        return {
            1: 'car',
            2: 'truck',
            3: 'bus',
            4: 'motorcycle',
            5: 'bicycle'
        }
    
    def detect(self, frame):
        """Detect vehicles in a frame
        
        Args:
            frame: Image frame from camera
            
        Returns:
            List of detections with class, confidence, bounding box and ID
        """
        if frame is None:
            return []
        
        # Update frame count
        self.performance['frame_count'] += 1
        
        # Skip detection based on frequency setting
        detection_frequency = self.config.get('detection_frequency', 1)
        use_detection = (self.performance['frame_count'] % detection_frequency == 0)
        
        # If tracking is enabled and not using detection this frame, update trackers
        if self.enable_tracking and not use_detection and self.trackers:
            return self._update_trackers(frame)
        
        # If model is not loaded, return empty list
        if self.model is None:
            return []
        
        # Preprocess image
        input_tensor = self._preprocess_image(frame)
        
        # Run inference with timing
        start_time = time.time()
        detections = self._run_inference(input_tensor)
        inference_time = time.time() - start_time
        
        # Track performance
        self.performance['inference_times'].append(inference_time)
        if len(self.performance['inference_times']) > self.performance['max_inference_times']:
            self.performance['inference_times'].pop(0)
        
        self.performance['last_detection_time'] = datetime.now()
        
        # Process detections
        processed_detections = self._process_detections(detections, frame.shape)
        
        # Update detection counts
        for detection in processed_detections:
            class_name = detection['class']
            if class_name in self.performance['detection_counts']:
                self.performance['detection_counts'][class_name] += 1
            else:
                self.performance['detection_counts'][class_name] = 1
        
        # Update trackers if tracking is enabled
        if self.enable_tracking:
            self._initialize_trackers(frame, processed_detections)
        
        return processed_detections
    
    def _preprocess_image(self, image):
        """Preprocess image for the model"""
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor
    def _run_inference(self, input_tensor):
        """Run model inference on the input tensor"""
        output = self.model(input_tensor)
        return output
    
    def _process_detections(self, detections, image_shape):
        """Process raw detections into a structured format"""
        processed_results = []
        
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        
        # Apply non-max suppression if configured
        nms_threshold = self.config.get('nms_threshold', 0.45)
        if nms_threshold > 0:
            # Convert to format expected by NMS
            detection_boxes = []
            detection_scores = []
            detection_classes = []
            
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    detection_boxes.append(boxes[i])
                    detection_scores.append(scores[i])
                    detection_classes.append(classes[i])
            
            if detection_boxes:
                # Convert to numpy arrays
                detection_boxes = np.array(detection_boxes)
                detection_scores = np.array(detection_scores)
                detection_classes = np.array(detection_classes)
                
                # Apply NMS
                selected_indices = tf.image.non_max_suppression(
                    detection_boxes, 
                    detection_scores, 
                    max_output_size=100, 
                    iou_threshold=nms_threshold
                ).numpy()
                
                # Filter based on selected indices
                boxes = detection_boxes[selected_indices]
                scores = detection_scores[selected_indices]
                classes = detection_classes[selected_indices]
        
        h, w, _ = image_shape
        
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                # Get class name
                class_id = classes[i]
                class_name = self.classes.get(class_id, 'unknown')
                
                # Only include vehicle classes
                if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    # Convert box coordinates to pixels
                    ymin, xmin, ymax, xmax = boxes[i]
                    box = {
                        'xmin': int(xmin * w),
                        'ymin': int(ymin * h),
                        'xmax': int(xmax * w),
                        'ymax': int(ymax * h)
                    }
                    
                    # Calculate box area and dimensions
                    width = box['xmax'] - box['xmin']
                    height = box['ymax'] - box['ymin']
                    area = width * height
                    
                    processed_results.append({
                        'class': class_name,
                        'confidence': float(scores[i]),
                        'box': box,
                        'area': area,
                        'width': width,
                        'height': height,
                        'center': (
                            (box['xmin'] + box['xmax']) // 2,
                            (box['ymin'] + box['ymax']) // 2
                        )
                    })
        
        return processed_results
    
    def _initialize_trackers(self, frame, detections):
        """Initialize object trackers for detected vehicles"""
        # Clear existing trackers
        self.trackers = {}
        
        # Initialize new trackers for each detection
        for detection in detections:
            box = detection['box']
            tracker = self.create_tracker()
            
            # OpenCV tracker uses (x, y, width, height) format
            tracker_box = (
                box['xmin'],
                box['ymin'],
                box['xmax'] - box['xmin'],
                box['ymax'] - box['ymin']
            )
            
            # Initialize tracker
            success = tracker.init(frame, tracker_box)
            
            if success:
                # Assign a unique ID to this detection
                track_id = self.next_track_id
                self.next_track_id += 1
                
                # Store tracker and detection info
                self.trackers[track_id] = {
                    'tracker': tracker,
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'box': box,
                    'last_seen': self.performance['frame_count'],
                    'age': 0
                }
                
                # Add ID to detection
                detection['id'] = track_id
                
                # Initialize track history
                self.track_history[track_id] = {
                    'positions': [(detection['center'][0], detection['center'][1])],
                    'class': detection['class'],
                    'first_seen': self.performance['frame_count'],
                    'last_seen': self.performance['frame_count']
                }
    
    def _update_trackers(self, frame):
        """Update object trackers and return tracked objects"""
        tracked_objects = []
        trackers_to_remove = []
        
        # Update each tracker
        for track_id, tracker_info in self.trackers.items():
            tracker = tracker_info['tracker']
            
            # Update tracker
            success, box = tracker.update(frame)
            
            if success:
                # Update tracker info
                x, y, w, h = [int(v) for v in box]
                updated_box = {
                    'xmin': x,
                    'ymin': y,
                    'xmax': x + w,
                    'ymax': y + h
                }
                
                # Calculate center
                center_x = (updated_box['xmin'] + updated_box['xmax']) // 2
                center_y = (updated_box['ymin'] + updated_box['ymax']) // 2
                
                # Update tracker info
                tracker_info['box'] = updated_box
                tracker_info['last_seen'] = self.performance['frame_count']
                tracker_info['age'] += 1
                
                # Add to track history
                if track_id in self.track_history:
                    self.track_history[track_id]['positions'].append((center_x, center_y))
                    self.track_history[track_id]['last_seen'] = self.performance['frame_count']
                
                # Create detection-like object
                tracked_objects.append({
                    'id': track_id,
                    'class': tracker_info['class'],
                    'confidence': tracker_info['confidence'],
                    'box': updated_box,
                    'center': (center_x, center_y),
                    'width': w,
                    'height': h,
                    'area': w * h,
                    'tracked': True
                })
            else:
                # Mark for removal if tracker failed
                tracker_info['age'] += 1
                if tracker_info['age'] > self.max_track_age:
                    trackers_to_remove.append(track_id)
        
        # Remove old trackers
        for track_id in trackers_to_remove:
            if track_id in self.trackers:
                del self.trackers[track_id]
        
        return tracked_objects
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detected vehicles
        
        Args:
            frame: Image frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with annotations
        """
        annotated_frame = frame.copy()
        
        # Get colors from config
        class_colors = self.config.get('class_colors', {
            'car': (0, 255, 0),
            'truck': (0, 0, 255),
            'bus': (255, 0, 0),
            'motorcycle': (255, 255, 0),
            'bicycle': (0, 255, 255)
        })
        
        # Default color if class not found
        default_color = (255, 255, 255)
        
        # Draw each detection
        for detection in detections:
            box = detection['box']
            class_name = detection['class']
            confidence = detection['confidence']
            track_id = detection.get('id', None)
            
            # Get color for this class
            color = class_colors.get(class_name, default_color)
            
            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (box['xmin'], box['ymin']),
                (box['xmax'], box['ymax']),
                color,
                2
            )
            
            # Create label with class, ID and confidence
            label_parts = []
            label_parts.append(class_name)
            
            if track_id is not None:
                label_parts.append(f"ID:{track_id}")
            
            label_parts.append(f"{confidence:.2f}")
            label = " | ".join(label_parts)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated_frame,
                (box['xmin'], box['ymin'] - label_size[1] - 5),
                (box['xmin'] + label_size[0], box['ymin']),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (box['xmin'], box['ymin'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            
            # Draw track history if available
            if self.enable_tracking and track_id in self.track_history:
                positions = self.track_history[track_id]['positions']
                for i in range(1, len(positions)):
                    if i > 1:  # Skip first point as it's the current position
                        # Draw line between consecutive positions
                        cv2.line(
                            annotated_frame,
                            positions[i-1],
                            positions[i],
                            color,
                            1
                        )
        
        # Add performance info
        if self.performance['inference_times']:
            avg_time = sum(self.performance['inference_times']) / len(self.performance['inference_times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            performance_text = f"Inference: {avg_time*1000:.1f}ms | FPS: {fps:.1f}"
            cv2.putText(
                annotated_frame,
                performance_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        return annotated_frame
    
    def get_performance_stats(self):
        """Get detector performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'frame_count': self.performance['frame_count'],
            'detection_counts': self.performance['detection_counts'].copy(),
            'tracked_objects': len(self.trackers)
        }
        
        # Calculate average inference time
        if self.performance['inference_times']:
            avg_time = sum(self.performance['inference_times']) / len(self.performance['inference_times'])
            stats['avg_inference_time'] = avg_time
            stats['fps'] = 1.0 / avg_time if avg_time > 0 else 0
        else:
            stats['avg_inference_time'] = 0
            stats['fps'] = 0
        
        # Add last detection time
        if self.performance['last_detection_time']:
            stats['last_detection_time'] = self.performance['last_detection_time'].isoformat()
        
        return stats
    
    def save_config(self, config_path=None):
        """Save current configuration to file"""
        if config_path is None:
            config_path = 'config/vehicle_detection_config.json'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Vehicle detection configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
