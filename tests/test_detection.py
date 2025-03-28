import unittest
import numpy as np
import cv2
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.vehicle_detection import VehicleDetector

class TestVehicleDetection(unittest.TestCase):
    def setUp(self):
        # Create a mock model path - in real testing you'd use a test model
        self.model_path = "models/saved_models/detection_model"
        
        # Skip tests if model doesn't exist
        if not os.path.exists(self.model_path):
            self.skipTest("Test model not found")
            
        self.detector = VehicleDetector(self.model_path)
        
    def test_detect_empty_frame(self):
        # Create an empty frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Should return empty list for empty frame
        self.assertEqual(len(detections), 0)
        
    def test_detection_format(self):
        # Create a test frame with a vehicle-like object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a rectangle that might look like a car
        cv2.rectangle(frame, (100, 100), (300, 200), (255, 255, 255), -1)
        
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # If any detections, check format
        for detection in detections:
            self.assertIn('class', detection)
            self.assertIn('confidence', detection)
            self.assertIn('box', detection)
            
            # Check box format
            box = detection['box']
            self.assertIn('xmin', box)
            self.assertIn('ymin', box)
            self.assertIn('xmax', box)
            self.assertIn('ymax', box)

if __name__ == '__main__':
    unittest.main()