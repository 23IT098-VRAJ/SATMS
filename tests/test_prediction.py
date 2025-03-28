import unittest
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_traffic_management.src.prediction.traffic_predictor_tf import TrafficPredictor

class TestTrafficPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = TrafficPredictor()
        
    def test_build_model(self):
        # Test model building
        input_shape = (10, 5)  # 10 time steps, 5 features
        model = self.predictor.build_model(input_shape)
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 10, 5))
        
    def test_predict(self):
        # Build a model
        input_shape = (10, 5)
        self.predictor.build_model(input_shape)
        
        # Create dummy input data
        X = np.random.random((1, 10, 5))
        
        # Make prediction
        prediction = self.predictor.predict(X)
        
        # Check prediction shape
        self.assertEqual(prediction.shape, (1, 1))
        
    def test_save_load_model(self):
        # Build a model
        input_shape = (10, 5)
        self.predictor.build_model(input_shape)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            # Save model
            self.predictor.save_model(tmp.name)
            
            # Create a new predictor
            new_predictor = TrafficPredictor()
            
            # Load the model
            new_predictor.load_model(tmp.name)
            
            # Check that model was loaded
            self.assertIsNotNone(new_predictor.model)
            
            # Test prediction with loaded model
            X = np.random.random((1, 10, 5))
            prediction = new_predictor.predict(X)
            self.assertEqual(prediction.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()