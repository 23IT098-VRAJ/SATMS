import unittest
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.traffic_light_controller import TrafficLightController

class TestTrafficLightController(unittest.TestCase):
    def setUp(self):
        # Create test intersection configuration
        self.config = {
            'id': 'test_intersection',
            'phases': [
                {'id': 1, 'directions': ['north', 'south'], 'min_time': 10, 'max_time': 30},
                {'id': 2, 'directions': ['east', 'west'], 'min_time': 10, 'max_time': 30}
            ],
            'yellow_time': 3,
            'all_red_time': 2
        }
        
        self.controller = TrafficLightController(self.config)
        
    def test_initialization(self):
        # Test initial state
        self.assertEqual(self.controller.current_phase, 0)
        self.assertEqual(self.controller.get_current_phase(), 1)  # First phase ID
        
    def test_phase_times(self):
        # Test default phase times
        self.assertEqual(self.controller.phase_times[1], 10)  # min_time
        self.assertEqual(self.controller.phase_times[2], 10)  # min_time
        
    def test_update_phase_times(self):
        # Test updating phase times based on traffic predictions
        traffic_predictions = {
            'zone_1': 20,  # north
            'zone_2': 10,  # south
            'zone_3': 5,   # east
            'zone_4': 5    # west
        }
        
        # Set direction to zone mapping
        self.controller.set_direction_to_zone_mapping({
            'north': 'zone_1',
            'south': 'zone_2',
            'east': 'zone_3',
            'west': 'zone_4'
        })
        
        # Update phase times
        updated_times = self.controller.update_phase_times(traffic_predictions)
        
        # Check that phase times were updated
        self.assertGreater(updated_times[1], updated_times[2])  # Phase 1 should get more time
        
    def test_phase_change(self):
        # Test phase changing
        initial_phase = self.controller.get_current_phase()
        
        # Change phase
        new_phase = self.controller.change_phase()
        
        # Check that phase changed
        self.assertNotEqual(initial_phase, new_phase)
        
    def test_should_change_phase(self):
        # Set a short phase time for testing
        self.controller.phase_times[self.controller.get_current_phase()] = 0.1
        
        # Wait for phase to expire
        time.sleep(0.2)
        
        # Should indicate phase change needed
        self.assertTrue(self.controller.should_change_phase())

if __name__ == '__main__':
    unittest.main()