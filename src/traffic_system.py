import threading
import time
from datetime import datetime
import json
import os

from utils.data_generator import TrafficDataGenerator
from processing.data_processor import TrafficDataProcessor
from control.traffic_controller import TrafficLightController

class TrafficSystem:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrafficSystem, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.update_interval = 5  # seconds
        self.running = False
        self.data_generator = TrafficDataGenerator()
        self.data_processor = TrafficDataProcessor()
        self.traffic_controller = TrafficLightController()
        self.current_state = None
        self._initialized = True
        
    def start(self):
        """Start the traffic management system"""
        if self.running:
            return
            
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop(self):
        """Stop the traffic management system"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Generate traffic data
                traffic_data = self.data_generator.generate_traffic_data()
                
                # Process traffic data
                processed_state = self.data_processor.process_traffic_data(traffic_data)
                
                # Update traffic lights
                light_state = self.traffic_controller.update_traffic_lights(processed_state)
                
                # Save states
                self.data_generator.save_traffic_data(traffic_data)
                self.data_processor.save_processed_data()
                self.traffic_controller.save_state()
                
                # Update current state
                self.current_state = {
                    'timestamp': datetime.now().isoformat(),
                    'traffic_data': traffic_data,
                    'processed_state': processed_state,
                    'traffic_light_state': light_state
                }
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in processing loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def get_current_state(self):
        """Get the current state of the entire system"""
        if self.current_state is None:
            return {
                'timestamp': datetime.now().isoformat(),
                'traffic_data': self.data_generator.generate_traffic_data(),
                'processed_state': self.data_processor.get_current_state(),
                'traffic_light_state': self.traffic_controller.get_current_state()
            }
        return self.current_state
    
    def get_traffic_data(self):
        """Get current traffic data"""
        return self.data_generator.generate_traffic_data()
    
    def get_processed_state(self):
        """Get current processed state"""
        return self.data_processor.get_current_state()
    
    def get_traffic_light_state(self):
        """Get current traffic light state"""
        return self.traffic_controller.get_current_state()
    
    def get_historical_data(self, direction, hours=1):
        """Get historical data for a specific direction"""
        return self.data_processor.get_historical_data(direction, hours) 