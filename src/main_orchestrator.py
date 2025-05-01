import time
import threading
import logging
from datetime import datetime
import json
import os

from utils.data_generator import TrafficDataGenerator
from processing.data_processor import TrafficDataProcessor
from control.traffic_controller import TrafficLightController

class TrafficSystemOrchestrator:
    def __init__(self, update_interval=5):
        self.update_interval = update_interval
        self.running = False
        self.data_generator = TrafficDataGenerator()
        self.data_processor = TrafficDataProcessor()
        self.traffic_controller = TrafficLightController()
        
        # Configure logging
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/traffic_system.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start the traffic management system"""
        self.running = True
        self.logger.info("Starting Traffic Management System")
        
        # Start the main processing loop in a separate thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()
        
    def stop(self):
        """Stop the traffic management system"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        self.logger.info("Traffic Management System stopped")
        
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Generate traffic data
                traffic_data = self.data_generator.generate_traffic_data()
                self.logger.debug(f"Generated traffic data: {json.dumps(traffic_data, indent=2)}")
                
                # Process traffic data
                processed_state = self.data_processor.process_traffic_data(traffic_data)
                self.logger.debug(f"Processed traffic state: {json.dumps(processed_state, indent=2)}")
                
                # Update traffic lights
                light_state = self.traffic_controller.update_traffic_lights(processed_state)
                self.logger.debug(f"Updated traffic light state: {json.dumps(light_state, indent=2)}")
                
                # Save states
                self.data_generator.save_traffic_data(traffic_data)
                self.data_processor.save_processed_data()
                self.traffic_controller.save_state()
                
                # Log summary
                self._log_summary(traffic_data, processed_state, light_state)
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def _log_summary(self, traffic_data, processed_state, light_state):
        """Log a summary of the current system state"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_vehicles': traffic_data['total_vehicles'],
            'average_speed': traffic_data['average_speed'],
            'congestion_level': traffic_data['congestion_level'],
            'current_phase': light_state['phase'],
            'phase_duration': light_state['phase_duration'],
            'active_directions': [
                d for d, state in light_state['lights'].items()
                if state == 'green'
            ]
        }
        self.logger.info(f"System Summary: {json.dumps(summary, indent=2)}")
    
    def get_current_state(self):
        """Get the current state of the entire system"""
        return {
            'timestamp': datetime.now().isoformat(),
            'traffic_data': self.data_generator.generate_traffic_data(),
            'processed_state': self.data_processor.get_current_state(),
            'traffic_light_state': self.traffic_controller.get_current_state()
        }

if __name__ == "__main__":
    # Create and start the orchestrator
    orchestrator = TrafficSystemOrchestrator(update_interval=5)
    try:
        orchestrator.start()
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop() 