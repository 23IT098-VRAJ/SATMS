import numpy as np
from datetime import datetime, timedelta
import json
import os

class TrafficLightController:
    def __init__(self, config_path='config/traffic_light_config.json'):
        self.config_path = config_path
        self.load_config()
        self.current_phase = 0
        self.phase_start_time = datetime.now()
        self.last_update = datetime.now()
        
    def load_config(self):
        """Load or create default traffic light configuration"""
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            default_config = {
                'min_phase_duration': 30,  # seconds
                'max_phase_duration': 120,  # seconds
                'yellow_duration': 3,  # seconds
                'phases': [
                    {'north': 'green', 'south': 'green', 'east': 'red', 'west': 'red'},
                    {'north': 'yellow', 'south': 'yellow', 'east': 'red', 'west': 'red'},
                    {'north': 'red', 'south': 'red', 'east': 'green', 'west': 'green'},
                    {'north': 'red', 'south': 'red', 'east': 'yellow', 'west': 'yellow'}
                ],
                'congestion_thresholds': {
                    'low': 30,
                    'medium': 60,
                    'high': 80
                }
            }
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def calculate_phase_duration(self, traffic_state):
        """Calculate optimal phase duration based on traffic state"""
        current_phase = self.config['phases'][self.current_phase]
        active_directions = [d for d, state in current_phase.items() if state == 'green']
        
        if not active_directions:
            return self.config['min_phase_duration']
        
        # Calculate weighted congestion for active directions
        total_weight = 0
        weighted_congestion = 0
        
        for direction in active_directions:
            if direction in traffic_state['directions']:
                direction_data = traffic_state['directions'][direction]
                congestion = direction_data['congestion_level']
                vehicle_count = direction_data['vehicle_count']
                
                # Weight based on vehicle count
                weight = max(1, vehicle_count)
                total_weight += weight
                weighted_congestion += congestion * weight
        
        if total_weight == 0:
            return self.config['min_phase_duration']
        
        average_congestion = weighted_congestion / total_weight
        
        # Calculate duration based on congestion
        if average_congestion < self.config['congestion_thresholds']['low']:
            duration = self.config['min_phase_duration']
        elif average_congestion < self.config['congestion_thresholds']['medium']:
            duration = self.config['min_phase_duration'] + (self.config['max_phase_duration'] - self.config['min_phase_duration']) * 0.3
        elif average_congestion < self.config['congestion_thresholds']['high']:
            duration = self.config['min_phase_duration'] + (self.config['max_phase_duration'] - self.config['min_phase_duration']) * 0.6
        else:
            duration = self.config['max_phase_duration']
        
        return int(duration)
    
    def should_change_phase(self, traffic_state):
        """Determine if phase should be changed based on traffic state and timing"""
        current_time = datetime.now()
        phase_duration = self.calculate_phase_duration(traffic_state)
        
        # Check if minimum phase duration has passed
        if (current_time - self.phase_start_time).total_seconds() < self.config['min_phase_duration']:
            return False
        
        # Check if maximum phase duration has been reached
        if (current_time - self.phase_start_time).total_seconds() >= self.config['max_phase_duration']:
            return True
        
        # Check traffic conditions for phase change
        current_phase = self.config['phases'][self.current_phase]
        active_directions = [d for d, state in current_phase.items() if state == 'green']
        inactive_directions = [d for d, state in current_phase.items() if state == 'red']
        
        # Calculate average congestion for active and inactive directions
        active_congestion = 0
        inactive_congestion = 0
        
        for direction in active_directions:
            if direction in traffic_state['directions']:
                active_congestion += traffic_state['directions'][direction]['congestion_level']
        
        for direction in inactive_directions:
            if direction in traffic_state['directions']:
                inactive_congestion += traffic_state['directions'][direction]['congestion_level']
        
        # Change phase if inactive directions have significantly higher congestion
        if inactive_directions and active_directions:
            if inactive_congestion > active_congestion * 1.5 and (current_time - self.phase_start_time).total_seconds() >= phase_duration:
                return True
        
        return False
    
    def update_traffic_lights(self, traffic_state):
        """Update traffic light states based on traffic conditions"""
        current_time = datetime.now()
        
        if self.should_change_phase(traffic_state):
            # Move to next phase
            self.current_phase = (self.current_phase + 1) % len(self.config['phases'])
            self.phase_start_time = current_time
        
        self.last_update = current_time
        return self.get_current_state()
    
    def get_current_state(self):
        """Get current traffic light state"""
        return {
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase,
            'phase_duration': (datetime.now() - self.phase_start_time).total_seconds(),
            'lights': self.config['phases'][self.current_phase]
        }
    
    def save_state(self, output_dir='data/traffic_light_states'):
        """Save current traffic light state"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/traffic_light_state_{timestamp}.json"
        
        state = self.get_current_state()
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4) 