import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

class TrafficDataGenerator:
    def __init__(self, config_path='config/traffic_config.json'):
        self.config_path = config_path
        self.load_config()
        self.current_time = datetime.now()
        self.vehicle_types = ['car', 'bus', 'truck', 'motorcycle']
        self.directions = ['north', 'south', 'east', 'west']
        self.vehicle_counts = {direction: 0 for direction in self.directions}
        self.max_vehicles_per_direction = 100  # Maximum vehicles per direction
        self.vehicle_removal_rate = 0.1  # Rate at which vehicles are removed
        
    def load_config(self):
        """Load or create default configuration"""
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            default_config = {
                'traffic_density': {
                    'low': {'min': 1, 'max': 5},
                    'medium': {'min': 6, 'max': 15},
                    'high': {'min': 16, 'max': 30}
                },
                'time_weights': {
                    'morning': {'north': 0.4, 'south': 0.3, 'east': 0.2, 'west': 0.1},
                    'afternoon': {'north': 0.2, 'south': 0.3, 'east': 0.4, 'west': 0.1},
                    'evening': {'north': 0.3, 'south': 0.4, 'east': 0.2, 'west': 0.1},
                    'night': {'north': 0.25, 'south': 0.25, 'east': 0.25, 'west': 0.25}
                },
                'traffic_patterns': {
                    'morning_rush': {
                        'start_hour': 7,
                        'end_hour': 9,
                        'multiplier': 1.5
                    },
                    'evening_rush': {
                        'start_hour': 17,
                        'end_hour': 19,
                        'multiplier': 1.5
                    },
                    'night': {
                        'start_hour': 22,
                        'end_hour': 5,
                        'multiplier': 0.5
                    }
                }
            }
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def get_time_of_day(self):
        """Determine time of day based on current time"""
        hour = self.current_time.hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def get_traffic_multiplier(self):
        """Get traffic multiplier based on time of day"""
        hour = self.current_time.hour
        patterns = self.config['traffic_patterns']
        
        for pattern, details in patterns.items():
            if pattern == 'night':
                if hour >= details['start_hour'] or hour < details['end_hour']:
                    return details['multiplier']
            elif details['start_hour'] <= hour < details['end_hour']:
                return details['multiplier']
        
        return 1.0  # Default multiplier

    def generate_vehicle_data(self):
        """Generate simulated vehicle detection data"""
        time_of_day = self.get_time_of_day()
        weights = self.config['time_weights'][time_of_day]
        traffic_multiplier = self.get_traffic_multiplier()
        
        # Generate number of vehicles based on time of day and traffic multiplier
        if time_of_day in ['morning', 'evening']:
            density = 'high'
        elif time_of_day == 'afternoon':
            density = 'medium'
        else:
            density = 'low'
            
        base_vehicles = random.randint(
            self.config['traffic_density'][density]['min'],
            self.config['traffic_density'][density]['max']
        )
        
        # Apply traffic multiplier
        num_vehicles = int(base_vehicles * traffic_multiplier)
        
        vehicles = []
        for _ in range(num_vehicles):
            direction = random.choices(
                self.directions,
                weights=[weights[d] for d in self.directions]
            )[0]
            
            # Check if we've reached the maximum vehicles for this direction
            if self.vehicle_counts[direction] >= self.max_vehicles_per_direction:
                continue
                
            vehicle = {
                'id': f"V{random.randint(1000, 9999)}",
                'type': random.choice(self.vehicle_types),
                'direction': direction,
                'speed': random.uniform(20, 60),  # km/h
                'timestamp': self.current_time.isoformat(),
                'confidence': random.uniform(0.85, 0.99)
            }
            vehicles.append(vehicle)
            self.vehicle_counts[direction] += 1
            
        # Remove some vehicles to prevent infinite growth
        for direction in self.directions:
            if self.vehicle_counts[direction] > 0:
                vehicles_to_remove = int(self.vehicle_counts[direction] * self.vehicle_removal_rate)
                self.vehicle_counts[direction] = max(0, self.vehicle_counts[direction] - vehicles_to_remove)
            
        return vehicles

    def generate_traffic_data(self):
        """Generate complete traffic data for all directions"""
        self.current_time = datetime.now()
        traffic_data = {
            'timestamp': self.current_time.isoformat(),
            'directions': {},
            'total_vehicles': 0,
            'average_speed': 0,
            'congestion_level': 0
        }
        
        total_speed = 0
        total_vehicles = 0
        
        for direction in self.directions:
            vehicles = [v for v in self.generate_vehicle_data() if v['direction'] == direction]
            total_vehicles += len(vehicles)
            if vehicles:
                total_speed += sum(v['speed'] for v in vehicles)
            
            traffic_data['directions'][direction] = {
                'vehicle_count': len(vehicles),
                'average_speed': sum(v['speed'] for v in vehicles) / len(vehicles) if vehicles else 0,
                'vehicles': vehicles
            }
        
        traffic_data['total_vehicles'] = total_vehicles
        traffic_data['average_speed'] = total_speed / total_vehicles if total_vehicles > 0 else 0
        
        # Calculate congestion level (0-100)
        max_vehicles = self.config['traffic_density']['high']['max'] * 4  # Maximum possible vehicles
        traffic_data['congestion_level'] = int((total_vehicles / max_vehicles) * 100)
        
        return traffic_data

    def save_traffic_data(self, data, output_dir='data/traffic_data'):
        """Save generated traffic data to file"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = self.current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/traffic_data_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        return filename 