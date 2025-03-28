import numpy as np
import json
import os
from collections import defaultdict
from datetime import datetime

class TrafficDensityAnalyzer:
    """Analyze traffic density from vehicle detections"""
    
    def __init__(self, zones, config_path=None):
        """Initialize traffic density analyzer
        
        Args:
            zones: Dictionary of traffic zones with coordinates
                {zone_id: {'polygon': [(x1,y1), (x2,y2), ...], 'direction': 'north'}}
            config_path: Path to configuration file
        """
        self.zones = zones
        self.vehicle_counts = defaultdict(int)
        self.vehicle_types = defaultdict(lambda: defaultdict(int))
        self.density_history = defaultdict(list)
        self.speed_history = defaultdict(list)
        self.flow_history = defaultdict(list)
        self.max_history = 30  # Keep 30 frames of history
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Apply configuration
        if 'max_history' in self.config:
            self.max_history = self.config['max_history']
        
        # Initialize metrics
        self.metrics = {
            'density': defaultdict(float),
            'speed': defaultdict(float),
            'flow': defaultdict(float),
            'occupancy': defaultdict(float)
        }
        
        # For speed estimation
        self.vehicle_tracking = defaultdict(dict)
        self.last_update_time = datetime.now()
    
    def _load_config(self, config_path=None):
        """Load configuration from file"""
        default_config = {
            'max_history': 30,
            'density_smoothing': True,
            'speed_estimation': True,
            'flow_calculation': True,
            'zone_capacities': {},
            'speed_limits': {},
            'vehicle_weights': {
                'car': 1.0,
                'truck': 2.5,
                'bus': 3.0,
                'motorcycle': 0.5,
                'bicycle': 0.3
            }
        }
        
        # If no config path provided, use default
        if config_path is None:
            config_path = 'config/traffic_density_config.json'
        
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
                print(f"Error loading traffic density configuration: {e}")
                return default_config
        else:
            # Create default configuration file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                print(f"Created default traffic density configuration at {config_path}")
            except Exception as e:
                print(f"Error creating default configuration: {e}")
            
            return default_config
    
    def update(self, detections, timestamp=None):
        """Update traffic density based on new detections
        
        Args:
            detections: List of vehicle detections with bounding boxes
            timestamp: Optional timestamp for this update
            
        Returns:
            Dictionary with traffic metrics for each zone
        """
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate time delta for speed estimation
        time_delta = (timestamp - self.last_update_time).total_seconds()
        self.last_update_time = timestamp
        
        # Reset counts for this frame
        self.vehicle_counts = defaultdict(int)
        self.vehicle_types = defaultdict(lambda: defaultdict(int))
        
        # Track vehicles for speed estimation
        current_vehicles = defaultdict(list)
        
        # Count vehicles in each zone
        for detection in detections:
            box = detection['box']
            vehicle_class = detection['class']
            vehicle_id = detection.get('id', None)
            
            # Get center point of the vehicle
            center_x = (box['xmin'] + box['xmax']) // 2
            center_y = (box['ymin'] + box['ymax']) // 2
            
            # Calculate vehicle size (area of bounding box)
            vehicle_area = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])
            
            # Check which zone the vehicle is in
            for zone_id, zone_info in self.zones.items():
                if self._point_in_polygon(center_x, center_y, zone_info['polygon']):
                    # Apply vehicle weight based on type
                    vehicle_weight = self.config['vehicle_weights'].get(vehicle_class, 1.0)
                    
                    # Increment weighted count
                    self.vehicle_counts[zone_id] += vehicle_weight
                    
                    # Track vehicle type
                    self.vehicle_types[zone_id][vehicle_class] += 1
                    
                    # Track vehicle for speed estimation
                    if vehicle_id is not None and self.config['speed_estimation']:
                        current_vehicles[zone_id].append({
                            'id': vehicle_id,
                            'position': (center_x, center_y),
                            'class': vehicle_class,
                            'area': vehicle_area
                        })
        
        # Estimate speeds for tracked vehicles
        if self.config['speed_estimation'] and time_delta > 0:
            for zone_id, vehicles in current_vehicles.items():
                zone_speeds = []
                
                for vehicle in vehicles:
                    vehicle_id = vehicle['id']
                    current_pos = vehicle['position']
                    
                    # Check if we've seen this vehicle before
                    if vehicle_id in self.vehicle_tracking[zone_id]:
                        prev_pos = self.vehicle_tracking[zone_id][vehicle_id]['position']
                        
                        # Calculate displacement
                        dx = current_pos[0] - prev_pos[0]
                        dy = current_pos[1] - prev_pos[1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # Convert pixel distance to estimated real-world distance
                        # This would require camera calibration for accuracy
                        # Here we use a simple scaling factor
                        scale_factor = self.zones[zone_id].get('scale_factor', 0.1)  # meters per pixel
                        real_distance = distance * scale_factor
                        
                        # Calculate speed (m/s)
                        speed = real_distance / time_delta
                        
                        # Convert to mph for consistency with simulation
                        speed_mph = speed * 2.237
                        
                        # Add to zone speeds
                        zone_speeds.append(speed_mph)
                    
                    # Update tracking with current position
                    self.vehicle_tracking[zone_id][vehicle_id] = {
                        'position': current_pos,
                        'timestamp': timestamp,
                        'class': vehicle['class'],
                        'area': vehicle['area']
                    }
                
                # Calculate average speed for the zone
                if zone_speeds:
                    avg_speed = sum(zone_speeds) / len(zone_speeds)
                    # Apply speed limit if configured
                    speed_limit = self.config['speed_limits'].get(zone_id, 60)
                    avg_speed = min(avg_speed, speed_limit)
                else:
                    # If no vehicles with speed, use previous or default
                    avg_speed = self.metrics['speed'][zone_id] if self.metrics['speed'][zone_id] > 0 else 30
                
                # Update speed metric
                self.metrics['speed'][zone_id] = avg_speed
                
                # Add to speed history
                self.speed_history[zone_id].append(avg_speed)
                if len(self.speed_history[zone_id]) > self.max_history:
                    self.speed_history[zone_id].pop(0)
        
        # Calculate metrics for all zones
        for zone_id in self.zones:
            # Get zone capacity (max vehicles)
            zone_capacity = self.config['zone_capacities'].get(zone_id, 20)
            
            # Calculate density (vehicles / capacity)
            density = min(1.0, self.vehicle_counts[zone_id] / zone_capacity)
            self.metrics['density'][zone_id] = density
            
            # Add to density history
            self.density_history[zone_id].append(density)
            if len(self.density_history[zone_id]) > self.max_history:
                self.density_history[zone_id].pop(0)
            
            # Calculate flow rate (vehicles per minute)
            # Flow = density * speed * constant
            if self.config['flow_calculation']:
                speed = self.metrics['speed'][zone_id]
                flow = density * speed * 0.1  # Simple flow calculation
                self.metrics['flow'][zone_id] = flow
                
                # Add to flow history
                self.flow_history[zone_id].append(flow)
                if len(self.flow_history[zone_id]) > self.max_history:
                    self.flow_history[zone_id].pop(0)
            
            # Calculate occupancy (percentage of zone occupied by vehicles)
            # This would require knowing zone area in pixels
            # For now, we use a simplified calculation based on density
            self.metrics['occupancy'][zone_id] = density * 100  # percentage
        
        # Return current metrics
        return self.get_current_metrics()
    
    def get_current_metrics(self):
        """Get the current traffic metrics for all zones
        
        Returns:
            Dictionary with traffic metrics
        """
        # Apply smoothing if enabled
        if self.config['density_smoothing']:
            for zone_id in self.zones:
                # Smooth density using exponential moving average
                if self.density_history[zone_id]:
                    smoothed_density = sum(self.density_history[zone_id]) / len(self.density_history[zone_id])
                    self.metrics['density'][zone_id] = smoothed_density
                
                # Smooth speed
                if self.speed_history[zone_id]:
                    smoothed_speed = sum(self.speed_history[zone_id]) / len(self.speed_history[zone_id])
                    self.metrics['speed'][zone_id] = smoothed_speed
                
                # Smooth flow
                if self.flow_history[zone_id]:
                    smoothed_flow = sum(self.flow_history[zone_id]) / len(self.flow_history[zone_id])
                    self.metrics['flow'][zone_id] = smoothed_flow
        
        # Create comprehensive metrics dictionary
        metrics = {
            'density': dict(self.metrics['density']),
            'speed': dict(self.metrics['speed']),
            'flow': dict(self.metrics['flow']),
            'occupancy': dict(self.metrics['occupancy']),
            'vehicle_counts': dict(self.vehicle_counts),
            'vehicle_types': {k: dict(v) for k, v in self.vehicle_types.items()}
        }
        
        return metrics
    
    def get_density_history(self, zone_id=None):
        """Get density history for a specific zone or all zones
        
        Args:
            zone_id: Optional zone ID (if None, return all zones)
            
        Returns:
            Dictionary with density history
        """
        if zone_id:
            return {zone_id: self.density_history[zone_id]}
        else:
            return {k: v for k, v in self.density_history.items()}
    
    def get_speed_history(self, zone_id=None):
        """Get speed history for a specific zone or all zones"""
        if zone_id:
            return {zone_id: self.speed_history[zone_id]}
        else:
            return {k: v for k, v in self.speed_history.items()}
    
    def get_flow_history(self, zone_id=None):
        """Get flow history for a specific zone or all zones"""
        if zone_id:
            return {zone_id: self.flow_history[zone_id]}
        else:
            return {k: v for k, v in self.flow_history.items()}
    
    def _point_in_polygon(self, x, y, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def save_config(self, config_path=None):
        """Save current configuration to file"""
        if config_path is None:
            config_path = 'config/traffic_density_config.json'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Traffic density configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")