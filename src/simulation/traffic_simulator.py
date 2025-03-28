import numpy as np
import time
from collections import defaultdict
import random
import json
import os
from datetime import datetime, timedelta

class TrafficSimulator:
    """Simulate traffic flow for testing the traffic management system"""
    
    def __init__(self, config=None, config_path=None):
        """Initialize traffic simulator
        
        Args:
            config: Configuration dictionary with intersection and zones info
            config_path: Path to load configuration from file
        """
        # Load configuration
        if config is None and config_path is not None:
            self._load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            # Use default configuration
            self.config = self._create_default_config()
        
        # Extract key configuration elements
        self.zones = self.config['zones']
        self.intersection = self.config['intersection']
        
        # Initialize simulation state
        self.current_phase = 0
        self.vehicles = defaultdict(int)
        self.vehicle_types = defaultdict(lambda: defaultdict(int))
        self.simulation_time = 0  # Internal simulation time in seconds
        self.time_step = 1  # seconds per step
        self.real_time_factor = 1.0  # Simulation speed multiplier
        
        # Traffic speed tracking
        self.traffic_speeds = defaultdict(lambda: 30.0)  # Default 30 mph
        
        # Traffic patterns - vehicles per hour by time of day
        self.traffic_patterns = {
            'morning_rush': {
                'north': 1200,
                'south': 800,
                'east': 1000,
                'west': 1000
            },
            'midday': {
                'north': 600,
                'south': 600,
                'east': 500,
                'west': 500
            },
            'evening_rush': {
                'north': 800,
                'south': 1200,
                'east': 1000,
                'west': 1000
            },
            'night': {
                'north': 200,
                'south': 200,
                'east': 200,
                'west': 200
            }
        }
        
        # Vehicle type distributions
        self.vehicle_distributions = {
            'car': 0.75,
            'truck': 0.15,
            'bus': 0.05,
            'motorcycle': 0.05
        }
        
        # Traffic light phase timings (in seconds)
        self.phase_timings = self.config.get('phase_timings', {})
        if not self.phase_timings:
            # Default timings if not specified
            self.phase_timings = {
                'morning_rush': [30, 20, 30, 20],  # N-S gets more time
                'midday': [25, 25, 25, 25],  # Equal time
                'evening_rush': [20, 30, 20, 30],  # E-W gets more time
                'night': [15, 15, 15, 15]  # Shorter cycles
            }
        
        # Phase timing tracking
        self.current_phase_start_time = 0
        self.current_phase_duration = 30  # Default duration
        
        # Traffic density tracking
        self.traffic_density = defaultdict(float)
        
        # History tracking
        self.history = {
            'vehicles': [],
            'phases': [],
            'speeds': [],
            'densities': []
        }
        self.max_history_length = 1000
        
        # Statistics
        self.stats = {
            'total_vehicles_generated': 0,
            'total_vehicles_processed': 0,
            'avg_wait_time': 0,
            'max_queue_length': defaultdict(int)
        }
        
        # Initialize with random vehicles
        self._initialize_vehicles()
        
        # Simulation clock
        self.start_time = datetime.now()
        self.simulated_start_time = datetime.now().replace(
            hour=7, minute=0, second=0, microsecond=0
        )  # Start at 7 AM by default
    
    def _create_default_config(self):
        """Create a default configuration if none is provided"""
        default_config = {
            'zones': {
                'zone_1': {'direction': 'north', 'polygon': [(100, 300), (150, 300), (150, 400), (100, 400)]},
                'zone_2': {'direction': 'south', 'polygon': [(100, 0), (150, 0), (150, 100), (100, 100)]},
                'zone_3': {'direction': 'east', 'polygon': [(300, 100), (400, 100), (400, 150), (300, 150)]},
                'zone_4': {'direction': 'west', 'polygon': [(0, 100), (100, 100), (100, 150), (0, 150)]}
            },
            'intersection': {
                'center': [125, 125],
                'phases': [
                    {'id': 1, 'directions': ['north', 'south']},
                    {'id': 2, 'directions': ['east', 'west']},
                    {'id': 3, 'directions': ['north']},
                    {'id': 4, 'directions': ['east']}
                ]
            }
        }
        return default_config
    
    def _load_config(self, config_path):
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self.config = self._create_default_config()
            print("Using default configuration")
    
    def save_config(self, config_path):
        """Save current configuration to a JSON file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def _initialize_vehicles(self):
        """Initialize random number of vehicles in each zone"""
        for zone_id in self.zones:
            direction = self.zones[zone_id].get('direction', 'unknown')
            
            # Start with a random number of vehicles
            vehicle_count = random.randint(0, 10)
            self.vehicles[zone_id] = vehicle_count
            
            # Distribute among vehicle types
            for vehicle_type, distribution in self.vehicle_distributions.items():
                # Allocate vehicles proportionally to the distribution
                type_count = int(vehicle_count * distribution)
                # Add remaining vehicles to cars
                if vehicle_type == 'car':
                    type_count += vehicle_count - sum(
                        int(vehicle_count * d) for d in self.vehicle_distributions.values()
                    )
                self.vehicle_types[zone_id][vehicle_type] = type_count
            
            # Initialize traffic density based on vehicle count
            # Simple formula: density = vehicles / (lane_length * num_lanes)
            lane_length = 100  # Default lane length in meters
            num_lanes = 1  # Default number of lanes
            if 'lane_length' in self.zones[zone_id]:
                lane_length = self.zones[zone_id]['lane_length']
            if 'num_lanes' in self.zones[zone_id]:
                num_lanes = self.zones[zone_id]['num_lanes']
            
            self.traffic_density[zone_id] = min(1.0, vehicle_count / (lane_length * num_lanes * 10))
            
            # Initialize traffic speed based on density
            # Simple formula: speed decreases as density increases
            self.traffic_speeds[zone_id] = max(5, 50 - (self.traffic_density[zone_id] * 45))
    
    def _get_current_pattern(self):
        """Get the current traffic pattern based on time of day"""
        # Get current simulated time
        current_time = self.get_simulated_datetime()
        hour = current_time.hour
        
        if 6 <= hour < 9:
            return 'morning_rush'
        elif 9 <= hour < 16:
            return 'midday'
        elif 16 <= hour < 19:
            return 'evening_rush'
        else:
            return 'night'
    
    def get_simulated_datetime(self):
        """Get the current simulated date and time"""
        # Calculate elapsed simulation time
        elapsed_seconds = self.simulation_time
        
        # Add elapsed time to simulated start time
        return self.simulated_start_time + timedelta(seconds=elapsed_seconds)
    
    def _generate_vehicles(self):
        """Generate new vehicles based on traffic patterns"""
        pattern = self._get_current_pattern()
        
        for zone_id, zone_info in self.zones.items():
            direction = zone_info.get('direction', 'unknown')
            
            if direction in self.traffic_patterns[pattern]:
                # Vehicles per hour for this direction
                vph = self.traffic_patterns[pattern][direction]
                
                # Apply any special events or weather effects
                # (could be expanded in the future)
                
                # Convert to vehicles per second and apply randomness
                vps = vph / 3600.0
                
                # Probability of a new vehicle in this time step
                prob = vps * self.time_step
                
                # Generate vehicle with calculated probability
                if random.random() < prob:
                    # Increment total vehicle count
                    self.vehicles[zone_id] += 1
                    self.stats['total_vehicles_generated'] += 1
                    
                    # Distribute by vehicle type
                    vehicle_type = self._generate_vehicle_type()
                    self.vehicle_types[zone_id][vehicle_type] += 1
                    
                    # Update traffic density
                    self._update_traffic_density(zone_id)
    
    def _generate_vehicle_type(self):
        """Generate a random vehicle type based on distributions"""
        r = random.random()
        cumulative = 0
        for vehicle_type, probability in self.vehicle_distributions.items():
            cumulative += probability
            if r <= cumulative:
                return vehicle_type
        return 'car'  # Default fallback
    
    def _update_traffic_density(self, zone_id):
        """Update traffic density based on vehicle count"""
        # Get zone parameters
        lane_length = self.zones[zone_id].get('lane_length', 100)
        num_lanes = self.zones[zone_id].get('num_lanes', 1)
        
        # Calculate density as vehicles per meter per lane
        # Limit density to a maximum of 1.0 (complete congestion)
        self.traffic_density[zone_id] = min(1.0, self.vehicles[zone_id] / (lane_length * num_lanes * 10))
        
        # Update speed based on density (simple linear relationship)
        # Speed decreases as density increases
        self.traffic_speeds[zone_id] = max(5, 50 - (self.traffic_density[zone_id] * 45))
    
    def _move_vehicles(self):
        """Move vehicles through the intersection based on traffic light phase"""
        current_phase = self.intersection['phases'][self.current_phase]
        allowed_directions = current_phase['directions']
        
        # Calculate how many vehicles can move through in one time step
        # based on saturation flow rate (vehicles per hour green time)
        saturation_flow = 1800  # typical value for one lane
        vehicles_per_second = saturation_flow / 3600.0
        max_vehicles_per_step = vehicles_per_second * self.time_step
        
        for zone_id, zone_info in self.zones.items():
            direction = zone_info.get('direction', 'unknown')
            
            if direction in allowed_directions:
                # This zone has a green light
                # Number of vehicles that can move is affected by traffic density
                # Higher density = slower movement
                flow_reduction = 1.0 - (0.5 * self.traffic_density[zone_id])
                effective_max = max_vehicles_per_step * flow_reduction
                
                vehicles_to_move = min(
                    self.vehicles[zone_id],
                    int(effective_max) + (1 if random.random() < (effective_max % 1) else 0)
                )
                
                # Move vehicles
                self.vehicles[zone_id] -= vehicles_to_move
                self.stats['total_vehicles_processed'] += vehicles_to_move
                
                # Update vehicle types proportionally
                total_types = sum(self.vehicle_types[zone_id].values())
                if total_types > 0:
                    for vehicle_type in self.vehicle_types[zone_id]:
                        type_ratio = self.vehicle_types[zone_id][vehicle_type] / total_types
                        type_vehicles_to_move = int(vehicles_to_move * type_ratio)
                        self.vehicle_types[zone_id][vehicle_type] -= min(
                            type_vehicles_to_move, 
                            self.vehicle_types[zone_id][vehicle_type]
                        )
                
                # Update traffic density after vehicles move
                self._update_traffic_density(zone_id)
    
    def _update_phase_timing(self):
        """Update traffic light phase timing based on current traffic pattern"""
        pattern = self._get_current_pattern()
        phase_durations = self.phase_timings.get(pattern, [30, 30, 30, 30])
        
        # Get duration for current phase
        if self.current_phase < len(phase_durations):
            self.current_phase_duration = phase_durations[self.current_phase]
        else:
            self.current_phase_duration = 30  # Default
        
        # Check if phase should change
        elapsed_in_phase = self.simulation_time - self.current_phase_start_time
        if elapsed_in_phase >= self.current_phase_duration:
            # Move to next phase
            self.current_phase = (self.current_phase + 1) % len(self.intersection['phases'])
            self.current_phase_start_time = self.simulation_time
    
    def step(self):
        """Run one simulation step
        
        Returns:
            Current traffic state dictionary
        """
        # Generate new vehicles
        self._generate_vehicles()
        
        # Move vehicles through intersection
        self._move_vehicles()
        
        # Update phase timing
        self._update_phase_timing()
        
        # Advance time
        self.simulation_time += self.time_step
        
        # Update history
        self._update_history()
        
        # Update statistics
        self._update_statistics()
        
        # Return current state
        return self.get_state()
    
    def _update_history(self):
        """Update simulation history"""
        # Add current state to history
        self.history['vehicles'].append(dict(self.vehicles))
        self.history['phases'].append(self.current_phase)
        self.history['speeds'].append(dict(self.traffic_speeds))
        self.history['densities'].append(dict(self.traffic_density))
        
        # Trim history if it gets too long
        if len(self.history['vehicles']) > self.max_history_length:
            self.history['vehicles'].pop(0)
            self.history['phases'].pop(0)
            self.history['speeds'].pop(0)
            self.history['densities'].pop(0)
    
    def _update_statistics(self):
        """Update simulation statistics"""
        # Update maximum queue length
        for zone_id, count in self.vehicles.items():
            self.stats['max_queue_length'][zone_id] = max(
                self.stats['max_queue_length'][zone_id],
                count
            )
        
        # Other statistics could be added here
    
    def get_state(self):
        """Get current traffic state
        
        Returns:
            Dictionary with traffic state information
        """
        # Get current simulated time
        current_time = self.get_simulated_datetime()
        
        state = {
            'time': self.simulation_time,
            'timestamp': current_time.isoformat(),
            'hour': current_time.hour,
            'minute': current_time.minute,
            'day_of_week': current_time.weekday(),
            'vehicles': dict(self.vehicles),
            'vehicle_types': {k: dict(v) for k, v in self.vehicle_types.items()},
            'traffic_density': dict(self.traffic_density),
            'traffic_speeds': dict(self.traffic_speeds),
            'current_phase': self.current_phase,
            'phase_info': self.intersection['phases'][self.current_phase],
            'pattern': self._get_current_pattern(),
            'phase_remaining_time': self.current_phase_duration - (self.simulation_time - self.current_phase_start_time)
        }
        
        return state
    
    def set_traffic_lights(self, phase_id):
        """Set traffic light phase
        
        Args:
            phase_id: ID of the phase to set
        """
        # Find phase index by ID
        for i, phase in enumerate(self.intersection['phases']):
            if phase['id'] == phase_id:
                self.current_phase = i
                self.current_phase_start_time = self.simulation_time
                return
        
        # If phase not found
        print(f"Warning: Phase ID {phase_id} not found")
    
    def set_traffic_pattern(self, pattern_name, pattern_data):
        """Set or update a traffic pattern
        
        Args:
            pattern_name: Name of the pattern (e.g., 'morning_rush')
            pattern_data: Dictionary with traffic volumes by direction
        """
        if pattern_name and pattern_data:
            self.traffic_patterns[pattern_name] = pattern_data
            print(f"Traffic pattern '{pattern_name}' updated")
    
    def set_real_time_factor(self, factor):
        """Set the simulation speed
        
        Args:
            factor: Real-time factor (1.0 = real time, 2.0 = twice as fast)
        """
        self.real_time_factor = max(0.1, factor)
        print(f"Simulation speed set to {self.real_time_factor}x")
    
    def run_for_duration(self, duration_seconds, callback=None):
        """Run simulation for a specified duration
        
        Args:
            duration_seconds: Duration to run in seconds
            callback: Optional callback function to call after each step
            
        Returns:
            List of states from each step
        """
        steps = int(duration_seconds / self.time_step)
        states = []
        
        for _ in range(steps):
            state = self.step()
            states.append(state)
            
            if callback:
                callback(state)
        
        return states
    
    def get_statistics(self):
        """Get simulation statistics
        
        Returns:
            Dictionary with simulation statistics
        """
        # Calculate average wait time if we have processed vehicles
        if self.stats['total_vehicles_processed'] > 0:
            # Simple estimate based on total vehicles and current queue
            current_queue = sum(self.vehicles.values())
            self.stats['avg_wait_time'] = (current_queue * self.time_step) / self.stats['total_vehicles_processed']
        
        return self.stats
    
    def reset(self):
        """Reset the simulation to initial state"""
        self.vehicles = defaultdict(int)
        self.vehicle_types = defaultdict(lambda: defaultdict(int))
        self.simulation_time = 0
        self.current_phase = 0
        self.current_phase_start_time = 0
        self.traffic_density = defaultdict(float)
        self.traffic_speeds = defaultdict(lambda: 30.0)
        
        # Reset history
        self.history = {
            'vehicles': [],
            'phases': [],
            'speeds': [],
            'densities': []
        }
        
        # Reset statistics
        self.stats = {
            'total_vehicles_generated': 0,
            'total_vehicles_processed': 0,
            'avg_wait_time': 0,
            'max_queue_length': defaultdict(int)
        }
        
        # Initialize with random vehicles
        self._initialize_vehicles()
        
        # Reset simulation clock
        self.start_time = datetime.now()
        
        print("Simulation reset")