import numpy as np
from collections import defaultdict
import time

# Import optimization strategies
class FixedTimeStrategy:
    """Fixed time strategy - equal time for all phases"""
    
    def __init__(self, cycle_time=60):
        """Initialize fixed time strategy
        
        Args:
            cycle_time: Total cycle time in seconds
        """
        self.name = "FixedTime"
        self.cycle_time = cycle_time
        
    def optimize(self, traffic_data, intersection_config):
        """Assign equal time to all phases
        
        Args:
            traffic_data: Current traffic data (ignored)
            intersection_config: Intersection configuration
            
        Returns:
            Dictionary of phase timings
        """
        phases = intersection_config['phases']
        num_phases = len(phases)
        
        # Equal time distribution
        phase_time = self.cycle_time / num_phases
        
        # Create phase timings dictionary
        phase_timings = {}
        for phase in phases:
            phase_id = phase['id']
            # Ensure time is within min/max bounds
            min_time = phase.get('min_time', 10)
            max_time = phase.get('max_time', 120)
            phase_timings[phase_id] = max(min_time, min(max_time, phase_time))
            
        return phase_timings


class ProportionalStrategy:
    """Proportional strategy - time proportional to traffic volume"""
    
    def __init__(self, min_phase_time=10, max_cycle_time=120):
        """Initialize proportional strategy
        
        Args:
            min_phase_time: Minimum time for any phase
            max_cycle_time: Maximum total cycle time
        """
        self.name = "Proportional"
        self.min_phase_time = min_phase_time
        self.max_cycle_time = max_cycle_time
        
    def optimize(self, traffic_data, intersection_config):
        """Assign time proportional to traffic volume
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            
        Returns:
            Dictionary of phase timings
        """
        phases = intersection_config['phases']
        direction_to_zone = intersection_config.get('direction_to_zone', {})
        
        # Calculate traffic volume for each phase
        phase_volumes = {}
        for phase in phases:
            phase_id = phase['id']
            directions = phase.get('directions', [])
            
            # Sum traffic for all directions in this phase
            volume = 0
            for direction in directions:
                zone_id = direction_to_zone.get(direction)
                if zone_id and zone_id in traffic_data:
                    volume += traffic_data[zone_id]
                    
            phase_volumes[phase_id] = max(1, volume)  # Ensure non-zero
            
        # Calculate proportional times
        total_volume = sum(phase_volumes.values())
        total_time = min(self.max_cycle_time, max(len(phases) * self.min_phase_time, total_volume / 2))
        
        # Assign times proportionally with minimum constraint
        phase_timings = {}
        
        for phase in phases:
            phase_id = phase['id']
            min_time = phase.get('min_time', self.min_phase_time)
            max_time = phase.get('max_time', total_time)
            
            # Calculate proportional time
            if total_volume > 0:
                proportion = phase_volumes[phase_id] / total_volume
                allocated_time = total_time * proportion
            else:
                allocated_time = total_time / len(phases)
                
            # Apply constraints
            allocated_time = max(min_time, min(max_time, allocated_time))
            
            phase_timings[phase_id] = allocated_time
            
        return phase_timings


class WebsterStrategy:
    """Webster's method for traffic light optimization"""
    
    def __init__(self, saturation_flow=1800, lost_time_per_phase=4):
        """Initialize Webster strategy
        
        Args:
            saturation_flow: Saturation flow rate (veh/h/lane)
            lost_time_per_phase: Lost time per phase (seconds)
        """
        self.name = "Webster"
        self.saturation_flow = saturation_flow
        self.lost_time_per_phase = lost_time_per_phase
        
    def optimize(self, traffic_data, intersection_config):
        """Optimize using Webster's method
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            
        Returns:
            Dictionary of phase timings
        """
        phases = intersection_config['phases']
        direction_to_zone = intersection_config.get('direction_to_zone', {})
        
        # Calculate traffic flow for each phase
        phase_flows = {}
        critical_flows = []
        
        for phase in phases:
            phase_id = phase['id']
            directions = phase.get('directions', [])
            
            # Find maximum flow ratio for this phase
            max_flow_ratio = 0
            for direction in directions:
                zone_id = direction_to_zone.get(direction)
                if zone_id and zone_id in traffic_data:
                    # Convert count to hourly flow
                    flow = traffic_data[zone_id] * 300  # Assuming count is for 12 seconds
                    flow_ratio = flow / self.saturation_flow
                    max_flow_ratio = max(max_flow_ratio, flow_ratio)
                    
            phase_flows[phase_id] = max_flow_ratio
            critical_flows.append(max_flow_ratio)
            
        # Calculate cycle time using Webster's formula
        total_lost_time = len(phases) * self.lost_time_per_phase
        sum_critical_flows = sum(critical_flows)
        
        if sum_critical_flows < 0.9:  # Check if intersection is not oversaturated
            # Webster's optimal cycle time formula
            cycle_time = (1.5 * total_lost_time + 5) / (1 - sum_critical_flows)
            cycle_time = min(120, max(40, cycle_time))  # Constrain between 40-120s
        else:
            # Oversaturated condition - use maximum cycle time
            cycle_time = 120
            
        # Calculate green times
        effective_green_time = cycle_time - total_lost_time
        
        phase_timings = {}
        for phase in phases:
            phase_id = phase['id']
            min_time = phase.get('min_time', 10)
            max_time = phase.get('max_time', 90)
            
            if sum_critical_flows > 0:
                green_time = (phase_flows[phase_id] / sum_critical_flows) * effective_green_time
            else:
                green_time = effective_green_time / len(phases)
                
            # Add lost time to get actual phase time
            phase_time = green_time + self.lost_time_per_phase
            
            # Apply constraints
            phase_time = max(min_time, min(max_time, phase_time))
            
            phase_timings[phase_id] = phase_time
            
        return phase_timings


class AdaptiveStrategy:
    """Adaptive strategy using real-time and predicted traffic data"""
    
    def __init__(self, prediction_weight=0.7, congestion_threshold=15):
        """Initialize adaptive strategy
        
        Args:
            prediction_weight: Weight given to predictions vs current data
            congestion_threshold: Vehicle count threshold for congestion
        """
        self.name = "Adaptive"
        self.prediction_weight = prediction_weight
        self.congestion_threshold = congestion_threshold
        self.last_optimization = time.time()
        self.optimization_interval = 30  # seconds
        self.last_result = {}  # Store last result
        
    def optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Optimize based on current and predicted traffic
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            predicted_data: Predicted traffic data (optional)
            
        Returns:
            Dictionary of phase timings
        """
        # Check if it's time to reoptimize
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval and self.last_result:
            # Reuse last optimization results
            return self.last_result
            
        self.last_optimization = current_time
        
        phases = intersection_config['phases']
        direction_to_zone = intersection_config.get('direction_to_zone', {})
        
        # Combine current and predicted data
        combined_data = {}
        for zone_id, current_value in traffic_data.items():
            predicted_value = predicted_data.get(zone_id, current_value) if predicted_data else current_value
            combined_data[zone_id] = (1 - self.prediction_weight) * current_value + self.prediction_weight * predicted_value
            
        # Identify congested zones
        congested_directions = []
        for direction, zone_id in direction_to_zone.items():
            if zone_id in combined_data and combined_data[zone_id] >= self.congestion_threshold:
                congested_directions.append(direction)
                
        # Calculate basic proportional times
        phase_volumes = {}
        for phase in phases:
            phase_id = phase['id']
            directions = phase.get('directions', [])
            
            # Sum traffic for all directions in this phase
            volume = 0
            has_congestion = False
            
            for direction in directions:
                zone_id = direction_to_zone.get(direction)
                if zone_id and zone_id in combined_data:
                    volume += combined_data[zone_id]
                    if direction in congested_directions:
                        has_congestion = True
                        
            # Apply congestion multiplier
            if has_congestion:
                volume *= 1.5  # Give 50% more time to congested phases
                
            phase_volumes[phase_id] = max(1, volume)  # Ensure non-zero
            
        # Calculate proportional times
        total_volume = sum(phase_volumes.values())
        
        # Base cycle time on total volume
        if total_volume < 20:
            total_time = 60  # Light traffic
        elif total_volume < 40:
            total_time = 90  # Medium traffic
        else:
            total_time = 120  # Heavy traffic
            
        # Assign times proportionally with minimum constraint
        phase_timings = {}
        
        for phase in phases:
            phase_id = phase['id']
            min_time = phase.get('min_time', 10)
            max_time = phase.get('max_time', 90)
            
            # Calculate proportional time
            if total_volume > 0:
                proportion = phase_volumes[phase_id] / total_volume
                allocated_time = total_time * proportion
            else:
                allocated_time = total_time / len(phases)
                
            # Apply constraints
            allocated_time = max(min_time, min(max_time, allocated_time))
            
            phase_timings[phase_id] = allocated_time
            
        self.last_result = phase_timings
        return phase_timings


class TrafficLightController:
    """Controller for traffic light timing based on traffic predictions"""
    
    def __init__(self, intersection_config):
        """Initialize traffic light controller
        
        Args:
            intersection_config: Dictionary with intersection configuration
                {
                    'id': 'intersection_1',
                    'phases': [
                        {'id': 1, 'directions': ['north', 'south'], 'min_time': 20, 'max_time': 90},
                        {'id': 2, 'directions': ['east', 'west'], 'min_time': 20, 'max_time': 90}
                    ],
                    'yellow_time': 3,
                    'all_red_time': 2
                }
        """
        self.config = intersection_config
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.phase_times = self._initialize_phase_times()
        self.direction_to_zone = {}  # Will be set by external mapping
        
        # Initialize optimization strategies
        self.strategies = {
            "fixed": FixedTimeStrategy(),
            "proportional": ProportionalStrategy(),
            "webster": WebsterStrategy(),
            "adaptive": AdaptiveStrategy()
        }
        
        # Default strategy
        self.current_strategy = "proportional"
        
    def _initialize_phase_times(self):
        """Initialize default phase times"""
        times = {}
        for phase in self.config['phases']:
            # Start with minimum time for each phase
            times[phase['id']] = phase['min_time']
        return times
        
    def set_direction_to_zone_mapping(self, mapping):
        """Set mapping from directions to traffic zones
        
        Args:
            mapping: Dictionary mapping directions to zone IDs
                {'north': 'zone_1', 'south': 'zone_2', ...}
        """
        self.direction_to_zone = mapping
        
    def set_optimization_strategy(self, strategy_name):
        """Set the optimization strategy to use
        
        Args:
            strategy_name: Name of the strategy to use
            
        Returns:
            True if strategy was set, False if strategy not found
        """
        if strategy_name in self.strategies:
            self.current_strategy = strategy_name
            return True
        return False
        
    def get_current_strategy(self):
        """Get the current optimization strategy name
        
        Returns:
            Name of the current strategy
        """
        return self.current_strategy
        
    def get_available_strategies(self):
        """Get list of available optimization strategies
        
        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())
        
    def update_phase_times(self, traffic_predictions):
        """Update traffic light phase times based on predictions
        
        Args:
            traffic_predictions: Dictionary with predicted traffic density by zone
                {'zone_1': 15.5, 'zone_2': 8.2, ...}
                
        Returns:
            Updated phase times dictionary
        """
        # Convert numpy array to dictionary if needed
        if not isinstance(traffic_predictions, dict):
            # If it's not a dictionary, create a default one
            traffic_dict = {}
            for i, zone_id in enumerate(['zone_1', 'zone_2', 'zone_3', 'zone_4']):
                if isinstance(traffic_predictions, (list, np.ndarray)) and len(traffic_predictions) > i:
                    traffic_dict[zone_id] = float(traffic_predictions[i])
                else:
                    # Default value if index is out of range
                    traffic_dict[zone_id] = 5.0
            traffic_predictions = traffic_dict
        
        # Use the selected optimization strategy
        strategy = self.strategies[self.current_strategy]
        
        # Get optimized phase times
        optimized_times = strategy.optimize(traffic_predictions, {
            'phases': self.config['phases'],
            'direction_to_zone': self.direction_to_zone
        })
        
        # Update phase times
        self.phase_times.update(optimized_times)
        
        return self.phase_times
        
    def get_current_phase(self):
        """Get the current active phase
        
        Returns:
            Current phase ID
        """
        return self.config['phases'][self.current_phase]['id']
        
    def should_change_phase(self):
        """Check if it's time to change the traffic light phase
        
        Returns:
            Boolean indicating if phase should change
        """
        current_phase_id = self.get_current_phase()
        current_phase_time = self.phase_times[current_phase_id]
        elapsed_time = time.time() - self.phase_start_time
        
        return elapsed_time >= current_phase_time
        
    def change_phase(self):
        """Change to the next traffic light phase
        
        Returns:
            New phase ID
        """
        # Update to next phase
        self.current_phase = (self.current_phase + 1) % len(self.config['phases'])
        self.phase_start_time = time.time()
        
        return self.get_current_phase()
        
    def get_phase_remaining_time(self):
        """Get remaining time in the current phase
        
        Returns:
            Remaining time in seconds
        """
        current_phase_id = self.get_current_phase()
        current_phase_time = self.phase_times[current_phase_id]
        elapsed_time = time.time() - self.phase_start_time
        
        return max(0, current_phase_time - elapsed_time)
        
    def get_phase_info(self):
        """Get information about the current phase
        
        Returns:
            Dictionary with phase information
        """
        phase = self.config['phases'][self.current_phase]
        return {
            'id': phase['id'],
            'directions': phase['directions'],
            'total_time': self.phase_times[phase['id']],
            'remaining_time': self.get_phase_remaining_time(),
            'elapsed_time': time.time() - self.phase_start_time
        }