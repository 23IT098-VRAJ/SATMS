import numpy as np
from collections import defaultdict
import time
import json
import os
import logging
from datetime import datetime

# Import optimization strategies
from optimization_strategies import (
    FixedTimeStrategy, 
    ProportionalStrategy, 
    WebsterStrategy, 
    AdaptiveStrategy,
    PredictiveStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/traffic_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("traffic_controller")

class TrafficLightController:
    """Controller for traffic light timing based on traffic predictions"""
    
    def __init__(self, intersection_config, config_path=None):
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
            config_path: Path to controller configuration file
        """
        self.config = intersection_config
        self.controller_config = self._load_config(config_path)
        
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.phase_times = self._initialize_phase_times()
        self.direction_to_zone = {}  # Will be set by external mapping
        
        # Initialize optimization strategies
        self.strategies = {
            "fixed": FixedTimeStrategy(),
            "proportional": ProportionalStrategy(),
            "webster": WebsterStrategy(),
            "adaptive": AdaptiveStrategy(),
            "predictive": PredictiveStrategy()
        }
        
        # Default strategy
        self.current_strategy = self.controller_config.get('default_strategy', "proportional")
        
        # Performance tracking
        self.performance = {
            'phase_history': [],
            'timing_history': [],
            'strategy_history': [],
            'max_history': 100,
            'last_update_time': time.time(),
            'total_updates': 0,
            'total_phase_changes': 0
        }
        
        # State tracking
        self.state = {
            'is_transitioning': False,
            'transition_start_time': None,
            'transition_type': None,  # 'yellow' or 'all_red'
            'next_phase': None
        }
        
        logger.info(f"Traffic light controller initialized for intersection {intersection_config['id']}")

    def _load_config(self, config_path=None):
        """Load controller configuration from file"""
        # Default configuration
        default_config = {
            'default_strategy': 'proportional',
            'auto_strategy_selection': True,
            'strategy_selection_interval': 300,  # seconds
            'max_history_length': 100,
            'log_level': 'INFO',
            'performance_tracking': True,
            'emergency_vehicle_priority': True,
            'pedestrian_priority': True,
            'transitioning_enabled': True,
            'yellow_time': 3,
            'all_red_time': 2
        }
        
        # If no config path provided, use default path
        if config_path is None:
            config_path = "config/traffic_controller_config.json"
        
        # Try to load configuration
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with default config
                    config = default_config.copy()
                    config.update(loaded_config)
                    logger.info(f"Loaded controller configuration from {config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading controller configuration: {e}")
                return default_config
        else:
            # Create default configuration file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info(f"Created default controller configuration at {config_path}")
            except Exception as e:
                logger.error(f"Error creating default configuration: {e}")
            
            return default_config
    
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
        logger.debug(f"Direction to zone mapping set: {mapping}")
    
    def set_optimization_strategy(self, strategy_name):
        """Set the optimization strategy to use
        
        Args:
            strategy_name: Name of the strategy to use
            
        Returns:
            True if strategy was set, False if strategy not found
        """
        if strategy_name in self.strategies:
            self.current_strategy = strategy_name
            logger.info(f"Optimization strategy set to {strategy_name}")
            
            # Add to history
            self.performance['strategy_history'].append({
                'time': time.time(),
                'strategy': strategy_name
            })
            
            # Trim history if needed
            if len(self.performance['strategy_history']) > self.performance['max_history']:
                self.performance['strategy_history'].pop(0)
            
            return True
        
        logger.warning(f"Strategy {strategy_name} not found")
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
    
    def auto_select_strategy(self, traffic_data):
        """Automatically select the best strategy based on traffic conditions
        
        Args:
            traffic_data: Current traffic data
            
        Returns:
            Selected strategy name
        """
        # Check if it's time to update strategy
        current_time = time.time()
        last_selection_time = self.performance.get('last_strategy_selection_time', 0)
        
        if current_time - last_selection_time < self.controller_config['strategy_selection_interval']:
            # Not time to update yet
            return self.current_strategy
        
        # Store current time
        self.performance['last_strategy_selection_time'] = current_time
        
        # Calculate total traffic volume
        total_volume = 0
        max_volume = 0
        min_volume = float('inf')
        
        for zone_id, value in traffic_data.items():
            if isinstance(value, dict):
                # Use density if available
                volume = value.get('density', 0) * 20  # Scale to vehicle count
            else:
                volume = value
            
            total_volume += volume
            max_volume = max(max_volume, volume)
            min_volume = min(min_volume, volume) if volume > 0 else min_volume
        
        # Calculate volume imbalance (ratio of max to min)
        imbalance = max_volume / min_volume if min_volume > 0 else float('inf')
        
        # Select strategy based on conditions
        if total_volume < 10:  # Very light traffic
            selected_strategy = "fixed"
        elif imbalance > 3:  # Highly imbalanced traffic
            selected_strategy = "adaptive"
        elif total_volume > 50:  # Heavy traffic
            selected_strategy = "webster"
        else:  # Medium traffic
            selected_strategy = "proportional"
        
        # If we have prediction data, use predictive strategy for medium to heavy traffic
        if hasattr(self, 'prediction_available') and self.prediction_available and total_volume > 20:
            selected_strategy = "predictive"
        
        # Set the selected strategy
        if selected_strategy != self.current_strategy:
            logger.info(f"Auto-selecting strategy: {selected_strategy} (traffic volume: {total_volume:.1f}, imbalance: {imbalance:.1f})")
            self.set_optimization_strategy(selected_strategy)
        
        return selected_strategy
    
    def update_phase_times(self, traffic_data, predicted_data=None):
        """Update traffic light phase times based on current and predicted traffic
        
        Args:
            traffic_data: Dictionary with traffic density by zone
                {'zone_1': 15.5, 'zone_2': 8.2, ...}
            predicted_data: Dictionary with predicted traffic (optional)
            
        Returns:
            Updated phase times dictionary
        """
        # Track update time
        self.performance['last_update_time'] = time.time()
        self.performance['total_updates'] += 1
        
        # Convert numpy array to dictionary if needed
        if not isinstance(traffic_data, dict):
            # If it's not a dictionary, create a default one
            traffic_dict = {}
            for i, zone_id in enumerate(['zone_1', 'zone_2', 'zone_3', 'zone_4']):
                if isinstance(traffic_data, (list, np.ndarray)) and len(traffic_data) > i:
                    traffic_dict[zone_id] = float(traffic_data[i])
                else:
                    # Default value if index is out of range
                    traffic_dict[zone_id] = 5.0
            traffic_data = traffic_dict
        
        # Flag if prediction data is available
        self.prediction_available = predicted_data is not None
        
        # Auto-select strategy if enabled
        if self.controller_config['auto_strategy_selection']:
            self.auto_select_strategy(traffic_data)
        
        # Use the selected optimization strategy
        strategy = self.strategies[self.current_strategy]
        
        # Get optimized phase times
        optimized_times = strategy.optimize(
            traffic_data, 
            {
                'phases': self.config['phases'],
                'direction_to_zone': self.direction_to_zone
            },
            predicted_data
        )
        
        # Update phase times
        self.phase_times.update(optimized_times)
        
        # Add to timing history
        self.performance['timing_history'].append({
            'time': time.time(),
            'phase_times': self.phase_times.copy(),
            'strategy': self.current_strategy
        })
        
        # Trim history if needed
        if len(self.performance['timing_history']) > self.performance['max_history']:
            self.performance['timing_history'].pop(0)
        
        logger.debug(f"Updated phase times: {self.phase_times}")
        
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
        # If we're in a transition (yellow or all-red), check if it's complete
        if self.state['is_transitioning']:
            transition_elapsed = time.time() - self.state['transition_start_time']
            
            if self.state['transition_type'] == 'yellow':
                # If yellow phase is complete, move to all-red
                if transition_elapsed >= self.config.get('yellow_time', 3):
                    self.state['transition_type'] = 'all_red'
                    self.state['transition_start_time'] = time.time()
                    logger.debug(f"Changing from yellow to all-red transition")
                    return False
            
            elif self.state['transition_type'] == 'all_red':
                # If all-red phase is complete, complete the transition
                if transition_elapsed >= self.config.get('all_red_time', 2):
                    # Complete the transition to the next phase
                    self.current_phase = self.state['next_phase']
                    self.phase_start_time = time.time()
                    self.state['is_transitioning'] = False
                    
                    # Log the phase change
                    logger.info(f"Changed to phase {self.get_current_phase()}")
                    
                    # Add to phase history
                    self.performance['phase_history'].append({
                        'time': time.time(),
                        'phase': self.get_current_phase(),
                        'duration': self.phase_times[self.get_current_phase()]
                    })
                    
                    # Trim history if needed
                    if len(self.performance['phase_history']) > self.performance['max_history']:
                        self.performance['phase_history'].pop(0)
                    
                    self.performance['total_phase_changes'] += 1
                    
                    return False
            
            # Still in transition
            return False
        
        # Regular phase - check if time is up
        current_phase_id = self.get_current_phase()
        current_phase_time = self.phase_times[current_phase_id]
        elapsed_time = time.time() - self.phase_start_time
        
        return elapsed_time >= current_phase_time
    
    def change_phase(self):
        """Change to the next traffic light phase
        
        Returns:
            New phase ID
        """
        # If transitions are enabled, start yellow phase
        if self.controller_config['transitioning_enabled']:
            # Start transition with yellow light
            self.state['is_transitioning'] = True
            self.state['transition_start_time'] = time.time()
            self.state['transition_type'] = 'yellow'
            
            # Calculate next phase
            self.state['next_phase'] = (self.current_phase + 1) % len(self.config['phases'])
            
            logger.debug(f"Starting yellow transition from phase {self.get_current_phase()} to {self.config['phases'][self.state['next_phase']]['id']}")
            
            # Return current phase ID (we're still in this phase, just showing yellow)
            return self.get_current_phase()
        else:
            # Direct phase change without transition
            self.current_phase = (self.current_phase + 1) % len(self.config['phases'])
            self.phase_start_time = time.time()
            
            # Log the phase change
            logger.info(f"Changed to phase {self.get_current_phase()}")
            
            # Add to phase history
            self.performance['phase_history'].append({
                'time': time.time(),
                'phase': self.get_current_phase(),
                'duration': self.phase_times[self.get_current_phase()]
            })
            
            # Trim history if needed
            if len(self.performance['phase_history']) > self.performance['max_history']:
                self.performance['phase_history'].pop(0)
            
            self.performance['total_phase_changes'] += 1
            
            return self.get_current_phase()
    
    def get_phase_remaining_time(self):
        """Get remaining time in the current phase
        
        Returns:
            Remaining time in seconds
        """
        # If we're in a transition, return remaining transition time
        if self.state['is_transitioning']:
            transition_elapsed = time.time() - self.state['transition_start_time']
            
            if self.state['transition_type'] == 'yellow':
                yellow_time = self.config.get('yellow_time', 3)
                return max(0, yellow_time - transition_elapsed)
            
            elif self.state['transition_type'] == 'all_red':
                all_red_time = self.config.get('all_red_time', 2)
                return max(0, all_red_time - transition_elapsed)
        
        # Regular phase - calculate remaining time
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
        remaining_time = self.get_phase_remaining_time()
        elapsed_time = time.time() - self.phase_start_time
        
        info = {
            'id': phase['id'],
            'directions': phase['directions'],
            'total_time': self.phase_times[phase['id']],
            'remaining_time': remaining_time,
            'elapsed_time': elapsed_time
        }
        
        # Add transition information if applicable
        if self.state['is_transitioning']:
            info['is_transitioning'] = True
            info['transition_type'] = self.state['transition_type']
            info['next_phase'] = self.config['phases'][self.state['next_phase']]['id'] if self.state['next_phase'] is not None else None
        else:
            info['is_transitioning'] = False
        
        return info
    
    def handle_emergency_vehicle(self, direction):
        """Handle emergency vehicle approaching from a specific direction
        
        Args:
            direction: Direction from which emergency vehicle is approaching
            
        Returns:
            True if phase was changed, False otherwise
        """
        if not self.controller_config['emergency_vehicle_priority']:
            return False
        
        # Find phase that allows traffic from this direction
        target_phase = None
        target_phase_index = None
        
        for i, phase in enumerate(self.config['phases']):
            if direction in phase['directions']:
                target_phase = phase
                target_phase_index = i
                break
        
        if target_phase is None:
            logger.warning(f"No phase found for emergency vehicle from direction {direction}")
            return False
        
        # If we're already in this phase, extend it
        if self.current_phase == target_phase_index and not self.state['is_transitioning']:
            # Extend current phase
            emergency_extension = self.controller_config.get('emergency_extension', 30)
            self.phase_times[target_phase['id']] = max(
                self.phase_times[target_phase['id']],
                self.get_phase_remaining_time() + emergency_extension
            )
            logger.info(f"Extended phase {target_phase['id']} for emergency vehicle")
            return True
        
        # Otherwise, transition to this phase immediately
        if self.state['is_transitioning']:
            # If already transitioning, change the target
            self.state['next_phase'] = target_phase_index
            logger.info(f"Redirecting transition to phase {target_phase['id']} for emergency vehicle")
        else:
            # Start a new transition to the emergency phase
            self.state['is_transitioning'] = True
            self.state['transition_start_time'] = time.time()
            self.state['transition_type'] = 'yellow'
            self.state['next_phase'] = target_phase_index
            logger.info(f"Starting transition to phase {target_phase['id']} for emergency vehicle")
        
        return True
    
    def handle_pedestrian(self, direction):
        """Handle pedestrian request from a specific direction
        
        Args:
            direction: Direction from which pedestrian is requesting to cross
            
        Returns:
            True if handled, False otherwise
        """
        if not self.controller_config['pedestrian_priority']:
            return False
        
        # Find phase that allows pedestrians to cross this direction
        # For simplicity, we assume pedestrians cross perpendicular to traffic flow
        perpendicular_directions = {
            'north': ['east', 'west'],
            'south': ['east', 'west'],
            'east': ['north', 'south'],
            'west': ['north', 'south']
        }
        
        crossing_directions = perpendicular_directions.get(direction, [])
        
        if not crossing_directions:
            logger.warning(f"No crossing directions found for pedestrian from {direction}")
            return False
        
        # Find phase that allows traffic from the crossing directions
        target_phase = None
        target_phase_index = None
        
        for i, phase in enumerate(self.config['phases']):
            if any(d in phase['directions'] for d in crossing_directions):
                target_phase = phase
                target_phase_index = i
                break
        
        if target_phase is None:
            logger.warning(f"No phase found for pedestrian crossing from {direction}")
            return False
        
        # If we're already in this phase, extend it for pedestrian
        if self.current_phase == target_phase_index and not self.state['is_transitioning']:
            # Extend current phase
            pedestrian_extension = self.controller_config.get('pedestrian_extension', 15)
            self.phase_times[target_phase['id']] = max(
                self.phase_times[target_phase['id']],
                self.get_phase_remaining_time() + pedestrian_extension
            )
            logger.info(f"Extended phase {target_phase['id']} for pedestrian crossing")
            return True
        
        # Otherwise, prioritize this phase in the next cycle
        # We don't immediately change for pedestrians, but reduce wait time
        logger.info(f"Prioritizing phase {target_phase['id']} for pedestrian crossing in next cycle")
        
        # If the wait is too long, consider changing phase
        if self.get_phase_remaining_time() > self.controller_config.get('max_pedestrian_wait', 60):
            # Start transition to the pedestrian phase
            if not self.state['is_transitioning']:
                self.state['is_transitioning'] = True
                self.state['transition_start_time'] = time.time()
                self.state['transition_type'] = 'yellow'
                self.state['next_phase'] = target_phase_index
                logger.info(f"Starting transition to phase {target_phase['id']} for pedestrian (long wait)")
            
            return True
        
        return False
    
    def get_performance_stats(self):
        """Get controller performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'total_updates': self.performance['total_updates'],
            'total_phase_changes': self.performance['total_phase_changes'],
            'current_strategy': self.current_strategy,
            'avg_phase_duration': 0
        }
        
        # Calculate average phase duration
        if self.performance['phase_history']:
            durations = [p['duration'] for p in self.performance['phase_history']]
            stats['avg_phase_duration'] = sum(durations) / len(durations)
            stats['min_phase_duration'] = min(durations)
            stats['max_phase_duration'] = max(durations)
        
        # Get strategy performance
        strategy_counts = {}
        for entry in self.performance['strategy_history']:
            strategy = entry['strategy']
            if strategy in strategy_counts:
                strategy_counts[strategy] += 1
            else:
                strategy_counts[strategy] = 1
        
        stats['strategy_usage'] = strategy_counts
        
        # Get current strategy performance
        if self.current_strategy in self.strategies:
            strategy = self.strategies[self.current_strategy]
            stats['strategy_metrics'] = strategy.get_performance_metrics()
        
        return stats
    
    def save_state(self, state_path=None):
        """Save controller state to file
        
        Args:
            state_path: Path to save state (default: config/controller_state.json)
            
        Returns:
            True if successful, False otherwise
        """
        if state_path is None:
            state_path = "config/controller_state.json"
        
        # Create state object
        state = {
            'current_phase': self.current_phase,
            'phase_times': self.phase_times,
            'current_strategy': self.current_strategy,
            'phase_start_time': self.phase_start_time,
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'total_updates': self.performance['total_updates'],
                'total_phase_changes': self.performance['total_phase_changes']
            },
            'transition_state': self.state
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
            logger.info(f"Controller state saved to {state_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving controller state: {e}")
            return False
    
    def load_state(self, state_path=None):
        """Load controller state from file
        
        Args:
            state_path: Path to load state from (default: config/controller_state.json)
            
        Returns:
            True if successful, False otherwise
        """
        if state_path is None:
            state_path = "config/controller_state.json"
        
        if not os.path.exists(state_path):
            logger.warning(f"Controller state file not found: {state_path}")
            return False
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.current_phase = state['current_phase']
            self.phase_times = state['phase_times']
            self.current_strategy = state['current_strategy']
            
            # Adjust phase start time based on saved time and elapsed time
            saved_time = datetime.fromisoformat(state['timestamp'])
            elapsed_since_save = (datetime.now() - saved_time).total_seconds()
            self.phase_start_time = state['phase_start_time'] + elapsed_since_save
            
            # Restore performance counters
            if 'performance' in state:
                self.performance['total_updates'] = state['performance'].get('total_updates', 0)
                self.performance['total_phase_changes'] = state['performance'].get('total_phase_changes', 0)
            
            # Restore transition state
            if 'transition_state' in state:
                self.state = state['transition_state']
                
                # Adjust transition start time if in transition
                if self.state['is_transitioning'] and self.state['transition_start_time'] is not None:
                    self.state['transition_start_time'] += elapsed_since_save
            
            logger.info(f"Controller state loaded from {state_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading controller state: {e}")
            return False    
