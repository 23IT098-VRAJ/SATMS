import numpy as np
from scipy.optimize import minimize
import time
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optimization_strategies")

class OptimizationStrategy:
    """Base class for traffic light optimization strategies"""
    
    def __init__(self, name="BaseStrategy", config_path=None):
        """Initialize the optimization strategy
        
        Args:
            name: Strategy name
            config_path: Path to configuration file
        """
        self.name = name
        self.config = self._load_config(config_path)
        self.last_optimization_time = None
        self.last_result = None
        self.optimization_count = 0
        self.performance_metrics = {
            'avg_execution_time': 0,
            'total_execution_time': 0,
            'min_phase_time': float('inf'),
            'max_phase_time': 0
        }
    
    def _load_config(self, config_path=None):
        """Load configuration from file"""
        # Default configuration
        default_config = {
            'min_phase_time': 10,
            'max_phase_time': 120,
            'min_cycle_time': 60,
            'max_cycle_time': 180,
            'optimization_interval': 30,  # seconds
            'logging_enabled': True,
            'performance_tracking': True
        }
        
        # If no config path provided, use default path
        if config_path is None:
            config_path = f"config/optimization/{self.name.lower()}_config.json"
        
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
                logger.error(f"Error loading configuration for {self.name}: {e}")
                return default_config
        else:
            # Create default configuration file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info(f"Created default configuration for {self.name} at {config_path}")
            except Exception as e:
                logger.error(f"Error creating default configuration: {e}")
            
            return default_config
    
    def optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Optimize traffic light timings
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            predicted_data: Optional predicted traffic data
            
        Returns:
            Dictionary of optimized phase timings
        """
        # Track optimization time
        start_time = time.time()
        
        # Check if we should reuse last result
        current_time = time.time()
        if (self.last_optimization_time is not None and 
            current_time - self.last_optimization_time < self.config['optimization_interval'] and
            self.last_result is not None):
            logger.debug(f"Reusing last optimization result for {self.name}")
            return self.last_result
        
        # Implement optimization (should be overridden by subclasses)
        result = self._optimize(traffic_data, intersection_config, predicted_data)
        
        # Update performance metrics
        execution_time = time.time() - start_time
        self.optimization_count += 1
        self.performance_metrics['total_execution_time'] += execution_time
        self.performance_metrics['avg_execution_time'] = (
            self.performance_metrics['total_execution_time'] / self.optimization_count
        )
        
        # Track min/max phase times
        for phase_id, time_value in result.items():
            self.performance_metrics['min_phase_time'] = min(
                self.performance_metrics['min_phase_time'], time_value
            )
            self.performance_metrics['max_phase_time'] = max(
                self.performance_metrics['max_phase_time'], time_value
            )
        
        # Log optimization result if enabled
        if self.config['logging_enabled']:
            logger.info(f"{self.name} optimization completed in {execution_time:.3f}s")
            logger.debug(f"Optimized phase times: {result}")
        
        # Store result and time
        self.last_result = result
        self.last_optimization_time = current_time
        
        return result
    
    def _optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Implementation of optimization algorithm (to be overridden)"""
        raise NotImplementedError("Subclasses must implement _optimize()")
    
    def get_name(self):
        """Get the name of this strategy"""
        return self.name
    
    def get_performance_metrics(self):
        """Get performance metrics for this strategy"""
        metrics = self.performance_metrics.copy()
        metrics['optimization_count'] = self.optimization_count
        return metrics
    
    def save_config(self, config_path=None):
        """Save current configuration to file"""
        if config_path is None:
            config_path = f"config/optimization/{self.name.lower()}_config.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


class FixedTimeStrategy(OptimizationStrategy):
    """Fixed time strategy - equal time for all phases"""
    
    def __init__(self, cycle_time=60, config_path=None):
        """Initialize fixed time strategy
        
        Args:
            cycle_time: Total cycle time in seconds
            config_path: Path to configuration file
        """
        super().__init__("FixedTime", config_path)
        
        # Set cycle time from config or parameter
        self.cycle_time = self.config.get('cycle_time', cycle_time)
    
    def _optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Assign equal time to all phases
        
        Args:
            traffic_data: Current traffic data (ignored)
            intersection_config: Intersection configuration
            predicted_data: Optional predicted traffic data (ignored)
            
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
            min_time = phase.get('min_time', self.config['min_phase_time'])
            max_time = phase.get('max_time', self.config['max_phase_time'])
            phase_timings[phase_id] = max(min_time, min(max_time, phase_time))
        
        return phase_timings


class ProportionalStrategy(OptimizationStrategy):
    """Proportional strategy - time proportional to traffic volume"""
    
    def __init__(self, min_phase_time=10, max_cycle_time=120, config_path=None):
        """Initialize proportional strategy
        
        Args:
            min_phase_time: Minimum time for any phase
            max_cycle_time: Maximum total cycle time
            config_path: Path to configuration file
        """
        super().__init__("Proportional", config_path)
        
        # Set parameters from config or arguments
        self.min_phase_time = self.config.get('min_phase_time', min_phase_time)
        self.max_cycle_time = self.config.get('max_cycle_time', max_cycle_time)
    
    def _optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Assign time proportional to traffic volume
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            predicted_data: Optional predicted traffic data
            
        Returns:
            Dictionary of phase timings
        """
        phases = intersection_config['phases']
        direction_to_zone = intersection_config.get('direction_to_zone', {})
        
        # Use predicted data if available
        if predicted_data is not None:
            # Blend current and predicted data
            blend_ratio = self.config.get('prediction_weight', 0.5)
            blended_data = {}
            for zone_id, current_value in traffic_data.items():
                predicted_value = predicted_data.get(zone_id, current_value)
                blended_data[zone_id] = (1 - blend_ratio) * current_value + blend_ratio * predicted_value
            
            data_to_use = blended_data
        else:
            data_to_use = traffic_data
        
        # Calculate traffic volume for each phase
        phase_volumes = {}
        for phase in phases:
            phase_id = phase['id']
            directions = phase.get('directions', [])
            
            # Sum traffic for all directions in this phase
            volume = 0
            for direction in directions:
                zone_id = direction_to_zone.get(direction)
                if zone_id and zone_id in data_to_use:
                    # Check if we have multi-metric data
                    if isinstance(data_to_use[zone_id], dict) and 'density' in data_to_use[zone_id]:
                        # Use density as primary metric
                        volume += data_to_use[zone_id]['density'] * 20  # Scale to reasonable vehicle count
                    else:
                        volume += data_to_use[zone_id]
            
            phase_volumes[phase_id] = max(1, volume)  # Ensure non-zero
        
        # Calculate proportional times
        total_volume = sum(phase_volumes.values())
        
        # Determine total cycle time based on traffic volume
        min_cycle = self.config.get('min_cycle_time', 60)
        max_cycle = self.config.get('max_cycle_time', 120)
        
        # Scale cycle time based on total volume
        volume_scale_factor = self.config.get('volume_scale_factor', 0.5)
        total_time = min(max_cycle, max(min_cycle, total_volume * volume_scale_factor))
        
        # Assign times proportionally with minimum constraint
        phase_timings = {}
        remaining_time = total_time
        remaining_phases = len(phases)
        
        for phase in phases:
            phase_id = phase['id']
            min_time = phase.get('min_time', self.min_phase_time)
            max_time = phase.get('max_time', self.config['max_phase_time'])
            
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


class WebsterStrategy(OptimizationStrategy):
    """Webster's method for traffic light optimization"""
    
    def __init__(self, saturation_flow=1800, lost_time_per_phase=4, config_path=None):
        """Initialize Webster strategy
        
        Args:
            saturation_flow: Saturation flow rate (veh/h/lane)
            lost_time_per_phase: Lost time per phase (seconds)
            config_path: Path to configuration file
        """
        super().__init__("Webster", config_path)
        
        # Set parameters from config or arguments
        self.saturation_flow = self.config.get('saturation_flow', saturation_flow)
        self.lost_time_per_phase = self.config.get('lost_time_per_phase', lost_time_per_phase)
    
    def _optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Optimize using Webster's method
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            predicted_data: Optional predicted traffic data
            
        Returns:
            Dictionary of phase timings
        """
        phases = intersection_config['phases']
        direction_to_zone = intersection_config.get('direction_to_zone', {})
        
        # Use predicted data if available
        if predicted_data is not None:
            # Blend current and predicted data
            blend_ratio = self.config.get('prediction_weight', 0.5)
            blended_data = {}
            for zone_id, current_value in traffic_data.items():
                predicted_value = predicted_data.get(zone_id, current_value)
                blended_data[zone_id] = (1 - blend_ratio) * current_value + blend_ratio * predicted_value
            
            data_to_use = blended_data
        else:
            data_to_use = traffic_data
        
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
                if zone_id and zone_id in data_to_use:
                    # Check if we have multi-metric data
                    if isinstance(data_to_use[zone_id], dict):
                        if 'flow_rate' in data_to_use[zone_id]:
                            # Use flow rate directly if available
                            flow = data_to_use[zone_id]['flow_rate'] * 60  # Convert to hourly flow
                        elif 'density' in data_to_use[zone_id]:
                            # Calculate flow from density and speed
                            density = data_to_use[zone_id]['density']
                            speed = data_to_use[zone_id].get('speed', 30)  # Default 30 mph if not available
                            flow = density * speed * 60  # Simple flow calculation
                        else:
                            # Use raw value
                            flow = data_to_use[zone_id] * 300  # Assuming count is for 12 seconds
                    else:
                        # Convert count to hourly flow
                        flow = data_to_use[zone_id] * 300  # Assuming count is for 12 seconds
                    
                    # Calculate flow ratio
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
            min_cycle = self.config.get('min_cycle_time', 40)
            max_cycle = self.config.get('max_cycle_time', 120)
            cycle_time = min(max_cycle, max(min_cycle, cycle_time))  # Constrain between min-max
        else:
            # Oversaturated condition - use maximum cycle time
            cycle_time = self.config.get('max_cycle_time', 120)
        
        # Calculate green times
        effective_green_time = cycle_time - total_lost_time
        
        phase_timings = {}
        for phase in phases:
            phase_id = phase['id']
            min_time = phase.get('min_time', self.config['min_phase_time'])
            max_time = phase.get('max_time', self.config['max_phase_time'])
            
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


class AdaptiveStrategy(OptimizationStrategy):
    """Adaptive strategy using real-time and predicted traffic data"""
    
    def __init__(self, prediction_weight=0.7, congestion_threshold=15, config_path=None):
        """Initialize adaptive strategy
        
        Args:
            prediction_weight: Weight given to predictions vs current data
            congestion_threshold: Vehicle count threshold for congestion
            config_path: Path to configuration file
        """
        super().__init__("Adaptive", config_path)
        
        # Set parameters from config or arguments
        self.prediction_weight = self.config.get('prediction_weight', prediction_weight)
        self.congestion_threshold = self.config.get('congestion_threshold', congestion_threshold)
    
    def _optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Optimize based on current and predicted traffic
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            predicted_data: Predicted traffic data (optional)
            
        Returns:
            Dictionary of phase timings
        """
        phases = intersection_config['phases']
        direction_to_zone = intersection_config.get('direction_to_zone', {})
        
        # Combine current and predicted data
        combined_data = {}
        
        for zone_id, current_value in traffic_data.items():
            # Handle multi-metric data
            if isinstance(current_value, dict) and predicted_data and zone_id in predicted_data:
                # Combine each metric separately
                combined_data[zone_id] = {}
                for metric, value in current_value.items():
                    if isinstance(predicted_data[zone_id], dict) and metric in predicted_data[zone_id]:
                        predicted_value = predicted_data[zone_id][metric]
                        combined_data[zone_id][metric] = (
                            (1 - self.prediction_weight) * value + 
                            self.prediction_weight * predicted_value
                        )
                    else:
                        combined_data[zone_id][metric] = value
            else:
                # Simple value combination
                predicted_value = predicted_data.get(zone_id, current_value) if predicted_data else current_value
                if isinstance(predicted_value, dict) and isinstance(current_value, (int, float)):
                    # Handle case where prediction is multi-metric but current is single value
                    # Use density as the main metric
                    predicted_value = predicted_value.get('density', current_value)
                elif isinstance(current_value, dict) and not isinstance(predicted_value, dict):
                    # Handle case where current is multi-metric but prediction is single value
                    # Use current structure and update density
                    combined_data[zone_id] = current_value.copy()
                    combined_data[zone_id]['density'] = (
                        (1 - self.prediction_weight) * current_value.get('density', 0) + 
                        self.prediction_weight * predicted_value
                    )
                    continue
                
                combined_data[zone_id] = (
                    (1 - self.prediction_weight) * current_value + 
                    self.prediction_weight * predicted_value
                )
        
        # Identify congested zones
        congested_directions = []
        
        for direction, zone_id in direction_to_zone.items():
            if zone_id in combined_data:
                # Check if we have multi-metric data
                if isinstance(combined_data[zone_id], dict):
                    density = combined_data[zone_id].get('density', 0)
                    # Convert density (0-1) to vehicle count equivalent
                    vehicle_count = density * 20  # Approximate scaling
                else:
                    vehicle_count = combined_data[zone_id]
                
                if vehicle_count >= self.congestion_threshold:
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
                    # Handle multi-metric data
                    if isinstance(combined_data[zone_id], dict):
                        # Use density as primary metric, scaled to vehicle count
                        density = combined_data[zone_id].get('density', 0)
                        volume += density * 20  # Scale to reasonable vehicle count
                    else:
                        volume += combined_data[zone_id]
                    
                    if direction in congested_directions:
                        has_congestion = True
            
            # Apply congestion multiplier
            congestion_multiplier = self.config.get('congestion_multiplier', 1.5)
            if has_congestion:
                volume *= congestion_multiplier  # Give more time to congested phases
            
            phase_volumes[phase_id] = max(1, volume)  # Ensure non-zero
        
        # Calculate proportional times
        total_volume = sum(phase_volumes.values())
        
        # Base cycle time on total volume
        light_threshold = self.config.get('light_traffic_threshold', 20)
        medium_threshold = self.config.get('medium_traffic_threshold', 40)
        
        light_cycle_time = self.config.get('light_cycle_time', 60)
        medium_cycle_time = self.config.get('medium_cycle_time', 90)
        heavy_cycle_time = self.config.get('heavy_cycle_time', 120)
        
        if total_volume < light_threshold:
            total_time = light_cycle_time  # Light traffic
        elif total_volume < medium_threshold:
            total_time = medium_cycle_time  # Medium traffic
        else:
            total_time = heavy_cycle_time  # Heavy traffic
        
        # Assign times proportionally with minimum constraint
        phase_timings = {}
        
        for phase in phases:
            phase_id = phase['id']
            min_time = phase.get('min_time', self.config['min_phase_time'])
            max_time = phase.get('max_time', self.config['max_phase_time'])
            
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


class PredictiveStrategy(OptimizationStrategy):
    """Predictive strategy that optimizes based on future traffic patterns"""
    
    def __init__(self, prediction_horizon=3, config_path=None):
        """Initialize predictive strategy
        
        Args:
            prediction_horizon: Number of time steps to look ahead
            config_path: Path to configuration file
        """
        super().__init__("Predictive", config_path)
        
        # Set parameters from config or arguments
        self.prediction_horizon = self.config.get('prediction_horizon', prediction_horizon)
        self.time_weights = self.config.get('time_weights', [0.6, 0.3, 0.1])  # Weights for current, near, far future
    
    def _optimize(self, traffic_data, intersection_config, predicted_data=None):
        """Optimize based on current and multiple future predictions
        
        Args:
            traffic_data: Current traffic data
            intersection_config: Intersection configuration
            predicted_data: List of predicted traffic data for future time steps
            
        Returns:
            Dictionary of phase timings
        """
        # If no predictions available, fall back to proportional strategy
        if predicted_data is None or not isinstance(predicted_data, list):
            logger.warning("No prediction sequence available, falling back to proportional strategy")
            fallback = ProportionalStrategy()
            return fallback._optimize(traffic_data, intersection_config)
        
        # Limit predictions to horizon
        predictions = predicted_data[:min(self.prediction_horizon, len(predicted_data))]
        
        # Adjust weights if we have fewer predictions than expected
        weights = self.time_weights[:len(predictions) + 1]  # +1 for current data
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        
        phases = intersection_config['phases']
        direction_to_zone = intersection_config.get('direction_to_zone', {})
        
        # Calculate weighted traffic volumes for each phase
        phase_volumes = {phase['id']: 0 for phase in phases}
        
        # Process current data (weight[0])
        self._add_weighted_volumes(
            phase_volumes, traffic_data, intersection_config, 
            weights[0], direction_to_zone
        )
        
        # Process each prediction with its corresponding weight
        for i, prediction in enumerate(predictions):
            self._add_weighted_volumes(
                phase_volumes, prediction, intersection_config, 
                weights[i + 1], direction_to_zone
            )
        
        # Calculate proportional times
        total_volume = sum(phase_volumes.values())
        
        # Determine cycle time based on total weighted volume
        min_cycle = self.config.get('min_cycle_time', 60)
        max_cycle = self.config.get('max_cycle_time', 120)
        volume_scale_factor = self.config.get('volume_scale_factor', 0.5)
        
        total_time = min(max_cycle, max(min_cycle, total_volume * volume_scale_factor))
        
        # Assign times proportionally with minimum constraint
        phase_timings = {}
        
        for phase in phases:
            phase_id = phase['id']
            min_time = phase.get('min_time', self.config['min_phase_time'])
            max_time = phase.get('max_time', self.config['max_phase_time'])
            
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
    
    def _add_weighted_volumes(self, phase_volumes, traffic_data, intersection_config, weight, direction_to_zone):
        """Add weighted traffic volumes to phase volumes"""
        phases = intersection_config['phases']
        
        for phase in phases:
            phase_id = phase['id']
            directions = phase.get('directions', [])
            
            # Sum traffic for all directions in this phase
            volume = 0
            for direction in directions:
                zone_id = direction_to_zone.get(direction)
                if zone_id and zone_id in traffic_data:
                    # Handle multi-metric data
                    if isinstance(traffic_data[zone_id], dict):
                        # Use density as primary metric, scaled to vehicle count
                        density = traffic_data[zone_id].get('density', 0)
                        volume += density * 20  # Scale to reasonable vehicle count
                    else:
                        volume += traffic_data[zone_id]
            
            # Add weighted volume to phase
            phase_volumes[phase_id] += volume * weight