import numpy as np
from scipy.optimize import minimize

class TrafficOptimizer:
    """Optimize traffic light timings using mathematical optimization"""
    
    def __init__(self, min_phase_time=10, max_phase_time=120, cycle_time_range=(60, 180)):
        """Initialize traffic optimizer
        
        Args:
            min_phase_time: Minimum time for any phase (seconds)
            max_phase_time: Maximum time for any phase (seconds)
            cycle_time_range: Tuple with (min, max) cycle time in seconds
        """
        self.min_phase_time = min_phase_time
        self.max_phase_time = max_phase_time
        self.min_cycle_time, self.max_cycle_time = cycle_time_range
        
    def optimize_timings(self, traffic_volumes, saturation_flows, yellow_times):
        """Optimize traffic light timings using Webster's method
        
        Args:
            traffic_volumes: List of traffic volumes for each phase [veh/h]
            saturation_flows: List of saturation flows for each phase [veh/h]
            yellow_times: List of yellow times for each phase [s]
            
        Returns:
            Dictionary with optimized timings
        """
        n_phases = len(traffic_volumes)
        
        # Calculate flow ratios (y = volume/saturation)
        flow_ratios = [v/s for v, s in zip(traffic_volumes, saturation_flows)]
        
        # Calculate critical flow ratio sum
        critical_y_sum = sum(flow_ratios)
        
        # Check if intersection is overs