import numpy as np
from scipy.optimize import minimize
import time
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/traffic_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("traffic_optimization")

class TrafficOptimizer:
    """Optimize traffic light timings using mathematical optimization"""
    
    def __init__(self, min_phase_time=10, max_phase_time=120, cycle_time_range=(60, 180), config_path=None):
        """Initialize traffic optimizer
        
        Args:
            min_phase_time: Minimum time for any phase (seconds)
            max_phase_time: Maximum time for any phase (seconds)
            cycle_time_range: Tuple with (min, max) cycle time in seconds
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set parameters from config or arguments
        self.min_phase_time = self.config.get('min_phase_time', min_phase_time)
        self.max_phase_time = self.config.get('max_phase_time', max_phase_time)
        self.min_cycle_time = self.config.get('min_cycle_time', cycle_time_range[0])
        self.max_cycle_time = self.config.get('max_cycle_time', cycle_time_range[1])
        
        # Performance tracking
        self.performance = {
            'optimization_count': 0,
            'total_execution_time': 0,
            'avg_execution_time': 0,
            'last_optimization_time': None,
            'optimization_results': []
        }
    
    def _load_config(self, config_path=None):
        """Load configuration from file"""
        # Default configuration
        default_config = {
            'min_phase_time': 10,
            'max_phase_time': 120,
            'min_cycle_time': 60,
            'max_cycle_time': 180,
            'webster_coefficient': 1.5,
            'saturation_flow_rate': 1800,
            'lost_time_per_phase': 4,
            'yellow_time': 3,
            'all_red_time': 2,
            'optimization_method': 'webster',  # 'webster', 'delay', 'queue'
            'log_level': 'INFO',
            'performance_tracking': True
        }
        
        # If no config path provided, use default path
        if config_path is None:
            config_path = "config/traffic_optimizer_config.json"
        
        # Try to load configuration
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with default config
                    config = default_config.copy()
                    config.update(loaded_config)
                    logger.info(f"Loaded optimizer configuration from {config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading optimizer configuration: {e}")
                return default_config
        else:
            # Create default configuration file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info(f"Created default optimizer configuration at {config_path}")
            except Exception as e:
                logger.error(f"Error creating default configuration: {e}")
            
            return default_config
    
    def optimize_timings(self, traffic_volumes, saturation_flows=None, yellow_times=None):
        """Optimize traffic light timings using Webster's method
        
        Args:
            traffic_volumes: List of traffic volumes for each phase [veh/h]
            saturation_flows: List of saturation flows for each phase [veh/h]
            yellow_times: List of yellow times for each phase [s]
            
        Returns:
            Dictionary with optimized timings
        """
        # Track performance
        start_time = time.time()
        self.performance['optimization_count'] += 1
        self.performance['last_optimization_time'] = datetime.now()
        
        n_phases = len(traffic_volumes)
        
        # Set default values if not provided
        if saturation_flows is None:
            saturation_flows = [self.config['saturation_flow_rate']] * n_phases
        
        if yellow_times is None:
            yellow_times = [self.config['yellow_time']] * n_phases
        
        # Calculate flow ratios (y = volume/saturation)
        flow_ratios = [v/s for v, s in zip(traffic_volumes, saturation_flows)]
        
        # Calculate critical flow ratio sum
        critical_y_sum = sum(flow_ratios)
        
        # Calculate lost time
        lost_time = sum(yellow_times) + (n_phases * self.config['all_red_time'])
        
        # Select optimization method based on configuration
        method = self.config['optimization_method']
        
        if method == 'webster':
            result = self._optimize_webster(flow_ratios, critical_y_sum, lost_time, n_phases)
        elif method == 'delay':
            result = self._optimize_delay(traffic_volumes, saturation_flows, lost_time, n_phases)
        elif method == 'queue':
            result = self._optimize_queue(traffic_volumes, saturation_flows, lost_time, n_phases)
        else:
            # Default to Webster
            result = self._optimize_webster(flow_ratios, critical_y_sum, lost_time, n_phases)
        
        # Track execution time
        execution_time = time.time() - start_time
        self.performance['total_execution_time'] += execution_time
        self.performance['avg_execution_time'] = (
            self.performance['total_execution_time'] / self.performance['optimization_count']
        )
        
        # Store result
        self.performance['optimization_results'].append({
            'time': datetime.now().isoformat(),
            'method': method,
            'result': result,
            'execution_time': execution_time
        })
        
        # Trim results history
        max_results = self.config.get('max_results_history', 100)
        if len(self.performance['optimization_results']) > max_results:
            self.performance['optimization_results'] = self.performance['optimization_results'][-max_results:]
        
        logger.debug(f"Optimization completed in {execution_time:.3f}s using {method} method")
        
        return result
    
    def _optimize_webster(self, flow_ratios, critical_y_sum, lost_time, n_phases):
        """Optimize using Webster's method"""
        # Check if intersection is oversaturated
        if critical_y_sum >= 0.95:
            logger.warning(f"Intersection is oversaturated (critical_y_sum = {critical_y_sum:.2f})")
            cycle_time = self.max_cycle_time
        else:
            # Calculate Webster's optimal cycle time
            webster_coef = self.config['webster_coefficient']
            cycle_time = (webster_coef * lost_time) / (1 - critical_y_sum)
            
            # Constrain cycle time to configured limits
            cycle_time = max(self.min_cycle_time, min(self.max_cycle_time, cycle_time))
        
        # Calculate effective green time
        effective_green = cycle_time - lost_time
        
        # Calculate green times for each phase
        green_times = []
        phase_times = {}
        
        for i, y in enumerate(flow_ratios):
            # Allocate green time proportional to flow ratio
            if critical_y_sum > 0:
                green = (y / critical_y_sum) * effective_green
            else:
                green = effective_green / len(flow_ratios)
            
            # Apply minimum green time constraint
            green = max(self.min_phase_time, green)
            
            green_times.append(green)
            phase_times[i+1] = green  # Phase IDs typically start at 1
        
        # Adjust if sum of green times exceeds effective green
        total_green = sum(green_times)
        if total_green > effective_green:
            # Scale down proportionally
            scale_factor = effective_green / total_green
            for i in range(len(green_times)):
                green_times[i] *= scale_factor
                phase_times[i+1] = green_times[i]
        
        # Add results to the returned dictionary
        result = {
            'cycle_time': cycle_time,
            'lost_time': lost_time,
            'effective_green': effective_green,
            'critical_y_sum': critical_y_sum,
            'phase_times': phase_times,
            'method': 'webster'
        }
        
        return result
    
    def _optimize_delay(self, traffic_volumes, saturation_flows, lost_time, n_phases):
        """Optimize to minimize total delay"""
        # Define objective function to minimize delay
        def objective(x):
            cycle_time = sum(x) + lost_time
            
            # Calculate delay for each phase using Webster's delay formula
            total_delay = 0
            for i in range(n_phases):
                green_time = x[i]
                volume = traffic_volumes[i]
                saturation = saturation_flows[i]
                
                # Webster's delay formula components
                y = volume / saturation  # Flow ratio
                g_c = green_time / cycle_time  # Green ratio
                
                # First term: uniform delay
                d1 = 0.5 * cycle_time * (1 - g_c)**2 / (1 - min(0.95, y * g_c))
                
                # Second term: overflow delay (simplified)
                d2 = 900 * 0.25 * ((y - 1) + ((y - 1)**2 + 8 * min(0.5, y) / (saturation * 0.25))**0.5)
                
                # Total delay for this phase
                phase_delay = volume * (d1 + max(0, d2))
                total_delay += phase_delay
            
            return total_delay
        
        # Define constraints
        def constraint_cycle(x):
            # Cycle time within bounds
            return self.max_cycle_time - (sum(x) + lost_time)
        
        def constraint_min_cycle(x):
            # Minimum cycle time
            return (sum(x) + lost_time) - self.min_cycle_time
        
        # Initial guess based on equal distribution
        initial_green = [(self.min_cycle_time - lost_time) / n_phases] * n_phases
        
        # Bounds for each phase
        bounds = [(self.min_phase_time, self.max_phase_time) for _ in range(n_phases)]
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': constraint_cycle},
            {'type': 'ineq', 'fun': constraint_min_cycle}
        ]
        
        # Run optimization
        try:
            solution = minimize(
                objective,
                initial_green,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'disp': False}
            )
            
            if solution.success:
                # Extract optimized green times
                green_times = solution.x
                cycle_time = sum(green_times) + lost_time
                
                # Create result dictionary
                phase_times = {i+1: max(self.min_phase_time, g) for i, g in enumerate(green_times)}
                
                result = {
                    'cycle_time': cycle_time,
                    'lost_time': lost_time,
                    'effective_green': cycle_time - lost_time,
                    'phase_times': phase_times,
                    'method': 'delay',
                    'objective_value': solution.fun
                }
                
                return result
            else:
                logger.warning(f"Delay optimization failed: {solution.message}")
                # Fall back to Webster method
                flow_ratios = [v/s for v, s in zip(traffic_volumes, saturation_flows)]
                critical_y_sum = sum(flow_ratios)
                return self._optimize_webster(flow_ratios, critical_y_sum, lost_time, n_phases)
        
        except Exception as e:
            logger.error(f"Error in delay optimization: {e}")
            # Fall back to Webster method
            flow_ratios = [v/s for v, s in zip(traffic_volumes, saturation_flows)]
            critical_y_sum = sum(flow_ratios)
            return self._optimize_webster(flow_ratios, critical_y_sum, lost_time, n_phases)
    
    def _optimize_queue(self, traffic_volumes, saturation_flows, lost_time, n_phases):
        """Optimize to minimize maximum queue length"""
        # Define objective function to minimize maximum queue length
        def objective(x):
            cycle_time = sum(x) + lost_time
            
            # Calculate maximum queue length for each phase
            queue_lengths = []
            for i in range(n_phases):
                green_time = x[i]
                volume = traffic_volumes[i]
                saturation = saturation_flows[i]
                
                # Calculate arrival rate (vehicles per second)
                arrival_rate = volume / 3600
                
                # Calculate service rate during green (vehicles per second)
                service_rate = saturation / 3600
                
                # Calculate red time
                red_time = cycle_time - green_time
                
                # Calculate queue length at end of red time
                # Queue = arrival rate * red time
                queue = arrival_rate * red_time
                
                # Check if queue can be cleared during green time
                # If arrival_rate >= service_rate, queue will grow indefinitely
                if arrival_rate >= service_rate:
                    queue = 1000  # Large value to indicate oversaturation
                else:
                    # Factor in queue dissipation during green
                    # Time to clear queue = queue / (service_rate - arrival_rate)
                    clear_time = queue / (service_rate - arrival_rate)
                    
                    # If clear time > green time, residual queue remains
                    if clear_time > green_time:
                        residual = queue - (service_rate - arrival_rate) * green_time
                        queue = max(queue, residual)
                
                queue_lengths.append(queue)
            
            # Objective is to minimize maximum queue across all phases
            return max(queue_lengths)
        
        # Define constraints
        def constraint_cycle(x):
            # Cycle time within bounds
            return self.max_cycle_time - (sum(x) + lost_time)
        
        def constraint_min_cycle(x):
            # Minimum cycle time
            return (sum(x) + lost_time) - self.min_cycle_time
        
        # Initial guess based on equal distribution
        initial_green = [(self.min_cycle_time - lost_time) / n_phases] * n_phases
        
        # Bounds for each phase
        bounds = [(self.min_phase_time, self.max_phase_time) for _ in range(n_phases)]
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': constraint_cycle},
            {'type': 'ineq', 'fun': constraint_min_cycle}
        ]
        
        # Run optimization
        try:
            solution = minimize(
                objective,
                initial_green,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'disp': False}
            )
            
            if solution.success:
                # Extract optimized green times
                green_times = solution.x
                cycle_time = sum(green_times) + lost_time
                
                # Create result dictionary
                phase_times = {i+1: max(self.min_phase_time, g) for i, g in enumerate(green_times)}
                
                result = {
                    'cycle_time': cycle_time,
                    'lost_time': lost_time,
                    'effective_green': cycle_time - lost_time,
                    'phase_times': phase_times,
                    'method': 'queue',
                    'objective_value': solution.fun
                }
                
                return result
            else:
                logger.warning(f"Queue optimization failed: {solution.message}")
                # Fall back to Webster method
                flow_ratios = [v/s for v, s in zip(traffic_volumes, saturation_flows)]
                critical_y_sum = sum(flow_ratios)
                return self._optimize_webster(flow_ratios, critical_y_sum, lost_time, n_phases)
        
        except Exception as e:
            logger.error(f"Error in queue optimization: {e}")
            # Fall back to Webster method
            flow_ratios = [v/s for v, s in zip(traffic_volumes, saturation_flows)]
            critical_y_sum = sum(flow_ratios)
            return self._optimize_webster(flow_ratios, critical_y_sum, lost_time, n_phases)
    
    def get_performance_stats(self):
        """Get optimizer performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'optimization_count': self.performance['optimization_count'],
            'avg_execution_time': self.performance['avg_execution_time'],
            'last_optimization_time': (
                self.performance['last_optimization_time'].isoformat() 
                if self.performance['last_optimization_time'] else None
            )
        }
        
        # Add method-specific statistics
        method_stats = {}
        for result in self.performance['optimization_results']:
            method = result['method']
            if method not in method_stats:
                method_stats[method] = {
                    'count': 0,
                    'avg_cycle_time': 0,
                    'total_cycle_time': 0
                }
            
            method_stats[method]['count'] += 1
            method_stats[method]['total_cycle_time'] += result['result']['cycle_time']
            method_stats[method]['avg_cycle_time'] = (
                method_stats[method]['total_cycle_time'] / method_stats[method]['count']
            )
        
        stats['method_stats'] = method_stats
        
        return stats
    
    def save_config(self, config_path=None):
        """Save current configuration to file"""
        if config_path is None:
            config_path = "config/traffic_optimizer_config.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False