import numpy as np
from datetime import datetime, timedelta
from collections import deque
import logging

class TrafficPredictor:
    """Simple traffic prediction system using historical patterns and time-series analysis"""
    
    def __init__(self, history_size=60):  # 1 hour of history with 1-minute intervals
        self.history_size = history_size
        self.traffic_history = {
            'north': deque(maxlen=history_size),
            'south': deque(maxlen=history_size),
            'east': deque(maxlen=history_size),
            'west': deque(maxlen=history_size)
        }
        self.time_patterns = self._initialize_time_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_time_patterns(self):
        """Initialize typical traffic patterns based on time of day"""
        return {
            'morning_rush': {  # 7:00-9:00
                'north': 0.8,  # 80% capacity
                'south': 0.9,  # 90% capacity
                'east': 0.7,   # 70% capacity
                'west': 0.85   # 85% capacity
            },
            'evening_rush': {  # 16:00-19:00
                'north': 0.9,   # 90% capacity
                'south': 0.85,  # 85% capacity
                'east': 0.8,    # 80% capacity
                'west': 0.9     # 90% capacity
            },
            'normal': {  # Regular hours
                'north': 0.5,  # 50% capacity
                'south': 0.5,
                'east': 0.4,
                'west': 0.45
            },
            'night': {  # Late night/early morning
                'north': 0.2,  # 20% capacity
                'south': 0.2,
                'east': 0.15,
                'west': 0.15
            }
        }
    
    def _get_time_pattern(self, current_time=None):
        """Get the traffic pattern based on time of day"""
        if current_time is None:
            current_time = datetime.now()
        
        hour = current_time.hour
        
        if 7 <= hour < 9:
            return self.time_patterns['morning_rush']
        elif 16 <= hour < 19:
            return self.time_patterns['evening_rush']
        elif 23 <= hour or hour < 5:
            return self.time_patterns['night']
        else:
            return self.time_patterns['normal']
    
    def update_history(self, traffic_data):
        """Update traffic history with new data
        
        Args:
            traffic_data: Dictionary with traffic densities for each direction
        """
        try:
            for direction, density in traffic_data.items():
                self.traffic_history[direction].append(density)
        except Exception as e:
            self.logger.error(f"Error updating traffic history: {e}")
    
    def predict_next_interval(self, current_time=None, prediction_window=5):
        """Predict traffic density for the next interval
        
        Args:
            current_time: Current datetime (uses system time if None)
            prediction_window: Minutes to look ahead
            
        Returns:
            Dictionary with predicted densities for each direction
        """
        if current_time is None:
            current_time = datetime.now()
            
        predictions = {}
        time_pattern = self._get_time_pattern(current_time + timedelta(minutes=prediction_window))
        
        try:
            for direction in self.traffic_history:
                if len(self.traffic_history[direction]) > 0:
                    # Get recent history
                    recent_history = list(self.traffic_history[direction])
                    
                    # Calculate trend from recent data
                    if len(recent_history) >= 3:
                        trend = np.mean(np.diff(recent_history[-3:]))
                    else:
                        trend = 0
                    
                    # Current average
                    current_avg = np.mean(recent_history[-3:]) if len(recent_history) >= 3 else recent_history[-1]
                    
                    # Combine historical trend with time-based pattern
                    base_prediction = current_avg + (trend * prediction_window)
                    time_factor = time_pattern[direction]
                    
                    # Weight the prediction (70% trend, 30% time pattern)
                    prediction = (0.7 * base_prediction) + (0.3 * time_factor)
                    
                    # Ensure prediction is within bounds
                    predictions[direction] = max(0.1, min(1.0, prediction))
                else:
                    # If no history, use time pattern
                    predictions[direction] = time_pattern[direction]
                    
        except Exception as e:
            self.logger.error(f"Error in traffic prediction: {e}")
            # Fallback to time patterns if prediction fails
            predictions = time_pattern
        
        return predictions
    
    def get_congestion_probability(self, direction, prediction):
        """Calculate probability of congestion
        
        Args:
            direction: Traffic direction
            prediction: Predicted traffic density
            
        Returns:
            float: Probability of congestion (0-1)
        """
        try:
            # Consider historical patterns
            recent_history = list(self.traffic_history[direction])
            if len(recent_history) >= 5:
                # Calculate volatility
                volatility = np.std(recent_history[-5:])
                
                # Higher prediction + higher volatility = higher congestion probability
                base_probability = prediction * (1 + volatility)
                
                # Consider time patterns
                time_pattern = self._get_time_pattern()[direction]
                
                # Combine factors (60% prediction, 40% time pattern)
                probability = (0.6 * base_probability) + (0.4 * time_pattern)
                
                return max(0.0, min(1.0, probability))
            else:
                # If insufficient history, base on prediction only
                return prediction
        except Exception as e:
            self.logger.error(f"Error calculating congestion probability: {e}")
            return prediction  # Fallback to raw prediction
    
    def get_recommended_green_time(self, direction, prediction):
        """Calculate recommended green light duration
        
        Args:
            direction: Traffic direction
            prediction: Predicted traffic density
            
        Returns:
            int: Recommended green light duration in seconds
        """
        try:
            base_time = 30  # Base green light duration
            congestion_prob = self.get_congestion_probability(direction, prediction)
            
            # Adjust time based on congestion probability
            if congestion_prob > 0.8:
                return 60  # Maximum time for high congestion
            elif congestion_prob > 0.6:
                return 45  # Extended time for moderate congestion
            elif congestion_prob > 0.4:
                return base_time
            else:
                return max(15, int(base_time * congestion_prob))  # Minimum 15 seconds
                
        except Exception as e:
            self.logger.error(f"Error calculating green time: {e}")
            return 30  # Return default time on error 